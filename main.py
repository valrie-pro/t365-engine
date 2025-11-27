from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import csv
import io
import requests
import os
import json
import logging

from openai import OpenAI

# ─────────────────────────────────────────
# Config & client OpenAI
# ─────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Le client lit OPENAI_API_KEY dans les variables d’environnement (Render)
client = OpenAI()

app = FastAPI(title="T365 Engine - Minimal")


# ─────────────────────────────────────────
# Models
# ─────────────────────────────────────────

class Intentions(BaseModel):
    # On garde ça générique pour l’instant
    q: Optional[Dict[str, Any]] = None


class AnalyzeRequest(BaseModel):
    docType: str = "bank"
    fileKind: str  # "csv" (on ajoutera pdf plus tard)
    csvText: Optional[str] = None
    fileName: Optional[str] = None
    intentions: Optional[Intentions] = None


class ErrorPayload(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────
# Routes simples
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "T365 Engine up & running"}


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug-openai")
def debug_openai():
    key = os.environ.get("OPENAI_API_KEY")
    # ⚠️ On NE renvoie PAS la clé complète (question de sécurité)
    return {
        "has_key": bool(key),
        "prefix": key[:5] + "..." if key else None,
    }

# ─────────────────────────────────────────
# Analyse bancaire (CSV canonical)
# ─────────────────────────────────────────

def parse_canonical_csv(csv_text: str):
    """
    Attend un CSV avec header 'date,label,amount'
    et renvoie une liste de dicts {date, label, amount}
    """
    csv_text = csv_text.replace("\r\n", "\n").strip()
    if not csv_text:
        return []

    f = io.StringIO(csv_text)
    reader = csv.DictReader(f)

    rows = []
    for row in reader:
        try:
            amount = float(str(row.get("amount", "")).replace(" ", "").replace(",", "."))
        except ValueError:
            continue

        rows.append(
            {
                "date": (row.get("date") or "").strip(),
                "label": (row.get("label") or "").strip(),
                "amount": amount,
            }
        )

    return rows


def analyze_bank_csv(canonical_csv: str, intentions: Optional[Intentions]) -> Dict[str, Any]:
    """
    Version Python simplifiée de ton analyse :
    - calcule revenus, charges, taux
    - construit un score + quelques messages génériques
    """
    rows = parse_canonical_csv(canonical_csv)

    revenus = 0.0
    charges_abs = 0.0

    for r in rows:
        amt = r["amount"]
        if amt > 0:
            revenus += amt
        else:
            charges_abs += abs(amt)

    debt_ratio = charges_abs / revenus if revenus > 0 else 0.0

    # Score très simple : plus le taux est bas, plus le score est haut
    if debt_ratio == 0:
        score = 90
    elif debt_ratio < 0.35:
        score = 80
    elif debt_ratio < 0.45:
        score = 65
    elif debt_ratio < 0.55:
        score = 50
    else:
        score = 35

    anomalies_count = 0
    if debt_ratio >= 0.50:
        anomalies_count += 1

    strengths: List[Dict[str, Any]] = []
    cautions: List[Dict[str, Any]] = []
    interpretation_notes: List[Dict[str, Any]] = []
    anomalies: List[Dict[str, Any]] = []

    # Points forts / attention de base
    if revenus > 0 and debt_ratio < 0.4:
        strengths.append(
            {
                "label": "Niveau de charges globalement maîtrisé par rapport à vos revenus.",
                "detail": "Vous pouvez aller plus loin en suivant vos catégories de dépenses (courses, abonnements, sorties…).",
            }
        )

    if debt_ratio >= 0.45:
        cautions.append(
            {
                "label": "Taux d’endettement dans une zone à surveiller.",
                "detail": "Un allègement de certaines charges récurrentes peut améliorer votre marge de manœuvre.",
            }
        )

    interpretation_notes.append(
        {
            "label": "Ce diagnostic repose uniquement sur les mouvements présents dans le fichier fourni.",
            "detail": "D’autres comptes, crédits ou charges externes ne sont pas visibles dans cette analyse.",
        }
    )

    if debt_ratio >= 0.50:
        anomalies.append(
            {
                "code": "debt_ratio",
                "severity": "high",
                "label": "Taux d’endettement élevé par rapport aux standards classiques.",
                "hint": "Viser un taux ≤ 35 % améliore nettement la perception de votre dossier.",
            }
        )

    summary_line = (
        f"Pour votre projet de compréhension de votre budget, "
        f"votre taux d’endettement est de {round(debt_ratio * 100)}%, "
        f"charges : {round(-charges_abs, 2)} €, revenus : {round(revenus, 2)} €."
    )

    return {
        "kpi": {
            "revenus": revenus,
            "charges": -charges_abs,
            "debtRatio": debt_ratio,
            "score": score,
            "anomaliesCount": anomalies_count,
        },
        "summaryLine": summary_line,
        "strengths": strengths,
        "cautions": cautions,
        "interpretationNotes": interpretation_notes,
        "anomalies": anomalies,
    }


# ─────────────────────────────────────────
# Helpers IA : intentions + prompt narration FREE
# ─────────────────────────────────────────

def extract_intentions(intentions: Optional[Intentions]) -> Dict[str, Optional[str]]:
    """
    Ramène les intentions à un format simple :
    { "goal": ..., "horizon": ..., "feeling": ... }

    Gère le cas où intentions.q contient les vraies valeurs.
    """
    if intentions is None:
        return {"goal": None, "horizon": None, "feeling": None}

    raw = intentions.q or {}

    goal = raw.get("bank_goal") or raw.get("goal")
    horizon = raw.get("bank_horizon") or raw.get("horizon")
    feeling = raw.get("bank_feel") or raw.get("feeling")

    return {
        "goal": goal,
        "horizon": horizon,
        "feeling": feeling,
    }


PROMPT_FREE_NARRATION = """
Tu es l’IA-conseiller interne de T365, un outil d’analyse de relevés bancaires.
Ton rôle est d’expliquer des chiffres financiers personnels en français simple, de manière professionnelle,
directe mais bienveillante.

Tu reçois ci-dessous :
- des indicateurs chiffrés (analysis),
- le contexte utilisateur (intentions).

Tu dois produire UN TEXTE UNIQUE en 3 blocs courts, dans cet ordre, SANS TITRE, SANS PUCE, SANS NUMÉROTATION,
et SANS EMOJIS dans ta réponse :

1) Bloc "lecture globale"
→ Tu expliques ce que montrent les chiffres : niveau de revenus mensuels, niveau de charges,
impression générale du budget (large marge / marge confortable / très serré / déficit),
niveau de taux d’endettement et ce que ça signifie concrètement pour le quotidien.

2) Bloc "risques à surveiller"
→ Tu décris les fragilités concrètes : absence de marge, dépendance à un seul salaire,
frais bancaires élevés, endettement trop important, difficulté probable à faire passer un projet
(crédit, location…) si c’est le cas. Tu adaptes le ton selon l’objectif et l’horizon.

3) Bloc "coach rassurant mais cash"
→ Tu parles comme un conseiller qui veut vraiment aider : direct, sans dramatiser,
mais sans minimiser les enjeux. Tu proposes 2–3 pistes concrètes adaptées
(ex : réduire certaines dépenses récurrentes, sécuriser un matelas, lisser un découvert,
préparer un dossier en plusieurs mois…).

RÈGLES DE STYLE (À RESPECTER STRICTEMENT) :
- Ton = professionnel, clair, cash mais jamais culpabilisant.
- Tu vouvoies l’utilisateur.
- Tu écris EXACTEMENT 3 paragraphes, chacun de 3–5 phrases maximum.
- Tu ne fais AUCUNE liste, AUCUNE puce, AUCUNE numérotation (pas de "1.", "2.", "-", "•") dans ta réponse.
- Tu n’ajoutes PAS de sous-titres ni de texte en majuscules.
- Tu NE cites PAS les champs `goal`, `horizon` ou `feeling` tels quels, tu les traduis en langage naturel.
- Tu mentionnes le score uniquement si ça apporte quelque chose à la compréhension
  (ex : score bas = profil fragile, score élevé = profil solide mais perfectible).

PERSONNALISATION SELON LES INTENTIONS :
- Si l’objectif est un crédit immo / conso ou un dossier de location, tu insistes davantage
  sur la capacité à faire passer un dossier et sur les seuils classiques de taux d’endettement (~35 %).
- Si l’objectif est "mieux comprendre / optimiser mon budget",
  tu parles davantage d’équilibre du quotidien et de marge de manœuvre.
- Si l’horizon est très court ("moins de 3 mois"), tu rappelles qu’il sera difficile
  de tout transformer rapidement et que chaque petite optimisation compte.
- Si l’horizon est long ("plus de 12 mois"), tu insistes sur le fait que c’est le bon moment
  pour lisser les changements.
- Si la personne se dit "inquiète", tu reconnais le stress mais tu donnes un angle d’action concret.
- Si elle se dit "rassurée", tu valides mais tu montres où rester vigilant.

DONNÉES EN ENTRÉE (JSON) :
{donnees}

FORMAT DE SORTIE :
- Tu renvoies uniquement le texte des 3 paragraphes, séparés par UNE LIGNE VIDE.
- Tu n’utilises jamais de listes ni de numérotation dans ta réponse.
- À la toute fin, tu ajoutes UNE SEULE phrase finale qui combine :
  1) une invitation douce à aller plus loin avec un rapport détaillé,
  2) et un rappel que cela ne remplace pas un conseil humain.

Exemple de dernière phrase (à adapter, ne la copie pas mot à mot) :
"Pour aller plus loin, un rapport détaillé vous aidera à poser tout ça noir sur blanc,
mais gardez en tête qu’aucun outil ne remplace l’avis d’un professionnel
(banque, courtier, conseiller financier) sur votre situation complète."
"""

def generate_free_narration(
    analysis: Dict[str, Any],
    intentions: Optional[Intentions] = None,
) -> Optional[str]:
    """
    Prend l'objet d'analyse (kpi, summaryLine, etc.)
    + les intentions utilisateur, et renvoie un texte IA (str) OU None en cas d’échec.
    Utilise l’endpoint chat.completions (structure simple et fiable).
    """

    # 0) Protection : pas de clé = pas d’IA
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY manquant : narration IA désactivée.")
        return None

    # 1) Préparation des données passées au prompt
    kpi = analysis.get("kpi") or {}

    data_for_prompt = {
        "analysis": {
            "revenus": kpi.get("revenus"),
            "charges": kpi.get("charges"),
            "debtRatio": kpi.get("debtRatio"),
            "score": kpi.get("score"),
            "anomaliesCount": kpi.get("anomaliesCount", 0),
            "summaryLine": analysis.get("summaryLine"),
        },
        "intentions": extract_intentions(intentions),
    }

    prompt = PROMPT_FREE_NARRATION.format(
        donnees=json.dumps(data_for_prompt, ensure_ascii=False, indent=2)
    )

    try:
        # 2) Appel OpenAI via chat.completions
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # léger + fiable pour ce cas
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es l’IA-conseiller de T365. "
                        "Respecte STRICTEMENT les consignes du prompt utilisateur."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=700,
            temperature=0.5,
        )

        # 3) Extraction du texte
        if not resp.choices or not resp.choices[0].message:
            logger.warning("Réponse IA sans choix exploitable.")
            return None

        narration_text = resp.choices[0].message.content or ""
        narration_text = narration_text.strip()

        if not narration_text:
            logger.warning("Narration IA vide ou invalide.")
            return None

        logger.info(
            "Narration FREE générée avec succès (longueur=%d caractères).",
            len(narration_text),
        )
        return narration_text

    except Exception as e:
        logger.exception(f"Erreur lors de la génération de la narration FREE : {e}")
        return None

# ─────────────────────────────────────────
# Endpoint principal /analyze
# ─────────────────────────────────────────

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        if req.fileKind != "csv":
            raise HTTPException(
                status_code=400,
                detail=ErrorPayload(
                    code="UNSUPPORTED_FILE_KIND",
                    message="Pour l’instant, seul fileKind='csv' est supporté.",
                ).dict(),
            )

        if not req.csvText:
            raise HTTPException(
                status_code=400,
                detail=ErrorPayload(
                    code="MISSING_FIELD",
                    message="Le champ 'csvText' est obligatoire pour fileKind='csv'.",
                ).dict(),
            )

        # 1) Analyse chiffrée minimale
        analysis = analyze_bank_csv(req.csvText, req.intentions)

        # 2) Narration IA FREE (optionnelle : None si erreur ou pas de clé)
        narration = generate_free_narration(analysis, req.intentions)

        # 3) Réponse envoyée au front (Next)
        return {
            "analysisId": None,
            "meta": {
                "fileKind": req.fileKind,
                "fileName": req.fileName,
                "createdAt": datetime.utcnow().isoformat() + "Z",
            },
            "analysis": analysis,
            "narration": narration,
        }

    except HTTPException:
        raise
    except Exception as e:
        print("Unexpected /analyze error:", e)
        raise HTTPException(
            status_code=500,
            detail=ErrorPayload(
                code="INTERNAL_ERROR",
                message="Une erreur interne est survenue dans le moteur T365.",
            ).dict(),
        )
