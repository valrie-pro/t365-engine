from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import csv
import io
import os
import json
import logging
import base64
import re

from openai import OpenAI
from pypdf import PdfReader

# ─────────────────────────────────────────
# Config & client OpenAI
# ─────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI()  # lit OPENAI_API_KEY dans l'environnement

app = FastAPI(title="T365 Engine - Minimal")


# ─────────────────────────────────────────
# Models
# ─────────────────────────────────────────

class Intentions(BaseModel):
    q: Optional[Dict[str, Any]] = None


class AnalyzeRequest(BaseModel):
    docType: str = "bank"
    fileKind: str  # "csv" ou "pdf_base64"
    csvText: Optional[str] = None
    fileBase64: Optional[str] = None  # pour les PDF encodés
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


# ─────────────────────────────────────────
# Helpers CSV canonical
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
            amount_str = str(row.get("amount", "")).replace(" ", "").replace(",", ".")
            amount = float(amount_str)
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


# ─────────────────────────────────────────
# Analyse bancaire (chiffres)
# ─────────────────────────────────────────

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
        f"votre taux d’endettement est de {round(debt_ratio * 100)} %, "
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
# PDF → texte → CSV canonical
# ─────────────────────────────────────────

DATE_REGEX = re.compile(r"^(\d{2}/\d{2}/\d{4})\b")
AMOUNT_REGEX = re.compile(r"(-?\d[\d\s]*[.,]\d{2})\s*$")


def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    """
    Extrait le texte brut d’un PDF (texte, pas de scan image).
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    all_text_lines: List[str] = []

    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if not t:
            continue
        # On normalise les sauts de ligne
        lines = t.replace("\r\n", "\n").split("\n")
        all_text_lines.extend([line.rstrip() for line in lines])

    return "\n".join(all_text_lines)


def pdf_text_to_canonical_csv(text: str) -> str:
    """
    Très V1 : on cherche des lignes qui ressemblent à :
    '01/11/2025 ... LIBELLÉ ... -45,67'
    et on fabrique un CSV 'date,label,amount'.
    """
    lines = text.split("\n")
    rows: List[Dict[str, str]] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # 1) Date au début
        m_date = DATE_REGEX.match(line)
        if not m_date:
            continue
        date_str = m_date.group(1)

        # 2) Montant à la fin
        m_amount = AMOUNT_REGEX.search(line)
        if not m_amount:
            continue
        amount_raw = m_amount.group(1)

        # 3) Label = ce qu’il y a entre la date et le montant
        middle = line[m_date.end(): m_amount.start()].strip()
        # on nettoie un peu le label
        label = re.sub(r"\s{2,}", " ", middle)

        # 4) Normalisation du montant
        amount_clean = amount_raw.replace(" ", "").replace(",", ".")
        try:
            amount_val = float(amount_clean)
        except ValueError:
            continue

        rows.append(
            {
                "date": date_str,
                "label": label,
                "amount": f"{amount_val:.2f}",
            }
        )

    if not rows:
        raise ValueError("Impossible d'extraire des lignes de mouvements depuis ce PDF.")

    # Construction du CSV canonical
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["date", "label", "amount"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

    return output.getvalue()


def pdf_base64_to_canonical_csv(b64: str) -> str:
    """
    Décode un PDF en base64, extrait le texte, puis le convertit en CSV canonical.
    """
    try:
        pdf_bytes = base64.b64decode(b64)
    except Exception as e:
        raise ValueError("PDF en base64 invalide.") from e

    text = pdf_bytes_to_text(pdf_bytes)
    if not text.strip():
        raise ValueError("Aucun texte exploitable n'a été trouvé dans le PDF.")

    return pdf_text_to_canonical_csv(text)


# ─────────────────────────────────────────
# Helpers IA : intentions + prompt narration FREE
# ─────────────────────────────────────────

def extract_intentions(intentions: Optional[Intentions]) -> Dict[str, Optional[str]]:
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

Tu dois produire UN TEXTE UNIQUE en 4 blocs courts, dans cet ordre, SANS TITRE, SANS PUCE, SANS NUMÉROTATION,
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

4) Bloc "aller plus loin et limites de l’outil"
→ Tu expliques en quelques phrases ce que pourrait apporter un rapport plus détaillé
(vision plus fine des catégories de dépenses, des charges récurrentes, de la marge de manœuvre, etc.).
→ Tu termines ce paragraphe par une phrase qui invite simplement à approfondir l’analyse,
sans parler de prix, ni d’achat, ni de paiement.

RÈGLES DE STYLE (À RESPECTER STRICTEMENT) :
- Ton = professionnel, clair, cash mais jamais culpabilisant.
- Tu vouvoies l’utilisateur.
- Tu écris EXACTEMENT 4 paragraphes, chacun de 3–5 phrases maximum.
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
- Si elle se dit "rassurée", tu valides seulement si le taux d’endettement ne dépasse pas le seuil classique (~35 %).
  Si le taux d’endettement dépasse ce seuil, tu restes factuel et tu expliques où rester vigilant.

DONNÉES EN ENTRÉE (JSON) :
{donnees}

FORMAT DE SORTIE :
- Tu renvoies uniquement le texte des 4 paragraphes, séparés chacun par UNE LIGNE VIDE.
- Tu n’utilises jamais de listes ni de numérotation dans ta réponse.
"""


def generate_free_narration(
    analysis: Dict[str, Any],
    intentions: Optional[Intentions] = None,
) -> Optional[str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY manquant : narration IA désactivée.")
        return None

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
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es l’IA-conseiller de T365. "
                        "Respecte STRICTEMENT les consignes du prompt utilisateur."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=900,
            temperature=0.5,
        )

        if not resp.choices or not resp.choices[0].message:
            logger.warning("Réponse IA sans choix exploitable.")
            return None

        narration_text = (resp.choices[0].message.content or "").strip()
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
        # 1) CSV classique
        if req.fileKind == "csv":
            if not req.csvText:
                raise HTTPException(
                    status_code=400,
                    detail=ErrorPayload(
                        code="MISSING_FIELD",
                        message="Le champ 'csvText' est obligatoire pour fileKind='csv'.",
                    ).dict(),
                )

            canonical_csv = req.csvText
            analysis = analyze_bank_csv(canonical_csv, req.intentions)
            narration = generate_free_narration(analysis, req.intentions)

        # 2) PDF encodé en base64
        elif req.fileKind == "pdf_base64":
            if not req.fileBase64:
                raise HTTPException(
                    status_code=400,
                    detail=ErrorPayload(
                        code="MISSING_FIELD",
                        message="Le champ 'fileBase64' est obligatoire pour fileKind='pdf_base64'.",
                    ).dict(),
                )

            canonical_csv = pdf_base64_to_canonical_csv(req.fileBase64)
            analysis = analyze_bank_csv(canonical_csv, req.intentions)
            narration = generate_free_narration(analysis, req.intentions)

        else:
            raise HTTPException(
                status_code=400,
                detail=ErrorPayload(
                    code="UNSUPPORTED_FILE_KIND",
                    message=f"fileKind='{req.fileKind}' n'est pas supporté.",
                ).dict(),
            )

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
    except ValueError as e:
        # Erreurs “fonctionnelles” (PDF illisible, pas de lignes, etc.)
        logger.warning(f"Erreur fonctionnelle /analyze : {e}")
        raise HTTPException(
            status_code=400,
            detail=ErrorPayload(
                code="BAD_DOCUMENT_FORMAT",
                message=str(e),
            ).dict(),
        )
    except Exception as e:
        logger.exception("Unexpected /analyze error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=ErrorPayload(
                code="INTERNAL_ERROR",
                message="Une erreur interne est survenue dans le moteur T365.",
            ).dict(),
        )
