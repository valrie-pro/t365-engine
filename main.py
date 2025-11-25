from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import csv
import io

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


# ─────────────────────────────────────────
# Logique d'analyse bancaire (CSV canonical)
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

        analysis = analyze_bank_csv(req.csvText, req.intentions)

        return {
            "analysisId": None,
            "meta": {
                "fileKind": req.fileKind,
                "fileName": req.fileName,
                "createdAt": datetime.utcnow().isoformat() + "Z",
            },
            "analysis": analysis,
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
