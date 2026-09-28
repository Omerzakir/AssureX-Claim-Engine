"""AssureX Claim Engine API.

This is the orchestration layer for requirements I-L. The actual trained Python
and Teachable Machine artifacts are bundled with the project. /api/claims/analyze
executes the complete claim -> models -> rules -> decision pipeline when the runtime
dependencies are installed.
"""
from __future__ import annotations
import base64
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .security import identity, require_role

ROOT = Path(__file__).resolve().parents[1]
BACKEND = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("ASSUREX_PYTHON_MODEL_DIR", ROOT / "model_integration" / "models" / "python"))
TM_DIR = Path(os.getenv("ASSUREX_TM_MODEL_DIR", ROOT / "model_integration" / "models" / "teachable_machine"))
POLICY_FILE = ROOT / "config" / "warranty_policies.json"
MODEL_CONSISTENCY_FILE = ROOT / "config" / "model_consistency.json"
TM_SERVICE = BACKEND.parent / "model_integration" / "backend" / "teachable_machine" / "service.js"

REQUIRED_PY = ["best_model.pkl", "scaler.pkl", "label_encoders.pkl", "feature_columns.pkl", "label_map.pkl"]
REQUIRED_TM = ["model.json", "metadata.json", "weights.bin"]

app = FastAPI(title="AssureX Claim Engine", version="1.0.0")
allowed_origins = [x.strip() for x in os.getenv("ASSUREX_ALLOWED_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])

class ClaimAnalyzeRequest(BaseModel):
    claim: dict[str, Any]
    existing_claims: list[dict[str, Any]] = Field(default_factory=list)
    existing_documents: list[dict[str, Any]] = Field(default_factory=list)
    document: dict[str, Any] | None = None
    document_base64: str | None = None
    required_documents: list[str] = Field(default_factory=lambda: ["purchase_invoice", "warranty_document", "product_photo", "fault_evidence"])
    policy_id: str | None = None
    persist: bool = False

class ReviewRequest(BaseModel):
    claim_id: str
    reviewer_id: str | None = None  # legacy field ignored; actor is derived from verified token
    role: str | None = None  # legacy field ignored
    decision: str
    comments: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_readiness() -> dict[str, Any]:
    py_missing = [x for x in REQUIRED_PY if not (MODEL_DIR / x).is_file()]
    tm_missing = [x for x in REQUIRED_TM if not (TM_DIR / x).is_file()]
    return {
        "python": {"ready": not py_missing, "missing": py_missing, "version": "supplied-trained-model"},
        "teachable_machine": {"ready": not tm_missing, "missing": tm_missing, "version": "supplied-tm-export"},
        "complete": not py_missing and not tm_missing,
    }


def generate_summary_card(claim: dict[str, Any], output: Path) -> Path:
    """XX: create a neutral image card; never include predictions/decisions."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Pillow is required for Claim Summary Card generation") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1200, 900), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=28)
    body_font = ImageFont.load_default(size=20)
    draw.text((50, 35), "AssureX Claim Summary", fill="black", font=title_font)
    product = claim.get("product") or {}
    fault = claim.get("fault") or {}
    repair = claim.get("repairHistory") or {}
    evidence = claim.get("evidence") or {}
    rows = [
        ("Product", product.get("name") or claim.get("productName") or ""),
        ("Category", product.get("category") or claim.get("productCategory") or ""),
        ("Model", product.get("modelNumber") or claim.get("modelNumber") or ""),
        ("Serial Number", product.get("serialNumber") or claim.get("serialNumber") or ""),
        ("Purchase Date", (claim.get("purchase") or {}).get("date") or claim.get("purchase_date") or ""),
        ("Warranty Status", claim.get("warranty_status") or ""),
        ("Fault Type", fault.get("category") or claim.get("damage_type") or claim.get("damageCategory") or ""),
        ("Fault Description", fault.get("description") or claim.get("faultDescription") or ""),
        ("Previous Repair", repair.get("previouslyRepaired") or "No"),
        ("Repair Count", repair.get("repairCount") or claim.get("repair_history_count") or 0),
        ("Documents", len(evidence.get("documentIds") or [])),
        ("Serial Verification", claim.get("serial_number_match") if claim.get("serial_number_match") is not None else "Not evaluated"),
    ]
    y = 100
    for key, value in rows:
        draw.text((55, y), f"{key}: {value}", fill="black", font=body_font)
        y += 58
    image.save(output, "PNG")
    return output


def run_python(claim: dict[str, Any]) -> dict[str, Any]:
    from .model_integration_bridge import P
    from model.python_classifier import PythonClassifier
    return PythonClassifier(MODEL_DIR).predict(claim)


def run_tm(image_path: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["ASSUREX_TM_MODEL_DIR"] = str(TM_DIR)
    proc = subprocess.run(["node", str(TM_SERVICE)], input=json.dumps({"image_path": str(image_path)}) + "\n", text=True, capture_output=True, env=env, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "Teachable Machine runtime failed")
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    result = json.loads(line)
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "Teachable Machine inference failed")
    return result["result"]


def load_model_consistency_config() -> dict[str, Any]:
    if not MODEL_CONSISTENCY_FILE.is_file():
        return {}
    with MODEL_CONSISTENCY_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def run_rules(claim: dict[str, Any], req: ClaimAnalyzeRequest) -> dict[str, Any]:
    from .warranty.policies import PolicyStore
    from .warranty.rules import validate_warranty
    from .warranty.integration import attach_fraud_checks
    store = PolicyStore(POLICY_FILE)
    policy = store.get(req.policy_id)
    warranty = validate_warranty(claim, policy)
    return attach_fraud_checks(
        warranty, claim, claim.get("product") or {}, claim.get("extracted") or {},
        req.existing_claims, req.document, req.existing_documents, req.required_documents
    )


def final_decision(python_result: dict[str, Any], tm_result: dict[str, Any], comparison: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    from .decision.workflow import decide_claim, explain_decision, generate_claim_summary, prepare_claim
    decision = decide_claim(python_result, tm_result, comparison, rules, rules.get("fraud_checks") or rules)
    # Normalize to the specification's canonical vocabulary.
    mapping = {"Approve": "Likely Valid", "Reject": "Likely Invalid", "Manual Review": "Manual Review Required"}
    decision["decision"] = mapping.get(decision["decision"], decision["decision"])
    decision["finalDecision"] = decision["decision"]
    explanation = explain_decision(decision, rules, rules.get("fraud_checks") or rules, comparison)
    preparation = prepare_claim({}, rules, rules.get("fraud_checks") or rules)
    return {"decision": decision, "explanation": explanation, "preparation": preparation}


@app.get("/health")
def health():
    return {"status": "ok", "service": "AssureX Claim Engine", "model_readiness": model_readiness()}

@app.get("/api/models/readiness")
def readiness():
    return model_readiness()

@app.post("/api/claims/analyze")
def analyze(req: ClaimAnalyzeRequest, actor=Depends(identity)):
    readiness_state = model_readiness()
    if not readiness_state["complete"]:
        raise HTTPException(status_code=503, detail={
            "code": "MODEL_ARTIFACTS_NOT_INSTALLED",
            "message": "Required trained model artifacts are unavailable.",
            "model_readiness": readiness_state,
        })
    claim = dict(req.claim)
    owner = claim.get("userId") or claim.get("user_id")
    if actor["role"] in ("user", "employee") and owner != actor["uid"]:
        raise HTTPException(403, detail="You may analyze only your own claims.")
    claim["claim_submission_date"] = claim.get("claim_submission_date") or utc_now()
    rules = run_rules(claim, req)
    # Prevent the classifier silently interpreting absent verification results as
    # successful serial/document/duplicate checks. A manual-review queue is safer.
    required_verified = (
        "serial_number_match", "missing_documents_count", "fault_to_claim_delay_days",
        "contradiction_flag", "duplicate_claim_flag", "previous_repair_unauthorized"
    )
    absent = [key for key in required_verified if claim.get(key) is None]
    if absent:
        raise HTTPException(status_code=422, detail={
            "code": "VERIFIED_FEATURES_REQUIRED",
            "message": "Verified claim features are incomplete. Route this claim for manual review instead of inferring a model result.",
            "missing": absent,
        })
    with tempfile.TemporaryDirectory(prefix="assurex-card-") as td:
        card = generate_summary_card(claim, Path(td) / "claim-summary.png")
        python_result = run_python(claim)
        tm_result = run_tm(card)
    from .model_integration_bridge import P
    from model.comparison import compare_predictions
    comparison = compare_predictions(python_result, tm_result, load_model_consistency_config())
    decision = final_decision(python_result, tm_result, comparison, rules)
    result = {
        "analysisVersion": "1.0.0",
        "analyzedAt": utc_now(),
        "claimId": claim.get("claim_id") or claim.get("claimId"),
        "claimSummary": {"generated": True, "excludesPredictions": True, "excludesDecision": True},
        "pythonModel": python_result,
        "teachableMachine": tm_result,
        "comparison": comparison,
        "warrantyRules": rules,
        "decision": decision["decision"],
        "decisionExplanation": decision["explanation"],
        "claimPreparation": decision["preparation"],
        "modelVersions": {"python": python_result.get("model_version"), "teachableMachine": tm_result.get("model_version")},
    }
    return result

@app.post("/api/claims/review")
def review(req: ReviewRequest, actor=Depends(require_role("reviewer", "admin"))):
    """Server-validated review and audit record; never trust caller's role or reviewer ID."""
    from .decision.workflow import create_review_record, apply_review_override
    from firebase_admin import firestore
    if not req.comments.strip():
        raise HTTPException(422, detail="A review comment is required.")
    if req.decision not in ("Approve", "Reject", "Manual Review"):
        raise HTTPException(422, detail="Invalid review decision.")
    db = actor["db"]
    ref = db.collection("claims").document(req.claim_id)
    audit_ref = db.collection("audit_logs").document()
    notification_ref = db.collection("notifications").document()
    transaction = db.transaction()

    @firestore.transactional
    def save_review(tx):
        snap = ref.get(transaction=tx)
        if not snap.exists:
            raise HTTPException(404, detail="Claim not found.")
        previous = snap.to_dict() or {}
        if actor["role"] == "reviewer" and previous.get("predictedDecision") not in ("Manual Review", "Manual Review Required"):
            raise HTTPException(403, detail="Only manual-review claims can be reviewed by reviewers.")
        record = create_review_record(req.claim_id, actor["uid"], req.decision, req.comments)
        result = apply_review_override({"decision": previous.get("predictedDecision") or previous.get("finalDecision"), "claimId": req.claim_id}, record, actor["role"])
        now = firestore.SERVER_TIMESTAMP
        owner = previous.get("userId") or previous.get("user_id")
        tx.update(ref, {"reviewDecision": req.decision, "reviewerDecision": req.decision,
            "reviewComments": req.comments.strip(), "reviewedBy": actor["uid"],
            "reviewedByEmail": actor.get("email"), "reviewedAt": now,
            "finalDecision": result["canonicalReviewDecision"],
            "status": "Approved" if req.decision == "Approve" else "Rejected" if req.decision == "Reject" else "Manual Review",
            "updatedAt": now})
        tx.set(audit_ref, {"userId": actor["uid"], "actorRole": actor["role"],
            "action": "claim_review", "claimId": req.claim_id, "previousDecision": previous.get("finalDecision"),
            "aiDecision": previous.get("predictedDecision"), "reviewDecision": req.decision,
            "comments": req.comments.strip(), "timestamp": now})
        if owner:
            tx.set(notification_ref, {"userId": owner, "type": "claim_review", "claimId": req.claim_id,
                "title": "Claim review updated", "message": "Your claim has a new review decision.",
                "read": False, "createdAt": now})
        return result
    return save_review(transaction)

@app.get("/api/requirements/status")
def requirements_status():
    return {
        "trained_model_artifacts_bundled": model_readiness()["complete"],
        "excluded_artifacts": [],
        "pipeline_endpoint": "/api/claims/analyze",
        "review_endpoint": "/api/claims/review",
        "model_readiness": model_readiness(),
    }
