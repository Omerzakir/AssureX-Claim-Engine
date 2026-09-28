"""AssureX requirements XXXII-XXXVII decision workflow.

No new ML inference is performed here. The module consumes the outputs from the
existing Python/TM models and XXV-XXXI rule engine.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

ALLOWED_DECISIONS = {"Likely Valid", "Likely Invalid", "Manual Review Required", "Approve", "Reject", "Manual Review"}


def _s(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _label(model: Optional[Dict[str, Any]]) -> str:
    return _s((model or {}).get("label") or (model or {}).get("prediction"))


def _confidence(model: Optional[Dict[str, Any]]) -> Optional[float]:
    try:
        return float((model or {}).get("confidence"))
    except (TypeError, ValueError):
        return None


def generate_claim_summary(claim: Dict[str, Any]) -> Dict[str, Any]:
    """XXXII: factual claim summary only; never inserts model decisions."""
    product = claim.get("product") or {}
    fault = claim.get("fault") or {}
    repair = claim.get("repairHistory") or claim.get("repair_history") or {}
    evidence = claim.get("evidence") or {}
    parts = []
    name = _s(product.get("name") or claim.get("productName"))
    if name: parts.append(f"Product: {name}")
    serial = _s(product.get("serialNumber") or claim.get("serialNumber"))
    if serial: parts.append(f"Serial number: {serial}")
    category = _s(product.get("category") or claim.get("productCategory"))
    if category: parts.append(f"Category: {category}")
    damage = _s(fault.get("category") or claim.get("damageCategory") or claim.get("damage"))
    if damage: parts.append(f"Reported fault/damage: {damage}")
    desc = _s(fault.get("description") or claim.get("faultDescription"))
    if desc: parts.append(f"Description: {desc}")
    repaired = repair.get("previouslyRepaired")
    if repaired not in (None, "", False, "no", "No"):
        parts.append(f"Previous repair: {_s(repair.get('repairCount') or 'reported')}")
    doc_ids = evidence.get("documentIds") or []
    parts.append(f"Linked evidence documents: {len(doc_ids) if isinstance(doc_ids, list) else 0}")
    return {
        "version": "32.0.0",
        "type": "factual_claim_summary",
        "text": " ".join(parts) if parts else "Claim information is available for review.",
        "fields_used": [p.split(":", 1)[0] for p in parts],
        "excludes_predictions": True,
        "excludes_decision": True,
    }


def prepare_claim(claim: Dict[str, Any], warranty: Dict[str, Any], fraud: Dict[str, Any]) -> Dict[str, Any]:
    """XXXIII: deterministic preparation checklist for reviewer/decision engine."""
    missing = warranty.get("missing_documents") or fraud.get("missing_documents") or []
    contradictions = warranty.get("contradictions") or fraud.get("contradictions") or []
    serial = fraud.get("serial_verification") or {}
    duplicate = fraud.get("duplicate_claim") or {}
    docdup = fraud.get("duplicate_document") or {}
    return {
        "version": "33.0.0",
        "ready": not missing and not contradictions and not duplicate.get("is_duplicate") and not docdup.get("is_duplicate"),
        "checklist": {
            "required_documents_present": not bool(missing),
            "contradictions_clear": not bool(contradictions),
            "serial_verified": serial.get("status") in (None, "MATCH", "VERIFIED"),
            "no_duplicate_claim": not bool(duplicate.get("is_duplicate")),
            "no_duplicate_document": not bool(docdup.get("is_duplicate")),
            "warranty_evaluated": bool(warranty),
        },
        "missing_items": list(missing) if isinstance(missing, list) else [missing] if missing else [],
        "review_notes": [
            "Resolve every BLOCKED or MANUAL_REVIEW rule before approval." if (missing or contradictions) else "No outstanding rule finding was supplied."
        ],
    }


def decide_claim(python_model: Dict[str, Any], teachable_machine: Dict[str, Any], comparison: Dict[str, Any], warranty: Dict[str, Any], fraud: Dict[str, Any]) -> Dict[str, Any]:
    """XXXIV: deterministic SRS final decision from dual-model + rule evidence.

    The decision never invents model outputs. It consumes the Python and real
    Teachable Machine results, their comparison/confidence delta, and the
    warranty/fraud findings produced by XXV-XXXI.
    """
    warranty = warranty or {}
    fraud = fraud or {}
    comparison = comparison or {}

    hard = list(warranty.get("hard_block_reasons") or []) + list(fraud.get("hard_block_reasons") or [])
    hard += list(warranty.get("rules_failed") or []) + list(fraud.get("rules_failed") or [])
    manual = list(warranty.get("manual_review_flags") or []) + list(fraud.get("manual_review_flags") or [])

    missing = list(warranty.get("missing_documents") or []) + list(fraud.get("missing_documents") or [])
    contradictions = list(warranty.get("contradictions") or []) + list(fraud.get("contradictions") or [])
    duplicate_claim = warranty.get("duplicate_claim") or fraud.get("duplicate_claim") or {}
    duplicate_document = warranty.get("duplicate_document") or fraud.get("duplicate_document") or {}
    if duplicate_claim.get("is_duplicate"):
        manual.append("DUPLICATE_CLAIM")
    if duplicate_document.get("is_duplicate"):
        manual.append("DUPLICATE_DOCUMENT")
    if missing:
        manual.append("MISSING_DOCUMENTS")
    if contradictions:
        manual.append("CONTRADICTIONS")

    status = _s(comparison.get("consistency_status") or comparison.get("status") or comparison.get("consistency"))
    py_label, tm_label = _label(python_model), _label(teachable_machine)
    py_conf, tm_conf = _confidence(python_model), _confidence(teachable_machine)
    confidence_difference = comparison.get("confidence_difference")
    try:
        confidence_difference = float(confidence_difference) if confidence_difference is not None else None
    except (TypeError, ValueError):
        confidence_difference = None

    min_confidence = float(comparison.get("minimum_confidence", 0.55))
    low_confidence = (py_conf is not None and py_conf < min_confidence) or (tm_conf is not None and tm_conf < min_confidence)
    model_disagreement = py_label and tm_label and py_label != tm_label
    consistency_manual = status in {"Weak Match", "Model Disagreement", "Uncertain Result", "Inconsistent", "Uncertain"}

    evidence = {
        "python_prediction": py_label,
        "python_confidence": py_conf,
        "teachable_machine_prediction": tm_label,
        "teachable_machine_confidence": tm_conf,
        "predictions_match": comparison.get("classes_match", py_label == tm_label),
        "confidence_difference": confidence_difference,
        "consistency_status": status or "Uncertain Result",
        "minimum_confidence": min_confidence,
        "low_confidence": low_confidence,
        "warranty_rules_failed": list(warranty.get("rules_failed") or []),
        "missing_documents": missing,
        "contradictions": contradictions,
        "duplicate_claim": bool(duplicate_claim.get("is_duplicate")),
        "duplicate_document": bool(duplicate_document.get("is_duplicate")),
        "manual_review_flags": manual,
    }

    # Hard business-rule failures are invalid; evidence/rule ambiguity routes to review.
    if hard or warranty.get("blocked") is True or fraud.get("blocked") is True:
        decision = "Likely Invalid"
        basis = "A blocking warranty or fraud/business-rule condition was detected."
        automatic = True
    elif manual or consistency_manual or model_disagreement or low_confidence:
        decision = "Manual Review Required"
        basis = "The claim contains unresolved evidence, rule, confidence, or model-consistency conditions requiring human review."
        automatic = False
    elif py_label == tm_label == "Valid Claim" and py_conf is not None and tm_conf is not None and py_conf >= min_confidence and tm_conf >= min_confidence:
        decision = "Likely Valid"
        basis = "Both models independently classify the claim as valid with sufficient confidence and no blocking rule was detected."
        automatic = True
    elif py_label == tm_label == "Invalid Claim" and py_conf is not None and tm_conf is not None and py_conf >= min_confidence and tm_conf >= min_confidence:
        decision = "Likely Invalid"
        basis = "Both models independently classify the claim as invalid with sufficient confidence and no unresolved manual-review condition was detected."
        automatic = True
    else:
        decision = "Manual Review Required"
        basis = "The available model evidence is insufficient for a safe automatic final decision."
        automatic = False

    return {
        "version": "34.1.0",
        "decision": decision,
        "finalDecision": decision,
        "basis": basis,
        "automatic": automatic,
        "hard_block_count": len(hard),
        "manual_review_count": len(manual),
        "model_labels": {"python": py_label, "teachable_machine": tm_label},
        "model_confidences": {"python": py_conf, "teachable_machine": tm_conf},
        "rule_blockers": hard,
        "manual_review_flags": manual,
        "decision_evidence": evidence,
    }


def explain_decision(decision: Dict[str, Any], warranty: Dict[str, Any], fraud: Dict[str, Any], comparison: Dict[str, Any]) -> Dict[str, Any]:
    """XXXV: auditable explanation tied to concrete inputs."""
    reasons = []
    if decision.get("rule_blockers"): reasons.extend([f"Rule blocker: {x}" for x in decision["rule_blockers"]])
    if decision.get("manual_review_flags"): reasons.extend([f"Manual-review flag: {x}" for x in decision["manual_review_flags"]])
    if comparison.get("comparison"): reasons.append(f"Model comparison: {comparison['comparison']}")
    if comparison.get("confidence_difference") is not None: reasons.append(f"Confidence difference: {comparison['confidence_difference']}")
    if not reasons: reasons.append("No blocking or manual-review rule finding was supplied; decision follows the configured model outputs.")
    return {"version":"35.0.0", "decision":decision.get("decision"), "reasons":reasons, "source_sections":["models","warranty_rules","fraud_checks","model_comparison"], "generatedAt":datetime.now(timezone.utc).isoformat()}


def create_review_record(claim_id: str, reviewer_id: str, decision: str, comments: str = "") -> Dict[str, Any]:
    """XXXVI: canonical manual-review record."""
    if decision not in ALLOWED_DECISIONS: raise ValueError("Invalid review decision")
    if not _s(reviewer_id): raise ValueError("reviewer_id is required")
    return {"claimId":claim_id,"reviewerId":reviewer_id,"reviewDecision":decision,"reviewComments":_s(comments),"reviewStatus":"completed","reviewedAt":datetime.now(timezone.utc).isoformat(),"workflowVersion":"36.0.0"}


def apply_review_override(predicted: Dict[str, Any], review: Dict[str, Any], role: str) -> Dict[str, Any]:
    """XXXVII: only reviewer/admin may override a prediction; preserve original."""
    if role not in ("reviewer", "admin"): raise PermissionError("Only reviewer or admin can override a decision")
    review_decision = _s(review.get("reviewDecision"))
    if review_decision not in ALLOWED_DECISIONS: raise ValueError("Invalid review decision")
    original = predicted.get("decision") or predicted.get("finalDecision") or "Manual Review"
    return {
        **predicted,
        "finalDecision": review_decision,
        "reviewOverride": True,
        "originalDecision": original,
        "review": review,
        "canonicalReviewDecision": {"Approve":"Likely Valid","Reject":"Likely Invalid","Manual Review":"Manual Review Required"}.get(review_decision, review_decision),
        "decisionSource": "human_review_override",
        "workflowVersion": "37.0.0",
    }
