from __future__ import annotations
from ..fraud.detection import run_fraud_checks


def attach_fraud_checks(warranty_result: dict, claim: dict, product: dict | None = None,
                        extracted: dict | None = None, existing_claims=None,
                        document: dict | None = None, existing_documents=None,
                        required_documents=None):
    """Attach XXVII–XXXI deterministic checks to an existing XXV/XXVI result."""
    fraud = run_fraud_checks(
        claim, product, extracted, existing_claims, document,
        existing_documents, required_documents
    )
    result = dict(warranty_result or {})
    result['fraud_checks'] = fraud
    result['serial_verification'] = fraud['serial_verification']
    result['contradictions'] = fraud['contradictions']
    result['missing_documents'] = fraud['missing_documents']
    result['duplicate_claim'] = fraud['duplicate_claim']
    result['duplicate_document'] = fraud['duplicate_document']
    result['fraud_findings'] = fraud['findings']
    # Preserve warranty rule decisions while making the combined workflow state explicit.
    result['requires_manual_review'] = bool(result.get('requires_manual_review') or fraud['requires_manual_review'])
    result['blocked'] = bool(result.get('blocked') or fraud['blocked'])
    return result
