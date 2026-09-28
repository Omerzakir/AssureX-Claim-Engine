from __future__ import annotations
from datetime import date, datetime, timedelta
from .policies import WarrantyPolicy, VERSION


def _date(v):
    if not v: return None
    if isinstance(v, datetime): return v.date()
    if isinstance(v, date): return v
    return datetime.fromisoformat(str(v).replace('Z','+00:00')).date()


def _finding(code, severity, message, blocking=False):
    return {'code': code, 'severity': severity, 'message': message, 'blocking': blocking}


def validate_warranty(claim: dict, policy: WarrantyPolicy, today: date | None = None):
    today = today or date.today()
    findings, passed, failed, warnings, manual = [], [], [], [], []
    purchase = _date(claim.get('purchase_date'))
    fault = _date(claim.get('fault_date'))
    submitted = _date(claim.get('claim_submission_date') or claim.get('submission_date'))
    months = policy.effective_months(bool(claim.get('extended_warranty')))

    if not purchase:
        f=_finding('WARRANTY_PURCHASE_DATE_MISSING','error','Purchase date is required.',True); findings.append(f); failed.append(f['code'])
    else:
        # Month arithmetic without third-party dependencies.
        y = purchase.year + (purchase.month - 1 + months) // 12
        m = (purchase.month - 1 + months) % 12 + 1
        import calendar
        expiry = date(y, m, min(purchase.day, calendar.monthrange(y,m)[1]))
        expiry_with_grace = expiry + timedelta(days=max(0, policy.grace_period_days))
        if today <= expiry_with_grace:
            passed.append('WARRANTY_ACTIVE')
        else:
            f=_finding('WARRANTY_EXPIRED','error',f'Warranty expired on {expiry.isoformat()}.',True); findings.append(f); failed.append(f['code'])

    damage = str(claim.get('damage_type') or claim.get('fault_category') or '').strip()
    if damage and damage in policy.excluded_faults:
        f=_finding('EXCLUDED_DAMAGE','error',f'Damage type {damage!r} is excluded by policy.',True); findings.append(f); failed.append(f['code'])
    elif damage:
        passed.append('DAMAGE_COVERED')

    if policy.reporting_deadline_days is not None and fault and submitted:
        delay=(submitted-fault).days
        if delay < 0:
            f=_finding('INVALID_REPORTING_DATES','error','Claim submission precedes the fault date.',True); findings.append(f); failed.append(f['code'])
        elif delay > policy.reporting_deadline_days:
            f=_finding('REPORTING_DEADLINE','warning',f'Claim was submitted {delay} days after the fault; policy allows {policy.reporting_deadline_days} days.',False); findings.append(f); warnings.append(f['code'])
        else: passed.append('REPORTING_DEADLINE')

    unauthorized=bool(claim.get('previous_repair_unauthorized') or claim.get('repair_unauthorized') or claim.get('unauthorized_repair'))
    repair_count=int(claim.get('repair_history_count') or claim.get('repair_count') or 0)
    if unauthorized and repair_count >= policy.unauthorized_repair_count_threshold:
        sev='error' if policy.unauthorized_repair_blocks else 'warning'
        f=_finding('UNAUTHORIZED_REPAIR_THRESHOLD',sev,'Unauthorized repair count meets the policy threshold.',policy.unauthorized_repair_blocks); findings.append(f)
        (failed if policy.unauthorized_repair_blocks else warnings).append(f['code'])
    elif unauthorized:
        f=_finding('UNAUTHORIZED_REPAIR','warning','Unauthorized repair is present.',False); findings.append(f); warnings.append(f['code'])
    else: passed.append('REPAIR_HISTORY')

    if claim.get('serial_number_match') is False:
        f=_finding('SERIAL_MISMATCH','manual_review','Entered serial number does not match the registered record.',False); findings.append(f); manual.append(f['code'])
    else: passed.append('SERIAL_MATCH')

    missing=int(claim.get('missing_documents_count') or 0)
    if missing >= policy.missing_documents_manual_review_threshold:
        f=_finding('MISSING_DOCUMENTS','manual_review',f'{missing} required document(s) are missing.',False); findings.append(f); manual.append(f['code'])
    elif missing == 1:
        f=_finding('MISSING_DOCUMENT','warning','One required document is missing.',False); findings.append(f); warnings.append(f['code'])
    else: passed.append('REQUIRED_DOCUMENTS')

    contradictions=list(claim.get('contradictions') or [])
    if not contradictions:
        if purchase and fault and fault < purchase: contradictions.append('fault_before_purchase')
        if purchase and submitted and submitted < purchase: contradictions.append('claim_before_purchase')
        if fault and submitted and fault > submitted: contradictions.append('fault_after_submission')
    if contradictions:
        f=_finding('CONTRADICTION','manual_review','Contradictory claim dates or facts were detected.',False); findings.append(f); manual.append(f['code'])
    else: passed.append('CONTRADICTION_CHECK')

    if claim.get('duplicate_claim_flag') or claim.get('duplicate_document_flag'):
        f=_finding('DUPLICATE','manual_review','A duplicate claim or document was detected.',False); findings.append(f); manual.append(f['code'])
    else: passed.append('DUPLICATE_CHECK')

    if policy.authorized_service_centre_required and claim.get('service_centre_authorized') is False:
        f=_finding('UNAUTHORIZED_SERVICE_CENTRE','manual_review','Policy requires an authorized service centre.',False); findings.append(f); manual.append(f['code'])

    return {
        'engine_version': VERSION,
        'policy_id': policy.policy_id,
        'policy_name': policy.name,
        'effective_warranty_months': months,
        'rules_passed': passed,
        'rules_failed': failed,
        'warnings': warnings,
        'manual_review_flags': manual,
        'findings': findings,
        'valid': not failed and not manual,
        'requires_manual_review': bool(manual),
        'blocked': bool(failed),
    }
