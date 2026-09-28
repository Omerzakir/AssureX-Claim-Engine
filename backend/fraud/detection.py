from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import re
from typing import Any, Iterable

VERSION = '31.0.0'


def norm(v: Any) -> str:
    return re.sub(r'[^A-Z0-9]', '', str(v or '').upper())


def _date(v):
    if not v:
        return None
    if isinstance(v, datetime): return v.date()
    if isinstance(v, date): return v
    return datetime.fromisoformat(str(v).replace('Z', '+00:00')).date()


def _finding(code, severity, message, blocking=False, details=None):
    return {'code': code, 'severity': severity, 'message': message, 'blocking': blocking, 'details': details or {}}


def verify_serial_number(claim: dict, product: dict | None = None, extracted: dict | None = None):
    product = product or {}
    extracted = extracted or {}
    registered = product.get('serial_number') or product.get('serialNumber') or claim.get('registered_serial_number')
    entered = claim.get('serial_number') or claim.get('serialNumber')
    extracted_serial = extracted.get('serial_number') or extracted.get('serialNumber')
    values = {'registered': registered, 'entered': entered, 'extracted': extracted_serial}
    findings = []
    known = [(k, norm(v)) for k, v in values.items() if v]
    if not registered:
        return {'status':'REVIEW', 'match':None, 'findings':[_finding('SERIAL_REGISTERED_MISSING','manual_review','Registered product serial number is unavailable.')], 'values':values}
    mismatches = [(a,b) for i,(a,x) in enumerate(known) for b,y in known[i+1:] if x != y]
    if mismatches:
        findings.append(_finding('SERIAL_MISMATCH','manual_review','Serial-number values do not agree.', details={'mismatches':mismatches, 'normalized':{k:v for k,v in known}}))
        return {'status':'MISMATCH', 'match':False, 'findings':findings, 'values':values}
    if len(known) == 1:
        findings.append(_finding('SERIAL_UNCORROBORATED','manual_review','Only one serial-number source is available for verification.'))
        return {'status':'REVIEW', 'match':None, 'findings':findings, 'values':values}
    return {'status':'MATCH', 'match':True, 'findings':[], 'values':values}


def detect_contradictions(claim: dict, product: dict | None = None, extracted: dict | None = None):
    product = product or {}; extracted = extracted or {}
    findings=[]
    purchase=_date(product.get('purchase_date') or claim.get('purchase_date') or extracted.get('purchase_date'))
    fault=_date(claim.get('fault_date') or extracted.get('fault_date'))
    submitted=_date(claim.get('claim_submission_date') or claim.get('submission_date'))
    repair=_date(claim.get('repair_date') or extracted.get('repair_date'))
    if purchase and fault and fault < purchase: findings.append(_finding('FAULT_BEFORE_PURCHASE','manual_review','Fault date precedes purchase date.'))
    if purchase and submitted and submitted < purchase: findings.append(_finding('CLAIM_BEFORE_PURCHASE','manual_review','Claim submission date precedes purchase date.'))
    if fault and submitted and fault > submitted: findings.append(_finding('FAULT_AFTER_SUBMISSION','manual_review','Fault date occurs after claim submission date.'))
    if repair and fault and repair < fault: findings.append(_finding('REPAIR_BEFORE_FAULT','manual_review','Repair date precedes the reported fault date.'))
    if repair and submitted and repair > submitted and claim.get('repair_before_submission_required') is True:
        findings.append(_finding('REPAIR_AFTER_SUBMISSION','manual_review','Repair date conflicts with the configured repair timing rule.'))
    if claim.get('previously_repaired') in (False,'no','No') and (claim.get('repair_count') or claim.get('repair_history_count')):
        findings.append(_finding('REPAIR_STATUS_CONTRADICTION','manual_review','Claim says no previous repair but reports a repair count.'))
    if claim.get('warranty_status') == 'active' and claim.get('out_of_warranty') is True:
        findings.append(_finding('WARRANTY_STATUS_CONTRADICTION','manual_review','Warranty is marked active and out-of-warranty simultaneously.'))
    return {'detected':bool(findings), 'count':len(findings), 'findings':findings}


def detect_missing_documents(claim: dict, required: Iterable[str] | None = None):
    required=list(required or ['purchase_invoice','warranty_document','product_photo','fault_evidence'])
    docs=claim.get('documents') or claim.get('evidence',{}).get('documents') or {}
    present=set()
    for key,val in docs.items():
        if val and (not isinstance(val, list) or len(val)>0): present.add(key)
    aliases={
      'purchase_invoice':['invoice','purchase_invoice','receipt'],
      'warranty_document':['warranty','warranty_document','warranty_card'],
      'product_photo':['product_photo','product_image','photos'],
      'fault_evidence':['fault_evidence','damage_evidence','faultEvidence'],
      'serial_evidence':['serial_evidence','serialEvidence'],
      'diagnostic_report':['diagnostic_report','diagnosticReport'],
      'repair_history':['repair_history','repair_document','repairFile']
    }
    missing=[]
    for req in required:
        keys=aliases.get(req,[req])
        if not any(k in present for k in keys): missing.append(req)
    # Repair evidence becomes mandatory when the claim declares prior repair.
    repaired=claim.get('previously_repaired') in (True,'yes','Yes','true','True') or int(claim.get('repair_count') or 0)>0
    if repaired and 'repair_history' not in required and not any(k in present for k in aliases['repair_history']): missing.append('repair_history')
    return {'required':required,'present':sorted(present),'missing':missing,'count':len(missing), 'complete':not missing}


def claim_fingerprint(claim: dict):
    fields=(norm(claim.get('product_id') or claim.get('productId')), norm(claim.get('serial_number') or claim.get('serialNumber')), norm(claim.get('damage_type') or claim.get('fault_category') or claim.get('damageCategory')), norm(claim.get('fault_date')))
    return hashlib.sha256('|'.join(fields).encode()).hexdigest()


def detect_duplicate_claim(claim: dict, existing_claims: Iterable[dict] | None = None):
    fp=claim_fingerprint(claim); matches=[]
    for existing in existing_claims or []:
        if claim.get('claim_id') and existing.get('claim_id') == claim.get('claim_id'): continue
        if claim_fingerprint(existing)==fp: matches.append(existing.get('claim_id') or existing.get('id'))
    return {'duplicate':bool(matches),'fingerprint':fp,'matches':matches}


def document_hash(data: bytes | None = None, supplied_hash: str | None = None):
    if supplied_hash: return supplied_hash.lower()
    if data is None: return None
    return hashlib.sha256(data).hexdigest()


def detect_duplicate_document(document: dict, existing_documents: Iterable[dict] | None = None):
    h=document_hash(supplied_hash=document.get('sha256') or document.get('hash'))
    if not h: return {'duplicate':False,'hash':None,'matches':[],'reason':'hash_missing'}
    matches=[]
    for existing in existing_documents or []:
        eh=document_hash(supplied_hash=existing.get('sha256') or existing.get('hash'))
        if eh and eh==h and existing.get('id')!=document.get('id'):
            matches.append(existing.get('documentId') or existing.get('id'))
    return {'duplicate':bool(matches),'hash':h,'matches':matches}


def run_fraud_checks(claim: dict, product: dict | None=None, extracted: dict | None=None,
                     existing_claims: Iterable[dict]|None=None, document: dict|None=None,
                     existing_documents: Iterable[dict]|None=None, required_documents: Iterable[str]|None=None):
    serial=verify_serial_number(claim,product,extracted)
    contradictions=detect_contradictions(claim,product,extracted)
    missing=detect_missing_documents(claim,required_documents)
    duplicate_claim=detect_duplicate_claim(claim,existing_claims)
    duplicate_document=detect_duplicate_document(document or {},existing_documents)
    findings=serial['findings']+contradictions['findings']
    if missing['count']:
        findings.append(_finding('MISSING_DOCUMENTS','manual_review',f"{missing['count']} required document(s) are missing.",details={'missing':missing['missing']}))
    if duplicate_claim['duplicate']:
        findings.append(_finding('DUPLICATE_CLAIM','manual_review','A matching claim already exists.',details={'matches':duplicate_claim['matches']}))
    if duplicate_document['duplicate']:
        findings.append(_finding('DUPLICATE_DOCUMENT','manual_review','A document with the same SHA-256 hash already exists.',details={'matches':duplicate_document['matches']}))
    return {
      'engine_version':VERSION,
      'serial_verification':serial,
      'contradictions':contradictions,
      'missing_documents':missing,
      'duplicate_claim':duplicate_claim,
      'duplicate_document':duplicate_document,
      'findings':findings,
      'requires_manual_review':any(f['severity']=='manual_review' for f in findings),
      'blocked':any(f.get('blocking') for f in findings)
    }
