from __future__ import annotations
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import json
from pathlib import Path

VERSION = '26.0.0'

@dataclass
class WarrantyPolicy:
    policy_id: str
    name: str
    active: bool = True
    standard_months: int = 12
    extended_months: int = 0
    grace_period_days: int = 0
    reporting_deadline_days: int | None = 30
    covered_faults: list[str] = field(default_factory=list)
    excluded_faults: list[str] = field(default_factory=list)
    unauthorized_repair_blocks: bool = False
    unauthorized_repair_count_threshold: int = 3
    serial_mismatch_action: str = 'manual_review'
    missing_documents_manual_review_threshold: int = 2
    duplicate_action: str = 'manual_review'
    contradiction_action: str = 'manual_review'
    authorized_service_centre_required: bool = False
    replacement_conditions: list[str] = field(default_factory=list)

    def effective_months(self, extended=False) -> int:
        return max(0, self.standard_months) + (max(0, self.extended_months) if extended else 0)

    def to_dict(self):
        return asdict(self)


def default_policies() -> dict[str, WarrantyPolicy]:
    return {
        'standard-manufacturer': WarrantyPolicy(
            policy_id='standard-manufacturer', name='Standard Manufacturer',
            standard_months=12, extended_months=0, grace_period_days=0,
            reporting_deadline_days=30,
            excluded_faults=['Misuse', 'Accidents', 'Water Damage'],
        ),
        'extended-24': WarrantyPolicy(
            policy_id='extended-24', name='Extended 24 Month',
            standard_months=12, extended_months=12, grace_period_days=0,
            reporting_deadline_days=30,
            excluded_faults=['Misuse', 'Accidents', 'Water Damage'],
            active=False,
        ),
    }


class PolicyStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.policies = default_policies()
        if self.path and self.path.exists():
            self.load(self.path)

    def load(self, path: str | Path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        raw = data.get('policies', data)
        self.policies = {}
        for pid, value in raw.items():
            value = dict(value)
            value.setdefault('policy_id', pid)
            value.setdefault('name', pid)
            self.policies[pid] = WarrantyPolicy(**value)
        return self.policies

    def save(self, path: str | Path | None = None):
        target = Path(path or self.path)
        if not target:
            raise ValueError('No policy file path supplied')
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {'version': VERSION, 'policies': {k: v.to_dict() for k, v in self.policies.items()}}
        target.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    def get(self, policy_id: str | None = None) -> WarrantyPolicy:
        if policy_id:
            if policy_id not in self.policies:
                raise KeyError(f'Unknown warranty policy: {policy_id}')
            return deepcopy(self.policies[policy_id])
        active = [p for p in self.policies.values() if p.active]
        if len(active) != 1:
            raise ValueError('A policy_id is required when zero or multiple policies are active')
        return deepcopy(active[0])

    def upsert(self, policy: WarrantyPolicy):
        self.policies[policy.policy_id] = policy

    def activate(self, policy_id: str):
        if policy_id not in self.policies:
            raise KeyError(policy_id)
        for p in self.policies.values():
            p.active = p.policy_id == policy_id
