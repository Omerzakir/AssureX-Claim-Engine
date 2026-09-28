"""AssureX XVIII/XIX Python classifier runtime.

Loads the user's trained sklearn/joblib artifacts; it never trains or substitutes a model.
Expected artifacts: best_model.pkl, scaler.pkl, label_encoders.pkl, feature_columns.pkl,
label_map.pkl. The 19-feature contract is the XVII/XVI canonical contract.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np

VERSION = "18.0.0"
RAW_COLUMNS = [
    "Product_Category","Product_Age_Months","Warranty_Duration","Extended_Warranty",
    "Purchase_Price","Damage_Type","Repair_History_Count","Previous_Repair_Unauthorized",
    "Serial_Number_Match","Missing_Documents_Count","Fault_To_Claim_Delay_Days",
    "Contradiction_Flag","Duplicate_Claim_Flag",
]
ENGINEERED_COLUMNS = ["Remaining_Warranty_Months","Out_Of_Warranty","Documents_Complete","High_Risk_Repair","Late_Report","Risk_Score"]
MODEL_COLUMNS = RAW_COLUMNS + ENGINEERED_COLUMNS
DEFAULT_LABEL_MAP = {0:"Invalid Claim",1:"Valid Claim",2:"Manual Review"}

class ModelArtifactError(RuntimeError): pass

class PythonClassifier:
    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        required = ["best_model.pkl","scaler.pkl","label_encoders.pkl","feature_columns.pkl","label_map.pkl"]
        missing = [x for x in required if not (self.artifact_dir/x).is_file()]
        if missing:
            raise ModelArtifactError(f"Python model artifacts missing: {', '.join(missing)}. Expected directory: {self.artifact_dir}")
        self.model = joblib.load(self.artifact_dir/"best_model.pkl")
        self.scaler = joblib.load(self.artifact_dir/"scaler.pkl")
        self.encoders = joblib.load(self.artifact_dir/"label_encoders.pkl")
        self.feature_columns = list(joblib.load(self.artifact_dir/"feature_columns.pkl"))
        loaded_map = joblib.load(self.artifact_dir/"label_map.pkl")
        self.label_map = {int(k): str(v) for k,v in dict(loaded_map).items()} if all(str(k).isdigit() for k in loaded_map) else {int(v): str(k) for k,v in dict(loaded_map).items()}
        if self.feature_columns != MODEL_COLUMNS:
            raise ModelArtifactError("feature_columns.pkl does not exactly match the XVII 19-feature contract")
        self._validate_model_contract()

    def _validate_model_contract(self):
        if not hasattr(self.model, "predict_proba"):
            raise ModelArtifactError("best_model.pkl must expose predict_proba() for confidence generation")
        if not hasattr(self.scaler, "transform"):
            raise ModelArtifactError("scaler.pkl must expose transform()")
        for col in ("Product_Category","Damage_Type"):
            if col not in self.encoders or not hasattr(self.encoders[col], "transform"):
                raise ModelArtifactError(f"label_encoders.pkl is missing fitted encoder: {col}")

    @staticmethod
    def _num(claim, key, default=0.0):
        try: return float(claim.get(key, default))
        except (TypeError, ValueError): return float(default)

    def canonical_features(self, claim: dict[str, Any]) -> dict[str, float | str]:
        raw = {k: claim.get(k) for k in RAW_COLUMNS}
        # Accept frontend aliases without changing the canonical model contract.
        aliases = {"product_category":"Product_Category","product_age_months":"Product_Age_Months","warranty_duration":"Warranty_Duration","extended_warranty":"Extended_Warranty","purchase_price":"Purchase_Price","damage_type":"Damage_Type","repair_history_count":"Repair_History_Count","previous_repair_unauthorized":"Previous_Repair_Unauthorized","serial_number_match":"Serial_Number_Match","missing_documents_count":"Missing_Documents_Count","fault_to_claim_delay_days":"Fault_To_Claim_Delay_Days","contradiction_flag":"Contradiction_Flag","duplicate_claim_flag":"Duplicate_Claim_Flag"}
        for src,dst in aliases.items():
            if raw[dst] is None and src in claim: raw[dst] = claim[src]
        category = str(raw["Product_Category"] or "").strip()
        damage = str(raw["Damage_Type"] or "").strip()
        age = self._num(raw,"Product_Age_Months"); warranty=self._num(raw,"Warranty_Duration"); ext=self._num(raw,"Extended_Warranty")
        repairs=self._num(raw,"Repair_History_Count"); unauthorized=self._num(raw,"Previous_Repair_Unauthorized")
        missing=self._num(raw,"Missing_Documents_Count"); delay=self._num(raw,"Fault_To_Claim_Delay_Days")
        remaining = warranty + (12.0 if ext else 0.0) - age
        out = 1.0 if remaining < 0 else 0.0
        complete = 1.0 if missing == 0 else 0.0
        high_risk = 1.0 if unauthorized and repairs >= 3 else 0.0
        late = 1.0 if delay > 30 else 0.0
        risk = (2.0 if out else 0.0) + (2.0 if high_risk else 0.0) + (1.0 if late else 0.0) + min(missing,3.0)*0.5 + (1.5 if not raw["Serial_Number_Match"] else 0.0) + (1.5 if raw["Contradiction_Flag"] else 0.0) + (1.5 if raw["Duplicate_Claim_Flag"] else 0.0)
        return {**raw,"Product_Category":category,"Damage_Type":damage,"Product_Age_Months":age,"Warranty_Duration":warranty,"Extended_Warranty":ext,"Purchase_Price":self._num(raw,"Purchase_Price"),"Repair_History_Count":repairs,"Previous_Repair_Unauthorized":unauthorized,"Serial_Number_Match":self._num(raw,"Serial_Number_Match",1),"Missing_Documents_Count":missing,"Fault_To_Claim_Delay_Days":delay,"Contradiction_Flag":self._num(raw,"Contradiction_Flag"),"Duplicate_Claim_Flag":self._num(raw,"Duplicate_Claim_Flag"),"Remaining_Warranty_Months":remaining,"Out_Of_Warranty":out,"Documents_Complete":complete,"High_Risk_Repair":high_risk,"Late_Report":late,"Risk_Score":risk}

    def transform(self, claim: dict[str, Any]) -> np.ndarray:
        f=self.canonical_features(claim)
        values=[]
        for col in MODEL_COLUMNS:
            if col in self.encoders:
                try: values.append(float(self.encoders[col].transform([f[col]])[0]))
                except Exception as exc: raise ModelArtifactError(f"Unknown/unencodable category for {col}: {f[col]!r}") from exc
            else: values.append(float(f[col]))
        x=np.asarray(values,dtype=float).reshape(1,-1)
        return self.scaler.transform(x)

    def predict(self, claim: dict[str, Any]) -> dict[str, Any]:
        x=self.transform(claim)
        probabilities=np.asarray(self.model.predict_proba(x)[0],dtype=float)
        classes=getattr(self.model,"classes_",np.arange(len(probabilities)))
        probs={str(self.label_map.get(int(c),c)):float(p) for c,p in zip(classes,probabilities)}
        idx=int(np.argmax(probabilities)); class_id=int(classes[idx]); label=self.label_map.get(class_id,str(class_id)); confidence=float(probabilities[idx])
        return {"label":label,"class_id":class_id,"confidence":confidence,"probabilities":probs,"model_version":VERSION,"artifact_directory":str(self.artifact_dir)}
