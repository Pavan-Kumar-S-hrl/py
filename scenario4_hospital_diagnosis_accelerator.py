"""
============================================================
  SCENARIO 4: The "Hospital Patient Diagnosis Accelerator"
============================================================
A hospital emergency room has 500,000 incoming patient records.
Each patient has 6 vital signs: [Heart Rate, Blood Pressure,
O2 Saturation, Temperature, Respiratory Rate, Blood Sugar].

You also have a "Risk Scoring Matrix" (6 vitals × 4 risk categories):
[Cardiac Risk, Respiratory Risk, Diabetic Risk, Sepsis Risk].

THE PROBLEM:
- Use @ (Dot Product) to compute a 4-category risk score for all
  500,000 patients at once (matrix multiplication: 500k×6 @ 6×4)
- Transpose the Risk Matrix so dimensions align for multiplication
- Use element-wise * to apply an "urgency multiplier" — doubling
  the risk scores for patients whose temperature > 39°C (fever)
- Prove NumPy handles 500,000 patients in under 1 second vs Python
  which would take minutes
============================================================
"""

import numpy as np
import time

print("=" * 60)
print("SCENARIO 4: Hospital Patient Diagnosis Accelerator")
print("=" * 60)

NUM_PATIENTS = 500_000

# Simulate patient vitals data — 500,000 patients × 6 vitals
# [Heart Rate, Blood Pressure, O2 Sat, Temperature, Resp Rate, Blood Sugar]
np.random.seed(99)
patient_vitals = np.column_stack([
    np.random.normal(75,  15,  NUM_PATIENTS),   # Heart Rate (bpm)
    np.random.normal(120, 20,  NUM_PATIENTS),   # Blood Pressure (mmHg)
    np.random.normal(97,  2,   NUM_PATIENTS),   # O2 Saturation (%)
    np.random.normal(37,  1.5, NUM_PATIENTS),   # Temperature (°C)
    np.random.normal(16,  4,   NUM_PATIENTS),   # Respiratory Rate
    np.random.normal(100, 30,  NUM_PATIENTS),   # Blood Sugar (mg/dL)
])

# Risk Scoring Matrix: 4 risk categories × 6 vitals
# Rows = [Cardiac, Respiratory, Diabetic, Sepsis]
# Cols = [HR, BP, O2, Temp, RespRate, BloodSugar]
risk_weights = np.array([
    [0.40, 0.35, 0.10, 0.05, 0.05, 0.05],  # Cardiac — HR & BP dominant
    [0.10, 0.05, 0.45, 0.15, 0.20, 0.05],  # Respiratory — O2 & RespRate
    [0.05, 0.10, 0.05, 0.05, 0.05, 0.70],  # Diabetic — BloodSugar dominant
    [0.15, 0.10, 0.20, 0.30, 0.20, 0.05],  # Sepsis — Temp & O2 & RespRate
])

print(f"Patient Vitals shape  : {patient_vitals.shape}")   # (500000, 6)
print(f"Risk Weights shape    : {risk_weights.shape}")      # (4, 6)

# Transpose risk_weights → (6×4) to align for: (500k×6) @ (6×4) = (500k×4)
risk_weights_T = risk_weights.T
print(f"Transposed Weights    : {risk_weights_T.shape}")    # (6, 4)

# ── NumPy approach ──────────────────────────────────────────
start_np = time.time()

# Dot product: compute all 4 risk scores for all 500k patients at once
risk_scores = patient_vitals @ risk_weights_T               # (500000, 4)

# Identify fever patients (Temperature > 39°C) — column index 3
fever_patients = (patient_vitals[:, 3] > 39.0).astype(float)

# Apply urgency multiplier: double ALL risk scores for fever patients
# Reshape fever flag to (500000, 1) so it broadcasts across 4 columns
urgency_multiplier = 1.0 + fever_patients.reshape(-1, 1)    # 2.0 if fever, 1.0 if not
risk_scores_final = risk_scores * urgency_multiplier

end_np = time.time()
np_time = end_np - start_np

# ── Python loop approach (on tiny 500-patient subset for comparison) ──
SUBSET = 500
start_py = time.time()
py_scores = []
for p in range(SUBSET):
    patient_risk = []
    for r in range(4):
        score = sum(patient_vitals[p, v] * risk_weights[r, v] for v in range(6))
        fever_mult = 2.0 if patient_vitals[p, 3] > 39.0 else 1.0
        patient_risk.append(score * fever_mult)
    py_scores.append(patient_risk)
end_py = time.time()
py_time_subset = end_py - start_py

# Extrapolate Python time for full 500k patients
py_time_estimated = py_time_subset * (NUM_PATIENTS / SUBSET)

# Results
fever_count = int(np.sum(fever_patients))
critical_cardiac = int(np.sum(risk_scores_final[:, 0] > np.percentile(risk_scores_final[:, 0], 95)))

print(f"\n  Risk Score Matrix shape    : {risk_scores_final.shape}")
print(f"  Fever patients flagged     : {fever_count:,} ({fever_count/NUM_PATIENTS*100:.1f}%)")
print(f"  Critical cardiac alerts    : {critical_cardiac:,} (top 5% risk score)")

print(f"\n  ⏱  NumPy — 500,000 patients : {np_time:.4f} seconds")
print(f"  ⏱  Python loop (estimated)  : {py_time_estimated:.1f} seconds")
speedup = py_time_estimated / np_time
print(f"  🚀 NumPy is ~{speedup:.0f}× faster than Python loop")

if np_time < 1.0:
    print("  ✅ HOSPITAL SLA MET — All patients triaged in under 1 second!")
else:
    print("  ❌ Too slow — patients could be waiting dangerously long.")
