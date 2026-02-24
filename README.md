# py
team project by Pavan Kumar S & Sanketh Rai
============================================================
  SCENARIO 3: The "Earthquake Seismic Wave Analyzer"
============================================================
A seismic monitoring station captures ground vibration data.
The sensor records 10,000,000 readings per second (10M samples)
as a flat 1D array. Each batch of 1,000 samples represents one
"time window" — so you have 10,000 windows of 1,000 samples each.

THE PROBLEM:
- Reshape the 10M flat signal into a (10000 × 1000) matrix
- Apply an "Alert Mask" (element-wise *) to isolate readings above
  the danger threshold (amplitude > 0.75 on a normalized scale)
- Compute the energy (sum of squares) per window using vectorized ops
- Find the top 5 most dangerous earthquake windows
- Prove NumPy processes all 10M points in under 0.5 seconds


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

