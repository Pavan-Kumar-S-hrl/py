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


