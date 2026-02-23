"""
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
"""

import numpy as np
import time

print("=" * 60)
print("SCENARIO 3: Earthquake Seismic Wave Analyzer")
print("=" * 60)

NUM_SAMPLES  = 10_000_000
WINDOWS      = 10_000
WINDOW_SIZE  = 1_000
THRESHOLD    = 0.75

# Simulate raw seismic sensor data (normalized -1.0 to 1.0)
np.random.seed(7)
raw_signal = np.random.uniform(-1.0, 1.0, size=NUM_SAMPLES)
print(f"Raw signal shape  : {raw_signal.shape}")  # (10000000,)

start = time.time()

# Step 1: Reshape into windows (10000 windows × 1000 samples each)
signal_matrix = raw_signal.reshape(WINDOWS, WINDOW_SIZE)
print(f"Reshaped matrix   : {signal_matrix.shape}")  # (10000, 1000)

# Step 2: Build danger alert mask — 1 where amplitude > threshold
alert_mask = (np.abs(signal_matrix) > THRESHOLD).astype(np.float32)

# Step 3: Element-wise multiply to extract only dangerous readings
danger_readings = signal_matrix * alert_mask

# Step 4: Compute energy (sum of squares) for each window
window_energy = np.sum(signal_matrix ** 2, axis=1)  # shape: (10000,)

# Step 5: Find top 5 most dangerous windows
top5_indices = np.argsort(window_energy)[-5:][::-1]

end = time.time()
elapsed = end - start

print(f"\n  Total dangerous readings   : {int(np.sum(alert_mask)):,}")
print(f"  Windows above threshold    : {int(np.sum(np.any(alert_mask > 0, axis=1))):,}")
print(f"\n  Top 5 Earthquake Windows (by energy):")
for rank, idx in enumerate(top5_indices, 1):
    print(f"    #{rank}  Window {idx:5d}  →  Energy = {window_energy[idx]:.4f}")

print(f"\n  Processing time for 10M samples: {elapsed:.4f} seconds")
if elapsed < 0.5:
    print("  ✅ REAL-TIME CAPABLE — Alert can be triggered instantly!")
else:
    print("  ⚠️  Too slow for real-time earthquake alerting.")
