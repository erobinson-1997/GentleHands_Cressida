# Combined RPM + GPS datasets (2026-11-09 sessions)

Files: `combined_downhill.csv`, `combined_uphill.csv` (one row per tach reading, 4 Hz).
Scripts that produced them: `scripts/racechrono_tach/` (run_clip.py → analyze.py).

## Columns
| column | type | source |
|---|---|---|
| unix_time, elapsed_s | measured | video frame time (RaceChrono clip start + frame PTS); elapsed is on the GPS session clock |
| rpm_tach_video | **measured (video), calibrated** | needle angle read from dashboard video, mapped to rpm (see calibration) |
| rpm_quality_ok | flag | 1 = registration ok, needle clearly detected, not a spike (>350 rpm off the median of its ±1 s neighbours). **Use only rows with 1.** |
| rpm_alt_mapping | alt. calibration | same needle angle under the other angle→rpm mapping (spread = calibration uncertainty) |
| gps_speed_mps/mph_lagcorr | measured (GPS, 1 Hz) | linearly interpolated from 1 Hz fixes to the tach timestamp; lag correction = 0 s (best fit) |
| gps_accel_g_1hz | calculated | gradient of 1 Hz GPS speed; noisy at ±0.03 G per sample |
| nv_rpm_per_mps | calculated | rpm / speed |
| est_gear | estimate | gear whose rpm/speed line is within 4% of this sample; blank = clutch in, shifting, coasting, or ambiguous |
| reg_ecc, needle_snr | QA | image registration score, needle contrast |
| gps_accuracy_m, altitude_m | measured (GPS) | altitude has 1 m resolution |

## Calibration
- Downhill (close-up view): piecewise-linear between numerals 0–8 measured around the needle pivot. The pivot was confirmed by intersecting needle lines from 7 frames.
- Uphill (wide view, numerals 3–5 hidden by the wheel rim): projective dial model fitted to numerals 0,1,2,6,7,8 with the dial spacing fixed at 28.0°/1000 rpm. That spacing is consistent across both camera views.
- Estimated rpm accuracy: about ±3% (±60–120 rpm at cruise). This comes from the needle drifting within one gear and from the two camera views disagreeing.

## Known gaps / unreliable data
- Clip handovers: 1.3–3.0 s with no video between each pair of clips.
- Uphill: the first 104 s has no video. The last 33 s clip was not processed, because the camera was turned to a different view.
- Quality-pass coverage: downhill 69% of samples, uphill 53% (sun glare, hand or wheel blocking the tach, camera moving).
- A few isolated single-frame misreads survive (e.g. a 7907 rpm reading). The p99 rpm is about 3900.
- GPS is 1 Hz, so acceleration and shift timing are only resolved to about 1 s.
