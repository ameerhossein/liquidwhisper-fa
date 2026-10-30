# Answered-customer manifest

This directory contains the offline content labels and 1/3-octave target used
to recalibrate the telephone `customer` profile. None of it is read by the
runtime simulator.

- Labels come from `MIT/ast-finetuned-audioset-10-10-0.4593`, run on 10 s
  windows at the first, middle, and last portion of each recording. Per file,
  the majority label family (music / speech / other) is stored as
  `content_class`.
- The music family includes `Music`, `Singing`, `Song`, and instrument
  subclasses. The speech family includes `Speech`, `Conversation`, and
  `Narration`. The classifier is a screening tool, not ground truth; spot-check
  a sample by listening before treating a label as final.
- `answered_customer_manifest.csv` has 320 rows: the 107 in-repo
  `RealRecordedPhoneCalls` files and the 213-file `Outside` sample.
- The `Outside` sample is all 163 readable `noisy` + `quiet_noisy` customer
  calls plus 50 random `gated` customer calls sampled with `random.seed(42)`.
- The 67-row calibration subset is `calibration=true`:
  `content_class=speech_only`, `is_gated=true`, `caller_class=mobile`.
- `answered_customer_bands.csv` stores absolute 1/3-octave loud-frame levels
  for those 67 files; the fit normalizes each band to the pooled 1 kHz median.
- `outside_spectral_features.csv` supplies the HF/centroid metrics for the
  Outside sample; the original AST labels only carried basic level/SNR fields.
- Rebuild with `build_answered_manifest.py` and `build_answered_bands.py`.
