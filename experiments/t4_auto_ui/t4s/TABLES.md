# T4s Verified Regression and Equivalence Report

- Working tree candidate source SHA256: `27ba01f3aa13d29905af25f17284b68bc99da2897a40c957c2a5f4d898b9efb5` (includes user's unmodified paths.py).
- Committed T4r runtime evidence source SHA256: `276fa74b10912b20a38f47ee8edd7db2cbe59412d02ae90b80ff05c14afca1a1` (distinct historical environment).
- Clean baseline source SHA256 extracted from fbd48a0: `16862d53f8873cdf82d9659674da0f372114c8ab6ad7a284fca2dd1a0c1158a7`.
- Equivalence target: Committed T4r (`fbd48a0`), five dev presets and standard cross.
- Statistical baseline: Committed T4o (`4cd76b7`).

Per-frame median ΔE2000, then median / mean / p90 across frames.

| Set | N | T4o | T4r | T4s | vs T4o >.25 | vs T4o >.05 | Max abs ΔE(std) vs T4r | T4r ΔE(std) exact (0.0) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev:nlp | 11 | 8.44 / 8.72 / 10.25 | 8.44 / 8.72 / 10.25 | 8.44 / 8.72 / 10.25 | 0 / 0 | 0 / 0 | 0.000000 | YES (0.0) |
| dev:shop | 21 | 7.20 / 7.91 / 11.41 | 7.20 / 7.91 / 11.41 | 7.20 / 7.91 / 11.41 | 0 / 0 | 0 / 0 | 0.000000 | YES (0.0) |
| dev:blueneg | 27 | 13.62 / 13.49 / 16.08 | 13.62 / 13.45 / 16.08 | 13.62 / 13.45 / 16.08 | 1 / 0 | 1 / 0 | 0.000000 | YES (0.0) |
| dev:ext_own | 126 | 8.75 / 8.91 / 13.26 | 8.65 / 8.82 / 13.16 | 8.65 / 8.82 / 13.16 | 4 / 0 | 4 / 0 | 0.000000 | YES (0.0) |
| dev:all | 185 | 9.03 / 9.45 / 14.14 | 9.01 / 9.38 / 14.09 | 9.01 / 9.38 / 14.09 | 5 / 0 | 5 / 0 | 0.000000 | YES (0.0) |
| cross:all | 123 | 13.84 / 14.43 / 20.73 | 13.77 / 14.31 / 20.63 | 13.77 / 14.31 / 20.63 | 9 / 0 | 10 / 0 | 0.000000 | YES (0.0) |

## dev: presets vs T4o

| Preset | T4o | T4s | Better / worse >.25 | Better / worse >.05 | Max regression |
|---|---:|---:|---:|---:|---:|
| standard | 9.03 / 9.45 / 14.14 | 9.01 / 9.38 / 14.09 | 5 / 0 | 5 / 0 | +0.0000 |
| natural | 9.38 / 10.04 / 14.77 | 9.21 / 9.95 / 14.63 | 9 / 0 | 10 / 0 | +0.0000 |
| portrait | 9.56 / 10.65 / 16.57 | 9.52 / 10.50 / 16.18 | 14 / 0 | 16 / 0 | +0.0000 |
| vivid | 9.22 / 9.92 / 14.94 | 9.04 / 9.86 / 14.84 | 4 / 0 | 4 / 0 | +0.0000 |
| print_classic | 8.46 / 9.07 / 13.61 | 8.45 / 9.03 / 13.61 | 4 / 0 | 4 / 0 | +0.0093 |

### dev: preset regressions >.05 vs T4o

- **standard**: none
- **natural**: none
- **portrait**: none
- **vivid**: none
- **print_classic**: none

## dev: Equivalence check vs T4r

- Identically equal: **True**
- Max absolute diff params: `0.000000e+00`
- Max absolute diff metrics: `0.000000e+00`
- Number of mismatching records: `0`

## cross: presets vs T4o

| Preset | T4o | T4s | Better / worse >.25 | Better / worse >.05 | Max regression |
|---|---:|---:|---:|---:|---:|
| standard | 13.84 / 14.43 / 20.73 | 13.77 / 14.31 / 20.63 | 9 / 0 | 10 / 0 | +0.0000 |

### cross: preset regressions >.05 vs T4o

- **standard**: none

## cross: Equivalence check vs T4r

- Identically equal: **True**
- Max absolute diff params: `0.000000e+00`
- Max absolute diff metrics: `0.000000e+00`
- Number of mismatching records: `0`

## Protected frames

- ext:A008: unchanged, ΔE 8.720626.
- ext:A012: unchanged, ΔE 4.460079.
- ext:A013: unchanged, ΔE 6.661887.
- shop:DSC_3833: unchanged, ΔE 4.664065.
- nlp:_DSF8948: unchanged, ΔE 8.343404.

## Device Verification Summary

- All device checks passed: **False**
- A1296 (1600px preview): status `ok`, center < 1.35: `True`, band Blue < 1.0: `False` (candidate B=1.0749907493591309 vs baseline B=10583540.0), hash match: `True`
- Acros: 28 frames all ok: `True`, 4 designated roll_nat targets span >= 0.24 & flags empty: `True`
- TIFF: 8 scanner frames all ok & finite: `True`

### Device Verification Warnings / Failures
- **[FAILED]** A1296 upper_band_p99 Blue 1.0749907493591309 > 1.0 (hard acceptance failed; baseline=10583540.0)

### Boundary Appendix: A1296 1000px Preview Scale Sensitivity

- Note: Evaluated on 1000px scale to locate task prediction differences; does not replace 1600px primary device gate.
- Status: `ok`, Shape: `[645, 966, 3]`
- Center p99: `[1.1311981678009033, 1.1575795412063599, 1.148000717163086]`, Upper band p99 Blue: `0.9649619460105896` (1000px preview) vs `1.0749907493591309` (1600px primary device preview).

## Full pytest

399 passed in 602.18s, peak memory 2.98 GiB.

## Service Check

- Status: `ok`
- Sample: `_DSF9239.RAF`, EV: `0.5492`
- HTTP Config Rev: `t4s-2026-11-02`
- Source Thumb Rev: `t4s`, Source Config Rev: `t4s-2026-11-02`
- Render SHA256: `b3f4c6c3616115ab02f3b7119dac75e8d1768242ee4c2b3e347133111dca75ff`

## Preset Configuration Snapshots (configs/)

- Source Config Rev: `t4s-2026-11-02`
- All presets match runtime `preset_config()`: **True**
- `configs/noref_v3_natural.json`: matches runtime = True, SHA256 = `c443c7fba04a123c17fc4c2dbef5b3b91ab47562fad6c28135fff6363451c5d5`
- `configs/noref_v3_portrait.json`: matches runtime = True, SHA256 = `4b5af56e3187d68cbe86184ad7f2af53bec82fbae6ffd0561f52a8f451d75f15`
- `configs/noref_v3_print_classic.json`: matches runtime = True, SHA256 = `37a3eb29f2949001925ddc5fa5f787e81de260bd92d4ecc3057591f7e5ee778c`
- `configs/noref_v3.json`: matches runtime = True, SHA256 = `511d58602754ae74a51ef1d8f99db48f80085f2bc71c1a83a7115c351913f58a`
- `configs/noref_v3_vivid.json`: matches runtime = True, SHA256 = `5839eb325d4835accf4788fd1526278a4fd6e01d268f0ccaa1ca2196fc54b354`

## Peak memory footprint (GiB)

- candidate_a: 5.33 GiB
- candidate_b: 4.97 GiB
- candidate_cross: 7.89 GiB
- devices_baseline: 3.89 GiB
- devices_candidate: 3.40 GiB
- run_eval: 0.02 GiB
- tests: 2.98 GiB
- tiff_smoke: 0.10 GiB
