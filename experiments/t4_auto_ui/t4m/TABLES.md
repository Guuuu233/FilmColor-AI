# T4m reproducible evaluation tables

Baseline is a fresh run of e8a8c3d, not the research side's quick evaluation.
The BlueNeg check set contains other frames of existing development rolls; it is not a set of unseen rolls.

Development validity: baseline 185, candidate 185, paired 185; membership difference: none.

Cross validity: baseline 123, candidate 123, paired 123; membership difference: none.

## Development set

Per-frame median ΔE2000, then median / mean / p90 over frames. Negative change helps.

| Subset | N | T4l e8a8c3d | T4m | Better / worse > .25 | Better / worse > .05 | median mean_dL T4l → T4m |
|---|---:|---:|---:|---:|---:|---:|
| nlp | 11 | 8.44 / 8.80 / 10.12 | 8.44 / 8.72 / 10.25 | 2 / 0 | 2 / 1 | +3.23 → +1.76 |
| shop | 21 | 7.20 / 8.11 / 11.41 | 7.20 / 7.91 / 11.41 | 1 / 0 | 3 / 1 | +2.29 → +2.29 |
| blueneg | 27 | 13.62 / 13.60 / 16.57 | 13.62 / 13.58 / 16.57 | 1 / 0 | 2 / 0 | +10.81 → +10.81 |
| ext_own | 126 | 8.80 / 9.29 / 14.16 | 8.75 / 8.92 / 13.26 | 14 / 0 | 15 / 0 | +0.86 → +0.86 |
| all | 185 | 9.20 / 9.76 / 14.97 | 9.03 / 9.47 / 14.14 | 18 / 0 | 22 / 2 | +2.34 → +2.29 |

## BlueNeg check set

Per-frame median ΔE2000, then median / mean / p90 over frames. Negative change helps.

| Subset | N | T4l e8a8c3d | T4m | Better / worse > .25 | Better / worse > .05 | median mean_dL T4l → T4m |
|---|---:|---:|---:|---:|---:|---:|
| blueneg | 123 | 13.96 / 14.49 / 20.84 | 13.84 / 14.43 / 20.73 | 5 / 2 | 6 / 2 | +9.56 → +9.56 |
| all | 123 | 13.96 / 14.49 / 20.84 | 13.84 / 14.43 / 20.73 | 5 / 2 | 6 / 2 | +9.56 → +9.56 |

BlueNeg candidates 124, valid 123; excluded: blueneg:199807A-21-orlando-disney.

## Protected and named frames

| Frame | T4l | T4m | ΔE change | EV T4l → T4m | T4m flags |
|---|---:|---:|---:|---:|---|
| ext:A008 | 8.72 | 8.72 | +0.00 | -0.250 → -0.250 | white_point_unrepresentative |
| ext:A012 | 4.46 | 4.46 | +0.00 | +0.929 → +0.929 | white_point_unrepresentative, backlit_midtone |
| ext:A013 | 6.66 | 6.66 | +0.00 | +0.525 → +0.525 |  |
| shop:DSC_3833 | 4.66 | 4.66 | +0.00 | +0.652 → +0.652 |  |
| nlp:_DSF8948 | 8.36 | 8.34 | -0.02 | -0.040 → -0.044 |  |
| shop:DSC_3827 | 15.87 | 11.89 | -3.98 | -0.049 → -0.038 |  |
| ext:A006 | 15.22 | 13.52 | -1.71 | -0.200 → -0.200 |  |
| ext:A011 | 10.00 | 8.58 | -1.41 | -0.037 → -0.037 |  |
| ext:A196 | 12.76 | 12.39 | -0.37 | -0.095 → -0.095 |  |
| ext:A216 | 8.60 | 7.22 | -1.37 | -0.200 → -0.200 |  |
| ext:A193 | 9.78 | 9.78 | +0.00 | -0.148 → -0.148 |  |
| ext:A199 | 11.34 | 10.13 | -1.21 | +0.136 → +0.020 |  |
| ext:A208 | 14.14 | 9.00 | -5.14 | +0.486 → -0.158 |  |
| ext:A209 | 14.96 | 9.37 | -5.60 | +0.270 → -0.301 |  |
| ext:A222 | 12.81 | 10.69 | -2.11 | -0.200 → -0.414 | white_point_unrepresentative |
| ext:A223 | 17.33 | 10.37 | -6.95 | +0.673 → -0.334 |  |
| ext:A224 | 20.41 | 12.27 | -8.14 | +0.032 → -0.110 |  |
| ext:A225 | 15.50 | 10.30 | -5.20 | +0.000 → -0.450 |  |
| ext:A226 | 14.17 | 12.47 | -1.71 | +0.000 → -0.183 |  |
| ext:A227 | 18.29 | 14.03 | -4.26 | +0.150 → +0.038 |  |
| ext:A228 | 17.48 | 15.41 | -2.07 | +0.000 → -0.297 |  |
| nlp:_DSF9239 | 10.12 | 10.25 | +0.12 | +0.559 → +0.549 |  |
| blueneg:19960812D-09-las-vegas | 12.91 | 12.43 | -0.48 | +0.328 → +0.260 |  |

## Estimator ablations (on minus off, standard preset)

- dev thin_t4l: >.25 11 / 0, >.05 15 / 2.
- dev no_cyan: >.25 3 / 0, >.05 3 / 0.
- dev blue guard: 2: ext:A216, shop:DSC_3827.
- dev cyan confidence: 3: ext:A006, ext:A011, ext:A196.
- dev thin cap: 25: blueneg:19960812D-09-las-vegas, blueneg:19970619A-14-white-chapel, ext:A006, ext:A011, ext:A015, ext:A199, ext:A204, ext:A207, ext:A208, ext:A209, ext:A214, ext:A221, ext:A222, ext:A223, ext:A224, ext:A225, ext:A226, ext:A227, ext:A228, nlp:_DSF9239, shop:DSC_3827, shop:DSC_3828, shop:DSC_3829, shop:DSC_3837, shop:DSC_3859.
- dev backlit suppressed: 20: blueneg:19960812D-09-las-vegas, ext:A006, ext:A011, ext:A015, ext:A199, ext:A207, ext:A208, ext:A209, ext:A222, ext:A223, ext:A224, ext:A225, ext:A226, ext:A227, ext:A228, shop:DSC_3827, shop:DSC_3828, shop:DSC_3829, shop:DSC_3837, shop:DSC_3859.
- cross thin_t4l: >.25 5 / 1, >.05 6 / 1.
- cross no_cyan: >.25 0 / 1, >.05 0 / 1.
- cross blue guard: 0: .
- cross cyan confidence: 1: blueneg:199807B-08-orlando-seaworld.
- cross thin cap: 10: blueneg:19941221D-30-kyoto, blueneg:19941222E-02-nara, blueneg:199807A-10-orlando-siggraph, blueneg:199807A-14-orlando-universal-studio, blueneg:199807A-27-orlando-disney, blueneg:199807A-28-orlando-disney, blueneg:199807A-30-orlando-disney, blueneg:199807B-05-orlando-seaworld, blueneg:199807B-06-orlando-seaworld, blueneg:199910-14-hk-maipo.
- cross backlit suppressed: 9: blueneg:19941221D-30-kyoto, blueneg:199807A-10-orlando-siggraph, blueneg:199807A-14-orlando-universal-studio, blueneg:199807A-27-orlando-disney, blueneg:199807A-28-orlando-disney, blueneg:199807A-30-orlando-disney, blueneg:199807B-05-orlando-seaworld, blueneg:199807B-06-orlando-seaworld, blueneg:199910-14-hk-maipo.

## Largest standard changes vs T4l

| Frame | T4l | T4m | Change |
|---|---:|---:|---:|
| ext:A224 | 20.41 | 12.27 | -8.14 |
| ext:A223 | 17.33 | 10.37 | -6.95 |
| ext:A209 | 14.96 | 9.37 | -5.60 |
| ext:A225 | 15.50 | 10.30 | -5.20 |
| ext:A208 | 14.14 | 9.00 | -5.14 |
| ext:A227 | 18.29 | 14.03 | -4.26 |
| shop:DSC_3827 | 15.87 | 11.89 | -3.98 |
| ext:A222 | 12.81 | 10.69 | -2.11 |
| ext:A228 | 17.48 | 15.41 | -2.07 |
| ext:A226 | 14.17 | 12.47 | -1.71 |
| ext:A006 | 15.22 | 13.52 | -1.71 |
| ext:A011 | 10.00 | 8.58 | -1.41 |
| ext:A216 | 8.60 | 7.22 | -1.37 |
| ext:A199 | 11.34 | 10.13 | -1.21 |
| nlp:_DSF9229 | 8.29 | 7.67 | -0.61 |
| blueneg:19960812D-09-las-vegas | 12.91 | 12.43 | -0.48 |
| ext:A196 | 12.76 | 12.39 | -0.37 |
| nlp:_DSF9239 | 10.12 | 10.25 | +0.12 |
| shop:DSC_3828 | 6.79 | 6.89 | +0.10 |
| ext:A015 | 5.76 | 5.77 | +0.01 |
| shop:DSC_3837 | 7.48 | 7.49 | +0.01 |
| nlp:_DSF8951 | 8.44 | 8.44 | +0.00 |
| shop:DSC_3856 | 7.21 | 7.21 | +0.00 |
| shop:DSC_3855 | 6.82 | 6.82 | +0.00 |
| shop:DSC_3853 | 9.03 | 9.03 | +0.00 |
| shop:DSC_3849 | 6.44 | 6.44 | +0.00 |
| shop:DSC_3847 | 11.37 | 11.37 | +0.00 |

## BlueNeg changes exceeding .05

| Frame | T4l | T4m | Change | Thin on minus off | Cyan on minus off |
|---|---:|---:|---:|---:|---:|
| blueneg:199807A-14-orlando-universal-studio | 14.75 | 11.96 | -2.80 | -2.80 | +0.00 |
| blueneg:199910-14-hk-maipo | 22.55 | 19.89 | -2.66 | -2.66 | +0.00 |
| blueneg:199807A-27-orlando-disney | 11.32 | 9.46 | -1.86 | -1.86 | +0.00 |
| blueneg:199807A-10-orlando-siggraph | 10.81 | 10.48 | -0.33 | -0.33 | +0.00 |
| blueneg:199807B-06-orlando-seaworld | 13.96 | 13.69 | -0.27 | -0.27 | +0.00 |
| blueneg:199807A-30-orlando-disney | 22.18 | 21.96 | -0.22 | -0.22 | +0.00 |
| blueneg:199807B-08-orlando-seaworld | 16.02 | 16.59 | +0.57 | +0.00 | +0.57 |
| blueneg:19941221D-30-kyoto | 16.93 | 17.79 | +0.85 | +0.85 | +0.00 |

## Device samples (no reference)

| Sample | T4l | T4m | Holder T4l → T4m | T4m film_box |
|---|---|---|---:|---|
| A1089.DNG | ValueError: /private/tmp/fc_t4m_devices/rolls/A1089/A1089.DNG: only single-plane CFA RAW files are supported | ok | — → 0.00000 | None |
| A322.cr2 | ok | ok | 0.07553 → 0.34988 | [142, 205, 747, 1118] |
| A1106.CR2 | ok | ok | 0.20076 → 0.20510 | [16, 69, 854, 1301] |
| A1283.DNG | ok | ok | 0.02985 → 0.03303 | [5, 0, 981, 1477] |
| A1141.NEF | manual_review_required | manual_review_required | 0.00017 → 0.00083 | [0, 0, 920, 1381] |
| A223.NEF | ok | ok | 0.00025 → 0.00039 | [0, 0, 920, 1381] |
| A224.NEF | ok | ok | 0.00040 → 0.00053 | [0, 0, 920, 1381] |
| DSC_3827.NEF | ok | ok | 0.00000 → 0.00001 | [0, 0, 920, 1381] |

## Source fingerprints

- baseline_a: `25d86621c42b6431e0396fc37105eed69309d1de19082933cae415ffdf69191c`.
- candidate_a: `ac291d97a8848ddd93f6344030485bcdd538960c5ecc6648fdc6393bde8c1024`.
- baseline_cross: `25d86621c42b6431e0396fc37105eed69309d1de19082933cae415ffdf69191c`.
- candidate_cross: `ac291d97a8848ddd93f6344030485bcdd538960c5ecc6648fdc6393bde8c1024`.
- devices_baseline: `25d86621c42b6431e0396fc37105eed69309d1de19082933cae415ffdf69191c`.
- devices_candidate: `ac291d97a8848ddd93f6344030485bcdd538960c5ecc6648fdc6393bde8c1024`.

## Peak memory footprint (GiB, /usr/bin/time -l)

- baseline_a: 4.69 GiB (largest recorded run, including resumed invocations).
- baseline_b: 4.84 GiB (largest recorded run, including resumed invocations).
- baseline_cross: 5.68 GiB (largest recorded run, including resumed invocations).
- candidate_a: 5.68 GiB (largest recorded run, including resumed invocations).
- candidate_b: 4.89 GiB (largest recorded run, including resumed invocations).
- candidate_cross: 5.60 GiB (largest recorded run, including resumed invocations).
- devices_baseline: 3.52 GiB (largest recorded run, including resumed invocations).
- devices_candidate: 3.88 GiB (largest recorded run, including resumed invocations).
- tests: 2.99 GiB (largest recorded run, including resumed invocations).
