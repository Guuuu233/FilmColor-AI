# T4o vs c76e588 (T4n) reproducible results

Per-frame median ΔE2000, then median / mean / p90 across frames; lower is closer to the reference.

| Set | N | T4n | T4o | Better / worse > .25 | Better / worse > .05 |
|---|---:|---:|---:|---:|---:|
| dev:nlp | 11 | 8.44 / 8.72 / 10.25 | 8.44 / 8.72 / 10.25 | 0 / 0 | 0 / 0 |
| dev:shop | 21 | 7.20 / 7.91 / 11.41 | 7.20 / 7.91 / 11.41 | 0 / 0 | 0 / 0 |
| dev:blueneg | 27 | 13.62 / 13.58 / 16.57 | 13.62 / 13.49 / 16.08 | 1 / 0 | 1 / 0 |
| dev:ext_own | 126 | 8.75 / 8.92 / 13.26 | 8.75 / 8.91 / 13.26 | 3 / 0 | 3 / 0 |
| dev:all | 185 | 9.03 / 9.48 / 14.14 | 9.03 / 9.45 / 14.14 | 4 / 0 | 4 / 0 |
| cross:all | 123 | 13.84 / 14.43 / 20.73 | 13.84 / 14.43 / 20.73 | 1 / 0 | 1 / 0 |

## dev: mechanism checks

- no_rescue: parameters changed 2; on minus off >.25 2 / 0, >.05 2 / 0: blueneg:19970620C-10-st-michel, ext:A103.
- no_spatial_gate: parameters changed 0; on minus off >.25 0 / 0, >.05 0 / 0: none.
- Holder white-point filter: 0: none.
- Rescue: 2: blueneg:19970620C-10-st-michel, ext:A103.
- Cool cast: 13; clip_share > .0021: 0: none.

## dev: metric changes > .000001

| Frame | T4n | T4o | ΔE change | mean_dL T4n → T4o |
|---|---:|---:|---:|---:|
| blueneg:19970620C-10-st-michel | 16.963776 | 14.441746 | -2.522030 | -11.831 → -6.556 |
| ext:A103 | 10.327800 | 9.095242 | -1.232557 | -8.632 → -6.384 |
| ext:A104 | 9.610570 | 9.080654 | -0.529916 | -7.454 → -6.816 |
| ext:A138 | 11.840010 | 11.546544 | -0.293465 | -12.616 → -11.749 |

## cross: mechanism checks

- no_rescue: parameters changed 0; on minus off >.25 0 / 0, >.05 0 / 0: none.
- no_spatial_gate: parameters changed 0; on minus off >.25 0 / 0, >.05 0 / 0: none.
- Holder white-point filter: 0: none.
- Rescue: 0: none.
- Cool cast: 9; clip_share > .0021: 0: none.

## cross: metric changes > .000001

| Frame | T4n | T4o | ΔE change | mean_dL T4n → T4o |
|---|---:|---:|---:|---:|
| blueneg:199807B-08-orlando-seaworld | 16.592069 | 16.020896 | -0.571173 | -15.935 → -15.628 |

## Protected and named dev frames

| Frame | T4n | T4o |
|---|---:|---:|
| ext:A008 | 8.720626 | 8.720626 |
| ext:A012 | 4.460079 | 4.460079 |
| ext:A013 | 6.661887 | 6.661887 |
| shop:DSC_3833 | 4.664065 | 4.664065 |
| nlp:_DSF8948 | 8.343404 | 8.343404 |
| ext:A104 | 9.610570 | 9.080654 |
| ext:A138 | 11.840010 | 11.546544 |
| ext:A006 | 13.516800 | 13.516800 |
| ext:A011 | 8.582616 | 8.582616 |
| ext:A196 | 12.393907 | 12.393907 |
| ext:A103 | 10.327800 | 9.095242 |
| blueneg:19970620C-10-st-michel | 16.963776 | 14.441746 |

## Devices (no paired reference)

| Sample | T4n status | T4o status | EV T4n → T4o | Temp T4n → T4o |
|---|---|---|---:|---:|
| A1195.CR2 | manual_review_required | ok | +0.0000 → -0.1958 | -0.0000 → -0.0395 |
| A240.CR3 | ok | ok | +1.6103 → +1.6103 | +0.0021 → +0.0021 |
| A328.CR3 | ok | ok | +1.6103 → +1.6103 | +0.0021 → +0.0021 |
| A329.ARW | ok | ok | +2.1000 → +2.1000 | -0.0356 → -0.0356 |
| A099.NEF | ok | ok | +1.2694 → +1.2694 | -0.0554 → -0.0554 |
| A1094.NEF | ok | ok | +2.4548 → +2.4548 | -0.0743 → -0.0743 |
| A1170.NEF | ok | ok | +1.2328 → +1.2328 | +0.5000 → +0.5000 |
| A1095.RAF | ok | ok | -0.1609 → -0.1609 | -0.0916 → -0.0916 |
| A1042.RAF | ok | ok | -0.1609 → -0.1609 | -0.0916 → -0.0916 |
| A277.dng | ok | ok | +0.6726 → +0.6726 | +0.0707 → +0.0707 |
| A281.RAF | ok | ok | +1.2153 → +1.2153 | -0.0676 → -0.0676 |
| A1005.CR2 | ok | ok | +0.0377 → +0.0377 | +0.0422 → +0.0422 |
| A1089.DNG | ok | ok | +0.9604 → +0.9604 | +0.3711 → +0.3711 |
| A322.cr2 | ok | ok | +0.8770 → +0.8770 | +0.0436 → +0.0436 |

## Check-set rescue targets: post-Step-5 inputs

| Frame | span G | textured median | q25 | q90 |
|---|---:|---:|---:|---:|
| blueneg:199807A-19-orlando-disney | 1.152295 | 0.238993 | 0.110628 | 0.435240 |
| blueneg:199807A-33-orlando-disney | 1.217829 | 0.588722 | 0.172840 | 0.679100 |

## Peak memory footprint (GiB)

- candidate_a: 4.66
- candidate_b: 4.60
- candidate_cross: 5.73
- devices_baseline: 2.97
- devices_candidate: 3.48
- tests: 2.96

Candidate source SHA256: `7d076c33c97f46d54c8a6e764228e2795e54adbdd363a30aa3ea398e379cf211`.
