# T4r verified regression

Per-frame median ΔE2000, then median / mean / p90 across frames.

| Set | N | T4o | T4q | T4r | vs T4o >.25 | vs T4o >.05 | vs T4q >.25 | vs T4q >.05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev:nlp | 11 | 8.44 / 8.72 / 10.25 | 8.44 / 8.72 / 10.25 | 8.44 / 8.72 / 10.25 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| dev:shop | 21 | 7.20 / 7.91 / 11.41 | 7.20 / 7.91 / 11.41 | 7.20 / 7.91 / 11.41 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| dev:blueneg | 27 | 13.62 / 13.49 / 16.08 | 13.62 / 13.45 / 16.08 | 13.62 / 13.45 / 16.08 | 1 / 0 | 1 / 0 | 0 / 0 | 0 / 0 |
| dev:ext_own | 126 | 8.75 / 8.91 / 13.26 | 8.65 / 8.82 / 13.16 | 8.65 / 8.82 / 13.16 | 4 / 0 | 4 / 0 | 0 / 0 | 0 / 0 |
| dev:all | 185 | 9.03 / 9.45 / 14.14 | 9.01 / 9.38 / 14.09 | 9.01 / 9.38 / 14.09 | 5 / 0 | 5 / 0 | 0 / 0 | 0 / 0 |
| cross:all | 123 | 13.84 / 14.43 / 20.73 | 13.77 / 14.31 / 20.63 | 13.77 / 14.31 / 20.63 | 9 / 0 | 10 / 0 | 0 / 0 | 0 / 0 |

## dev: changes vs T4q (standard)

| Frame | T4q | T4r | Change | mean_dL before → after |
|---|---:|---:|---:|---:|

## dev: changes vs T4o (standard)

| Frame | T4o | T4r | Change | mean_dL before → after |
|---|---:|---:|---:|---:|
| ext:A212 | 15.420377 | 10.948056 | -4.472321 | +15.390 → +10.600 |
| ext:A218 | 10.968294 | 7.841977 | -3.126317 | +13.406 → +7.358 |
| ext:A211 | 11.934733 | 9.376038 | -2.558696 | +15.470 → +11.272 |
| blueneg:199807A-09-orlando-siggraph | 10.271164 | 9.078063 | -1.193101 | -9.323 → -2.937 |
| ext:A105 | 14.973155 | 13.810336 | -1.162819 | +16.867 → +15.473 |

## dev: presets vs T4o

| Preset | T4o | T4r | Better / worse >.25 | Better / worse >.05 | Max regression |
|---|---:|---:|---:|---:|---:|
| standard | 9.03 / 9.45 / 14.14 | 9.01 / 9.38 / 14.09 | 5 / 0 | 5 / 0 | +0.0000 |
| natural | 9.38 / 10.04 / 14.77 | 9.21 / 9.95 / 14.63 | 9 / 0 | 10 / 0 | +0.0000 |
| portrait | 9.56 / 10.65 / 16.57 | 9.52 / 10.50 / 16.18 | 14 / 0 | 16 / 0 | +0.0000 |
| vivid | 9.22 / 9.92 / 14.94 | 9.04 / 9.86 / 14.84 | 4 / 0 | 4 / 0 | +0.0000 |
| print_classic | 8.46 / 9.07 / 13.61 | 8.45 / 9.03 / 13.61 | 4 / 0 | 4 / 0 | +0.0093 |

## dev: presets vs T4q

| Preset | T4q | T4r | Better / worse >.25 | Better / worse >.05 | Max regression |
|---|---:|---:|---:|---:|---:|
| standard | 9.01 / 9.38 / 14.09 | 9.01 / 9.38 / 14.09 | 0 / 0 | 0 / 0 | +0.0000 |
| natural | 9.21 / 9.96 / 14.63 | 9.21 / 9.95 / 14.63 | 2 / 0 | 3 / 0 | +0.0000 |
| portrait | 9.52 / 10.51 / 16.18 | 9.52 / 10.50 / 16.18 | 1 / 1 | 3 / 1 | +0.3142 |
| vivid | 9.04 / 9.86 / 14.84 | 9.04 / 9.86 / 14.84 | 0 / 0 | 0 / 0 | +0.0000 |
| print_classic | 8.45 / 9.03 / 13.61 | 8.45 / 9.03 / 13.61 | 0 / 0 | 0 / 0 | +0.0000 |

### dev: preset regressions >.05 vs T4q

- **standard**: none
- **natural**: none
- **portrait**: ext:A094 +0.314241
- **vivid**: none
- **print_classic**: none

### dev: preset regressions >.05 vs T4o

- **standard**: none
- **natural**: none
- **portrait**: none
- **vivid**: none
- **print_classic**: none

## dev: old_wb_gate ablation (standard WB new gate vs old gate)

Touched frames: 0: none. >.25 0 / 0, >.05 0 / 0.

## dev: over-lift wide-bright exemption ablation (new standard vs old_overlift_*)

- **standard**: touched 0 frames: none; >.25 0 / 0, >.05 0 / 0.
- **natural**: touched 3 frames: ext:A033, ext:A038, ext:A123; >.25 2 / 0, >.05 3 / 0.
  - `ext:A033`: with exemption 5.403680 vs without 6.861180 (-1.457500)
  - `ext:A038`: with exemption 5.119303 vs without 5.561717 (-0.442415)
  - `ext:A123`: with exemption 5.785353 vs without 5.939132 (-0.153779)
- **portrait**: touched 4 frames: ext:A033, ext:A038, ext:A094, ext:A123; >.25 1 / 1, >.05 3 / 1.
  - `ext:A033`: with exemption 4.904283 vs without 6.150341 (-1.246058)
  - `ext:A038`: with exemption 5.663124 vs without 5.803148 (-0.140025)
  - `ext:A094`: with exemption 8.317867 vs without 8.003626 (+0.314241)
  - `ext:A123`: with exemption 5.891131 vs without 6.063005 (-0.171874)
- **vivid**: touched 0 frames: none; >.25 0 / 0, >.05 0 / 0.
- **print_classic**: touched 0 frames: none; >.25 0 / 0, >.05 0 / 0.

### dev: base diagnostics and spike clamp statistics

- `bare_backlight_excluded` active on 0 frames: none.
- `spike_clamp` active on 185 frames, total clamped pixels: 32652220.
- Spike-clamp report changed vs T4q: blueneg:19970619B-09-notre-dame.
- Final clip > .002: blueneg:199807A-09-orlando-siggraph 0.002780.

## cross: changes vs T4q (standard)

| Frame | T4q | T4r | Change | mean_dL before → after |
|---|---:|---:|---:|---:|

## cross: changes vs T4o (standard)

| Frame | T4o | T4r | Change | mean_dL before → after |
|---|---:|---:|---:|---:|
| blueneg:199807B-01-orlando-seaworld | 15.869348 | 12.199066 | -3.670282 | +17.127 → +11.841 |
| blueneg:19960812D-27-las-vegas | 15.775887 | 13.520462 | -2.255425 | +14.781 → +12.134 |
| blueneg:199807A-33-orlando-disney | 20.862067 | 18.681808 | -2.180259 | -12.984 → -5.514 |
| blueneg:19960808B-07-new-orleans-siggraph96 | 19.314945 | 17.967643 | -1.347302 | +15.925 → +14.405 |
| blueneg:199807A-19-orlando-disney | 15.246108 | 13.909698 | -1.336410 | -17.437 → -14.120 |
| blueneg:199807A-36-orlando-disney | 12.097995 | 10.850510 | -1.247485 | +5.808 → +2.787 |
| blueneg:19960808B-10-new-orleans-siggraph96 | 19.804979 | 18.777104 | -1.027875 | +4.411 → -0.140 |
| blueneg:19941220D-22-kamakura | 12.601108 | 12.151222 | -0.449887 | +14.555 → +13.537 |
| blueneg:199807B-03-orlando-seaworld | 16.575918 | 16.138214 | -0.437705 | +17.868 → +17.284 |
| blueneg:199807A-37-orlando-disney | 12.092827 | 11.861910 | -0.230917 | +0.535 → -1.613 |
| blueneg:19960811B-36-laughlin | 5.955943 | 5.955885 | -0.000058 | +0.819 → -0.515 |

## cross: presets vs T4o

| Preset | T4o | T4r | Better / worse >.25 | Better / worse >.05 | Max regression |
|---|---:|---:|---:|---:|---:|
| standard | 13.84 / 14.43 / 20.73 | 13.77 / 14.31 / 20.63 | 9 / 0 | 10 / 0 | +0.0000 |

## cross: presets vs T4q

| Preset | T4q | T4r | Better / worse >.25 | Better / worse >.05 | Max regression |
|---|---:|---:|---:|---:|---:|
| standard | 13.77 / 14.31 / 20.63 | 13.77 / 14.31 / 20.63 | 0 / 0 | 0 / 0 | +0.0000 |

### cross: preset regressions >.05 vs T4q

- **standard**: none

### cross: preset regressions >.05 vs T4o

- **standard**: none

## cross: old_wb_gate ablation (standard WB new gate vs old gate)

Touched frames: 0: none. >.25 0 / 0, >.05 0 / 0.

## cross: over-lift wide-bright exemption ablation (new standard vs old_overlift_*)

- **standard**: touched 0 frames: none; >.25 0 / 0, >.05 0 / 0.

### cross: base diagnostics and spike clamp statistics

- `bare_backlight_excluded` active on 0 frames: none.
- `spike_clamp` active on 123 frames, total clamped pixels: 21281915.
- Spike-clamp report changed vs T4q: none.
- Final clip > .002: none.

## Protected frames

- ext:A008: unchanged, ΔE 8.720626.
- ext:A012: unchanged, ΔE 4.460079.
- ext:A013: unchanged, ΔE 6.661887.
- shop:DSC_3833: unchanged, ΔE 4.664065.
- nlp:_DSF8948: unchanged, ΔE 8.343404.

## Devices

| File / mode | 4d31a38 baseline status | T4r status | Full finite | Preserves pos | EV | Temp | Clip |
|---|---|---|---|---|---:|---:|---:|
| A1195.CR2 / color | ok | ok | True | True | -0.1958 | -0.0395 | 0.000020 |
| A240.CR3 / color | ok | ok | True | True | +1.6103 | +0.0021 | 0.000740 |
| A328.CR3 / color | ok | ok | True | True | +1.6103 | +0.0021 | 0.000740 |
| A329.ARW / color | ok | ok | True | True | +2.1000 | -0.0356 | 0.001900 |
| A099.NEF / color | ok | ok | True | True | +1.2694 | -0.0554 | 0.001340 |
| A1094.NEF / color | ok | ok | True | True | +2.4548 | -0.0743 | 0.001700 |
| A1170.NEF / color | ok | ok | True | True | +1.2445 | +0.2508 | 0.000000 |
| A1095.RAF / color | ok | ok | True | True | -0.1609 | -0.0916 | 0.000020 |
| A1042.RAF / color | ok | ok | True | True | -0.1609 | -0.0916 | 0.000020 |
| A277.dng / color | ok | ok | True | True | +0.6726 | +0.0707 | 0.000060 |
| A281.RAF / color | ok | ok | True | True | +1.2153 | -0.0676 | 0.000000 |
| A1005.CR2 / color | ok | ok | True | True | +0.0377 | +0.0422 | 0.000160 |
| A1089.DNG / color | ok | ok | True | True | +0.9604 | +0.3711 | 0.000020 |
| A322.cr2 / color | ok | ok | True | True | +0.8770 | +0.0436 | 0.000000 |
| A1036.RW2 / color | ok | ok | True | True | -0.0915 | +0.3448 | 0.000220 |
| A1296.CR2 / color | manual_review_required | ok | True | True | -0.2500 | +0.0458 | 0.000640 |
| A1102.ARW / color | ok | ok | True | True | +0.3742 | +0.2357 | 0.000260 |
| B086.dng / color | ok | ok | True | True | +1.0466 | +0.0000 | 0.000000 |
| B087.dng / color | ok | ok | True | True | +0.8882 | +0.0000 | 0.000000 |
| B091.RW2 / color | ok | ok | True | True | +0.9787 | +0.1745 | 0.001960 |
| B086.dng / bw | ok | ok | True | True | +1.0466 | +0.0000 | 0.000000 |
| B087.dng / bw | ok | ok | True | True | +0.8882 | +0.0000 | 0.000000 |
| B091.RW2 / bw | ok | ok | True | True | +0.2349 | +0.0000 | 0.004560 |

## Device spatial and over-lift diagnostics

- A1036.RW2: `{"bare_backlight_excluded": false, "flags": [], "holder_share": 0.21811837423425406, "overlift": null, "own_base": [0.25462695956230164, 0.3387156128883362, 0.20058158040046692], "positive_q": [[0.15251459926366806, 0.2757265120744705, 0.5701816976070404], [0.6736966687440874, 0.6669176101684572, 0.9201716732978821], [1.3409537976980583, 1.4535547878743194, 1.7167336126567052], [4.608442306518555, 4.610048294067383, 5.950956344604492]], "status": "ok", "wb": {"bright_px_share": 0.16748, "split_sky": true, "zone_disagreement": 0.2677956521511078}}`
- A1102.ARW: `{"bare_backlight_excluded": false, "flags": [], "holder_share": 0.07099244577712562, "overlift": null, "own_base": [0.19140954315662384, 0.29453572630882263, 0.17794553935527802], "positive_q": [[0.020932571962475777, 0.08023940771818161, 0.08856526017189026], [0.7781441617012024, 0.8083353257179263, 0.7997677302360535], [1.2720060265064574, 1.2368143928051265, 1.2649423694610828], [15.635048866271973, 4.616933822631836, 2.7216744422912598]], "status": "ok", "wb": {"bright_px_share": 0.04435855329034325, "split_sky": false, "zone_disagreement": 0.6224226355552673}}`
- A1170.NEF: `{"bare_backlight_excluded": false, "flags": [], "holder_share": 7.870793061108838e-07, "overlift": null, "own_base": [0.10602755099534988, 0.09965334832668304, 0.04002755135297775], "positive_q": [[0.28668221831321716, 0.4823896884918213, 0.7537619173526764], [0.771783776879307, 0.8305235171318053, 1.0220677995681762], [1.1157365325689739, 1.09353466427329, 1.1270865504741896], [1.5552006959915161, 1.4589835405349731, 2.523036003112793]], "status": "ok", "wb": {"bright_px_share": 0.5373149628900644, "split_sky": true, "zone_disagreement": 0.29347220063209534}}`
- A1296.CR2: `{"bare_backlight_excluded": true, "flags": ["white_point_unrepresentative"], "holder_share": 0.12657375945897145, "overlift": null, "own_base": [0.014196043834090233, 0.009091353043913841, 0.002718671690672636], "positive_q": [[0.5712475776672363, 0.5858709216117859, 0.6457433700561523], [1.1812818932533269, 1.215037972927094, 1.246609988212586], [1.3981621868610536, 1.46648540306093, 1.5506432664394494], [5.102916717529297, 58074960.0, 31750616.0]], "status": "ok", "wb": {"bright_px_share": 0.63712, "split_sky": false, "top_bottom_spread": 0.023675059899687767, "zone_disagreement": 0.04073610156774521}}`
- A277.dng: `{"bare_backlight_excluded": false, "flags": [], "holder_share": 0.0, "overlift": null, "own_base": [0.697440505027771, 0.375554621219635, 0.31903335452079773], "positive_q": [[0.23166946321725845, 0.4321540296077728, 0.29275448620319366], [0.7832855749130249, 0.886133517622947, 0.9501806664466858], [1.1309524887800217, 1.1340506575107576, 1.040377071261426], [12.92824935913086, 3.9064087867736816, 3.642617702484131]], "status": "ok", "wb": {"bright_px_share": 0.4448949350676079, "split_sky": true, "zone_disagreement": 0.7423045635223389}}`

## Full pytest

385 passed in 389.71s.

## Peak memory footprint (GiB)

- candidate_a: 5.95
- candidate_b: 5.10
- candidate_cross: 7.86
- devices_baseline_bw: 1.08
- devices_baseline_color: 3.14
- devices_candidate_bw: 1.08
- devices_candidate_color: 3.22
- tests: 2.97

## Service

`{"ev": 0.5492, "flagged": 0, "highlight_hold": {"ev_lowered": 0.0, "max_drop": 0.6}, "render_png_bytes": 3362409, "render_sha256": "b3f4c6c3616115ab02f3b7119dac75e8d1768242ee4c2b3e347133111dca75ff", "sample": "_DSF9239.RAF", "source_config_rev": "t4r-2026-11-01", "source_thumb_rev": "t4r", "status": "ok", "sub_rolls": 6, "url": "http://127.0.0.1:62872"}`

Source SHA256: `276fa74b10912b20a38f47ee8edd7db2cbe59412d02ae90b80ff05c14afca1a1`.
