# T4q verified regression

Per-frame median ΔE2000, then median / mean / p90 across frames.

| Set | N | T4o | T4q | Better / worse >.25 | Better / worse >.05 |
|---|---:|---:|---:|---:|---:|
| dev:nlp | 11 | 8.44 / 8.72 / 10.25 | 8.44 / 8.72 / 10.25 | 0 / 0 | 0 / 0 |
| dev:shop | 21 | 7.20 / 7.91 / 11.41 | 7.20 / 7.91 / 11.41 | 0 / 0 | 0 / 0 |
| dev:blueneg | 27 | 13.62 / 13.49 / 16.08 | 13.62 / 13.45 / 16.08 | 1 / 0 | 1 / 0 |
| dev:ext_own | 126 | 8.75 / 8.91 / 13.26 | 8.65 / 8.82 / 13.16 | 4 / 0 | 4 / 0 |
| dev:all | 185 | 9.03 / 9.45 / 14.14 | 9.01 / 9.38 / 14.09 | 5 / 0 | 5 / 0 |
| cross:all | 123 | 13.84 / 14.43 / 20.73 | 13.77 / 14.31 / 20.63 | 9 / 0 | 10 / 0 |

## dev: changes vs T4o

| Frame | T4o | T4q | Change | mean_dL before → after |
|---|---:|---:|---:|---:|
| ext:A212 | 15.420377 | 10.948056 | -4.472321 | +15.390 → +10.600 |
| ext:A218 | 10.968294 | 7.841977 | -3.126317 | +13.406 → +7.358 |
| ext:A211 | 11.934733 | 9.376038 | -2.558696 | +15.470 → +11.272 |
| blueneg:199807A-09-orlando-siggraph | 10.271164 | 9.078063 | -1.193101 | -9.323 → -2.937 |
| ext:A105 | 14.973155 | 13.810336 | -1.162819 | +16.867 → +15.473 |

## dev: over-lift ablation

Applied 4; >.25 4 / 0, >.05 4 / 0.

| Frame | Off ΔE | On ΔE | Diagnostic |
|---|---:|---:|---|
| ext:A105 | 14.973155 | 13.810336 | `{"applied": true, "blacks_lowered": 0.012253834629058834, "ev_after": 1.921682466816471, "ev_before": 2.044220813107059, "pull": 0.12253834629058834, "q25_before": 0.36829323321580887}` |
| ext:A211 | 11.934733 | 9.376038 | `{"applied": true, "blacks_lowered": 0.03809961023330688, "ev_after": 1.685348441784034, "ev_before": 2.066344544117103, "pull": 0.3809961023330688, "q25_before": 0.449061281979084}` |
| ext:A212 | 15.420377 | 10.948056 | `{"applied": true, "blacks_lowered": 0.03703846397399902, "ev_after": 1.779626967784523, "ev_before": 2.150011607524513, "pull": 0.3703846397399902, "q25_before": 0.44574519991874695}` |
| ext:A218 | 10.968294 | 7.841977 | `{"applied": true, "blacks_lowered": 0.05766087474822998, "ev_after": 1.9052714405059814, "ev_before": 2.4818801879882812, "pull": 0.5766087474822997, "q25_before": 0.5101902335882187}` |

Module A alone vs T4o: >.25 1 / 0, >.05 1 / 0.
Module A parameter changes: blueneg:199807A-09-orlando-siggraph.
Applied frames with strictly lower / higher ΔE (no tolerance): 4 / 0.
Final clip > .002: blueneg:199807A-09-orlando-siggraph 0.002780.

## dev: presets vs T4o

| Preset | T4o | T4q | Better / worse >.25 | Better / worse >.05 |
|---|---:|---:|---:|---:|
| standard | 9.03 / 9.45 / 14.14 | 9.01 / 9.38 / 14.09 | 5 / 0 | 5 / 0 |
| natural | 9.38 / 10.04 / 14.77 | 9.21 / 9.96 / 14.63 | 9 / 2 | 10 / 3 |
| portrait | 9.56 / 10.65 / 16.57 | 9.52 / 10.51 / 16.18 | 15 / 1 | 17 / 3 |
| vivid | 9.22 / 9.92 / 14.94 | 9.04 / 9.86 / 14.84 | 4 / 0 | 4 / 0 |
| print_classic | 8.46 / 9.07 / 13.61 | 8.45 / 9.03 / 13.61 | 4 / 0 | 4 / 0 |

natural regressions >.05: ext:A033 +1.457500, ext:A038 +0.442415, ext:A123 +0.153779.

portrait regressions >.05: ext:A033 +1.246058, ext:A038 +0.140025, ext:A123 +0.171874.

vivid regressions >.05: none.

print_classic regressions >.05: none.

WB parameter changes vs T4o: none.
Cool-cast count: 13.

Prior T4p regressions >.05 remaining: none.

T4q vs uncommitted T4p: >.25 6 / 2, >.05 11 / 4.
Historical T4p vs T4o recomputed: >.25 3 / 2, >.05 6 / 6.

## cross: changes vs T4o

| Frame | T4o | T4q | Change | mean_dL before → after |
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

## cross: over-lift ablation

Applied 9; >.25 7 / 0, >.05 8 / 0.

| Frame | Off ΔE | On ΔE | Diagnostic |
|---|---:|---:|---|
| blueneg:19941220D-22-kamakura | 12.601108 | 12.151222 | `{"applied": true, "blacks_lowered": 0.009536761665344236, "ev_after": 1.9932913071765725, "ev_before": 2.088658923830015, "pull": 0.09536761665344234, "q25_before": 0.35980238020420074}` |
| blueneg:19960808B-07-new-orleans-siggraph96 | 19.314945 | 17.967643 | `{"applied": true, "blacks_lowered": 0.013224787139892575, "ev_after": 2.367752128601074, "ev_before": 2.5, "pull": 0.13224787139892574, "q25_before": 0.3713274598121643}` |
| blueneg:19960808B-10-new-orleans-siggraph96 | 19.804979 | 18.777104 | `{"applied": true, "blacks_lowered": 0.04444205800330365, "ev_after": 1.1441544105538497, "ev_before": 1.5885749905868862, "pull": 0.44442058003303647, "q25_before": 0.5706331729888916}` |
| blueneg:19960811B-36-laughlin | 5.955943 | 5.955885 | `{"applied": true, "blacks_lowered": 0.01919134435075923, "ev_after": 1.4748948840396965, "ev_before": 1.6668083275472887, "pull": 0.1919134435075923, "q25_before": 0.4117497131228447}` |
| blueneg:19960812D-27-las-vegas | 15.775887 | 13.520462 | `{"applied": true, "blacks_lowered": 0.02354046764373779, "ev_after": 1.989044804350341, "ev_before": 2.224449480787719, "pull": 0.2354046764373779, "q25_before": 0.4035639613866806}` |
| blueneg:199807A-36-orlando-disney | 12.097995 | 10.850510 | `{"applied": true, "blacks_lowered": 0.029309060001373288, "ev_after": 1.6549645595436169, "ev_before": 1.9480551595573499, "pull": 0.2930906000137329, "q25_before": 0.42159081250429153}` |
| blueneg:199807A-37-orlando-disney | 12.092827 | 11.861910 | `{"applied": true, "blacks_lowered": 0.02308096590042114, "ev_after": 1.8674759654035378, "ev_before": 2.0982856244077492, "pull": 0.23080965900421138, "q25_before": 0.40212801843881607}` |
| blueneg:199807B-01-orlando-seaworld | 15.869348 | 12.199066 | `{"applied": true, "blacks_lowered": 0.04323419575683165, "ev_after": 1.312303515838536, "ev_before": 1.7446454734068526, "pull": 0.43234195756831645, "q25_before": 0.48192650079727173}` |
| blueneg:199807B-03-orlando-seaworld | 16.575918 | 16.138214 | `{"applied": true, "blacks_lowered": 0.00491058025905864, "ev_after": 1.479129326098695, "ev_before": 1.5282351286892815, "pull": 0.049105802590586396, "q25_before": 0.3636178821325302}` |

Module A alone vs T4o: >.25 2 / 0, >.05 2 / 0.
Module A parameter changes: blueneg:199807A-19-orlando-disney, blueneg:199807A-33-orlando-disney.
Applied frames with strictly lower / higher ΔE (no tolerance): 9 / 0.
Final clip > .002: none.

## cross: presets vs T4o

| Preset | T4o | T4q | Better / worse >.25 | Better / worse >.05 |
|---|---:|---:|---:|---:|
| standard | 13.84 / 14.43 / 20.73 | 13.77 / 14.31 / 20.63 | 9 / 0 | 10 / 0 |

WB parameter changes vs T4o: none.
Cool-cast count: 9.

Prior T4p regressions >.05 remaining: none.

T4q vs uncommitted T4p: >.25 14 / 0, >.05 19 / 5.
Historical T4p vs T4o recomputed: >.25 2 / 6, >.05 7 / 12.

## Protected frames

- ext:A008: unchanged, ΔE 8.720626.
- ext:A012: unchanged, ΔE 4.460079.
- ext:A013: unchanged, ΔE 6.661887.
- shop:DSC_3833: unchanged, ΔE 4.664065.
- nlp:_DSF8948: unchanged, ΔE 8.343404.

## Devices

| File / mode | 4cd76b7 source status (probed during T4p) | T4q status | EV | Temp | Clip |
|---|---|---|---:|---:|---:|
| A1195.CR2 / color | ok | ok | -0.1958 | -0.0395 | 0.000020 |
| A240.CR3 / color | ok | ok | +1.6103 | +0.0021 | 0.000740 |
| A328.CR3 / color | ok | ok | +1.6103 | +0.0021 | 0.000740 |
| A329.ARW / color | ok | ok | +2.1000 | -0.0356 | 0.001900 |
| A099.NEF / color | ok | ok | +1.2694 | -0.0554 | 0.001340 |
| A1094.NEF / color | ok | ok | +2.4548 | -0.0743 | 0.001700 |
| A1170.NEF / color | ok | ok | +1.2445 | +0.2508 | 0.000000 |
| A1095.RAF / color | ok | ok | -0.1609 | -0.0916 | 0.000020 |
| A1042.RAF / color | ok | ok | -0.1609 | -0.0916 | 0.000020 |
| A277.dng / color | ok | ok | +0.6726 | +0.0707 | 0.000060 |
| A281.RAF / color | ok | ok | +1.2153 | -0.0676 | 0.000000 |
| A1005.CR2 / color | ok | ok | +0.0377 | +0.0422 | 0.000160 |
| A1089.DNG / color | ok | ok | +0.9604 | +0.3711 | 0.000020 |
| A322.cr2 / color | ok | ok | +0.8770 | +0.0436 | 0.000000 |
| A1036.RW2 / color | ok | ok | -0.0915 | +0.5000 | 0.000160 |
| B086.dng / color | IndexError: index 6 is out of bounds for axis 0 with size 4 | ok | +1.0466 | +0.0000 | 0.000000 |
| B087.dng / color | IndexError: index 6 is out of bounds for axis 0 with size 4 | ok | +0.8882 | +0.0000 | 0.000000 |
| B091.RW2 / color | ok | ok | +0.9787 | +0.1745 | 0.001960 |
| B086.dng / bw | IndexError: index 6 is out of bounds for axis 0 with size 4 | ok | +1.0466 | +0.0000 | 0.000000 |
| B087.dng / bw | IndexError: index 6 is out of bounds for axis 0 with size 4 | ok | +0.8882 | +0.0000 | 0.000000 |
| B091.RW2 / bw | ok | ok | +0.2349 | +0.0000 | 0.004560 |

## Device over-lift and spatial diagnostics

- A1170.NEF: `{"overlift": null, "wb": {"bright_px_share": 0.5373149628900644, "split_sky": true, "top_bottom_spread": null, "zone_disagreement": 0.29347220063209534}}`.
- A277.dng: `{"overlift": null, "wb": {"bright_px_share": 0.4448949350676079, "split_sky": false, "top_bottom_spread": null, "zone_disagreement": 0.7423045635223389}}`.
- A1036.RW2: `{"overlift": null, "wb": {"bright_px_share": 0.16748, "split_sky": false, "top_bottom_spread": null, "zone_disagreement": 0.2677956521511078}}`.

## Full pytest

377 passed in 339.71s.

## Peak memory footprint (GiB)

- candidate_a: 5.75
- candidate_cross: 6.37
- devices_candidate_bw: 1.08
- devices_candidate_color: 3.43
- tests: 2.77
- candidate_b: unavailable (no peak-memory statistic in log).

## Service

`{"ev": 0.5492, "flagged": 0, "highlight_hold": {"ev_lowered": 0.0, "max_drop": 0.6}, "render_png_bytes": 3362409, "render_sha256": "b3f4c6c3616115ab02f3b7119dac75e8d1768242ee4c2b3e347133111dca75ff", "sample": "_DSF9239.RAF", "source_config_rev": "t4q-2026-11-01", "source_thumb_rev": "t4q", "status": "ok", "sub_rolls": 6, "url": "http://127.0.0.1:62872"}`

Source SHA256: `8abc02b946fcd9408cf2f3eaff137837781a4841387facb6f874259ceadf5f4c`.
