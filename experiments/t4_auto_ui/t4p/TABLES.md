# T4p regression vs 4cd76b7

Per-frame median ΔE2000, then median / mean / p90 across frames.

| Set | N | T4o | T4p | Better / worse >.25 | Better / worse >.05 |
|---|---:|---:|---:|---:|---:|
| dev:nlp | 11 | 8.44 / 8.72 / 10.25 | 8.57 / 8.73 / 10.25 | 0 / 0 | 0 / 1 |
| dev:shop | 21 | 7.20 / 7.91 / 11.41 | 7.20 / 7.91 / 11.41 | 0 / 0 | 1 / 1 |
| dev:blueneg | 27 | 13.62 / 13.49 / 16.08 | 13.58 / 13.45 / 16.08 | 1 / 0 | 1 / 0 |
| dev:ext_own | 126 | 8.75 / 8.91 / 13.26 | 8.71 / 8.91 / 13.26 | 2 / 2 | 4 / 4 |
| dev:all | 185 | 9.03 / 9.45 / 14.14 | 9.01 / 9.45 / 14.15 | 3 / 2 | 6 / 6 |
| cross:all | 123 | 13.84 / 14.43 / 20.73 | 13.85 / 14.43 / 20.63 | 2 / 6 | 7 / 12 |

## dev: changes

| Frame | T4o | T4p | ΔE change | mean_dL before → after |
|---|---:|---:|---:|---:|
| blueneg:199807A-09-orlando-siggraph | 10.271164 | 9.162867 | -1.108297 | -9.323 → -2.953 |
| ext:A100 | 10.003059 | 9.739646 | -0.263413 | +9.820 → +9.402 |
| ext:A075 | 9.811174 | 9.550235 | -0.260939 | +4.843 → +4.005 |
| shop:DSC_3853 | 9.030893 | 8.847961 | -0.182932 | +8.042 → +8.112 |
| ext:A211 | 11.934733 | 11.832464 | -0.102269 | +15.470 → +15.480 |
| ext:A137 | 8.783843 | 8.705956 | -0.077887 | +0.864 → +0.898 |
| blueneg:19960812D-17-las-vegas | 13.622840 | 13.577164 | -0.045676 | +9.800 → +9.519 |
| blueneg:19960809B-14-new-orleans-siggraph96 | 19.706261 | 19.715198 | +0.008937 | +17.144 → +17.212 |
| blueneg:19941220D-20-kamakura | 14.143900 | 14.160224 | +0.016324 | +13.877 → +13.881 |
| ext:A086 | 7.122091 | 7.225169 | +0.103078 | -5.535 → -5.510 |
| nlp:_DSF8951 | 8.443679 | 8.570184 | +0.126505 | +1.316 → +1.246 |
| ext:A031 | 13.961106 | 14.094463 | +0.133357 | -11.840 → -11.853 |
| shop:DSC_3856 | 7.212537 | 7.454473 | +0.241935 | +4.851 → +4.919 |
| ext:A150 | 6.767307 | 7.242958 | +0.475651 | +0.849 → +0.902 |
| ext:A149 | 6.733339 | 7.425905 | +0.692566 | +4.442 → +5.307 |

## dev: ablations and highlight budget

- old_highlight_drop: >.25 1 / 0, >.05 1 / 0; parameters changed 1: blueneg:199807A-09-orlando-siggraph.
- old_sky_gate: >.25 0 / 0, >.05 0 / 0; parameters changed 0: none.
- clip_share > .002: blueneg:199807A-09-orlando-siggraph 0.002740.
- Cool-cast count: 13.
- Step 5 EV reductions: blueneg:199807A-09-orlando-siggraph 0.6000, ext:A035 0.1000.
- Active capped rescues: blueneg:19970620C-10-st-michel 0.598986, ext:A103 0.261435.
- WB parameters changed vs T4o: blueneg:19941220D-20-kamakura, blueneg:19960809B-14-new-orleans-siggraph96, blueneg:19960812D-17-las-vegas, blueneg:199807A-09-orlando-siggraph, ext:A031, ext:A075, ext:A086, ext:A100, ext:A137, ext:A149, ext:A150, ext:A211, nlp:_DSF8951, shop:DSC_3853, shop:DSC_3856.

## dev: preset results

| Preset | N | T4o | T4p | Better / worse >.25 | Better / worse >.05 |
|---|---:|---:|---:|---:|---:|
| standard | 185 | 9.03 / 9.45 / 14.14 | 9.01 / 9.45 / 14.15 | 3 / 2 | 6 / 6 |
| natural | 185 | 9.38 / 10.04 / 14.77 | 9.30 / 10.04 / 14.77 | 1 / 2 | 5 / 8 |
| portrait | 185 | 9.56 / 10.65 / 16.57 | 9.61 / 10.65 / 16.57 | 1 / 2 | 4 / 8 |
| vivid | 185 | 9.22 / 9.92 / 14.94 | 9.39 / 9.93 / 14.97 | 1 / 4 | 3 / 11 |
| print_classic | 185 | 8.46 / 9.07 / 13.61 | 8.47 / 9.07 / 13.61 | 2 / 3 | 4 / 8 |

## cross: changes

| Frame | T4o | T4p | ΔE change | mean_dL before → after |
|---|---:|---:|---:|---:|
| blueneg:199807A-33-orlando-disney | 20.862067 | 18.676226 | -2.185841 | -12.984 → -5.440 |
| blueneg:199807A-19-orlando-disney | 15.246108 | 13.909698 | -1.336410 | -17.437 → -14.120 |
| blueneg:199807B-05-orlando-seaworld | 12.075301 | 11.841853 | -0.233448 | -12.891 → -12.848 |
| blueneg:199807B-11-orlando-seaworld | 11.637817 | 11.429143 | -0.208675 | +7.207 → +7.208 |
| blueneg:199807B-09-orlando-seaworld | 5.891787 | 5.694343 | -0.197444 | -3.303 → -3.238 |
| blueneg:199807B-08-orlando-seaworld | 16.020896 | 15.875964 | -0.144932 | -15.628 → -15.536 |
| blueneg:199807A-20-orlando-disney | 13.393286 | 13.334207 | -0.059078 | -17.356 → -17.330 |
| blueneg:199807A-29-orlando-disney | 17.248382 | 17.233402 | -0.014980 | +18.012 → +18.017 |
| blueneg:19970619B-07-notre-dame | 13.843007 | 13.862647 | +0.019640 | +8.460 → +8.462 |
| blueneg:199807A-37-orlando-disney | 12.092827 | 12.134693 | +0.041866 | +0.535 → +0.545 |
| blueneg:19960808B-10-new-orleans-siggraph96 | 19.804979 | 19.868029 | +0.063050 | +4.411 → +4.404 |
| blueneg:199807B-06-orlando-seaworld | 13.685348 | 13.798014 | +0.112666 | +4.851 → +4.870 |
| blueneg:19960812D-32-las-vegas | 6.922634 | 7.058278 | +0.135645 | +7.497 → +7.508 |
| blueneg:19960812D-30-las-vegas | 21.040998 | 21.186446 | +0.145448 | +18.264 → +18.299 |
| blueneg:199807B-15-orlando-seaworld | 7.513821 | 7.675998 | +0.162177 | -3.920 → -3.917 |
| blueneg:199807A-08-hk | 15.915491 | 16.081108 | +0.165618 | +5.583 → +5.586 |
| blueneg:19941222E-10-nara | 13.520061 | 13.814218 | +0.294157 | +12.198 → +12.195 |
| blueneg:19960808B-08-new-orleans-siggraph96 | 8.543236 | 8.860527 | +0.317291 | +0.574 → +0.638 |
| blueneg:19941222E-02-nara | 14.380432 | 14.734955 | +0.354523 | -0.798 → -0.780 |
| blueneg:19960812D-35-las-vegas | 12.042438 | 12.457170 | +0.414733 | +11.377 → +11.689 |
| blueneg:19960812D-11-las-vegas | 19.013078 | 19.520566 | +0.507488 | +14.094 → +14.653 |
| blueneg:199807A-22-orlando-disney | 14.233180 | 15.946044 | +1.712863 | -5.145 → -5.105 |

## cross: ablations and highlight budget

- old_highlight_drop: >.25 0 / 0, >.05 0 / 0; parameters changed 0: none.
- old_sky_gate: >.25 0 / 0, >.05 0 / 0; parameters changed 0: none.
- clip_share > .002: none.
- Cool-cast count: 8.
- Step 5 EV reductions: none.
- Active capped rescues: blueneg:199807A-19-orlando-disney 0.400000, blueneg:199807A-33-orlando-disney 0.850000.
- WB parameters changed vs T4o: blueneg:19941222E-02-nara, blueneg:19941222E-10-nara, blueneg:19960808B-08-new-orleans-siggraph96, blueneg:19960808B-10-new-orleans-siggraph96, blueneg:19960812D-11-las-vegas, blueneg:19960812D-30-las-vegas, blueneg:19960812D-32-las-vegas, blueneg:19960812D-35-las-vegas, blueneg:19970619B-07-notre-dame, blueneg:199807A-08-hk, blueneg:199807A-20-orlando-disney, blueneg:199807A-22-orlando-disney, blueneg:199807A-29-orlando-disney, blueneg:199807A-33-orlando-disney, blueneg:199807A-37-orlando-disney, blueneg:199807B-05-orlando-seaworld, blueneg:199807B-06-orlando-seaworld, blueneg:199807B-08-orlando-seaworld, blueneg:199807B-09-orlando-seaworld, blueneg:199807B-11-orlando-seaworld, blueneg:199807B-15-orlando-seaworld.

## cross: preset results

| Preset | N | T4o | T4p | Better / worse >.25 | Better / worse >.05 |
|---|---:|---:|---:|---:|---:|
| standard | 123 | 13.84 / 14.43 / 20.73 | 13.85 / 14.43 / 20.63 | 2 / 6 | 7 / 12 |

## Additional comparison vs T4n

| Set | T4n | T4p | Better / worse >.25 | Better / worse >.05 |
|---|---:|---:|---:|---:|
| dev | 9.03 / 9.48 / 14.14 | 9.01 / 9.45 / 14.15 | 7 / 2 | 10 / 6 |
| cross | 13.84 / 14.43 / 20.73 | 13.85 / 14.43 / 20.63 | 3 / 6 | 7 / 12 |

## Protected frames

| Frame | T4o | T4p |
|---|---:|---:|
| ext:A008 | 8.720626 | 8.720626 |
| ext:A012 | 4.460079 | 4.460079 |
| ext:A013 | 6.661887 | 6.661887 |
| shop:DSC_3833 | 4.664065 | 4.664065 |
| nlp:_DSF8948 | 8.343404 | 8.343404 |

## Devices by mode

| Sample / mode | Before status | After status | EV before → after | Temp before → after | final clip |
|---|---|---|---:|---:|---:|
| A1195.CR2 / color | ok | ok | -0.1958 → -0.1958 | -0.0395 → -0.0395 | 2e-05 |
| A240.CR3 / color | ok | ok | +1.6103 → +1.6103 | +0.0021 → +0.0021 | 0.00074 |
| A328.CR3 / color | ok | ok | +1.6103 → +1.6103 | +0.0021 → +0.0021 | 0.00074 |
| A329.ARW / color | ok | ok | +2.1000 → +2.1000 | -0.0356 → -0.0356 | 0.0019 |
| A099.NEF / color | ok | ok | +1.2694 → +1.2694 | -0.0554 → -0.0554 | 0.00134 |
| A1094.NEF / color | ok | ok | +2.4548 → +2.4548 | -0.0743 → -0.0743 | 0.0017 |
| A1170.NEF / color | ok | ok | +1.2328 → +1.2445 | +0.5000 → +0.2508 | 0.0 |
| A1095.RAF / color | ok | ok | -0.1609 → -0.1609 | -0.0916 → -0.0916 | 2e-05 |
| A1042.RAF / color | ok | ok | -0.1609 → -0.1609 | -0.0916 → -0.0916 | 2e-05 |
| A277.dng / color | ok | ok | +0.6726 → +0.6726 | +0.0707 → +0.0530 | 6e-05 |
| A281.RAF / color | ok | ok | +1.2153 → +1.2153 | -0.0676 → -0.0676 | 0.0 |
| A1005.CR2 / color | ok | ok | +0.0377 → +0.0377 | +0.0422 → +0.0422 | 0.00016 |
| A1089.DNG / color | ok | ok | +0.9604 → +0.9604 | +0.3711 → +0.3711 | 2e-05 |
| A322.cr2 / color | ok | ok | +0.8770 → +0.8770 | +0.0436 → +0.0436 | 0.0 |
| A1036.RW2 / color | ok | ok | -0.0915 → -0.0915 | +0.5000 → +0.5000 | 0.00016 |
| B086.dng / color | IndexError: index 6 is out of bounds for axis 0 with size 4 | ok | — → +1.0466 | — → +0.0000 | 0.0 |
| B087.dng / color | IndexError: index 6 is out of bounds for axis 0 with size 4 | ok | — → +0.8882 | — → +0.0000 | 0.0 |
| B091.RW2 / color | ok | ok | +0.9787 → +0.9787 | +0.1745 → +0.1745 | 0.00196 |
| B086.dng / bw | IndexError: index 6 is out of bounds for axis 0 with size 4 | ok | — → +1.0466 | — → +0.0000 | 0.0 |
| B087.dng / bw | IndexError: index 6 is out of bounds for axis 0 with size 4 | ok | — → +0.8882 | — → +0.0000 | 0.0 |
| B091.RW2 / bw | ok | ok | -1.7651 → +0.2349 | +0.0000 → +0.0000 | 0.00456 |

## Device parameter changes and rescue evidence

- A240.CR3: parameter changes `{"midtone": [0.2777, 0.277], "shadows": [0.0954, 0.0939], "shoulder": [0.3365, 0.3355]}`; rescue weight 0.596183 → 0.587163.
- A328.CR3: parameter changes `{"midtone": [0.2777, 0.277], "shadows": [0.0954, 0.0939], "shoulder": [0.3365, 0.3355]}`; rescue weight 0.596183 → 0.587163.
- A329.ARW: parameter changes `{"midtone": [0.262, 0.298], "shadows": [0.064, 0.136]}`; rescue weight 0.400000 → 0.850000.
- A099.NEF: parameter changes `{}`; rescue weight 0.850000 → 0.850000.
- A1094.NEF: parameter changes `{}`; rescue weight 0.850000 → 0.850000.
- A1170.NEF: parameter changes `{"ev": [1.2328, 1.2445], "highlights": [-0.485, -0.65], "temp": [0.5, 0.2508], "tint": [0.0429, 0.051]}`; rescue weight 0.000000 → 0.000000.
- A277.dng: parameter changes `{"temp": [0.0707, 0.053], "tint": [0.0624, 0.0505]}`; rescue weight 0.000000 → 0.000000.
- B086.dng: parameter changes `{"chroma_preserve": [null, 0.4], "contrast": [null, 1.4], "ev": [null, 1.0466], "highlights": [null, -0.4], "matrix": [null, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], "midtone": [null, 0.32], "sat": [null, 1.02], "shoulder": [null, 0.37], "temp": [null, 0.0], "tint": [null, 0.015], "vibrance": [null, 0.08]}`; rescue weight 0.000000 → 0.000000.
- B087.dng: parameter changes `{"chroma_preserve": [null, 0.4], "contrast": [null, 1.4], "ev": [null, 0.8882], "highlights": [null, -0.478], "matrix": [null, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], "midtone": [null, 0.32], "sat": [null, 1.02], "shoulder": [null, 0.37], "temp": [null, 0.0], "tint": [null, 0.015], "vibrance": [null, 0.08]}`; rescue weight 0.000000 → 0.000000.

## Target diagnostics

- A1170.NEF / color: `{"display_rgb_q50_after": [0.7090400457382202, 0.8285925388336182, 0.9415260553359985], "display_rgb_q50_before": [0.7447077035903931, 0.8380891680717468, 0.9209932088851929], "highlight_hold": {"ev_lowered": 0.0, "max_drop": 0.6}, "raw_sha256": "d13ec77247acb2fa38b4bef37991d3a64ae0d374cab9368238d75b2ecb4cc230", "wb": {"bright_px_share": 0.5373149628900644, "cool_cast": false, "split_sky": true, "top_bottom_spread": 0.12469103932380676, "zone_disagreement": 0.29347220063209534}}`.
- A1036.RW2 / color: `{"display_rgb_q50_after": [0.44619715213775635, 0.591437041759491, 0.7481209635734558], "display_rgb_q50_before": [0.44619715213775635, 0.591437041759491, 0.7481209635734558], "highlight_hold": {"ev_lowered": 0.0, "max_drop": 0.6}, "raw_sha256": "e03ec5ffdd973cb78d2490f370a9c03ab8d237b20a4c9fe7a02389b3a43d913f", "wb": {"bright_px_share": 0.16748, "cool_cast": true, "split_sky": false, "top_bottom_spread": 0.04731471836566925, "zone_disagreement": 0.2677956521511078}}`.
- B091.RW2 / bw: `{"display_rgb_q50_after": [0.34950220584869385, 0.34950220584869385, 0.34950220584869385], "display_rgb_q50_before": [0.2045443207025528, 0.2045443207025528, 0.2045443207025528], "highlight_hold": {"ev_lowered": 0.6, "max_drop": 0.6}, "raw_sha256": "5b0a872f8dce4d04b5ee8e88f5dc6f75259fe692738e1168ef66af5828e9c5e4", "wb": {"bright_px_share": 0.10128322639780019, "cool_cast": false, "split_sky": false, "top_bottom_spread": 0.0, "zone_disagreement": 0.0}}`.

## Peak memory footprint (GiB)

- candidate_a: 5.36
- candidate_b: 6.08
- candidate_cross: 7.49
- devices_baseline_bw: 0.72
- devices_baseline_color: 4.03
- devices_candidate_bw: 1.05
- devices_candidate_color: 2.76
- tests: 2.99

## Full pytest

376 passed in 394.26s; peak memory footprint 2.99 GiB.

Candidate source SHA256: `45c7818133e3fb33b6f5aca24fcc3814783e3a905b1f591cc699cdb35f3efbd6`.

## Local service

`{"ev": 0.5492, "flagged": 0, "highlight_hold": {"ev_lowered": 0.0, "max_drop": 0.6}, "render_png_bytes": 3362409, "render_sha256": "b3f4c6c3616115ab02f3b7119dac75e8d1768242ee4c2b3e347133111dca75ff", "sample": "_DSF9239.RAF", "source_config_rev": "t4p-2026-10-31", "source_thumb_rev": "t4p", "status": "ok", "sub_rolls": 6, "url": "http://127.0.0.1:62872"}`.
