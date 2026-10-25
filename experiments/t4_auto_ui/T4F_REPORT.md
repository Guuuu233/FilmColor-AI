# T4f (2026-10-25): sky-frame underexposure fix and five-preset retuning — results

Code commit `163fb63` (314 tests pass). The evaluation in `t4f/eval_t4f_{a,b}.json` ran on the committed code
(`git_commit=163fb63`, `src_dirty=false`). Script, frame list and alignment are the same as T4d/T4e
(`scripts/t4d_source_eval.py`, 191 dev frames; 6 reference JPEGs have no ICC profile and are left out, leaving 185 frames,
all with alignment NCC ≥ 0.7), so the results compare directly with the T4c baseline in `t4d/` and with `t4e/`.
`render.py`, `invert.py`, `preview.py` and `base.py` are unchanged between `bf445cf` (T4e) and `163fb63`.
Tables are produced by `scripts/t4d_summary.py`, which gains a dE p90 column (90th percentile, across frames, of each
frame's median dE2000) and `--ev-below`.

Images are not committed:
- `a013_presets_check.jpg`: A013 with T4e standard, then the five T4f presets, then the reference.
- `presets_showcase.jpg`: T4f, 8 frames (A013 added) × five presets + reference.
- The T4e sheet is renamed `presets_showcase_t4e.jpg`.

Parameters are in the `.json` file of the same name. Tiles are shown in scan orientation (`rot90=0`): the scripts open
symlinked roll folders under `/tmp`, so they do not pick up orientations set in the app. T4e behaved the same way.

## 1. Conclusions and open decisions

1. **Bug 1 is fixed (the main cause).** All presets on all 185 frames now have EV ≥ −0.25; T4e had 63 frames below
   −0.30.
   - **A013:** EV −1.08 → +0.58 on all five presets; ΔE 19.62 → 5.84 (standard); L\* difference −28.0 → −4.3.
   - **The 63 frames:** ΔE median 14.67 → 10.01, mean 14.90 → 9.97, p90 21.65 → 14.54; 44 better and 17 worse than T4c.
   - **NLP `_DSF9231`:** 15.64 → 6.37. **Shop `DSC_3843`:** 18.18 → 7.03. **A002:** 23.63 → 5.80.
2. **A013's EV is not inside the task list's expected −0.10 to +0.35.** It is +0.58, and at +0.58 it is still
   4.3 L\* darker than the reference. That range does not match the data. I did not force EV into it.
3. **Beyond the task list: the full-frame p98 ceiling also limits the anchor EV** (`ev_anchor_p98_cap`, on by default).
   - The task list puts the ceiling only on the "refinement brightening" step. But on backlit frames the textured anchor
     alone gives +2.50 EV, so A012 and A023 went to +2.50 EV (ΔE 6.12 / 8.35, L\* +9.0 / +11.8).
   - With the ceiling: A012 +0.95 EV, ΔE 4.09; A023 +0.88 EV, ΔE 3.86.
   - 185-frame ΔE median 10.32 → 10.01, mean 11.52 → 10.90, p90 17.52 → 16.95 (section 3, column `task_literal`).
   - Please confirm whether to keep this.
4. **The new preset table makes images too bright overall.** Standard on 185 frames: ΔE median 10.44 → 10.01, mean
   12.04 → 10.90, p90 20.28 → 16.95. But the dL median went from −2.34 to **+4.50** (L\* 64.5 vs reference 58.1).
   - **The 122 frames that were not underexposed got worse:** ΔE median 9.42 → 10.05, 35 better / 74 worse.
   - **Same T4f algorithm with T4e's tone and colour values** (column `t4e_table`: band 0.45–0.52, midtone 0.23,
     contrast 1.30, shoulder 0.20, sat 1.02, vibrance 0.08, chroma_preserve / camera matrix 0.35):
     - 185 frames: median 9.72, mean 10.33, p90 15.76, 89 better / 72 worse, dL +2.49;
     - the 122 frames: median 9.46, close to T4e's 9.42.
   - So the gain comes from the Bug 1 fix; the new table only adds brightness on top. Same pattern as T4d (band
     0.48–0.55 / midtone 0.25).
   - **NLP 11 frames:** new table 8.79 (dL +5.15); T4e values 9.30 (dL +2.49); T4e 9.27 (dL −0.38).
   - **Please decide:** keep the T4f table, or keep the T4f algorithm and return to T4e's tone values.
5. **BlueNeg got worse:** 13.35 → 15.78, dL +12.60, 8 better / 19 worse. BlueNeg references are darker prints. The EV
   median rose 0.28 and the band and midtone rose too. With `t4e_table` it is still worse than T4e (section 3).
6. **The Bug 3 white-balance change does not help on these references.** The change is clear-sky damping 0.35 plus
   strength 0.65.
   - A013 temp went from +0.19 to +0.13 as intended, but ΔE went from 5.44 (T4e white balance) to 5.84.
   - On 185 frames, compared with T4e white balance (`wb_t4e`): temp changed by more than 0.02 on 71 frames, with
     **7 better / 27 worse**; median 9.93 → 10.01.
   - It is implemented as the task list specifies, and setting `wb_v3_strength=0.85, wb_v3_sky_weight=1.0` restores the
     old behaviour. **Please decide whether to keep it.**
7. **Bug 2 (wider backlit weight):** the weight is > 0 on 56 frames for standard (T4e 26).
   - On a plain 4-stop log-uniform test scene it now fires with w = 0.06, so the T4b test was switched to 3.5 stops.
   - The task list's ramp numbers disagree: the formula uses 0.10 for the p25 ramp, the parameter says
     `backlit_ramp = 0.08`. I used 0.72–0.80 for p90 (`backlit_ramp=0.08`) and 0.40−p25 over 0–0.10
     (`backlit_ramp_p25=0.10`), following the formula.
8. **Midtones are now distinct.** Off backlit frames the five presets have 0.22 / 0.24 / 0.25 / 0.26 / 0.29.
   - On the synthetic backlit scene, natural reaches 0.329 and vivid stops at its search cap of 0.330 (0.001 apart), so
     the T4e test that "midtones on backlit frames are strictly distinct" now asserts contrast only.
   - On A012: 0.34 / 0.35 / 0.38 / 0.33 / 0.31 (standard, natural, portrait, vivid, print).
9. **A010:** T4e EV −1.02, ΔE 10.37, L\* −3.5 (it was not underexposed relative to its reference). T4f EV −0.20 (the
   downward-step limit), ΔE 11.04, L\* +9.9: now too bright. As predicted, not every low-EV frame was a bug.

## 2. Results by source (185 frames; better / worse = per-frame ΔE better / worse than T4c by more than 0.25)

### nlp (11 frames)

| Column | ΔE median | ΔE mean | ΔE p90 | dL median | \|dL\| median | Chroma ratio | L\* ours / ref | C\* ours / ref | Better / worse | Backlit frames |
|---|---|---|---|---|---|---|---|---|---|---|
| T4c | 9.11 | 11.99 | 19.94 | −0.37 | 1.74 | 0.69 | 54.3 / 57.0 | 5.1 / 5.3 | - | 2 |
| T4e standard | 9.27 | 11.65 | 19.72 | −0.38 | 1.99 | 0.67 | 54.2 / 57.0 | 5.3 / 5.3 | 3 / 4 | 3 |
| **T4f standard** | 8.79 | 10.66 | 15.46 | +5.15 | 5.15 | 0.77 | 64.5 / 57.0 | 6.7 / 5.3 | 3 / 5 | 5 |
| Natural | 9.46 | 11.04 | 15.07 | +6.10 | 6.10 | 0.65 | 64.7 / 57.0 | 5.8 / 5.3 | 3 / 7 | 4 |
| Portrait | 9.65 | 11.58 | 15.65 | +8.62 | 8.62 | 0.76 | 67.2 / 57.0 | 7.2 / 5.3 | 3 / 7 | 3 |
| Vivid | 8.25 | 10.61 | 15.74 | +4.32 | 5.00 | 0.95 | 63.7 / 57.0 | 8.0 / 5.3 | 2 / 6 | 5 |
| Print classic | 9.36 | 9.91 | 13.53 | +1.84 | 4.62 | 0.74 | 62.3 / 57.0 | 6.4 / 5.3 | 7 / 2 | 5 |

### shop (21 frames)

| Column | ΔE median | ΔE mean | ΔE p90 | dL median | \|dL\| median | Chroma ratio | L\* ours / ref | C\* ours / ref | Better / worse | Backlit frames |
|---|---|---|---|---|---|---|---|---|---|---|
| T4c | 7.83 | 10.10 | 18.00 | −3.45 | 4.90 | 0.82 | 54.1 / 59.9 | 9.5 / 14.2 | - | 0 |
| T4e standard | 7.81 | 10.03 | 18.18 | −3.74 | 4.81 | 0.80 | 54.1 / 59.9 | 9.4 / 14.2 | 5 / 4 | 0 |
| **T4f standard** | 8.13 | 8.58 | 11.93 | +6.20 | 6.49 | 1.00 | 65.3 / 59.9 | 11.4 / 14.2 | 10 / 11 | 2 |
| Natural | 8.66 | 9.34 | 13.83 | +6.12 | 6.61 | 0.84 | 65.8 / 59.9 | 9.7 / 14.2 | 8 / 11 | 0 |
| Portrait | 9.02 | 10.29 | 15.72 | +8.25 | 8.25 | 0.96 | 68.7 / 59.9 | 11.4 / 14.2 | 10 / 11 | 0 |
| Vivid | 8.80 | 8.64 | 11.54 | +5.83 | 6.11 | 1.20 | 66.1 / 59.9 | 13.9 / 14.2 | 10 / 11 | 2 |
| Print classic | 7.32 | 7.39 | 9.97 | +2.72 | 4.42 | 0.97 | 62.0 / 59.9 | 11.1 / 14.2 | 13 / 7 | 5 |

### blueneg (27 frames)

| Column | ΔE median | ΔE mean | ΔE p90 | dL median | \|dL\| median | Chroma ratio | L\* ours / ref | C\* ours / ref | Better / worse | Backlit frames |
|---|---|---|---|---|---|---|---|---|---|---|
| T4c | 13.49 | 14.23 | 19.64 | +7.80 | 9.71 | 0.69 | 49.6 / 48.3 | 9.2 / 8.7 | - | 5 |
| T4e standard | 13.35 | 14.09 | 20.24 | +6.21 | 9.23 | 0.65 | 48.7 / 48.3 | 8.5 / 8.7 | 7 / 7 | 8 |
| **T4f standard** | 15.78 | 15.64 | 20.30 | +12.60 | 12.60 | 0.75 | 58.2 / 48.3 | 9.6 / 8.7 | 8 / 19 | 11 |
| Natural | 15.78 | 16.02 | 21.26 | +14.92 | 14.92 | 0.66 | 59.0 / 48.3 | 8.7 / 8.7 | 6 / 21 | 8 |
| Portrait | 17.37 | 17.52 | 23.53 | +17.26 | 17.26 | 0.73 | 61.8 / 48.3 | 9.8 / 8.7 | 4 / 21 | 7 |
| Vivid | 15.59 | 15.78 | 19.47 | +11.63 | 11.70 | 0.86 | 57.3 / 48.3 | 10.9 / 8.7 | 7 / 19 | 12 |
| Print classic | 13.91 | 14.48 | 18.68 | +10.03 | 12.12 | 0.75 | 54.8 / 48.3 | 9.1 / 8.7 | 10 / 14 | 17 |

### ext_own (126 frames, reference = `04_已反转成品_对照答案`)

| Column | ΔE median | ΔE mean | ΔE p90 | dL median | \|dL\| median | Chroma ratio | L\* ours / ref | C\* ours / ref | Better / worse | Backlit frames |
|---|---|---|---|---|---|---|---|---|---|---|
| T4c | 10.38 | 11.95 | 21.07 | −2.97 | 8.39 | 0.89 | 54.1 / 61.6 | 10.5 / 9.8 | - | 14 |
| T4e standard | 10.35 | 11.98 | 21.19 | −3.02 | 8.42 | 0.88 | 54.1 / 61.6 | 10.4 / 9.8 | 25 / 40 | 15 |
| **T4f standard** | 9.41 | 10.29 | 14.97 | +2.25 | 6.78 | 1.05 | 66.1 / 61.6 | 12.3 / 9.8 | 62 / 52 | 38 |
| Natural | 9.30 | 10.49 | 16.08 | +4.01 | 6.87 | 0.89 | 66.4 / 61.6 | 10.6 / 9.8 | 58 / 52 | 27 |
| Portrait | 9.71 | 11.29 | 18.61 | +6.07 | 7.30 | 1.04 | 69.5 / 61.6 | 12.6 / 9.8 | 50 / 67 | 27 |
| Vivid | 9.59 | 10.68 | 15.49 | +2.29 | 7.01 | 1.27 | 66.2 / 61.6 | 14.8 / 9.8 | 56 / 62 | 42 |
| Print classic | 8.83 | 9.77 | 14.37 | +0.08 | 7.28 | 1.02 | 63.8 / 61.6 | 11.8 / 9.8 | 60 / 53 | 46 |

### All (185 frames)

| Column | ΔE median | ΔE mean | ΔE p90 | dL median | \|dL\| median | Chroma ratio | L\* ours / ref | C\* ours / ref | Better / worse | Backlit frames |
|---|---|---|---|---|---|---|---|---|---|---|
| T4c | 10.94 | 12.08 | 20.20 | −2.37 | 7.80 | 0.84 | 53.7 / 58.1 | 10.1 / 9.6 | - | 21 |
| T4e standard | 10.44 | 12.04 | 20.28 | −2.34 | 7.88 | 0.83 | 53.7 / 58.1 | 9.9 / 9.6 | 40 / 55 | 26 |
| **T4f standard** | 10.01 | 10.90 | 16.95 | +4.50 | 7.19 | 0.98 | 64.5 / 58.1 | 11.6 / 9.6 | 83 / 87 | 56 |
| Natural | 9.95 | 11.20 | 17.60 | +5.15 | 7.40 | 0.84 | 64.9 / 58.1 | 10.1 / 9.6 | 75 / 91 | 39 |
| Portrait | 10.40 | 12.11 | 19.82 | +7.69 | 8.19 | 0.97 | 68.0 / 58.1 | 11.6 / 9.6 | 67 / 106 | 37 |
| Vivid | 10.62 | 11.19 | 17.01 | +4.18 | 7.56 | 1.17 | 64.4 / 58.1 | 13.8 / 9.6 | 75 / 98 | 61 |
| Print classic | 9.56 | 10.20 | 15.13 | +1.82 | 7.14 | 0.95 | 62.2 / 58.1 | 11.2 / 9.6 | 90 / 76 | 73 |

Print classic now has the lowest ΔE (median 9.56, p90 15.13) because it is the darkest preset, which offsets some of the
new table's extra brightness.

## 3. Ablations (standard preset, T4f code; only the listed parameters changed)

| Column | What changed | ΔE median | ΔE mean | ΔE p90 | dL median | Better / worse | NLP ΔE median / dL |
|---|---|---|---|---|---|---|---|
| **standard** | T4f as committed | 10.01 | 10.90 | 16.95 | +4.50 | 83 / 87 | 8.79 / +5.15 |
| task_literal | `ev_anchor_p98_cap=False` (ceiling on the refinement only, as in the task list) | 10.32 | 11.52 | 17.52 | +6.49 | 72 / 100 | 8.79 / +5.53 |
| t4e_tone | T4e band, midtone, contrast, shoulder | 9.75 | 10.52 | 15.96 | +2.54 | 84 / 81 | 9.27 / +2.54 |
| t4e_table | T4e band, midtone, contrast, shoulder, sat, vibrance, chroma_preserve, camera matrix | 9.72 | 10.33 | 15.76 | +2.49 | 89 / 72 | 9.30 / +2.49 |
| wb_t4e | `wb_v3_strength=0.85`, `wb_v3_sky_weight=1.0` (T4e white balance) | 9.93 | 10.83 | 16.89 | +4.46 | 85 / 87 | 8.57 / +5.15 |

**The 63 frames with T4e EV < −0.30** (`--ev-below=t4e,-0.30`). After T4f, no preset has any frame below −0.30; the
minimum is −0.25.

| Column | ΔE median | ΔE mean | ΔE p90 | dL median | Better / worse vs T4c |
|---|---|---|---|---|---|
| T4c | 14.45 | 14.82 | 22.25 | −14.39 | - |
| T4e standard | 14.67 | 14.90 | 21.65 | −14.33 | 6 / 16 |
| **T4f standard** | 10.01 | 9.97 | 14.54 | +2.32 | 44 / 17 |
| task_literal | 10.09 | 10.17 | 14.54 | +4.44 | 44 / 17 |
| t4e_table | 9.79 | 9.53 | 14.12 | +0.64 | 46 / 15 |

The same 63 frames by source, T4e → T4f standard:

| Source | Frames | ΔE median, T4e → T4f | dL median, T4e → T4f |
|---|---|---|---|
| nlp | 3 | 15.64 → 9.22 | −15.0 → −1.0 |
| shop | 7 | 17.39 → 9.87 | −18.2 → +3.9 |
| blueneg | 5 | 11.72 → 11.75 | −6.4 → +10.8 |
| ext_own | 48 | 14.57 → 9.80 | −15.0 → +2.2 |

**The other 122 frames:**

| Column | ΔE median | ΔE mean | ΔE p90 | Better / worse vs T4e | Median L\* difference |
|---|---|---|---|---|---|
| T4e standard | 9.42 | 10.57 | 16.51 | - | +1.8 |
| T4f standard | 10.05 | 11.38 | 17.40 | 35 / 74 | +6.2 |
| t4e_table | 9.46 | 10.74 | 16.33 | 40 / 54 | +3.9 |

On these 122 frames, 20 had EV rise by more than 0.5 (shop 8, blueneg 5, ext_own 7), and their ΔE median rose by 2.53.
T4e's full-frame refinement used to pull these down after the anchor over-lifted them. Under the T4f task list, downward
moves are capped at 0.20 EV.

## 4. Key frames (app's opening method, no sliders moved; `a013_presets_check.json`, `presets_showcase.json`, eval JSON)

| Frame | Version / preset | EV | Temp | Midtone | Contrast | Shoulder | Backlit w | ΔE | L\* difference |
|---|---|---|---|---|---|---|---|---|---|
| A013 | T4e standard | −1.08 | +0.187 | 0.23 | 1.30 | 0.20 | 0 | 19.62 | −28.0 |
| | T4f standard | +0.58 | +0.130 | 0.25 | 1.34 | 0.25 | 0 | 5.84 | −4.3 |
| | Natural | +0.58 | +0.130 | 0.26 | 1.20 | 0.35 | 0 | 6.11 | −5.3 |
| | Portrait | +0.58 | +0.150 | 0.29 | 1.26 | 0.40 | 0 | 5.60 | −3.0 |
| | Vivid | +0.58 | +0.130 | 0.24 | 1.46 | 0.30 | 0 | 6.63 | −4.1 |
| | Print classic | +0.58 | +0.130 | 0.22 | 1.40 | 0.20 | 0 | 5.64 | −5.4 |
| A012 | T4e standard | +0.26 | +0.117 | 0.32 | 1.42 | 0.35 | 1 | 5.92 | −7.8 |
| | T4f standard / natural / portrait / vivid / print | +0.95 / +1.29 / +1.26 / +0.98 / +0.80 | +0.04 | 0.34 / 0.35 / 0.38 / 0.33 / 0.31 | 1.44 / 1.30 / 1.36 / 1.56 / 1.50 | | 1 | 4.09 / 3.73 / 4.30 / 4.87 / 4.33 | −0.6 / −0.2 / +1.3 / −0.1 / −2.6 |
| A023 | T4e standard | +0.69 | | 0.32 | 1.42 | | 1 | 5.04 | −6.4 |
| | T4f standard | +0.88 | | 0.34 | 1.44 | 0.40 | 1 | 3.86 | −3.4 |
| A019 | T4e → T4f standard | −0.01 → +0.04 | | | | | 0 | 17.59 → 15.68 | −20.2 → −17.4 |
| A021 | T4e → T4f standard | −0.25 → −0.25 | +0.177 → +0.145 | | | | 0 | 14.33 → 13.22 | −19.0 → −17.3 |

Where A013's +0.58 EV comes from: the trimean anchor of the textured cells is 0.958 D, and the anchor step gives +0.58.
The textured median, 0.528, is inside the band, so the refinement does not move it. T4e's anchor was 0.869 D (+0.29 EV),
and the full-frame refinement then pulled it down to −1.08 because the full-frame median sat in the sky.
A019 and A021 remain 17 to 19 L\* darker than the references. That is not caused by EV (A019 is +0.04); it is the
existing gap carried over since T4c.

## 5. Code changes (`163fb63`)

**`noref.py` (NOREF_V3 rev `t4f-2026-10-25`):**
- `_textured_cells`: `anchor_d = 0.5·p50 + 0.25·(p25 + p75)` (Tukey trimean); `p25_d` / `p75_d` added to diagnostics.
- **EV refinement** on the encoded median of `textured_rgb`.
  - Note: `textured_rgb` is the mean RGB of each 8×8 cell of the 4×4 estimation input, not individual pixels.
  - Below the band: raise to t_lo + 0.03, but stop where full-frame encoded p98 reaches `ev_refine_p98_max = 0.96`.
  - Above the band (0.76 for high-key frames): lower by at most `ev_refine_down_max = 0.20`, and never below −0.20.
  - `ev_anchor_p98_cap = True`: a positive anchor EV is first held to the same ceiling (never below 0). This is my
    addition beyond the task list (section 1, item 3).
  - Diagnostics are in `diagnostics.ev_refine`.
- **Backlit weight:** `w = ramp(p90; 0.72, 0.80) × ramp(0.40 − p25; 0, 0.10)`, target `backlit_p25_target = 0.34`,
  contrast step 0.10 (shoulder step still 0.15).
- **`estimate_wb_v3`:** if the bright zone's estimate has log2 R/G < −0.05 and log2 B/G > +0.05, its weight is multiplied
  by `wb_v3_sky_weight = 0.35` (diagnostics `clear_sky` / `weight`); `wb_v3_strength` 0.85 → 0.65.
- **Presets, per the T4f table:**
  - standard: band 0.48–0.55, midtone 0.25, contrast 1.34, shoulder 0.25, sat 1.04, vibrance 0.10, chroma_preserve and
    camera matrix 0.45;
  - portrait temp_offset becomes 0.02;
  - portrait no longer carries a separate `backlit_p25_target`, since the global value is now 0.34.

**App and scripts:**
- `app.py`: `THUMB_REV = "t4f"`.
- `index.html`: midtone slider UI 0 = 0.25 (left −0.15, right +0.20); camera matrix and chroma preserve default to 45.
- `t4d_summary.py`: dE p90 column, `--ev-below=NAME,EV`.
- `t4d_presets_showcase.py`: `--before=LABEL=EVAL_JSON:VARIANT`, which renders an earlier version's automatic parameters
  as the first column.

**Tests (314):**
- New: trimean anchor plus EV refinement ignoring a large sky (synthetic sky over grass); brightening under the p98
  ceiling; anchor p98 ceiling; clear-sky white-balance damping; distinct base midtones across the five presets.
- Changed: T4b normal scene 4 → 3.5 stops; portrait offsets read from the config; T4e backlit test asserts contrast only.

## 6. Cache and timing

- t4f thumbnails `thumb_color_t4f_*` are pre-warmed for both rolls (dsc6161 37 frames, dsc6233 38 frames; 75 in
  `$FILMCOLOR_DATA_ROOT/app_cache`).
- After a restart, rolls plus thumbnails open in 4.1 s (dsc6161) and 3.0 s (dsc6233); peak memory footprint 0.57 GiB.
- Old thumbnails (`_t4e_` 75 and earlier) are now stale; none deleted.
- Estimator time in the eval: median 0.19 s, p90 0.41 s (T4e 0.30 / 0.43 s; two processes running in parallel, so this
  only shows it did not get slower). Preset switching over HTTP was not re-timed.
- Evaluation peak memory footprint: 6.2 GiB (half a) and 4.8 GiB (half b).

## 7. Reproduce

```
t4d_source_eval.py src t4f/eval_t4f_{a,b}.json --ids=<half> '--variants={"standard":{"preset":"standard"},
  "natural":{"preset":"natural"},"portrait":{"preset":"portrait"},"vivid":{"preset":"vivid"},
  "print_classic":{"preset":"print_classic"},"task_literal":{"preset":"standard","cfg":{"ev_anchor_p98_cap":false}},
  "t4e_tone":{"preset":"standard","cfg":{"target_encoded_band":[0.45,0.52],"base_midtone":0.23,"base_contrast":1.30,
  "base_shoulder":0.20}},"t4e_table":{"preset":"standard","cfg":{"target_encoded_band":[0.45,0.52],"base_midtone":0.23,
  "base_contrast":1.30,"base_shoulder":0.20,"base_sat":1.02,"base_vibrance":0.08,"chroma_preserve":0.35,
  "camera_matrix_mix":0.35}},"wb_t4e":{"preset":"standard","cfg":{"wb_v3_strength":0.85,"wb_v3_sky_weight":1.0}}}'
t4d_summary.py t4c=t4d/eval_t4c_a.json,t4d/eval_t4c_b.json:t4c t4e=t4e/eval_t4e_a.json,t4e/eval_t4e_b.json:standard \
  standard=t4f/eval_t4f_a.json,t4f/eval_t4f_b.json:standard natural=...:natural ... [--ev-below=t4e,-0.30]
t4d_presets_showcase.py a013_presets_check.jpg --tile-h=360 --before=T4E=t4e/eval_t4e_b.json:standard ext:A013
t4d_presets_showcase.py presets_showcase.jpg --tile-h=300 ext:A013 nlp:_DSF9237 nlp:_DSF8948 ext:A012 ext:A019 \
  ext:A021 ext:A075 ext:A091
```

The two halves are the same as in T4d/T4e:
- a = nlp, shop, blueneg and the ext_own dsc8242 roll (96 frames);
- b = the rest of ext_own (95 frames).
