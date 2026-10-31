# T4n vs T4m full-resolution regression

T4m evidence is from commit 13db565; T4n is rerun with the same script/protocol.

| Set | N | T4m median / mean / p90 | T4n median / mean / p90 | better / worse > .25 | better / worse > .05 |
|---|---:|---:|---:|---:|---:|
| dev:nlp | 11 | 8.44 / 8.72 / 10.25 | 8.44 / 8.72 / 10.25 | 0 / 0 | 0 / 0 |
| dev:shop | 21 | 7.20 / 7.91 / 11.41 | 7.20 / 7.91 / 11.41 | 0 / 0 | 0 / 0 |
| dev:blueneg | 27 | 13.62 / 13.58 / 16.57 | 13.62 / 13.58 / 16.57 | 0 / 0 | 0 / 0 |
| dev:ext_own | 126 | 8.75 / 8.92 / 13.26 | 8.75 / 8.92 / 13.26 | 0 / 1 | 0 / 2 |
| dev:all | 185 | 9.03 / 9.47 / 14.14 | 9.03 / 9.48 / 14.14 | 0 / 1 | 0 / 2 |
| cross:all | 123 | 13.84 / 14.43 / 20.73 | 13.84 / 14.43 / 20.73 | 0 / 0 | 0 / 0 |

dev standard: params changed 2, metrics changed 2.
dev natural: params changed 2, metrics changed 2.
dev portrait: params changed 2, metrics changed 2.
dev vivid: params changed 2, metrics changed 2.
dev print_classic: params changed 2, metrics changed 2.

dev: parameter changes 2: ext:A104, ext:A138.
dev: base metadata changes 3: ext:A104, ext:A138, ext:A198.
dev: mode_seed triggers 2: ext:A104, ext:A198.
dev: relaxed_uniformity triggers 1: ext:A138.

## dev metric changes > .000001

| Frame | T4m | T4n | Difference |
|---|---:|---:|---:|
| ext:A138 | 11.662843 | 11.840010 | +0.177167 |
| ext:A104 | 9.092680 | 9.610570 | +0.517890 |

cross: parameter changes 0: none.
cross: base metadata changes 0: none.
cross: mode_seed triggers 0: none.
cross: relaxed_uniformity triggers 0: none.

## cross metric changes > .000001

| Frame | T4m | T4n | Difference |
|---|---:|---:|---:|

## Protected and cyan target frames

| Frame | T4m | T4n |
|---|---:|---:|
| ext:A008 | 8.720626 | 8.720626 |
| ext:A012 | 4.460079 | 4.460079 |
| ext:A013 | 6.661887 | 6.661887 |
| shop:DSC_3833 | 4.664065 | 4.664065 |
| nlp:_DSF8948 | 8.343404 | 8.343404 |
| ext:A006 | 13.516800 | 13.516800 |
| ext:A011 | 8.582616 | 8.582616 |
| ext:A196 | 12.393907 | 12.393907 |

Target blueneg:199807B-08-orlando-seaworld: T4m 16.592069, T4n 16.592069; T4m no_cyan 16.020896.

## Fault samples

| Sample | T4m | T4n | Base | Span G | Finite positive |
|---|---|---|---|---:|---|
| A1095.RAF | ValueError: roll base unavailable for A1095.RAF: no frame in the sub-roll produced a valid automatic base | ok | [0.27907225489616394, 0.26045623421669006, 0.06788738816976547] | 0.9109 | True |
| A277.dng | ok | ok | [0.7074692845344543, 0.37794309854507446, 0.32568857073783875] | 1.1599 | True |
| A281.RAF | manual_review_required | ok | [0.23205921053886414, 0.1743343323469162, 0.04429693892598152] | 1.2670 | True |
| A1005.CR2 | ok | ok | [0.1753109097480774, 0.23362325131893158, 0.09887846559286118] | 0.6369 | True |
| A1234.DNG | ok | ok | [0.3192339837551117, 0.12056153267621994, 0.03834592178463936] | 0.6658 | True |

## Peak memory footprint (GiB)

- candidate_a: 5.10
- candidate_b: 4.92
- candidate_cross: 7.41
- devices_candidate: 2.58
- devices_candidate_final: 3.20
- tests: 3.05

Candidate source SHA256: `6c6956f2c05aae6ed8416b6d3bdb7e79a4622e62e5b9b8c52be811b1a4517e55`.
