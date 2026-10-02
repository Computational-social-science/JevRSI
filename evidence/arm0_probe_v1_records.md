# v1.0 environment, published spec, 1 step — evaluation records

Produced by: the v1.0 modules (their release's snapshot) + the published spec verbatim,
Qwen3-0.6B, seed 17, steps 1. Reproduced twice, agreeing to full float precision.

| target | role | option_order | pooled AURC | minDS | noul | choice | score |
|---|---|---|---|---|---|---|---|
| in_distribution | control | canonical | 0.3258 | 31.34 | 0.6048 | 0.5836 | 0.4240 |
| mmlu_pro_1k | control | canonical | 0.7146 | 8.36 | — | 0.2110 | — |
| typed_decisions | control | canonical | 0.4919 | -21.43 | 0.5567 | 0.3767 | 0.3063 |
| in_distribution | control | reversed | 0.4074 | 24.58 | 0.5083 | 0.5447 | 0.3114 |
| mmlu_pro_1k | control | reversed | 0.8238 | 0.81 | — | 0.1460 | — |
| typed_decisions | control | reversed | 0.5193 | -23.38 | 0.4933 | 0.3667 | 0.2838 |
| in_distribution | candidate | canonical | 0.5111 | 13.83 | 0.5342 | 0.2702 | 0.3039 |
| mmlu_pro_1k | candidate | canonical | 0.8631 | -2.21 | — | 0.1200 | — |
| typed_decisions | candidate | canonical | 0.5653 | -28.25 | 0.4817 | 0.3417 | 0.3438 |
| in_distribution | candidate | reversed | 0.5079 | 10.27 | 0.5280 | 0.2808 | 0.3340 |
| mmlu_pro_1k | candidate | reversed | 0.8448 | -2.21 | — | 0.1200 | — |
| typed_decisions | candidate | reversed | 0.5718 | -40.26 | 0.4967 | 0.2800 | 0.3237 |

Full per-question records (21,792 rows): E:/2026-AI4S/arms/arm0_probe_v1/out/
arm0probe.items.jsonl — kept out of the repository at 6.7 MB; regenerate with the
command in docs/reproduction_audit.md.
