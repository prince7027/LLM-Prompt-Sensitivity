# research_decisions.md

## G1 research brief (DRAFT: student must review and verify)
| Field | Entry |
|---|---|
| Review topic | Prompt sensitivity and stability of LLM-generated research conclusions (survey) |
| Key papers to verify from originals | Sclar et al. 2024 (FormatSpread, ICLR); Zhuo et al. 2024 (ProSA, EMNLP Findings); Hua et al. 2025 (EMNLP); Chatterjee et al. 2024 (POSIX); Gallo et al. 2025 (JAMIA) |
| Research gap | Review Sec. 3.2.2 / Table 1: formatting alone can cause large few-shot performance swings (Sclar), and few-shot examples may reduce sensitivity (Zhuo). Whether both hold for small modern instruction-tuned models is INFERRED/UNKNOWN. |
| Primary question | How large is the accuracy spread across plausible prompt formats for small open instruction-tuned models, and does the number of few-shot demonstrations (0,1,3,5) change it? |
| Hypotheses | H1: at k=0 the spread exceeds 5 pp for at least one model (threshold frozen now). H2: spread at k=5 is smaller than at k=0 (paired bootstrap 95% CI excludes 0). Contradicting result: no reduction, or an increase, at higher k. |
| Pathway | A: reproduction + extension (partial FormatSpread-style protocol, own implementation) |
| Data | SST-2 (GLUE): demos from train, 500 fixed validation examples. [TO VERIFY official source, license] |
| Baselines/comparators | (1) canonical single format (Sclar-style "Sentence/Label"); (2) two models of different size, same family; (3) k=0 vs k>0 as the extension control |
| Primary metric / controls | Accuracy per format and spread over 30 formats. Held constant: eval ids, demos per k, label tokens, formats across models |
| Compute / risks | Colab T4. About 120k short forward passes per model (estimate; MEASURE in env_check). Risk: spread depends on number of sampled formats (fixed at 30). If a model/dataset is unavailable, fall back to Qwen2.5-0.5B or another 2-class task. |
| First executable milestone | src/env_check.py: GPU, versions, SST-2 load, one forward pass, label tokens are single tokens |

## Evidence ledger
| Claim | Status | Source / next action |
|---|---|---|
| Sclar et al. report up to 76 pp spread (LLaMA-2-13B) | UNVERIFIED | Read original paper |
| Zhuo et al.: few-shot reduces sensitivity in several settings | UNVERIFIED | Read original; note which models/tasks |
| Original FormatSpread code is public | UNKNOWN | Check authors' repo and license; decide reuse vs own implementation |
| SST-2 and Qwen2.5 models accessible | UNKNOWN | Confirm via env_check output |
| Novelty of this exact setting | UNKNOWN | Run Prompt 04 searches; claim no novelty by default |

## Decision log
| Date | Decision | Reason | Evidence | Next action |
|---|---|---|---|---|
| 2026-10-20 | Chose gap B (FormatSpread reproduction + few-shot) over conclusion-stability-index idea | T4 compute, under 1 week, clean baseline | Student chose among 4 AI-proposed options | Verify papers; run env_check |

## Corrections needed in the review paper
- Hua 2025a/2025b are duplicate bib entries; Chen et al. 2024 entry is incomplete; Ahuja, Boiko, He, Razavi, Xie are listed but uncited.
