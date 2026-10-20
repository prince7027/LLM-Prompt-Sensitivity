# AI Assistance & Provenance Log (Prompt Audit)

**Course:** AI Tools for Research  
**Topic ID:** T-5  
**Student Name:** Prince Garg  
**Roll Number:** MNS2026002  

| # | Task | Tool | Prompt/version | AI Proposal | Human Decision | Verification | Outcome |
|---|---|---|---|---|---|---|---|
| 1 | Gap Selection | Gemini / Claude | Uploaded assignment + literature review | Four candidate study pathways (A-D) | Chose Pathway A: Reproduction + Extension (FormatSpread on small models) | Verified Sclar et al. (2024) & Zhuo et al. (2024) paper claims | Gate G1 Research Brief drafted in `research_decisions.md` |
| 2 | Review Audit & Citations | Gemini / Claude | Literature review reference audit | Flagged duplicate entries (Hua 2025a/b), incomplete references (Chen 2024), uncited bibliography entries | Verified discrepancies in source PDF document | Documented required bibliography corrections in `research_decisions.md` |
| 3 | Environment & Baseline Setup | Gemini / PyTorch | Executed `src/env_check.py` with `Qwen/Qwen2.5-1.5B-Instruct` | Proposed dataset `glue/sst2` via HuggingFace `datasets` API | Updated code to use `stanfordnlp/sst2` to resolve HuggingFace URI format errors | Console execution completed on Colab Tesla T4 GPU | VRAM memory usage (2.89 GiB) & single-token label encoding verified (`negative`: 8225, `positive`: 6785) |
| 4 | Repository Assembly | Python `zipfile` | Package workspace directory structure into runnable ZIP file | Automated script to build modular repository folders (`configs/`, `src/`, `scripts/`, `results/`) | Approved directory structure matching course guidelines | Downloaded `AITR_Lab02_MNS2026002_PromptSensitivity.zip` | Complete executable workspace environment ready for Gate G2 submission |
| 5 | Eval Subset Optimization | Gemini / PyTorch | Modify `run_eval.py` to add `--eval_subset_size` flag | Proposed reducing evaluation subset size from 500 to 100 samples to manage execution time on T4 GPU | Approved $N=100$ subset for Gate G3 empirical runs | Reduced run time from ~40 mins to ~6 mins while preserving metric trends | Fast evaluation pipeline verified and operational |
| 6 | Gate G3 Empirical Evaluation | Gemini / PyTorch | Executed `run_eval.py` across 30 formats for $k \in \{0, 1, 3, 5\}$ | Evaluated model accuracy, min/max spread, and standard deviation | Approved logged experimental outputs and CSV generation | Verified output in `/content/run_eval.csv` and `results/eval_results.csv` | Observed 33.8 percentage point format spread (53.60% to 87.40%), confirming Hypothesis H1 |