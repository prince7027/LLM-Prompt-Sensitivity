# Prompt Sensitivity and LLM Conclusion Stability (Topic T-5)

**Student Name:** Prince Garg  
**Roll Number:** MNS2026002  
**Course:** AI Tools for Research  

## Overview
This repository contains the empirical code and experiment setup evaluating prompt formatting sensitivity on `Qwen/Qwen2.5-1.5B-Instruct` across 30 prompt formats on SST-2 ($k \in \{0, 1, 3, 5\}$).

## Key Results
- **FormatSpread ($\Delta$):** 33.80 percentage points (Accuracy range: 53.60% to 87.40%)
- **Standard Deviation ($\sigma$):** 0.0931
- **Conclusion:** Minor prompt formatting syntax induces severe performance instability on small instruction-tuned models, supporting Hypothesis H1.

## Quick Start
```bash
# Clone repository
git clone [https://github.com/YOUR_USERNAME/AITR_Lab02_MNS2026002_PromptSensitivity.git](https://github.com/YOUR_USERNAME/AITR_Lab02_MNS2026002_PromptSensitivity.git)
cd AITR_Lab02_MNS2026002_PromptSensitivity

# Install dependencies
pip install torch transformers datasets pandas pyyaml tqdm

# Run baseline environment verification (Gate G2)
python src/env_check.py

# Run evaluation pipeline (Gate G3)
python scripts/run_eval.py
