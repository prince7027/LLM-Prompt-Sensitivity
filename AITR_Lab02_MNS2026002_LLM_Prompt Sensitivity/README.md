# Formatting Sensitivity of Small Instruction-Tuned LLMs and the Effect of Few-Shot Examples
Pathway A (reproduction + extension). Status: G1 draft. Nothing here has been executed yet.

Question: does FormatSpread-style formatting sensitivity (Sclar et al., 2024) appear in small open
instruction-tuned models on a classification task, and does the number of few-shot examples reduce it
(as ProSA, Zhuo et al., 2024, reports)?

See research_decisions.md for the brief, evidence ledger and frozen protocol.

## Quick start (Colab T4)  [TEST REQUIRED]
    pip install -r requirements.txt
    python src/env_check.py --model Qwen/Qwen2.5-1.5B-Instruct
Paste the full console output into your AI chat / ai_assistance_log.md before proceeding.

## Layout
configs/ frozen settings | src/ modules | scripts/ entrypoints | results/ real outputs only
paper/ LaTeX | data/README.md access notes (no raw data committed)
