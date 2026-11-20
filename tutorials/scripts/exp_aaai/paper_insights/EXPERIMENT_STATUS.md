# AAAI Paper Experiment Status

## Paper Title
"Pluggable, KV-Cache-Reusing Guardrails for RAG from a Single Composable Checkpoint"

## Completion Status

### ✅ Completed Experiments

#### 1. **Answerability Accuracy** (Table 2)
- ✅ Adapter mode - IRS queries: **87.5% accuracy** (90% P/R/F1)
- ✅ Adapter mode - Realistic queries: **95.0% accuracy** (90.9% P, 100% R, 95.2% F1)
- ✅ Prompting baseline - IRS queries: **56.3% accuracy** (80% P, 40% R, 53.3% F1)
- ✅ 3 seeds per configuration
- **Result**: Adapter beats prompting by **+31.2% accuracy**, **2.25× better recall**

#### 2. **Scripts Created**
- ✅ `run_answerability_accuracy.py` - Accuracy experiments with ChromaDB retrieval
- ✅ `run_latency_experiments.py` - Multi-seed latency comparison (aLoRA vs LoRA)
- ✅ `generate_race_html.py` - Merge results and generate animated visualizations
- ✅ `analyze_latency_results.py` - Extract statistics with error bars
- ✅ `debug_retrieval.py` - Debug ChromaDB retrieval issues
- ✅ `rag_101_queries.txt` - IRS/tax test queries (10 answerable, 6 unanswerable)
- ✅ `rag_101_queries_realistic.txt` - Government services queries (10 answerable, 10 unanswerable)

### ✅ All Core Experiments Complete!

#### 1. **Latency Experiments** (Figure 3) - ✅ COMPLETE
- ✅ 3-seed comparison completed
- ✅ aLoRA: **47.8 ± 1.7 seconds**
- ✅ LoRA: **88.8 ± 2.7 seconds**
- ✅ **Speedup: 1.86× faster**
- ✅ HTML race visualizations generated
- **Result**: aLoRA achieves **1.86× speedup** (46% latency reduction) through KV-cache reuse
- Data location: `exp/latency_results/` and `exp/race_visualizations/`

#### 2. **Additional Accuracy Experiments** (Optional)
- [ ] Query rewriting accuracy
- [ ] Harm detection accuracy
- [ ] Policy guardrails accuracy
- Status: Can use same infrastructure as answerability

### ❌ Not Required for Short Paper

- Full RAG generation quality (out of scope for guardrails paper)
- Citation accuracy (not a guardrail)
- Multi-model comparison (just need aLoRA results)

## Paper Structure Map

### Figure 1: System Architecture
- Status: Need to create diagram
- Shows: Granite Switch with embedded adapters, KV-cache reuse

### Figure 2: Control Token Mechanism
- Status: Need to create diagram
- Shows: How control tokens activate adapters mid-stream

### Figure 3: Latency Comparison (aLoRA vs LoRA)
- Status: ✅ Complete
- Shows: Overall pipeline latency across 3 seeds with error bars
- Main claim: "1.86× speedup from KV-cache reuse (47.8s vs 88.8s, 46% reduction)"
- Data: `exp/latency_results/` and `exp/race_visualizations/`

### Table 1: Model Configuration
- Status: ✅ Available from config files
- Shows: Granite 4.1 3B (2560 hidden size, 42 layers), 12 adapters, control tokens
- Key point: **Same 3B base model used across all experiments** (fair comparison)

### Table 2: Answerability Accuracy
- Status: ✅ Complete
- Shows: Adapter (87.5%) vs Prompting (56.3%)
- Main claim: "+31.2% accuracy, 2.25× better recall"

### Table 3: Per-Guardrail Breakdown (Optional)
- Status: Can reuse answerability infrastructure
- Shows: Accuracy for each guardrail type

## Timeline

### Priority 1 (Must Have) - ✅ ALL COMPLETE!
1. ✅ Answerability accuracy - DONE (87.5% vs 56.3%)
2. ✅ Latency experiments - DONE (1.86× speedup)
3. ⏳ Create figures/diagrams - **Next priority**
4. ⏳ Write the paper - All data ready!

### Priority 2 (Nice to Have)
1. Query rewriting accuracy
2. Harm detection accuracy
3. Additional query sets

### Priority 3 (Optional)
1. Policy guardrails accuracy
2. Ablation studies

## Key Results Summary (For Abstract)

> "We present Granite Switch, a system for pluggable RAG guardrails that reuses KV-cache across adapter transitions. Our answerability guardrail achieves **87.5% accuracy**, outperforming prompting baselines by **31.2 percentage points** with **2.25× better recall**. Through KV-cache reuse, aLoRA achieves **1.86× speedup** (47.8s vs 88.8s) over standard LoRA activation, reducing pipeline latency by 46% and enabling low-latency guardrail composition."

## Data Locations

### Completed
- Answerability (adapter): `exp/accuracy_results/answerability/adapter/`
- Answerability (realistic): `exp/accuracy_results/answerability/adapter_realistic/`
- Answerability (prompting): `exp/accuracy_results/answerability/prompting/`

### Pending
- Latency results: `latency_results/` (to be generated)
- HTML visualizations: `latency_results/visualizations/` (to be generated)

## Critical Path

1. ✅ **Run latency experiments** → Got Figure 3 + speedup claim (1.86×)
2. ✅ **Run accuracy experiments** → Got Table 2 (87.5% vs 56.3%)
3. ⏳ **Create diagrams** → Figure 1 (architecture), Figure 2 (control tokens), Figure 3 (latency chart)
4. ⏳ **Write paper** → Use ACCURACY_RESULTS.md + LATENCY_RESULTS.md
5. 🎯 **Submit to AAAI**

## Contact / Notes

- Experiments run on: aims-01 server
- ChromaDB corpus: 49,607 govt documents
- Models stored: `/data//granite-switch-benchmark/`
- Working directory: `/Users/veroniquedemers/git/granite/granite-switch/tutorials/scripts/exp_aaai/`

---

*Last updated: 2026-11-19*
