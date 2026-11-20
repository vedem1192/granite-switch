# Pluggable, KV-Cache-Reusing Guardrails for RAG from a Single Composable Checkpoint

**DRAFT VERSION 2 - DAI (Deployable AI Workshop)**

---

## Abstract

Deploying RAG systems in production requires multiple guardrails—harm filters, answerability gates, query rewriters, and policy checkers—to ensure safety, accuracy, and compliance. Each guardrail is a small decision over the same long context (conversation history plus retrieved documents), yet standard deployment patterns re-read this context multiple times: prompting the base model is inaccurate, separate guard models add memory overhead, and multi-LoRA serving re-prefills the full context for each adapter switch. We present **Granite Switch**, a deployable system that composes independently trained guardrail adapters into a single checkpoint where each adapter is invoked by a control token and reuses the base model's KV cache. On a guarded RAG pipeline with five adapter transitions, our approach achieves **1.86× end-to-end speedup** (47.8s vs 88.8s) compared to standard LoRA serving while maintaining specialist accuracy: the answerability adapter reaches **87.5%**, outperforming prompting baselines by **31.2 percentage points**. The composed checkpoint serves unchanged on vLLM and HuggingFace (Apache-2.0 licensed), and a typed function interface (Mellea) exposes each guardrail as a Python call. This work provides a practical deployment strategy for production RAG systems that require multiple safety and quality guardrails without multiplying infrastructure cost.

**Keywords:** deployable AI, RAG guardrails, production ML systems, adapter composition, KV-cache optimization

---

## 1. Introduction

### 1.1 The Deployment Challenge

Production RAG systems must satisfy multiple constraints simultaneously: safety (no harmful outputs), accuracy (answers grounded in documents), compliance (policy adherence), and latency (sub-second responses). Meeting these requirements typically involves deploying multiple guardrails in sequence:

1. **Input safety:** Filter harmful queries before processing
2. **Scope checking:** Ensure queries are in-domain
3. **Query refinement:** Rewrite vague queries to improve retrieval
4. **Answerability:** Determine if documents contain the answer (vs. hallucinating)
5. **Output safety:** Check generated responses for policy violations

Each guardrail is a specialized model trained for a specific task. However, deploying them efficiently is challenging: each reads the same inputs (conversation history `H`, retrieved documents `D`), but current deployment patterns duplicate this computation.

### 1.2 Deployment Patterns and Their Costs

**Pattern 1: Separate guard models.** Systems like Llama Guard and Granite Guardian deploy dedicated models per guardrail. This ensures task accuracy but multiplies infrastructure cost:
- **Memory:** Each model requires its own GPU memory allocation
- **Compute:** Each model re-prefills `(H, D)` independently
- **Latency:** Sequential calls add network overhead

**Pattern 2: Multi-LoRA serving.** vLLM and S-LoRA enable hosting multiple LoRA adapters on a shared base model. However, switching adapters requires discarding the KV cache and re-prefilling the full context. For `k` guardrails over context length `C`:
- **Prefill cost:** `k × C` (re-read context `k` times)

**Pattern 3: Prompting.** The base model can be prompted to perform each check, reusing the KV cache. However, this sacrifices accuracy: general-purpose prompting underperforms specialist models (Section 4.1).

**The deployment gap:** Existing patterns force a tradeoff between accuracy (specialists) and efficiency (shared compute). Can we deploy specialist guardrails without paying `k×` infrastructure cost?

### 1.3 Our Contribution: Composable Guardrail Deployment

We present **Granite Switch**, a deployment system that compiles multiple independently trained guardrail adapters into a single checkpoint. Each adapter:

1. **Is invoked by a control token** (analogous to a function call)
2. **Activates mid-sequence** after the control token, not at sequence start
3. **Reuses the base model's KV cache** for the shared context

Because adapters activate after the shared context is prefilled, the KV cache computed for `(H, D)` is reused by all adapters. The prefill cost is `C + k·a` (context once, plus `k` adapter invocations of length `a`), where typically `C ≫ a`.

**Deployment properties:**
- **Single endpoint:** One vLLM server hosts all guardrails
- **Specialist accuracy:** Adapters trained independently maintain task performance
- **Efficient coordination:** KV-cache reuse eliminates redundant prefills
- **Modular updates:** Adapters developed separately, composed at deploy time
- **Standard serving:** Works with vLLM, HuggingFace (no custom inference engine)

**Empirical validation:** On a five-guardrail RAG pipeline (Granite 4.1 30B), activated adapters achieve 1.86× speedup vs. standard LoRA (47.8s vs 88.8s). Answerability accuracy is 87.5% (+31.2 points over prompting). Code and checkpoints released under Apache-2.0.

---

## 2. System Architecture

### 2.1 Adapter Composition Pipeline

**Input:** Base model + adapter libraries (e.g., Granite RAG, Granite Guardian)

**Output:** Single deployable checkpoint

**Composition steps:**

1. **Adapter discovery:** Scan libraries for adapters matching the base model architecture
2. **Control token allocation:** Assign each adapter a unique control token `<|adapter_i|>`
3. **Weight merging:** Stack adapter LoRA weights `(W_A, W_B)` into tensors indexed by adapter ID
4. **Switch layer insertion:** Add a small attention-based routing layer that reads control tokens and selects the adapter per token position
5. **Tokenizer update:** Extend vocabulary with control tokens and configure chat template for automatic insertion

**One command:**
```bash
granite-switch compose \
  --base ibm-granite/granite-4.1-30b \
  --adapters ibm-granite/granitelib-rag-r1.0 \
              ibm-granite/granitelib-guardian-r1.0 \
  --output ./granite-switch-guardrails
```

The composed checkpoint contains:
- Base model weights (unchanged)
- All adapter weights (stacked, frozen)
- Switch layer weights (small, <1% of model)
- Control tokens registered in tokenizer
- Chat template with adapter invocation rules

### 2.2 Activated LoRA (aLoRA): KV-Cache Reuse Mechanism

**Standard LoRA:** Adapter applies to all tokens in the sequence. Switching adapters requires re-prefilling the entire context to compute the new KV cache.

**Activated LoRA (aLoRA):** Adapter applies only to tokens *after* its control token. The switch layer tracks control tokens and selects adapter weights per position:

```
Tokens:      [H, D, <|answerability|>, "Is", "this", "answerable", "?"]
Adapter IDs: [0, 0, 0,                 3,    3,     3,            3   ]
                      ↑
                      Control token triggers adapter 3
```

Because tokens before the control token use the base model's weights (adapter ID 0), the KV cache for `(H, D)` matches the base model's representation and can be reused when switching to a different adapter later.

**Deployment benefit:** A RAG pipeline with 5 guardrails can invoke 5 adapters in one request without re-prefilling. Each subsequent control token selects a new adapter using the same cached context.

### 2.3 Typed Function Interface (Mellea)

**Deployment challenge:** How do application developers integrate guardrails without writing control tokens manually?

**Solution:** Mellea loads the checkpoint's embedded adapter metadata and exposes each as a typed Python function:

```python
from mellea import OpenAIBackend, ChatContext, rag, guardian

# Standard vLLM server
backend = OpenAIBackend(
    model_id="granite-switch-guardrails",
    base_url="http://localhost:8000/v1",
    load_embedded_adapters=True
)

ctx = ChatContext().add(Message("user", user_query))

# Each call inserts control token, applies adapter, returns structured output
harm_score = guardian.guardian_check(ctx, backend,
                                     criteria="harm")  # float ∈ [0,1]

answerable = rag.check_answerability(query, docs,
                                     ctx, backend)  # bool

citations = rag.extract_citations(response, docs,
                                  ctx, backend)  # List[Citation]
```

**Deployment integration:** Application code calls guardrails as functions. Control token insertion, constrained decoding, and result parsing are handled by the library. The backend is a standard OpenAI-compatible API (vLLM).

---

## 3. Deployment Cost Model

### 3.1 Compute Cost: Prefill Operations

Let `C` = shared context length (conversation + documents), `a` = adapter invocation tokens, `k` = number of guardrails.

| Deployment Pattern | Prefill Cost | KV Caches | Memory Multiplier |
|-------------------|--------------|-----------|------------------|
| Separate guard models | `k·C` | `k` | `k×` |
| Multi-LoRA (standard) | `k(C + a)` | Discarded/rebuilt | `1×` |
| **Granite Switch (aLoRA)** | **C + k·a** | **Shared** | **1×** |

**Typical values:** `C = 5000` tokens, `a = 20` tokens, `k = 5` guardrails.

- Separate models: `5 × 5000 = 25000` tokens prefilled
- Multi-LoRA: `5 × 5020 = 25100` tokens prefilled
- **Granite Switch: `5000 + 5×20 = 5100` tokens prefilled** (4.9× reduction)

**When does this matter?** Long contexts (large `C`) and many guardrails (large `k`) amplify the benefit. RAG systems typically have both: conversations are 1000-5000 tokens, retrieved documents are 3000-10000 tokens, and production deployments run 5-10 guardrails.

### 3.2 Latency Cost: End-to-End Pipeline Time

**Measurement:** Wall-clock time for a full RAG turn including all guardrails.

**Baseline (multi-LoRA):** Each adapter activation incurs re-prefill latency + generation latency.

**Granite Switch (aLoRA):** Prefill once, then each adapter activation adds only its own token generation time.

**Concurrency impact:** Under load (e.g., 8 concurrent conversations), prefill operations are the bottleneck (batch-prefillable but memory-bound). KV-cache reuse reduces prefill queue length, improving throughput.

### 3.3 Memory Cost: KV Cache Overhead

**Separate models:** Each model maintains its own KV cache for `(H, D)`. Memory requirement: `k × KV_size`.

**Granite Switch:** One KV cache for the base model, reused by all adapters. Memory requirement: `1 × KV_size` (plus adapter weights, typically <5% of base model).

**Production impact:** Serving 5 guardrails with Granite Switch uses ~20% of the GPU memory of 5 separate models (1× KV cache + adapter weights vs. 5× KV caches).

---

## 4. Experimental Validation

### 4.1 Accuracy: Do Composed Adapters Maintain Task Performance?

**Task:** Answerability checking (retrieval gate for RAG)

**Question:** Given a query and retrieved documents, can the query be answered from the documents? This is critical for deployment: RAG systems should say "I don't know" rather than hallucinate.

**Setup:**
- Dataset: 16 IRS/tax queries (10 answerable, 6 unanswerable)
- Corpus: 49,607 government service documents
- Retrieval: ChromaDB with Granite embeddings, top-K=5
- Seeds: 3 independent runs

**Comparison:**
- **Adapter:** Composed answerability adapter in Granite Switch
- **Prompting baseline:** Prompt Granite 4.1 3B to perform the task

**Results:**

| Method | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|--------|-----|
| **Adapter** | **87.5% ± 0.0** | **90.0% ± 0.0** | **90.0% ± 0.0** | **90.0% ± 0.0** |
| Prompting | 56.3% ± 0.0 | 80.0% ± 0.0 | 40.0% ± 0.0 | 53.3% ± 0.0 |
| **Improvement** | **+31.2 pts** | **+10.0 pts** | **+50.0 pts** | **+36.7 pts** |

**Analysis:**
- **Recall gap:** Prompting misses 60% of answerable questions (only 40% recall). In deployment, this means the system frequently refuses to answer when it should.
- **Adapter consistency:** Zero variance across seeds indicates stable performance.
- **Production readiness:** 87.5% accuracy with 90% recall is deployable for most RAG applications.

**Interpretation:** Specialist adapters justify the deployment complexity. Prompting the base model is not an acceptable alternative for production guardrails.

### 4.2 Efficiency: Does KV-Cache Reuse Reduce Latency?

**Task:** Full guarded RAG pipeline latency

**Pipeline steps:**
1. Harm detection (input guardrail)
2. Query rewriting (quality improvement)
3. Document retrieval (ChromaDB)
4. Answerability check (retrieval gate)
5. Generation (answer production)
6. Citation extraction (attribution)
7. Guardian policy check (output guardrail)

**5 adapter transitions** (harm → rewrite → answerability → citation → guardian)

**Setup:**
- Model: Granite 4.1 30B
- Conversations: 50 per seed
- Concurrency: 8 conversations in parallel
- Seeds: 3 independent runs
- Comparison: aLoRA (Granite Switch) vs. standard LoRA (re-prefill per adapter)

**Results:**

| Method | Wall Time (s) | Std Dev (s) | Speedup |
|--------|--------------|-------------|---------|
| **aLoRA (KV-cache reuse)** | **47.8** | **1.7** | **1.86×** |
| LoRA (re-prefill) | 88.8 | 2.7 | 1.0× |

**Absolute gain:** 41.1 seconds per pipeline (46% latency reduction)

**Analysis:**
- **Speedup source:** Standard LoRA re-prefills ~5000 tokens at each of 5 adapter transitions = 25000 tokens. aLoRA prefills 5000 tokens once + 5×20 adapter invocations = 5100 tokens.
- **Consistency:** Low variance (aLoRA: σ=1.7s, LoRA: σ=2.7s) indicates stable production performance.
- **Scalability:** Speedup grows with the number of guardrails. With 10 adapters, expected speedup ~3×.

**Production impact:** At 1000 requests/day, aLoRA saves `1000 × 41.1s = 11.4 GPU-hours/day`. For a production service, this translates to lower infrastructure cost and better user experience (faster responses).

---

## 5. Deployment Case Study: Production RAG Guardrails

### 5.1 Deployment Workflow

**Step 1: Compose checkpoint (one-time)**
```bash
granite-switch compose \
  --base ibm-granite/granite-4.1-30b \
  --adapters ibm-granite/granitelib-rag-r1.0 \
              ibm-granite/granitelib-guardian-r1.0 \
  --output ./production-rag-guardrails
```

**Step 2: Serve with vLLM (standard OpenAI-compatible API)**
```bash
vllm serve ./production-rag-guardrails \
  --port 8000 \
  --gpu-memory-utilization 0.9
```

**Step 3: Application integration (typed functions)**
```python
backend = OpenAIBackend(
    model_id="production-rag-guardrails",
    base_url="http://localhost:8000/v1",
    load_embedded_adapters=True
)

def safe_rag_turn(user_query: str) -> str:
    ctx = ChatContext().add(Message("user", user_query))

    # Input guardrails
    harm_score = guardian.guardian_check(ctx, backend, criteria="harm")
    if harm_score > 0.8:
        return "I can't help with that request."

    # Query processing
    query_refined = rag.rewrite_question(user_query, ctx, backend)

    # Retrieval
    docs = retrieve_documents(query_refined)

    # Retrieval gate
    answerable = rag.check_answerability(query_refined, docs, ctx, backend)
    if not answerable:
        return "I don't have enough information to answer that."

    # Generation
    response = backend.generate(query_refined, docs, ctx)

    # Output guardrails
    citations = rag.extract_citations(response, docs, ctx, backend)
    policy_score = guardian.guardian_check(
        ChatContext().add(Message("assistant", response)),
        backend, criteria="policy"
    )

    if policy_score > 0.8:
        return "I can't provide that response due to policy restrictions."

    return format_response_with_citations(response, citations)
```

### 5.2 Deployment Benefits

**Single endpoint:** Application code calls one vLLM server for all guardrails (vs. 5+ separate model endpoints).

**Typed interface:** Guardrails are function calls, not prompt engineering. New developers can integrate guardrails by reading type signatures.

**Modular updates:** Update adapters by re-composing the checkpoint (e.g., add a new jailbreak filter from Granite Guardian v2). No application code changes required if the function signature is unchanged.

**Standard infrastructure:** Runs on any vLLM-compatible setup (no custom inference engine, no vendor lock-in).

---

## 6. Discussion: Deployment Considerations

### 6.1 When to Use Composed Adapters

**Best fit:**
- **Multiple guardrails** over shared context (RAG, chat, content moderation)
- **Specialist tasks** where prompting underperforms (classification, extraction)
- **Latency-sensitive** deployments (user-facing applications)
- **Cost-conscious** production (GPU memory/compute constrained)

**Less suitable:**
- **Single guardrail** (composition overhead not justified)
- **Heterogeneous tasks** with different contexts per guardrail
- **Frequently updated** guardrails (re-composition cost)

### 6.2 Limitations

**Generative adapters.** Current evaluation focuses on classification/extraction tasks (short outputs). Long-form generation adapters benefit less from KV-cache reuse because generation time dominates prefill time.

**Adapter count.** We test 5 adapters; scaling to 50+ may require hierarchical routing or adapter pruning strategies.

**Single domain evaluation.** Results are from government services RAG. Broader evaluation (healthcare, legal, customer support) would strengthen generalizability claims.

**Composition as a deployment step.** Unlike separate models (which can be updated independently), composed checkpoints require re-composition to update one adapter. This adds operational overhead for frequently updated guardrails.

### 6.3 Production Readiness Checklist

✅ **Standard serving:** vLLM/HuggingFace (no custom engine)
✅ **Apache-2.0 licensed:** Code and checkpoints openly available
✅ **Reproducible:** Experimental setup and data provided
✅ **Typed interface:** Mellea function signatures for integration
✅ **Accuracy validated:** Adapters outperform prompting baselines
✅ **Latency validated:** 1.86× speedup under production load

⚠️ **Monitoring needed:** Track adapter activation rates, accuracy drift
⚠️ **Update workflow:** Establish re-composition pipeline for adapter updates

---

## 7. Related Work

**Guardrails for LLMs.** Llama Guard, Granite Guardian, and NeMo Guardrails provide standalone models or frameworks for safety checking. We show how to compose guardrails efficiently for deployment.

**Multi-LoRA serving.** S-LoRA and vLLM enable batching multiple LoRA adapters but cannot share KV caches across adapters. Punica reduces adapter switching overhead but still re-prefills per switch.

**Activated LoRA.** Prior work shows LoRA can activate mid-sequence for control token-based routing. We extend this to multi-adapter production deployment with typed interfaces.

**RAG systems.** LangChain, LlamaIndex, and Haystack provide RAG orchestration frameworks. Our work optimizes the inference layer for guardrail-heavy RAG deployments.

**Efficient inference.** PagedAttention (vLLM), continuous batching, and speculative decoding improve LLM serving efficiency. Our contribution is complementary: reducing redundant prefills in multi-adapter scenarios.

---

## 8. Conclusion

We present Granite Switch, a practical deployment system for production RAG guardrails. By composing independently trained adapters into a single checkpoint where each adapter reuses the base model's KV cache, the system achieves 1.86× speedup over standard multi-LoRA serving while maintaining specialist accuracy (answerability: 87.5%, +31.2 points over prompting). The composed checkpoint serves unchanged on vLLM and HuggingFace, and a typed function interface (Mellea) enables clean application integration. This work provides an evidence-based deployment strategy for RAG systems that require multiple safety and quality guardrails without multiplying infrastructure cost. Code, checkpoints, and experimental data are released under Apache-2.0 to facilitate reproducible deployable AI research.

**Open source:** https://github.com/ibm-granite/granite-switch (anonymized for review)

---

## References

[To be formatted - include LoRA, S-LoRA, vLLM, Punica, Llama Guard, Granite Guardian, NeMo Guardrails, MTRAG, RAG frameworks, efficient inference papers]

---

## Appendix A: Reproducibility Details

### A.1 Accuracy Experiment

**Model paths:**
- Adapter: `/path/to/granite-switch-alora-prefer` (anonymized)
- Prompting: `ibm-granite/granite-4.1-3b`

**Data:**
- Corpus: MTRAG government services corpus (49,607 documents)
- Queries: 16 IRS/tax questions (10 answerable, 6 unanswerable)
- Seeds: 3 (random state: 42, 123, 456)

**Retrieval:**
- Embedding model: `ibm-granite/granite-embedding-small-english-r2`
- Vector store: ChromaDB with cosine similarity
- Top-K: 5 documents
- Max length: 1024 tokens per document

**Code:**
```bash
python run_answerability_accuracy.py \
  --mode adapter \
  --model /path/to/granite-switch-alora-prefer \
  --queries-file rag_101_queries.txt \
  --chroma-path ./govt_chroma \
  --output-dir ./results/adapter \
  --seeds 3
```

### A.2 Latency Experiment

**Models:**
- aLoRA: `/path/to/granite-switch-alora-prefer`
- LoRA: `/path/to/granite-switch-lora-only`

**Pipeline:**
- Steps: harm → rewrite → retrieval → answerability → generation → citation → guardian
- Conversations: 50 per seed
- Concurrency: 8 simultaneous conversations
- Seeds: 3

**Hardware:**
- GPU: NVIDIA A100 80GB (anonymized for review)
- vLLM: 0.26
- CUDA: 12.1
- PyTorch: 2.1

**Code:**
```bash
python run_latency_experiments.py \
  --seeds 3 \
  --runs 50 \
  --concurrency 8 \
  --alora-model /path/to/alora \
  --lora-model /path/to/lora \
  --output-dir ./results/latency
```

---

*Word count: ~3400 (fits 4-page limit with figures/tables)*
