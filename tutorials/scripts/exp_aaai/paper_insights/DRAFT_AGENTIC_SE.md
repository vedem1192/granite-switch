# Composable Guardrail Functions for Agentic RAG: Building AI Systems Like Software

**DRAFT VERSION - Agentic Software Engineering Workshop (AAAI-27)**

**Topics:** Software Safety and Integrity, Architectures for Agentic Software Systems

---

## Abstract

Agentic RAG systems orchestrate multiple AI components—harm filters, query rewriters, answerability gates, and citation checkers—much like software functions in a program. However, unlike traditional software where functions share memory and state efficiently, current AI agent architectures re-read the entire context at every function call, multiplying compute costs. We present **Granite Switch**, a system that enables composable guardrail functions to be built, tested, and deployed independently, then composed into a single checkpoint where each function reuses shared context through KV-cache preservation. Each guardrail is invoked by a control token (analogous to a function call), processes only its task-specific tokens, and maintains the base model's context representation for subsequent functions. On a five-guardrail RAG pipeline, this software-like composition achieves **1.86× speedup** (47.8s vs 88.8s) compared to standard multi-adapter serving, while maintaining specialist accuracy (answerability: 87.5%, +31.2 points over prompting). We provide a typed function interface (Mellea) that exposes each guardrail as a Python function with clear input/output contracts, enabling modular development and testing. This work demonstrates how software engineering principles—modularity, composition, typed interfaces—can be applied to build safer, more efficient agentic AI systems.

**Keywords:** agentic systems, software composition, guardrails, safety, modular AI

---

## 1. Introduction

### 1.1 Agentic RAG as a Software System

Modern RAG systems are *agentic*: they orchestrate multiple AI components, each making autonomous decisions about whether to proceed, rewrite, retrieve, or refuse. A production RAG turn executes a pipeline of specialized agents:

```
User Query
    ↓
[Harm Filter] ──→ REJECT if harmful
    ↓
[Query Rewriter] ──→ Resolve references, improve clarity
    ↓
[Retrieval] ──→ Fetch documents
    ↓
[Answerability Gate] ──→ REFUSE if unanswerable (prevent hallucination)
    ↓
[Generator] ──→ Produce answer
    ↓
[Citation Extractor] ──→ Add attributions
    ↓
[Policy Guardrail] ──→ REJECT if violates policy
    ↓
Response
```

Each component is a *function*: it takes inputs (query, documents, history), makes a decision, and returns a typed output (boolean, label, score, text). This is functionally equivalent to software engineering, where a program orchestrates function calls to accomplish a task.

### 1.2 The Gap: AI Agents Don't Compose Like Software Functions

In traditional software, functions share memory efficiently. When `function_a()` calls `function_b()`, they both access the same program state without copying data:

```python
def safe_rag_pipeline(query: str, docs: List[str]) -> str:
    if harm_filter(query, docs):  # reads (query, docs)
        return "Cannot process"

    if not answerable(query, docs):  # reads same (query, docs)
        return "No answer available"

    return generate(query, docs)  # reads same (query, docs)
```

**In current AI agent architectures, each function re-reads the entire context from scratch:**

- **Prompting:** The base model re-processes (query, docs) for every task (inaccurate)
- **Separate models:** Each guardrail model has its own weights and KV cache (expensive)
- **Multi-LoRA:** Switching adapters discards the KV cache and re-prefills the context

**The cost:** For `k` guardrails over context length `C`, current systems perform `k·C` prefill operations—equivalent to copying the entire program state `k` times.

### 1.3 Contribution: Software-Like Composition for AI Agents

We present **Granite Switch**, a system that enables AI guardrails to compose like software functions:

**1. Modular Development**
- Each guardrail is an independently trained LoRA adapter (like a library function)
- Adapters are developed, tested, and versioned separately
- Distributed as "adapter libraries" (e.g., Granite RAG Library, Granite Guardian Library)

**2. Typed Interfaces**
- Each adapter has a clear input/output contract (e.g., `check_answerability(query, docs) → bool`)
- Exposed as Python functions through Mellea interface
- Type safety and documentation built-in

**3. Efficient Composition**
- Multiple adapters compiled into one checkpoint (like linking libraries)
- Adapters invoked by control tokens (like function calls)
- Shared context computed once and reused (like shared memory)

**4. Standard Deployment**
- Composed checkpoint runs on vLLM/HuggingFace (standard serving infrastructure)
- Single endpoint hosts all guardrails
- OpenAI-compatible API

**Empirical validation:** On a five-guardrail pipeline (Granite 4.1 30B), software-like composition achieves 1.86× speedup while maintaining safety (answerability accuracy: 87.5%, +31.2 points over prompting). Code and checkpoints released under Apache-2.0.

---

## 2. System Architecture: Building AI Like Software

### 2.1 Adapter Functions: Independent Development

**Analogy:** Adapter = software library function

Each guardrail is a LoRA adapter trained for a specific task:

```python
# Granite RAG Library (independently trained)
adapters = {
    "check_answerability": LoRA(task="classify", classes=["answerable", "unanswerable"]),
    "rewrite_query": LoRA(task="rewrite", max_tokens=50),
    "extract_citations": LoRA(task="extract", output_type=List[Citation]),
}

# Granite Guardian Library (independently trained)
adapters = {
    "detect_harm": LoRA(task="score", output_range=[0, 1]),
    "check_policy": LoRA(task="classify", classes=["compliant", "violation"]),
}
```

**Development workflow:**
1. Define task specification (input/output contract)
2. Train adapter on task-specific dataset
3. Test adapter in isolation
4. Publish to adapter library (versioned, documented)

This mirrors software engineering: write function, unit test, publish to package registry.

### 2.2 Composition: Linking Libraries into One Binary

**Analogy:** Composition = static linking

The composer merges multiple adapter libraries with a base model into one deployable checkpoint:

```bash
granite-switch compose \
  --base ibm-granite/granite-4.1-30b \
  --adapters ibm-granite/granitelib-rag-r1.0 \
              ibm-granite/granitelib-guardian-r1.0 \
  --output ./granite-rag-safe
```

**What happens:**
1. **Adapter discovery:** Scan libraries for compatible adapters
2. **Control token assignment:** Each adapter gets a unique token `<|adapter_name|>`
3. **Weight merging:** Stack adapter weights into indexed tensors
4. **Switch layer insertion:** Add routing logic to select adapter per token position
5. **Tokenizer update:** Register control tokens and configure chat template

**Output:** Single checkpoint containing base model + all adapters + routing logic

**Analogy:** Like compiling a C program that links multiple `.a` libraries into one executable.

### 2.3 Activated LoRA: Shared Context Reuse

**Analogy:** aLoRA = shared memory access

**Problem:** Standard LoRA applies to all tokens in a sequence. Switching adapters requires re-computing the KV cache (like copying all program state).

**Solution:** Activated LoRA (aLoRA) applies adapter weights only after a control token:

```
Sequence: [History, Docs, <|check_answerability|>, "Is", "this", "answerable", "?"]
                                ↑
                        Control token activates adapter

Adapter IDs: [0, 0, 0, 3, 3, 3, 3]
             └─ base ─┘ └─ adapter 3 ─┘
```

**Key property:** Tokens before the control token use base weights (adapter ID 0), so their KV cache matches the base model's and can be reused by all subsequent adapters.

**Analogy:** Functions can read shared program state without copying it.

### 2.4 Typed Function Interface (Mellea)

**Analogy:** Mellea = function library API

Application code calls guardrails as typed Python functions:

```python
from mellea import OpenAIBackend, ChatContext, rag, guardian

backend = OpenAIBackend(model_id="granite-rag-safe",
                        base_url="http://localhost:8000/v1",
                        load_embedded_adapters=True)

def safe_rag_turn(user_query: str) -> str:
    """Agentic RAG pipeline with type-safe guardrails."""
    ctx = ChatContext().add(Message("user", user_query))

    # Type: guardian.guardian_check(...) -> float ∈ [0,1]
    harm_score = guardian.guardian_check(ctx, backend, criteria="harm")
    if harm_score > 0.8:
        return "I cannot process this request."

    # Type: rag.rewrite_question(...) -> str
    query_refined = rag.rewrite_question(user_query, ctx, backend)

    # Type: retrieve_documents(...) -> List[Document]
    docs = retrieve_documents(query_refined)

    # Type: rag.check_answerability(...) -> bool
    answerable = rag.check_answerability(query_refined, docs, ctx, backend)
    if not answerable:
        return "I don't have enough information to answer."

    # Type: backend.generate(...) -> str
    response = backend.generate(query_refined, docs, ctx)

    # Type: rag.extract_citations(...) -> List[Citation]
    citations = rag.extract_citations(response, docs, ctx, backend)

    return format_with_citations(response, citations)
```

**Software engineering properties:**
- **Type safety:** Each function has a clear signature
- **Composability:** Functions can be called in any order
- **Testability:** Each guardrail can be unit tested in isolation
- **Documentation:** Function signatures serve as documentation

---

## 3. Software Safety and Integrity

### 3.1 Modular Safety: Independent Guardrail Testing

**Traditional approach:** Test the entire agentic pipeline end-to-end.

**Problem:** When a test fails, which component is broken? Hard to isolate failures.

**Granite Switch approach:** Test each guardrail function in isolation.

```python
def test_answerability_precision():
    """Unit test for answerability guardrail."""
    backend = load_test_backend("granite-rag-safe")

    # Test case: clearly answerable query
    ctx = ChatContext().add(Message("user", "What is the tax rate?"))
    docs = ["The federal tax rate is 24%."]

    result = rag.check_answerability("What is the tax rate?", docs, ctx, backend)
    assert result == True, "Should detect answerable query"

def test_answerability_recall():
    """Unit test for answerability guardrail."""
    backend = load_test_backend("granite-rag-safe")

    # Test case: clearly unanswerable query
    ctx = ChatContext().add(Message("user", "What is Jupiter's mass?"))
    docs = ["The federal tax rate is 24%."]  # unrelated docs

    result = rag.check_answerability("What is Jupiter's mass?", docs, ctx, backend)
    assert result == False, "Should detect unanswerable query"
```

**Benefit:** When a guardrail fails, the unit test pinpoints exactly which function and which case.

### 3.2 Composition Safety: Verification of Multi-Function Pipelines

**Integration tests** verify that composed guardrails work together:

```python
def test_harm_blocks_unsafe_generation():
    """Integration test: harm filter should prevent unsafe outputs."""
    backend = load_test_backend("granite-rag-safe")

    unsafe_query = "How do I hack into a system?"
    result = safe_rag_turn(unsafe_query)

    assert "cannot process" in result.lower(), \
           "Harm filter should block unsafe query"

def test_unanswerable_returns_refusal():
    """Integration test: answerability gate should refuse when no docs match."""
    backend = load_test_backend("granite-rag-safe")

    query = "What is the weather in Tokyo?"
    result = safe_rag_turn(query)  # government docs corpus has no weather info

    assert "don't have enough information" in result.lower(), \
           "Should refuse unanswerable queries instead of hallucinating"
```

**Analogy:** Unit tests verify individual functions; integration tests verify function composition.

### 3.3 Preventing Hallucinated Outputs: Answerability Guardrail

**Software safety problem:** RAG systems hallucinate when they generate answers not grounded in retrieved documents.

**Solution:** Answerability gate (a classifier function) determines "can this query be answered from these documents?" before generation.

**Decision logic:**
```python
if not rag.check_answerability(query, docs, ctx, backend):
    return "I don't have enough information to answer."
else:
    return generate_answer(query, docs)
```

**Empirical validation (Section 4.1):** Answerability adapter achieves 87.5% accuracy with 90% recall. This means:
- **90% of answerable queries** are correctly identified → system generates answers
- **83% of unanswerable queries** are correctly identified → system refuses (no hallucination)

**Comparison:** Prompting baseline has only 40% recall (misses 60% of answerable queries) and would frequently refuse to answer when it should.

### 3.4 Supply Chain Integrity: Versioned Adapter Libraries

**Analogy:** Adapter libraries = software dependencies (like npm, pip)

Adapter libraries are versioned and published with metadata:

```yaml
# Granite RAG Library v1.0 metadata
library_name: "granitelib-rag"
version: "1.0.0"
license: "Apache-2.0"
base_model: "ibm-granite/granite-4.1-*"
adapters:
  - name: "check_answerability"
    task: "classification"
    accuracy: 0.875  # validated on benchmark
    input_schema: {query: str, docs: List[str]}
    output_schema: bool
  - name: "rewrite_query"
    task: "generation"
    bleu_score: 0.82
    input_schema: {query: str}
    output_schema: str
```

**Compose with version pinning:**
```bash
granite-switch compose \
  --adapters ibm-granite/granitelib-rag-r1.0==1.0.0  # pin version
```

**Safety benefit:** Reproducible builds, audit trail, rollback capability (like software dependency management).

---

## 4. Experimental Evaluation

### 4.1 Safety Validation: Answerability Guardrail Accuracy

**Research question:** Does the composed answerability function maintain safety (prevent hallucination) compared to prompting?

**Task:** Binary classification: "Can this query be answered from these documents?"

**Setup:**
- Dataset: 16 IRS/tax queries (10 answerable, 6 unanswerable)
- Corpus: 49,607 government documents
- Retrieval: ChromaDB, top-K=5
- Seeds: 3 independent runs

**Results:**

| Method | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|--------|-----|
| **Adapter (composed)** | **87.5% ± 0.0** | **90.0% ± 0.0** | **90.0% ± 0.0** | **90.0% ± 0.0** |
| Prompting baseline | 56.3% ± 0.0 | 80.0% ± 0.0 | 40.0% ± 0.0 | 53.3% ± 0.0 |
| **Improvement** | **+31.2 pts** | **+10.0 pts** | **+50.0 pts** | **+36.7 pts** |

**Safety interpretation:**
- **High recall (90%):** System correctly allows 90% of answerable queries → users get answers
- **High precision (90%):** When system says "answerable," it's right 90% of the time → low false positive rate
- **Prompting failure mode:** 40% recall means prompting baseline refuses 60% of valid questions → poor user experience *and* drives users to bypass the guardrail

**Analogy:** A safety function with 87.5% accuracy is like a type checker with 87.5% bug detection rate—not perfect, but far better than no checking (56.3%).

### 4.2 Efficiency: Composition Overhead

**Research question:** Does composing multiple guardrails into one model add overhead vs. serving them separately?

**Hypothesis:** Shared context reuse (KV-cache) should reduce overhead.

**Setup:**
- Pipeline: 5 guardrails (harm → rewrite → answerability → citation → policy)
- Model: Granite 4.1 30B
- Concurrency: 8 conversations
- Conversations: 50 per seed
- Seeds: 3 independent runs

**Comparison:**
- **aLoRA (Granite Switch):** Composed adapters with KV-cache reuse
- **Standard LoRA:** Separate adapter switches (re-prefill per switch)

**Results:**

| Method | Wall Time (s) | Speedup |
|--------|--------------|---------|
| **aLoRA (composed)** | **47.8 ± 1.7** | **1.86×** |
| LoRA (separate) | 88.8 ± 2.7 | 1.0× |

**Efficiency interpretation:**
- **41.1 seconds saved per pipeline** (46% reduction)
- **Speedup source:** Context prefilled once (5000 tokens) vs. 5 times (25,000 tokens)
- **Scales with guardrails:** More guardrails → bigger speedup (estimated 3× at 10 guardrails)

**Software analogy:** Shared memory is faster than copying state. Composition has *negative* overhead—it's faster than isolated functions.

---

## 5. Discussion: Software Engineering Principles for Agentic AI

### 5.1 Modularity: Develop and Test in Isolation

**Principle:** Functions should be developed and tested independently.

**Application:** Each adapter is trained on a task-specific dataset, evaluated on benchmarks, and published as a standalone component. Developers can test adapters in isolation before composing them.

**Benefit:** Parallel development (multiple teams can build adapters simultaneously), clear ownership (RAG team owns retrieval adapters, safety team owns harm detection), easier debugging (unit tests isolate failures).

### 5.2 Composition: Build Complex Systems from Simple Parts

**Principle:** Complex systems are built by composing simple, well-tested components.

**Application:** A production RAG system composes 5-10 guardrails, each handling one concern (safety, accuracy, compliance). The composer merges them into one deployable artifact.

**Benefit:** Flexibility (swap adapters without retraining base model), reusability (same answerability adapter used in multiple RAG applications), maintainability (update one adapter without touching others).

### 5.3 Typed Interfaces: Clear Contracts Between Components

**Principle:** Functions communicate through well-defined interfaces with type signatures.

**Application:** Each adapter has a schema specifying inputs (query, docs, history) and outputs (bool, score, label, text). Mellea enforces these contracts at runtime.

**Benefit:** Early error detection (type mismatches caught before deployment), self-documenting code (function signatures explain what each guardrail does), safer refactoring (interface changes break tests, not production).

### 5.4 Efficiency: Avoid Redundant Computation

**Principle:** Don't compute the same thing multiple times.

**Application:** Context is prefilled once and reused by all guardrails (KV-cache sharing). Equivalent to caching expensive function results.

**Benefit:** Lower latency (1.86× faster), lower cost (less GPU compute), better scalability (can add more guardrails without linear cost increase).

---

## 6. Related Work

**Agentic AI systems.** AutoGen, LangChain, MetaGPT orchestrate multiple LLM calls for complex tasks. Our work optimizes the inference layer for multi-component agentic systems.

**Multi-adapter serving.** S-LoRA and vLLM enable batching multiple LoRA adapters but cannot share KV caches. Punica reduces switching overhead but still re-prefills per adapter.

**Guardrails for LLMs.** Llama Guard, Granite Guardian, NeMo Guardrails provide standalone safety models. We show how to compose guardrails efficiently while maintaining safety guarantees.

**Software engineering for ML.** MLOps and LLMOps focus on training, deployment, and monitoring pipelines. Our contribution is at the inference architecture level: making composed AI systems behave like composed software.

**Formal verification of neural networks.** Prior work verifies properties of individual models. Our modular architecture enables unit testing individual guardrails and integration testing their composition.

---

## 7. Limitations and Future Work

**Generative adapters.** Current work focuses on classification/extraction tasks (short outputs). Long-form generation adapters (e.g., creative writing assistants) benefit less from KV-cache reuse.

**Formal verification.** Unit tests validate guardrails empirically. Future work could apply formal methods to *prove* properties (e.g., "harm filter blocks 100% of harmful queries matching pattern X").

**Adapter composition strategies.** We test sequential composition (5 adapters in a chain). Future work: parallel composition (multiple adapters vote), hierarchical composition (adapter DAGs).

**Automated testing.** Currently, developers write unit tests manually. Future work: automatically generate test cases from adapter specifications.

---

## 8. Conclusion

We present Granite Switch, a system that applies software engineering principles to agentic AI: modularity (independent adapter development), composition (merge into one checkpoint), typed interfaces (clear contracts), and efficiency (shared context reuse). On a five-guardrail RAG pipeline, this approach achieves 1.86× speedup while maintaining safety (answerability: 87.5%, +31.2 points over prompting). By treating AI guardrails as composable functions with type signatures and unit tests, we enable building agentic systems that are safer, more efficient, and easier to maintain. Code and checkpoints released under Apache-2.0 to facilitate reproducible research in agentic software engineering.

**Open source:** https://github.com/ibm-granite/granite-switch (anonymized for review)

---

## References

[To be formatted - include LoRA, S-LoRA, vLLM, AutoGen, LangChain, Llama Guard, Granite Guardian, NeMo Guardrails, software testing literature, formal verification papers]

---

## Appendix A: Reproducibility

### A.1 Answerability Safety Test

```bash
python run_answerability_accuracy.py \
  --mode adapter \
  --model /path/to/granite-switch-alora-prefer \
  --queries-file rag_101_queries.txt \
  --chroma-path ./govt_chroma \
  --output-dir ./results/adapter \
  --seeds 3
```

### A.2 Composition Efficiency Test

```bash
python run_latency_experiments.py \
  --seeds 3 \
  --runs 50 \
  --concurrency 8 \
  --alora-model /path/to/granite-switch-alora-prefer \
  --lora-model /path/to/granite-switch-lora-only \
  --output-dir ./results/latency
```

**Hardware:** NVIDIA A100 80GB, vLLM 0.26, PyTorch 2.1

---

*Word count: ~3800 (fits 4-page short paper with figures/tables)*
