# ZolttyAI

Causal language model for code generation and reasoning (250.9M default parameters with `zolt-mini` sub-network slice, 4096 base context extendable to 16384 tokens).

## Architecture

| Component | Specification |
|---|---|
| Model Type | Decoder-only autoregressive transformer |
| Parameter Count | 250,905,600 (~250.9M `zolt` default, 109.5M for `zolt-mini` preset) |
| Normalization | RMSNorm (epsilon = 1e-6) |
| FFN Activation | SwiGLU (hidden dimension = 3072 for `zolt`, 2048 for `zolt-mini`) |
| Positional Encoding | Rotary Position Embeddings (RoPE, theta = 10000.0, Linear and NTK-aware scaling) |
| Attention Type | Grouped-Query Attention (16 heads for `zolt`, 12 heads for `zolt-mini`, head dimension = 64) |
| Sparse / MatFormer Config | e4b nested sub-network dimension slicing (active dimensions: 512, 1024) |
| Special Tokens | `<pad>`, `<bos>`, `<eos>`, `<unk>`, `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, `<code>`, `</code>`, `<|im_start|>`, `<|im_end|>`, `<FILL>`, `<PREFIX>`, `<SUFFIX>`, `<search>`, `<replace>`, `<diff_end>`, `<uncertain>`, `<db_call>`, `</db_call>` |

## Repository Structure

```
zolt/
├── Makefile                       # Automation targets for setup, testing, data pipeline, and training
├── README.md                      # Repository documentation in English
├── README.txt                     # Repository documentation in Portuguese (plain text)
├── notebooks/
│   └── zolt_train.ipynb           # Interactive training and execution notebook
├── pyproject.toml                 # Package configuration, dependencies, and build metadata
├── pytest.ini                     # Pytest runner configuration
├── scripts/
│   ├── debug_train.py             # Synthetic token training loop test on CPU
│   └── setup_runpod.sh            # Automated provisioning script for cloud GPU instances
├── smoke_test.py                  # End-to-end CPU architecture and component verification script
├── tests/
│   ├── test_data.py               # Unit tests for filtering, quality heuristics, curriculum, and distillation
│   ├── test_db_call.py            # Unit tests for structured database call generation and validation
│   ├── test_diff_format.py        # Unit tests for search/replace native diff block parser and applicator
│   ├── test_inference_features.py # Unit tests for MatFormer routing and uncertainty scoring
│   ├── test_memory_session.py     # Unit tests for persistent intersession memory
│   ├── test_model.py              # Unit tests for transformer blocks, RoPE, RMSNorm, SwiGLU, and presets
│   ├── test_optimize_search.py    # Unit tests for hyperparameter search
│   ├── test_probe_classify.py     # Unit tests for classification probes
│   ├── test_probe_cluster.py      # Unit tests for K-means clustering over representations
│   ├── test_probe_regress.py      # Unit tests for regression probes (quality score and complexity)
│   ├── test_tokenizer.py          # Unit tests for Byte-Level BPE tokenizer and special tokens
│   ├── test_verify_js.py          # Unit tests for JavaScript verification and self-correction
│   ├── test_verify_python.py      # Unit tests for Python verification and self-correction
│   └── test_verify_ts.py          # Unit tests for TypeScript verification and self-correction
└── zolt/
    ├── __init__.py                # Package root exporting core classes and version
    ├── config.py                  # ZoltConfig dataclass defining 250M defaults and zolt-mini presets
    ├── data/
    │   ├── __init__.py            # Data package root exporting datasets, loaders, filters, and curriculum
    │   ├── curriculum.py          # Curriculum learning complexity scoring and sorting utilities
    │   ├── dataset.py             # PackedSequenceDataset and causal LM DataLoader with curriculum support
    │   ├── db_call_synth.py       # Synthetic data generation for structured database calls
    │   ├── distill.py             # Teacher model synthetic distillation pipeline and dataset mixing
    │   ├── download.py            # Dataset download utility for StarCoderData and The Stack v2
    │   ├── filter_code.py         # License, language, textbook quality score heuristics, and SHA256 dedup
    │   └── pipeline.py            # End-to-end data processing CLI orchestrator
    ├── eval.py                    # Evaluation utility for perplexity, syntax checks, and reasoning tag balance
    ├── inference/
    │   ├── __init__.py            # Inference package root exporting generator, verification, and diffs
    │   ├── db_call.py             # Structured database call validation and parser
    │   ├── diff_format.py         # Native search/replace diff parser and applicator
    │   ├── generator.py           # ZoltGenerator with MatFormer routing and uncertainty scoring
    │   ├── verify.py              # Language dispatcher for multi-language verification
    │   ├── verify_base.py         # Common types and retry loop for self-verification
    │   ├── verify_js.py           # JavaScript verification (node/eslint and heuristic fallback)
    │   ├── verify_python.py       # Python verification (ast.parse and mypy)
    │   └── verify_ts.py           # TypeScript verification (tsc and heuristic fallback)
    ├── memory/
    │   ├── __init__.py            # Intersession memory package root
    │   └── session.py             # Persistent key-value memory mapping embeddings to text
    ├── model.py                   # PyTorch implementation of RMSNorm, RoPE, SwiGLU, MatFormer, and ZoltForCausalLM
    ├── optimize/
    │   ├── __init__.py            # Optimization package root
    │   └── search.py              # Grid search and random search hyperparameter utilities
    ├── probe/
    │   ├── __init__.py            # Probes package root
    │   ├── classify.py            # Classification probe over hidden representations
    │   ├── cluster.py             # Mini-batch K-means clustering over representations
    │   └── regress.py             # Regression probes for quality score and complexity
    ├── rope_scaling.py            # Linear and NTK-aware RoPE context extension module
    ├── tokenizer/
    │   ├── __init__.py            # Tokenizer package root exporting ZoltTokenizer
    │   ├── train_tokenizer.py     # Byte-Level BPE tokenizer training script with special tokens
    │   └── zolt_tokenizer.py      # Tokenizer runtime interface with reasoning and ChatML formatting
    └── train.py                   # Autoregressive causal LM training loop with overtraining and curriculum support
```

## Development Quickstart

```bash
# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu --python .venv/bin/python
uv pip install tokenizers datasets pytest einops tqdm --python .venv/bin/python
uv pip install -e . --python .venv/bin/python

# Run architecture smoke test
python smoke_test.py

# Run unit test suite
pytest tests/ -v
```

## Training Pipeline

### Phase 1: Data Download

Download source code data from StarCoderData:

```bash
python -m zolt.data.download \
  --source starcoder \
  --output_dir data/raw \
  --langs javascript typescript python vue css html \
  --max_samples 300000
```

### Phase 2: Data Filtering and Tokenizer Training

Filter raw data by language, permissive license, quality heuristics, and exact deduplication:

```bash
python -m zolt.data.pipeline filter \
  --raw_dir data/raw \
  --filtered_dir data/filtered
```

Train custom Byte-Level BPE tokenizer (32000 vocabulary size) on filtered corpora:

```bash
python -m zolt.tokenizer.train_tokenizer \
  --data_dirs data/filtered \
  --output zolt_tokenizer.json \
  --vocab_size 32000
```

Tokenize filtered JSONL documents into continuous binary token files:

```bash
python -m zolt.data.pipeline tokenize \
  --filtered_dir data/filtered \
  --tokenizer zolt_tokenizer.json \
  --tokens_dir data/tokens
```

Validate total token volume across generated binary files:

```bash
python -m zolt.data.pipeline validate \
  --tokens_dir data/tokens
```

### Phase 3: Base Model Training (4096 Context Length)

Execute causal LM pre-training with AdamW, cosine learning rate schedule, and mixed precision:

```bash
python -m zolt.train \
  --token_files data/tokens/starcoder_javascript.bin data/tokens/starcoder_python.bin \
  --output_dir checkpoints/ \
  --max_seq_len 4096 \
  --batch_size 8 \
  --grad_accum 4 \
  --lr 0.0003 \
  --lr_min 0.00003 \
  --warmup_steps 500 \
  --total_steps 100000 \
  --save_every 1000 \
  --log_every 50 \
  --dtype bf16
```

### Phase 4: Context Extension (4096 to 16384 Tokens)

Apply NTK-aware RoPE scaling to extend context length to 16384 tokens:

```bash
python -m zolt.rope_scaling \
  --checkpoint checkpoints/ckpt-step0100000 \
  --output checkpoints/zolt-16k \
  --target_len 16384 \
  --method ntk
```

### Phase 5: Evaluation

Evaluate cross-entropy loss, perplexity, and syntax validity:

```bash
python -m zolt.eval \
  --checkpoint checkpoints/zolt-16k \
  --eval_jsonl data/eval.jsonl
```

### Phase 6: Interactive Inference

Run command-line streaming inference interface:

```bash
python -m zolt.inference \
  --checkpoint checkpoints/zolt-16k \
  --tokenizer zolt_tokenizer.json \
  --temp 0.7 \
  --top_p 0.9
```

## Recommended Compute

| Phase | Environment |
|---|---|
| Smoke tests and local development | Local CPU / Free tier compute |
| Data filtering and tokenizer training | Multi-core CPU |
| Base model training (4096 context, 100K steps) | Cloud GPU instance (NVIDIA RTX 3090, RTX 4090, or A100) |
| Context extension fine-tuning (16384 context) | Cloud GPU instance (NVIDIA RTX 4090 or A100) |
| Inference and evaluation | Local CPU or Single GPU |

## Focus Stack

- Languages: JavaScript, TypeScript, Python, Vue, TSX, JSX, CSS, SCSS, HTML, Markdown, JSON, YAML
- Frameworks and Libraries: React.js, Next.js, Vue.js, NestJS, Tailwind CSS

## License

- Apache-2.0 License
