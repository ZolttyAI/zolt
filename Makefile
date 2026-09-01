.DEFAULT_GOAL := help

PYTHON := .venv/bin/python
UV     := $(HOME)/.local/bin/uv

# ─── Setup ──────────────────────────────────────────────────────────────────

.PHONY: setup
setup: ## Criar venv e instalar dependências CPU (desenvolvimento local)
	$(UV) venv .venv
	$(UV) pip install torch --index-url https://download.pytorch.org/whl/cpu --python $(PYTHON)
	$(UV) pip install tokenizers datasets pytest einops tqdm --python $(PYTHON)
	@echo "✓ Ambiente pronto. Activa com: source .venv/bin/activate"

.PHONY: setup-gpu
setup-gpu: ## Instalar dependências com suporte GPU (para RunPod/Colab)
	$(UV) venv .venv
	$(UV) pip install torch --python $(PYTHON)
	$(UV) pip install tokenizers datasets pytest einops tqdm wandb --python $(PYTHON)

# ─── Testes ─────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Correr todos os testes unitários
	$(PYTHON) -m pytest tests/ -v

.PHONY: smoke
smoke: ## Smoke test rápido (valida arquitetura sem dados reais)
	$(PYTHON) smoke_test.py

# ─── Dados ──────────────────────────────────────────────────────────────────

.PHONY: download-data
download-data: ## Descarregar StarCoderData (fonte principal, sem gating)
	HF_TOKEN=$(HF_TOKEN) $(PYTHON) -m zolt.data.download \
		--source starcoder \
		--output_dir data/raw \
		--max_samples 300000

.PHONY: filter-data
filter-data: ## Filtrar dados crus (lang + licença + qualidade + dedup)
	$(PYTHON) -m zolt.data.pipeline filter \
		--raw_dir data/raw \
		--filtered_dir data/filtered

.PHONY: train-tokenizer
train-tokenizer: ## Treinar tokenizer BPE 32K sobre dados filtrados
	$(PYTHON) -m zolt.tokenizer.train_tokenizer \
		--data_dirs data/filtered \
		--output zolt_tokenizer.json \
		--vocab_size 32000

.PHONY: tokenize-data
tokenize-data: ## Tokenizar dados filtrados → ficheiros .bin
	$(PYTHON) -m zolt.data.pipeline tokenize \
		--filtered_dir data/filtered \
		--tokenizer zolt_tokenizer.json \
		--tokens_dir data/tokens

.PHONY: validate-data
validate-data: ## Validar contagem de tokens (alvo: ≥3B)
	$(PYTHON) -m zolt.data.pipeline validate \
		--tokens_dir data/tokens

.PHONY: pipeline
pipeline: filter-data train-tokenizer tokenize-data validate-data ## Pipeline completo de dados

# ─── Treino ─────────────────────────────────────────────────────────────────

.PHONY: train
train: ## Treinar zolt 250M (requer GPU — usar no RunPod/Colab)
	$(PYTHON) -m zolt.train \
		--token_files $$(ls data/tokens/*.bin | tr '\n' ' ') \
		--output_dir checkpoints/ \
		--max_seq_len 4096 \
		--batch_size 8 \
		--grad_accum 4 \
		--lr 3e-4 \
		--warmup_steps 500 \
		--total_steps 100000 \
		--dtype bf16 \
		--wandb

.PHONY: train-debug
train-debug: ## Treino mínimo local em CPU (smoke test do loop)
	$(PYTHON) -m zolt.train \
		--token_files data/tokens/debug.bin \
		--output_dir checkpoints/debug/ \
		--max_seq_len 512 \
		--batch_size 2 \
		--grad_accum 1 \
		--lr 3e-4 \
		--warmup_steps 10 \
		--total_steps 50 \
		--dtype fp32

# ─── Extensão de Contexto ───────────────────────────────────────────────────

.PHONY: extend-context
extend-context: ## Estender contexto 4K → 16K via NTK RoPE scaling
	$(PYTHON) -m zolt.rope_scaling \
		--checkpoint $(CKPT) \
		--output checkpoints/zolt-16k \
		--target_len 16384 \
		--method ntk

# ─── Avaliação ──────────────────────────────────────────────────────────────

.PHONY: eval
eval: ## Avaliar checkpoint (perplexidade + syntax check)
	$(PYTHON) -m zolt.eval \
		--checkpoint $(CKPT) \
		--eval_jsonl data/eval.jsonl

# ─── Utils ──────────────────────────────────────────────────────────────────

.PHONY: count-params
count-params: ## Contar parâmetros do modelo zolt
	$(PYTHON) -c "from zolt import ZoltConfig, ZoltForCausalLM; m=ZoltForCausalLM(ZoltConfig()); n=sum(p.numel() for p in m.parameters()); print(f'zolt params: {n:,} ({n/1e6:.1f}M)')"

.PHONY: clean
clean: ## Limpar caches e ficheiros temporários
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache 2>/dev/null; true

.PHONY: help
help: ## Mostrar ajuda
	@echo ""
	@echo "zolt — zolt.ai | Makefile"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Variáveis:"
	@echo "    HF_TOKEN=xxx   Token HuggingFace (para The Stack v2)"
	@echo "    CKPT=path      Caminho para checkpoint (para eval/extend-context)"
	@echo ""
