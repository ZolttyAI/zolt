#!/bin/bash
# ============================================================
# zolt zolt.ai — Setup Script para RunPod / Vast.ai
# Colar no terminal do pod após arranque
# ============================================================

set -e

echo "============================================================"
echo "zolt zolt.ai — RunPod Setup"
echo "============================================================"

# ─── 1. Verificar GPU ────────────────────────────────────────
echo ""
echo "[1/6] Verificar GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | BF16: {torch.cuda.is_bf16_supported()}')"

# ─── 2. Instalar uv ──────────────────────────────────────────
echo ""
echo "[2/6] Instalar uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# ─── 3. Clonar repositório zolt ──────────────────────────────
echo ""
echo "[3/6] Clonar zolt..."
# Substituir pelo URL real do repositório
# git clone https://github.com/SEU_USER/zolt.git /workspace/zolt
# cd /workspace/zolt

# Se já tens o código comprimido:
# tar -xzf zolt.tar.gz -C /workspace/

cd /workspace/zolt 2>/dev/null || { echo "Ajusta o caminho do repositório zolt"; exit 1; }

# ─── 4. Criar venv e instalar dependências ───────────────────
echo ""
echo "[4/6] Instalar dependências Python (GPU)..."
uv venv .venv
uv pip install torch --python .venv/bin/python
uv pip install tokenizers datasets einops tqdm wandb --python .venv/bin/python
source .venv/bin/activate

# ─── 5. Smoke test ───────────────────────────────────────────
echo ""
echo "[5/6] Smoke test..."
python smoke_test.py

# ─── 6. Configurar WandB (opcional) ─────────────────────────
echo ""
echo "[6/6] WandB (opcional)..."
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
    echo "WandB configurado."
else
    echo "WANDB_API_KEY não definida — logging local apenas."
    echo "Para activar: export WANDB_API_KEY=xxx && wandb login"
fi

echo ""
echo "============================================================"
echo "✓ Setup completo! Para iniciar o treino:"
echo ""
echo "  # 1. Descarregar dados"
echo "  make download-data"
echo ""
echo "  # 2. Pipeline de dados completo"
echo "  make pipeline"
echo ""
echo "  # 3. Treinar zolt 250M"
echo "  make train"
echo ""
echo "  # Ou passo-a-passo:"
echo "  make filter-data"
echo "  make train-tokenizer"
echo "  make tokenize-data"
echo "  make validate-data"
echo "  make train"
echo "============================================================"
