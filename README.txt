================================================================================
zolt
================================================================================

Modelo de linguagem causal para geracao de codigo e raciocinio (250.9M parametros por omissao com predefinicao zolt-mini, contexto base de 4096 tokens extensivel para 16384 tokens).


ARQUITECTURA
--------------------------------------------------------------------------------
Componente                      Especificacao
--------------------------------------------------------------------------------
Tipo de Modelo                  Transformador auto-regressivo decoder-only
Contagem de Parametros          250,905,600 (~250.9M padrao zolt, 109.5M para zolt-mini)
Normalizacao                    RMSNorm (epsilon = 1e-6)
Activacao FFN                   SwiGLU (dimensao oculta = 3072 para zolt, 2048 para zolt-mini)
Codificacao Posicional          Rotary Position Embeddings (RoPE, theta = 10000.0,
                                escalonamento Linear e NTK-aware)
Tipo de Atencao                 Grouped-Query Attention (16 cabecas para zolt,
                                12 cabecas para zolt-mini, dimensao da cabeca = 64)
Configuracao Esparsa/MatFormer  e4b corte de dimensao para sub-redes aninhadas
                                (dimensoes activas: 512, 1024)
Tokens Especiais                <pad>, <bos>, <eos>, <unk>, <think>, </think>,
                                <tool_call>, </tool_call>, <tool_response>,
                                </tool_response>, <code>, </code>,
                                <|im_start|>, <|im_end|>, <FILL>, <PREFIX>,
                                <SUFFIX>, <search>, <replace>, <diff_end>,
                                <uncertain>, <db_call>, </db_call>
--------------------------------------------------------------------------------


ESTRUTURA DO REPOSITORIO
--------------------------------------------------------------------------------
zolt/
├── Makefile                       # Alvos de automacao para configuracao, testes, pipeline de dados e treino
├── README.md                      # Documentacao do repositorio em ingles
├── README.txt                     # Documentacao do repositorio em portugues mocambicano (texto simples)
├── notebooks/
│   └── zolt_train.ipynb           # Caderno interactivo de treino e execucao
├── plano-z1-zoneai.md             # Documento inicial de especificacoes do projecto
├── pyproject.toml                 # Configuracao do pacote, dependencias e metadados de construcao
├── pytest.ini                     # Configuracao do executor de testes pytest
├── scripts/
│   ├── debug_train.py             # Teste do ciclo de treino com tokens sinteticos em CPU
│   └── setup_runpod.sh            # Script de provisionamento automatico para instancias GPU em nuvem
├── smoke_test.py                  # Script de verificacao completa de arquitectura e componentes em CPU
├── tests/
│   ├── test_data.py               # Testes unitarios para filtragem, qualidade heuristica, curriculum e destilacao
│   ├── test_db_call.py            # Testes unitarios para chamadas estruturadas de bases de dados
│   ├── test_diff_format.py        # Testes unitarios para blocos diff de busca/substituicao
│   ├── test_inference_features.py # Testes unitarios para roteamento MatFormer e incerteza
│   ├── test_memory_session.py     # Testes unitarios para memoria persistente entre sessoes
│   ├── test_model.py              # Testes unitarios para blocos do transformador, RoPE, RMSNorm, SwiGLU e predefinicoes
│   ├── test_optimize_search.py    # Testes unitarios para pesquisa de hiperparametros
│   ├── test_probe_classify.py     # Testes unitarios para sondas de classificacao
│   ├── test_probe_cluster.py      # Testes unitarios para agrupamento K-means sobre representacoes
│   ├── test_probe_regress.py      # Testes unitarios para sondas de regressao
│   ├── test_tokenizer.py          # Testes unitarios para o tokenizer BPE Byte-Level e tokens especiais
│   ├── test_verify_js.py          # Testes unitarios para auto-verificacao JavaScript
│   ├── test_verify_python.py      # Testes unitarios para auto-verificacao Python
│   └── test_verify_ts.py          # Testes unitarios para auto-verificacao TypeScript
└── zolt/
    ├── __init__.py                # Raiz do pacote exportando classes principais e versao
    ├── config.py                  # Dataclass ZoltConfig definindo padrao 250M e predefinicao zolt-mini
    ├── data/
    │   ├── __init__.py            # Raiz do pacote de dados exportando datasets, loaders, filtros e curriculum
    │   ├── curriculum.py          # Utilitarios de pontuacao de complexidade e ordenacao para aprendizagem por curriculum
    │   ├── dataset.py             # PackedSequenceDataset e DataLoader para causal LM com suporte a curriculum
    │   ├── db_call_synth.py       # Gerador sintetico de dados para chamadas a base de dados
    │   ├── distill.py             # Pipeline de destilacao sintetica de modelo professor e mistura de datasets
    │   ├── download.py            # Utilitario de descarregamento de datasets para StarCoderData e The Stack v2
    │   ├── filter_code.py         # Filtro de licencas, linguagens, pontuacao de qualidade tipo manual escolar e desduplicacao SHA256
    │   └── pipeline.py            # Orquestrador CLI de processamento de dados ponta a ponta
    ├── eval.py                    # Utilitario de avaliacao para perplexidade, sintaxe e equilibrio de tags de raciocinio
    ├── inference/
    │   ├── __init__.py            # Pacote de inferencia exportando gerador, verificacao e diffs
    │   ├── db_call.py             # Validacao e parsing de chamadas a base de dados
    │   ├── diff_format.py         # Analisador e aplicador de formato diff nativo de busca/substituicao
    │   ├── generator.py           # ZoltGenerator com roteamento MatFormer e pontuacao de incerteza
    │   ├── verify.py              # Despachador de linguagem para verificacao multi-linguagem
    │   ├── verify_base.py         # Tipos comuns e ciclo de repeticao para auto-verificacao
    │   ├── verify_js.py           # Auto-verificacao JavaScript (node/eslint e fallback heuristico)
    │   ├── verify_python.py       # Auto-verificacao Python (ast.parse e mypy)
    │   └── verify_ts.py           # Auto-verificacao TypeScript (tsc e fallback heuristico)
    ├── memory/
    │   ├── __init__.py            # Pacote de memoria intersessao
    │   └── session.py             # Armazenamento chave-valor persistente mapeando embeddings para texto
    ├── model.py                   # Implementacao PyTorch de RMSNorm, RoPE, SwiGLU, MatFormer e ZoltForCausalLM
    ├── optimize/
    │   ├── __init__.py            # Pacote de optimizacao de hiperparametros
    │   └── search.py              # Pesquisa em grelha e pesquisa aleatoria de hiperparametros
    ├── probe/
    │   ├── __init__.py            # Pacote de sondas
    │   ├── classify.py            # Sonda de classificacao sobre representacoes ocultas
    │   ├── cluster.py             # Agrupamento Mini-batch K-means sobre representacoes
    │   └── regress.py             # Sondas de regressao para pontuacao de qualidade e complexidade
    ├── rope_scaling.py            # Modulo de extensao de contexto RoPE Linear e NTK-aware
    ├── tokenizer/
    │   ├── __init__.py            # Raiz do pacote do tokenizer exportando ZoltTokenizer
    │   ├── train_tokenizer.py     # Script de treino de tokenizer BPE Byte-Level com tokens especiais
    │   └── zolt_tokenizer.py      # Interface de execucao do tokenizer com formatacao de raciocinio e ChatML
    └── train.py                   # Ciclo de treino causal LM com suporte a sobre-treino e curriculum


GUIA RAPIDO DE DESENVOLVIMENTO
--------------------------------------------------------------------------------
# Criar ambiente virtual e instalar dependencias
uv venv .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu --python .venv/bin/python
uv pip install tokenizers datasets pytest einops tqdm --python .venv/bin/python
uv pip install -e . --python .venv/bin/python

# Executar teste de fumo de arquitectura
python smoke_test.py

# Executar conjunto de testes unitarios
pytest tests/ -v


PIPELINE DE TREINO
--------------------------------------------------------------------------------

Fase 1: Descarregamento de Dados
---------------------------------
Descarregar dados de codigo-fonte do StarCoderData:

python -m zolt.data.download \
  --source starcoder \
  --output_dir data/raw \
  --langs javascript typescript python vue css html \
  --max_samples 300000


Fase 2: Filtragem de Dados e Treino do Tokenizer
-------------------------------------------------
Filtrar dados brutos por linguagem, licenca permissiva, heuristica de qualidade e desduplicacao exacta:

python -m zolt.data.pipeline filter \
  --raw_dir data/raw \
  --filtered_dir data/filtered

Treinar tokenizer BPE Byte-Level personalizado (vocabulario de 32000) nos corpora filtrados:

python -m zolt.tokenizer.train_tokenizer \
  --data_dirs data/filtered \
  --output zolt_tokenizer.json \
  --vocab_size 32000

Tokenizar documentos JSONL filtrados em ficheiros binarios continuos de tokens:

python -m zolt.data.pipeline tokenize \
  --filtered_dir data/filtered \
  --tokenizer zolt_tokenizer.json \
  --tokens_dir data/tokens

Validar volume total de tokens nos ficheiros binarios gerados:

python -m zolt.data.pipeline validate \
  --tokens_dir data/tokens


Fase 3: Treino do Modelo Base (Comprimento de Contexto de 4096 Tokens)
----------------------------------------------------------------------
Executar pre-treino de causal LM com AdamW, escalonamento de taxa de aprendizagem por cosseno e precisao mista:

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


Fase 4: Extensao de Contexto (4096 para 16384 Tokens)
------------------------------------------------------
Aplicar escalonamento RoPE NTK-aware para estender o comprimento de contexto para 16384 tokens:

python -m zolt.rope_scaling \
  --checkpoint checkpoints/ckpt-step0100000 \
  --output checkpoints/zolt-16k \
  --target_len 16384 \
  --method ntk


Fase 5: Avaliacao
------------------
Avaliar perda de entropia cruzada, perplexidade e validade sintactica:

python -m zolt.eval \
  --checkpoint checkpoints/zolt-16k \
  --eval_jsonl data/eval.jsonl


Fase 6: Inferencia Interactiva
-------------------------------
Executar interface de linha de comando para inferencia com fluxo continuo:

python -m zolt.inference \
  --checkpoint checkpoints/zolt-16k \
  --tokenizer zolt_tokenizer.json \
  --temp 0.7 \
  --top_p 0.9


COMPUTE RECOMENDADO
--------------------------------------------------------------------------------
Fase                                            Ambiente
--------------------------------------------------------------------------------
Testes de fumo e desenvolvimento local          CPU local / Nivel gratuito em nuvem
Filtragem de dados e treino do tokenizer        CPU multi-nucleo
Treino do modelo base (contexto 4096, 100K)     Instancia GPU em nuvem (NVIDIA RTX 3090, RTX 4090 ou A100)
Ajuste fino de extensao de contexto (16384)     Instancia GPU em nuvem (NVIDIA RTX 4090 ou A100)
Inferencia e avaliacao                          CPU local ou GPU individual
--------------------------------------------------------------------------------


STACK DE FOCO
--------------------------------------------------------------------------------
- Linguagens: JavaScript, TypeScript, Python, Vue, TSX, JSX, CSS, SCSS, HTML, Markdown, JSON, YAML
- Frameworks e Bibliotecas: React.js, Next.js, Vue.js, NestJS, Tailwind CSS
