# Plano z1 — zone.ai

Coding-agent e reasoning AI, treinado de zero como projecto interno de teste.

## 1. Visão geral

| | |
|---|---|
| **Nome / code name** | z1 |
| **Marca** | zone.ai |
| **Tipo** | Modelo helper de desenvolvimento (coding-agent + reasoning) |
| **Peso alvo** | ~125M parâmetros (protótipo único, sem fase de escala) |
| **Contexto alvo** | 16K tokens |
| **Classe de peso** | e4b (activação esparsa tipo matformer, inspirado no padrão Gemma 3n) |
| **Escopo** | Projecto interno de teste, não produto de mercado |
| **Foco de stack** | JavaScript, TypeScript, Python, Next.js, React.js, Vue.js, NestJS, Tailwind |

Objectivo: validar um pipeline de treino de zero, funcional e reproduzível, na escala de 125M. Prioridade é fechar o loop, não bater benchmark.

## 2. Arquitectura

Decoder-only, estilo Llama leve:

- **Posições**: RoPE (Rotary Position Embeddings) — permite extensão de contexto sem reescrever o modelo
- **Normalização**: RMSNorm
- **Activação**: SwiGLU
- **Base de código**: nanoGPT (Karpathy) ou litGPT (Lightning AI) — controlo total, sem sobrecarga de framework

## 3. Tokenizer

- BPE próprio, vocabulário 32K–50K
- Corpus de treino do tokenizer pesado em: JS/TS/Python, JSX/TSX, Vue SFC, padrões Next.js/NestJS
- Objectivo: reduzir tokens por ficheiro de código real e melhorar eficiência de contexto desde o início

## 4. Dados de treino

- **Fontes**: subsets filtrados de The Stack v2, StarCoderData, CodeParrot
- **Filtro**: apenas JS/TS/Python/Vue/React/Next/NestJS, licença permissiva
- **Processamento**: deduplicação obrigatória antes de qualquer run
- **Volume mínimo**: 3–6B tokens para ~125M parâmetros (ideal: mais, seguindo lógica Chinchilla)

## 5. Infraestrutura de compute

O Dell Sarien (Intel UHD 620, sem GPU dedicada) não é viável para treino do zero, mesmo em modelo pequeno. Plano:

| Fase | Compute |
|---|---|
| Testes de pipeline (escala mínima) | Colab / Kaggle free tier |
| Runs reais | RunPod ou Vast.ai, spot instances RTX 3090/4090 ou A100 |
| Precisão | bf16 mixed precision |
| Optimizador | AdamW, cosine LR schedule |

## 6. Fases de execução

### Fase 1 — Protótipo (125M, contexto curto)
Constrói o pipeline inteiro nesta escala: arquitectura, tokenizer, dados, loop de treino. Ciclos de iteração rápidos. Este é o alvo do projecto, não um passo intermédio.

### Fase 2 — Contexto curto (2K–4K)
Treino principal do protótipo a 2K–4K tokens. Custo de atenção cresce com o quadrado do contexto, por isso 16K nativo nesta fase não compensa.

### Fase 3 — Extensão de contexto
Depois do protótipo validado, aplicar RoPE scaling (linear ou NTK-aware) sobre dados de sequência longa para chegar aos 16K.

### Fase 4 — Avaliação contra uso próprio
Conjunto de avaliação construído a partir dos próprios repositórios (Mavula, Sablify), não só benchmarks genéricos tipo HumanEval. A métrica que importa é utilidade real no stack, não ranking público.

## 7. Riscos e mitigação

| Risco | Mitigação |
|---|---|
| Treino do zero sem GPU local | Compute alugado desde a Fase 1 |
| Contexto 16K nativo caro demais | Treinar curto primeiro, estender depois via RoPE scaling |
| Dados insuficientes ou contaminados | Deduplicação e filtro de licença antes de qualquer run |
| Modelo pequeno demais para reasoning real | Escopo do teste focado em utilidade prática no stack, não em reasoning genérico competitivo |

## 8. Próximos passos imediatos

1. Escolher entre nanoGPT e litGPT como base de código
2. Definir e filtrar as fontes de dados (The Stack v2 / StarCoderData / CodeParrot)
3. Treinar o tokenizer BPE próprio
4. Configurar conta RunPod ou Vast.ai e validar pipeline no Colab/Kaggle primeiro
5. Rodar Fase 1 (protótipo 125M, contexto curto)
