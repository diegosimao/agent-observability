# Guia de estudo: o que este projeto ensina, passo a passo

Este guia é para você **entender** o projeto a ponto de explicá-lo numa entrevista,
não só rodá-lo. Cada etapa tem quatro partes:

- **Conceito:** o que é, em português simples.
- **Onde está no código:** o arquivo para abrir e ler.
- **Faça você mesmo:** um comando ou experimento curto.
- **Pergunta de entrevista:** o que alguém pode te perguntar e como responder.

Siga na ordem. Cada etapa usa a anterior. Reserve 1 a 2 horas por etapa, com
calma, rodando os comandos.

---

## Mapa geral (leia primeiro)

O projeto conta uma história em 4 atos:

1. **Construir** um agente de IA que atende clientes de uma loja (etapas 1–3).
2. **Observar** esse agente em produção: velocidade, erros, custo, satisfação (etapas 4–7).
3. **Avaliar** se uma mudança melhorou ou piorou o agente (etapas 8–9).
4. **Melhorar** o agente com os próprios dados de produção, via fine-tuning (etapas 10–13).

E por fora, a **engenharia**: a API web (etapa 14) e os testes (etapa 15).

```
cliente ──► API FastAPI ──► agente (LangChain) ──► LLM + ferramentas
                                │
                                └── cada passo vira um "trace" ──► LangSmith
                                                                     │
                     dashboards ◄── métricas ◄───────────────────────┤
                     experimentos ◄── avaliação ◄────────────────────┤
                     fine-tuning ◄── dados curados ◄─────────────────┘
```

---

# ATO 1: Construir o agente

## Etapa 1: O que é um LLM e o que é um "agente"

**Conceito.** Um **LLM** (Large Language Model, como Gemini, GPT ou Claude) é um
modelo que recebe texto e devolve texto. Sozinho, ele não sabe o status do *seu*
pedido, porque não tem acesso ao seu banco de dados.

Um **agente** é um LLM que pode **usar ferramentas** (*tools*). O ciclo é:

1. O usuário pergunta: "cadê meu pedido 1002?"
2. O LLM pensa: "preciso da ferramenta `buscar_pedido` com `pedido_id=1002`".
3. O **seu código** executa a ferramenta (o LLM nunca executa nada sozinho).
4. O resultado volta para o LLM: `{"status": "em transporte"}`.
5. O LLM escreve a resposta final: "Seu pedido está em transporte."

Esse ciclo se chama **ReAct** (*Reason + Act*: raciocinar e agir). Ele pode
repetir várias vezes numa mesma pergunta.

**Onde está no código.** [`agent.py`](src/agent_observability/agent.py), função
`build_agent`. A linha `create_agent(modelo, TOOLS, system_prompt=...)` monta
esse ciclo inteiro.

**Faça você mesmo.**

```bash
uv run obs-ask "Qual o status do pedido 1002 e o frete para 30140-071?"
```

Repare na linha `tools: ['buscar_pedido', 'calcular_frete']`: o agente decidiu
sozinho usar duas ferramentas.

**Pergunta de entrevista.** *"Qual a diferença entre um chatbot e um agente?"*
O chatbot só gera texto. O agente decide **ações** (chamar ferramentas) e usa o
resultado delas. O LLM só escolhe a ação; quem executa é o código, e é aí que
ficam o controle e a segurança.

---

## Etapa 2: Ferramentas (tools) e tool calling

**Conceito.** Uma ferramenta é uma função Python comum com uma **descrição**. A
descrição (a docstring) é o que o LLM lê para decidir quando usá-la. Por isso
ela precisa ser clara: é praticamente um prompt.

Quando o LLM quer usar uma ferramenta, ele não devolve texto livre, e sim um
objeto estruturado chamado **tool call**:

```json
{"name": "buscar_pedido", "args": {"pedido_id": "1002"}, "id": "call_abc123"}
```

O `id` serve para casar a chamada com a resposta (`ToolMessage`), o que importa
quando o LLM chama várias ferramentas ao mesmo tempo.

**Onde está no código.** [`tools.py`](src/agent_observability/tools.py). Veja o
decorador `@tool` e as docstrings. Repare também que as ferramentas têm
**latência e falhas de propósito** (`_latencia`, `ToolException`): elas simulam
sistemas reais, que são lentos e instáveis.

**Faça você mesmo.** Mude a docstring de `calcular_frete` para algo vago, como
"faz um cálculo", e rode com o Gemini. O agente passa a errar quando usar a
ferramenta. Depois desfaça a mudança.

**Pergunta de entrevista.** *"O que acontece quando uma ferramenta falha?"*
Neste projeto, uma falha **esperada** (ex.: pedido não encontrado) volta para o
LLM como mensagem de erro, e ele se desculpa com o cliente. Uma falha
**inesperada** derruba a execução. Veja a função `erro_para_o_modelo` e a etapa 6.

---

## Etapa 3: Mensagens, system prompt e o LLM simulado (mock)

**Conceito.** A conversa com um LLM é uma lista de **mensagens** com papéis:

| Papel | Classe LangChain | Exemplo |
| --- | --- | --- |
| system | `SystemMessage` | "Você é o assistente da loja Vértice. Regras: ..." |
| user | `HumanMessage` | "Cadê meu pedido?" |
| assistant | `AIMessage` | texto **ou** tool calls |
| tool | `ToolMessage` | resultado da ferramenta |

O **system prompt** define o comportamento. Este projeto tem **duas versões**
(`v1` e `v2` em [`config.py`](src/agent_observability/config.py)) para
compararmos qual é melhor (etapa 7).

**O mock.** Chamar um LLM real custa dinheiro e tem limite de uso (já batemos no
limite diário do Gemini gratuito!). Então o projeto tem um **LLM falso**
([`mock_llm.py`](src/agent_observability/mock_llm.py)) que:

- escolhe ferramentas por palavras-chave (não é inteligente, e nem precisa ser);
- informa quantos **tokens** "gastou" (`usage_metadata`);
- se apresenta ao LangSmith como `gemini-3.8-flash` (`_get_ls_params`), então o
  LangSmith calcula custo como se fosse real.

A lição de engenharia: **o mock respeita o contrato** (a interface
`BaseChatModel`), não a inteligência. Qualquer coisa que siga esse contrato
funciona no resto do sistema. Mock, Gemini e o modelo local da etapa 12 são
intercambiáveis.

**Faça você mesmo.**

```bash
AGENT_MODE=mock uv run obs-ask "Como funciona a devolução?" --prompt v1
```

```bash
AGENT_MODE=mock uv run obs-ask "Como funciona a devolução?" --prompt v2
```

Compare o tamanho das respostas: a v2 tem a regra "no máximo 3 frases".

**Pergunta de entrevista.** *"Como você testa um sistema com LLM sem gastar com
API?"* Com um modelo falso que implementa a mesma interface. Assim os testes
ficam rápidos, grátis e determinísticos. O LLM real fica para a avaliação.

---

# ATO 2: Observar em produção

## Etapa 4: Observabilidade, traces e runs

**Conceito.** Em software comum, você olha logs. Com agentes, log não basta: uma
única pergunta gera várias chamadas ao LLM e às ferramentas, em sequência.
Você precisa ver a **árvore** inteira.

- **Run:** um passo (uma chamada ao LLM, uma ferramenta).
- **Trace:** a árvore de runs de uma pergunta, da entrada à resposta.

```
atendimento (trace raiz, 1.8s)
├── model (LLM decide chamar tools)        0.6s  420 tokens
├── buscar_pedido                          0.2s
├── calcular_frete                         0.5s  ← erro: timeout
└── model (LLM escreve resposta)           0.5s  380 tokens
```

O **LangSmith** é a ferramenta da LangChain que recebe, guarda e mostra esses
traces. Basta definir `LANGSMITH_TRACING=true` e a chave da API: o LangChain
envia tudo automaticamente, em segundo plano, sem deixar a resposta mais lenta.

**Onde está no código.** O envio é automático. O único cuidado está no fim do
simulador: `wait_for_all_tracers()` espera os traces terminarem de ser enviados
antes de o programa fechar.

**Faça você mesmo.** Rode o `obs-ask` acima e abra smith.langchain.com →
Projects → `agente-suporte-prod`. Clique no trace mais recente e expanda cada
passo. Veja entradas, saídas, tokens e tempos.

**Pergunta de entrevista.** *"Por que não usar só logs?"* Porque o problema de
um agente costuma estar **no meio** da cadeia (uma ferramenta lenta, um prompt
que fez o LLM escolher a ferramenta errada). O trace mostra causa e efeito
encadeados; o log mostra linhas soltas.

---

## Etapa 5: Tags, metadata e run_id (a instrumentação)

**Conceito.** Um trace sem contexto é difícil de analisar. Junto com cada
chamada enviamos três coisas:

- **`metadata`:** dimensões para **agrupar** gráficos. Ex.: `prompt_version`,
  `plano_cliente`, `canal`, `app_version`. Assim você responde "a latência no
  WhatsApp é pior que na web?".
- **`tags`:** rótulos para **filtrar** rápido. Ex.: `prompt:v2`, `simulacao`.
- **`run_id`:** um identificador que **nós** geramos antes de chamar o agente.
  Por quê? Porque depois o usuário clica 👍 ou 👎 e precisamos saber a qual
  trace aquele voto pertence.

Regra prática: dimensões com poucos valores possíveis (baixa
**cardinalidade**) servem para agrupar. Nunca coloque e-mail, texto livre ou id
de usuário em tag: vira milhões de valores e os gráficos ficam inúteis.

**Onde está no código.** [`agent.py`](src/agent_observability/agent.py), função
`_config`. Leia o docstring no topo do arquivo.

**Faça você mesmo.** No LangSmith, filtre os traces por `metadata.canal =
whatsapp`. Depois por tag `prompt:v1`.

**Pergunta de entrevista.** *"Como você liga o feedback do usuário ao trace?"*
Gero o `run_id` antes de invocar o agente, devolvo esse id na resposta da API, e
o front-end o envia de volta junto com o 👍/👎 (`POST /v1/feedback`).

---

## Etapa 6: As métricas que importam

**Conceito.** As quatro perguntas de produção e as métricas que respondem cada uma:

| Pergunta | Métrica | Como ler |
| --- | --- | --- |
| Está rápido? | latência **p50 / p95 / p99** | ver abaixo |
| Está quebrando? | **taxa de erro** (LLM e tools, separadas) | % de runs com erro |
| Quanto custa? | **tokens** e **custo** por conversa | tokens × preço do modelo |
| Está ajudando? | **feedback** do usuário e **contenção** | % de 👍, % resolvido sem humano |

**Percentis (p50, p95, p99), com calma, porque caem em entrevista.** Ordene
todas as latências da menor para a maior.

- **p50 (mediana):** metade das conversas foi mais rápida que isso. É a
  experiência "típica".
- **p95:** 95% foram mais rápidas; 5% foram piores. Mostra a cauda.
- **p99:** o 1% pior.

Por que não usar a **média**? Porque uma única conversa de 60 s esconde-se na
média, mas é exatamente o cliente que reclama. Em produção, **a cauda (p95/p99)
é o que importa**. No nosso incidente simulado, o p50 mal muda, mas o p95 dispara.

**Tokens e custo.** O LLM cobra por **token** (pedaço de palavra, ~4 caracteres
em português). Há tokens de **entrada** (prompt + histórico + resultados das
ferramentas) e de **saída** (resposta). A saída costuma ser mais cara. Um system
prompt longo é pago **em toda chamada**.

**O bug que pegamos (ótimo para contar em entrevista).** A primeira versão usava
`handle_tool_error=True`, o jeito "óbvio" do LangChain de tratar erros de
ferramenta. Lendo o código-fonte do `langchain-core`, descobrimos que esse
caminho registra a falha como **sucesso** no trace (`on_tool_end` em vez de
`on_tool_error`). O gráfico de erros por ferramenta ficaria **sempre zerado**,
uma métrica que mente. A correção foi o `ToolErrorMiddleware`, e hoje há um teste
que impede a regressão (`test_falha_de_tool_aparece_como_erro_no_trace...`).

Lição: **valide que sua métrica consegue mostrar o problema** antes de confiar
nela. Um gráfico sempre verde pode estar quebrado.

**Faça você mesmo.**

```bash
uv run obs-simulate --n 200 --duracao-min 20 --incidente
```

Abra Projects → `agente-suporte-prod` → **Monitoring** e encontre o incidente nos
gráficos de latência e erro.

**Pergunta de entrevista.** *"Que métricas você monitoraria num agente em
produção?"* Latência p50/p99 por trace e por ferramenta, taxa de erro separando
LLM e ferramentas, custo por conversa e por segmento de cliente, e um sinal de
qualidade: feedback explícito (👍/👎) e implícito (escalou para humano?).

---

## Etapa 7: Dashboards, A/B e alertas

**Conceito.**

- **Dashboard:** gráficos fixos que você olha todo dia. Veja a tabela de
  gráficos no [README](README.md#montando-os-dashboards-no-langsmith).
- **Teste A/B:** metade dos clientes recebe o prompt v1 e metade o v2. Comparamos
  latência, custo e satisfação **em produção, com usuários reais**.
- **A/B fixo por sessão:** o mesmo cliente sempre cai na mesma versão (um hash
  do id da sessão decide). Senão, o agente mudaria de personalidade no meio da
  conversa. Veja `variante()` em [`api.py`](src/agent_observability/api.py).
- **Alerta:** uma regra que te avisa sozinha. Ex.: "taxa de erro > 10% em 5
  minutos". Você não fica olhando o dashboard o dia todo.

**Faça você mesmo.** Monte no LangSmith o gráfico "Satisfação A/B": métrica
feedback `user_score`, agrupado por metadata `prompt_version`. Qual versão
ganha? Configure um alerta de taxa de erro e dispare-o com `--incidente`.

**Pergunta de entrevista.** *"Como você decide se um prompt novo é melhor?"* Em
duas frentes. **Offline**, antes de ir para produção: um dataset fixo de casos
com notas automáticas (etapa 8). **Online**, depois: A/B com usuários reais,
olhando satisfação, custo e latência. Uma frente não substitui a outra.

---

# ATO 3: Avaliar

## Etapa 8: Avaliação offline (datasets, avaliadores, experimentos)

**Conceito.** Antes de colocar uma mudança em produção, você a testa contra um
**dataset golden**: casos com a resposta esperada.

- **Dataset:** 10 perguntas, cada uma com a(s) ferramenta(s) esperada(s) e
  trechos que a resposta precisa conter.
- **Avaliador:** uma função que dá nota a uma resposta. Aqui todos são
  determinísticos (código simples):
  - `tool_correta`: usou exatamente as ferramentas certas?
  - `conteudo_esperado`: a resposta contém os trechos obrigatórios?
  - `concisao`: tem até 280 caracteres?
  - `sem_erro`: completou sem falha?
- **Experimento:** uma rodada do dataset inteiro com uma configuração (ex.:
  prompt v2). O LangSmith mostra experimentos lado a lado (**Compare**).

Existe também o **LLM-as-a-judge**: um LLM dá nota a outro LLM. É útil para
critérios subjetivos ("a resposta foi educada?"), mas custa caro e tem vieses
próprios. Prefira avaliadores em código sempre que o critério for objetivo.

**Onde está no código.** [`evaluate.py`](src/agent_observability/evaluate.py).

**Faça você mesmo.**

```bash
uv run obs-evaluate
```

No LangSmith: Datasets & Experiments → `suporte-golden` → selecione dois
experimentos → Compare. Clique numa linha em que eles discordam.

**A lição do 429.** A primeira avaliação com o Gemini mostrou a v2 **pior** que a
v1. Antes de acreditar, olhamos os erros dos runs: eram **429 (cota diária
esgotada)**. A v2 rodou depois, então pegou mais bloqueios. O resultado foi
descartado em vez de publicado. Lição: **antes de concluir algo de uma métrica,
confira se a diferença vem da coisa que você mudou.**

**Pergunta de entrevista.** *"Como você evita que uma melhoria quebre outra
coisa?"* Com um dataset golden de regressão rodado a cada mudança, comparando os
experimentos lado a lado no LangSmith.

---

## Etapa 9: pass^k: confiabilidade (um conceito de pesquisa)

**Conceito.** LLMs não são determinísticos: a mesma pergunta pode dar respostas
diferentes. Um agente que acerta 80% das vezes parece bom, mas **o mesmo cliente
volta e pergunta de novo**.

O artigo **τ-bench** (Yao et al., 2024, arXiv:2406.12045) propôs a métrica
**pass^k** (lê-se "pass hat k"): a probabilidade de o agente acertar a mesma
tarefa em **todas** as k tentativas.

Compare com a métrica clássica **pass@k**:

- **pass@k:** acerta **pelo menos uma vez** em k tentativas? Mede *capacidade*.
- **pass^k:** acerta **todas as vezes** em k tentativas? Mede *confiabilidade*.

**A conta.** Para uma tarefa rodada n vezes com c acertos:

```
pass^k = C(c, k) / C(n, k)
```

C(a, b) é "combinação de a escolhe b": de quantas formas escolher b itens entre a.
Essa fração é a chance de, sorteando k das n tentativas, todas serem acertos.
Depois tiramos a média entre as tarefas.

**Exemplo à mão** (está no teste `test_pass_hat_k_e_mais_rigoroso_que_pass_1`):
3 tarefas, 3 tentativas cada, com 3, 0 e 2 acertos.

- pass^1 = (3/3 + 0/3 + 2/3) / 3 = **0,56**
- pass^3 = (1 + 0 + 0) / 3 = **0,33**: só a tarefa que acertou sempre conta.

Se o pass^1 é alto mas o pass^3 é baixo, o agente é **inconsistente**: sabe
fazer, mas não dá para confiar.

**Onde está no código.** `pass_hat_k` e `confiabilidade` em
[`evaluate.py`](src/agent_observability/evaluate.py); `--k` define as repetições.

**Faça você mesmo.**

```bash
uv run obs-evaluate --k 5 --prompts v2
```

Veja `pass^1` até `pass^5` no resumo do experimento.

**Pergunta de entrevista.** *"Você acompanha pesquisa recente? Deu para aplicar
alguma?"* Sim: o pass^k do τ-bench. Implementei o estimador não viesado como
avaliador de resumo no LangSmith, porque em atendimento a consistência importa
mais que acertar uma vez.

---

# ATO 4: Melhorar com fine-tuning

## Etapa 10: O que é fine-tuning e quando vale a pena

**Conceito.** Há três formas de mudar o comportamento de um LLM, da mais barata
para a mais cara:

1. **Prompt:** mudar as instruções (o que fizemos de v1 para v2).
2. **RAG / ferramentas:** dar informação nova ao modelo na hora da pergunta.
3. **Fine-tuning:** **treinar** o modelo com exemplos, mudando os pesos dele.

Fine-tuning ensina **comportamento e formato** (ex.: "sempre chame a ferramenta
certa, responda em 3 frases"), não conhecimento que muda (para isso usamos as
ferramentas). Neste projeto, o objetivo é a **destilação**: fazer um modelo
**pequeno e local** (0,5 bilhão de parâmetros, roda no seu notebook) imitar o bom
comportamento do agente de produção. Isso dá custo zero por chamada, mais
privacidade e menos latência de rede.

**SFT** (*Supervised Fine-Tuning*) é o tipo usado aqui: mostramos exemplos de
conversa "ideal" e o modelo aprende a reproduzi-los.

**Pergunta de entrevista.** *"Quando você faria fine-tuning em vez de melhorar o
prompt?"* Quando o prompt já está no limite e preciso de comportamento
consistente num modelo menor ou mais barato, ou quando o formato precisa ser
muito rígido. Primeiro esgoto prompt e ferramentas, que são mais baratos de mudar.

---

## Etapa 11: O data flywheel: da produção ao dataset de treino

**Conceito.** *Flywheel* (volante) é o ciclo que se retroalimenta: o agente roda
em produção → gera traces com feedback → os melhores viram dados de treino → o
modelo melhora → gera traces melhores → ...

A parte difícil não é treinar: é **escolher bons dados** (curadoria). Cada
critério é um filtro com contagem de descartes, para que a curadoria seja
auditável:

| Filtro | Por quê |
| --- | --- |
| sem erro e sem falha de ferramenta | não ensinar o modelo a lidar mal com falhas |
| sem 👎 do usuário | o cliente disse que a resposta foi ruim |
| só prompt v2 | imitar a variante vencedora do A/B |
| **fora as perguntas do golden** | ver "vazamento" abaixo |
| deduplicado | 100 cópias da mesma pergunta não ensinam nada novo |

**Vazamento de dados (data leakage), conceito crítico.** Se uma pergunta do
dataset de avaliação também estiver no treino, o modelo vai **decorar** a
resposta e a avaliação vai parecer ótima sem o modelo ter aprendido nada. É como
dar a prova de véspera para o aluno. Por isso as perguntas do golden são
removidas do treino (39 foram descartadas por esse motivo na nossa rodada).

**Rastreabilidade.** Cada exemplo guarda o `run_id` do trace de origem. Se o
modelo aprender algo errado, você volta ao trace que o ensinou.

**Onde está no código.** [`flywheel.py`](src/agent_observability/flywheel.py).

**Faça você mesmo.**

```bash
uv run obs-export-ft --dias 1
```

Leia a contagem de descartes. Abra `data/sft_train.jsonl` e veja um exemplo.

**Pergunta de entrevista.** *"Como você garante que a avaliação do modelo
fine-tunado é honesta?"* Removo do treino tudo que está no dataset de avaliação,
e separo um conjunto de validação que o modelo não vê durante o treino.

---

## Etapa 12: LoRA e o treino na sua GPU

**Conceito.** Treinar **todos** os pesos de um modelo exige muita memória. O
**LoRA** (*Low-Rank Adaptation*, Hu et al., 2021) congela o modelo original e
treina só pequenas matrizes "adaptadoras" acopladas a ele: cerca de **1% dos
parâmetros**. Resultados:

- cabe numa GPU de notebook (sua RTX 4050 de 6 GB);
- o resultado é um arquivo pequeno (o "adaptador", alguns MB), aplicado por
  cima do modelo base;
- você pode ter vários adaptadores para um único modelo base.

O `rank` (16 aqui) controla o tamanho das matrizes: maior é mais expressivo e
mais pesado.

**Chat template.** Cada modelo espera as mensagens num formato de texto próprio.
O Qwen usa marcadores assim:

```
<|im_start|>user
Cadê meu pedido 1002?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "buscar_pedido", "arguments": {"pedido_id": "1002"}}
</tool_call><|im_end|>
```

O `tokenizer.apply_chat_template` converte a lista de mensagens nesse texto.
Treino e uso **precisam** usar o mesmo template.

**Máscara de loss: o detalhe que separa quem entende de quem copiou tutorial.**
A *loss* é o "erro" que o treino tenta diminuir. Se calcularmos a loss no texto
inteiro, o modelo também aprende a **gerar o resultado das ferramentas**, ou
seja, a inventar retornos de API, que é exatamente o que não queremos. Então
marcamos com `-100` (o valor que o PyTorch ignora) tudo que **não** é fala do
assistente. Verificamos isso no exemplo real: só a tool call e a resposta final
ficam com loss.

**Onde está no código.** [`finetune/train_lora.py`](finetune/train_lora.py),
função `tokenizar`.

**Termos que aparecem no log do treino:**

- **época:** uma passada completa pelos dados de treino;
- **learning rate (lr):** o tamanho do passo de cada ajuste;
- **train loss / eval loss:** erro no treino e na validação. Se o de treino cai e
  o de validação sobe, é **overfitting**: o modelo está decorando.
- **gradient accumulation:** soma vários lotes pequenos antes de ajustar, para
  simular um lote maior sem estourar a memória.

**Faça você mesmo.**

```bash
uv run python finetune/train_lora.py
```

Depois abra `finetune/adapters/qwen-suporte-lora/historico_treino.json` e
compare a loss do início com a do fim.

**Pergunta de entrevista.** *"Por que LoRA?"* Porque treina ~1% dos parâmetros,
cabe em hardware modesto, e o adaptador é pequeno e trocável. *"Onde você calcula
a loss?"* Só nos turnos do assistente, senão o modelo aprende a alucinar
resultados de ferramenta.

---

## Etapa 13: Fechando o ciclo: avaliar o modelo treinado

**Conceito.** Um modelo treinado só vale se for medido **no mesmo teste** que os
outros. O [`local_llm.py`](src/agent_observability/local_llm.py) implementa o
mesmo contrato `BaseChatModel` do mock e do Gemini (lembra da etapa 3?). Então
o modelo local entra no mesmo agente, na mesma avaliação e nos mesmos
dashboards, sem mudar nada no resto.

Comparamos três experimentos no LangSmith:

1. **Qwen base** (sem treino): a linha de base.
2. **Qwen + LoRA**: o modelo treinado.
3. O agente de referência.

A pergunta é: o LoRA fechou a distância entre o modelo pequeno e o de referência?

**O que aconteceu na nossa rodada** (detalhes no [README](README.md#resultados-rtx-4050-laptop-6-gb)):

| | Qwen base | Qwen + LoRA |
| --- | --- | --- |
| escolheu a ferramenta certa | 40% | **90%** |
| pass^1 | 0,27 | **0,70** |

Três coisas para você saber explicar sobre esses números:

1. **A loss de validação foi 0,004, quase zero. Isso não prova nada.** O
   "professor" (mock) responde sempre com as mesmas frases, então é fácil
   imitá-lo. A prova é o golden, cujas perguntas o modelo nunca viu.
2. **pass^1 ≈ pass^3** porque a geração é *greedy* (sempre escolhe a palavra mais
   provável), então o modelo responde igual nas 3 tentativas. Com amostragem
   (`do_sample=True`) as repetições variariam e o pass^k mostraria a
   inconsistência.
3. **O LoRA ficou mais lento (8,7 s contra 4,0 s)** porque o adaptador é
   calculado à parte em cada passo. `merge_and_unload()` funde o adaptador nos
   pesos antes de servir e elimina esse custo.

**Detalhe técnico do parser.** Modelos pequenos às vezes geram JSON quebrado
dentro de `<tool_call>`. O `extrair_tool_calls` descarta o bloco inválido em vez
de travar. Isso também aparece na nota `tool_correta`.

**Uma ressalva honesta para ter na ponta da língua.** Em modo mock, o "professor"
que gerou os dados é o LLM simulado. O modelo aprende o comportamento do mock,
não o do Gemini. O **pipeline** é o que está demonstrado; com o Gemini com
faturamento ativo, os mesmos comandos destilam o Gemini de verdade. Saber
apontar a limitação do próprio projeto conta pontos em entrevista.

**Faça você mesmo.**

```bash
AGENT_MODE=local LOCAL_MODEL=Qwen/Qwen2.5-0.5B-Instruct uv run obs-evaluate --prompts v2 --concorrencia 1
```

```bash
AGENT_MODE=local LOCAL_MODEL=finetune/adapters/qwen-suporte-lora uv run obs-evaluate --prompts v2 --concorrencia 1
```

Compare os dois experimentos no LangSmith.

---

# Engenharia de software

## Etapa 14: A API com FastAPI

**Conceito.** Ninguém usa um agente pelo terminal: ele fica atrás de uma **API
HTTP** que o site, o app ou o WhatsApp chamam.

- **FastAPI:** framework web Python. Você declara funções e ele cria as rotas, a
  validação e a documentação automática (`/docs`).
- **Pydantic** (`ChatIn`, `ChatOut`, `FeedbackIn`): define o formato dos dados.
  Se o cliente mandar `score: 5` quando só vale 0 ou 1, a API responde **422**
  sozinha, sem você escrever um `if`.
- **async e threadpool:** o FastAPI é assíncrono (atende muitos pedidos ao mesmo
  tempo num único processo). Mas o nosso agente é **síncrono** (bloqueia
  esperando o LLM). Se rodasse direto, travaria todos os outros pedidos. Por isso
  `run_in_threadpool` o executa numa thread separada.
- **SSE (Server-Sent Events):** em `/v1/chat/stream`, o servidor envia cada passo
  conforme acontece ("chamando buscar_pedido...", "pronto", "resposta"). É assim
  que os chats mostram o texto aparecendo aos poucos.
- **Códigos HTTP:** 200 (ok), 202 (aceito, processa depois), 404 (não existe), 422
  (dados inválidos), 503 (serviço indisponível, aqui quando o LLM cai, devolvendo
  o `run_id` para depuração).

**Faça você mesmo.**

```bash
uv run obs-api
```

Abra http://127.0.0.1:8000/docs, clique em `POST /v1/chat` → *Try it out* e
mande uma pergunta. Copie o `run_id` e use em `POST /v1/feedback`.

**Pergunta de entrevista.** *"Por que `run_in_threadpool`?"* Porque código
bloqueante dentro de uma rota `async` trava o event loop e, com ele, todas as
requisições. *"O que é SSE e quando usar em vez de WebSocket?"* SSE é
unidirecional (servidor → cliente) sobre HTTP comum, que é exatamente o caso do
streaming de resposta. WebSocket é bidirecional e mais complexo.

---

## Etapa 15: Testes automatizados

**Conceito.** Os testes ([`tests/`](tests)) rodam em modo mock, sem rede, em
poucos segundos. Três ideias valem aprender:

- **Fixture** (`sem_caos` em `conftest.py`): prepara o ambiente antes de cada
  teste, aqui desligando falhas aleatórias para o teste ser determinístico.
- **Teste de regressão:** trava um bug já corrigido para ele não voltar (o do
  `handle_tool_error`).
- **TestClient do FastAPI:** testa a API sem subir servidor.

**Faça você mesmo.**

```bash
uv run pytest -v
```

Depois quebre algo de propósito (troque o `ToolErrorMiddleware` de volta por
`handle_tool_error=True`) e veja o teste de regressão falhar.

---

# Como apresentar o projeto (roteiro de 2 minutos)

> "Construí um agente de atendimento com LangChain e o instrumentei para
> produção com LangSmith: cada conversa vira um trace com metadados de versão de
> prompt, plano e canal, e o feedback do usuário é ligado ao trace pelo run_id.
> Montei dashboards de latência p99, erro por ferramenta, custo por segmento e
> satisfação por variante de A/B, e simulei um incidente para mostrar como ele
> aparece.
>
> Na avaliação, além dos avaliadores clássicos, implementei o pass^k do τ-bench,
> que mede consistência e não só acerto.
>
> Depois fechei o ciclo: os traces bons de produção, com curadoria auditável e
> sem vazamento do conjunto de teste, viraram dataset para um fine-tuning LoRA de
> um Qwen de 0,5B na minha GPU, com loss só nos turnos do assistente. O modelo
> treinado entra no mesmo harness e é comparado no LangSmith: o acerto de
> ferramenta subiu de 40% para 90% em perguntas fora do treino.
>
> Tudo isso fica atrás de uma API FastAPI com streaming SSE e A/B fixo por
> sessão, com testes. E desenvolvi em par com o Claude Code. Um exemplo do que
> isso pegou: o tratamento de erro padrão do LangChain registrava falha de
> ferramenta como sucesso, e o gráfico de erros ficaria sempre zerado."

---

# Glossário rápido

| Termo | Em uma frase |
| --- | --- |
| LLM | modelo que recebe e gera texto |
| Agente | LLM que decide usar ferramentas em ciclo |
| Tool calling | o LLM pede uma ferramenta em formato estruturado |
| Token | pedaço de texto (~4 caracteres); unidade de cobrança |
| Trace / run | árvore de passos de uma pergunta / um passo dela |
| Metadata / tag | dimensão para agrupar / rótulo para filtrar |
| p50 / p95 / p99 | latência típica / da cauda / do 1% pior |
| A/B test | duas versões em paralelo com usuários reais |
| Dataset golden | casos fixos com a resposta esperada |
| Avaliador | função que dá nota a uma resposta |
| LLM-as-judge | um LLM dando nota a outro |
| pass^k | acertar a mesma tarefa em todas as k tentativas |
| SFT | fine-tuning supervisionado com exemplos ideais |
| LoRA | treina ~1% dos pesos via matrizes pequenas |
| Destilação | modelo pequeno aprende a imitar um maior |
| Data leakage | dado de teste vazou para o treino |
| Overfitting | o modelo decorou em vez de aprender |
| Chat template | formato de texto que cada modelo espera |
| Máscara de loss | só calcular o erro no que o modelo deve gerar |
| SSE | servidor envia eventos em sequência via HTTP |
| Mock | implementação falsa com a mesma interface |

# Para ir além (leituras)

- Yao et al., 2024: *τ-bench: A Benchmark for Tool-Agent-User Interaction in
  Real-World Domains*, arXiv:2406.12045 (pass^k).
- Hu et al., 2021: *LoRA: Low-Rank Adaptation of Large Language Models*,
  arXiv:2106.09685.
- Yao et al., 2022: *ReAct: Synergizing Reasoning and Acting in Language
  Models*, arXiv:2210.03629 (o ciclo do agente).
- Zheng et al., 2023: *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*,
  arXiv:2306.05685 (vieses de LLM como avaliador).
- Documentação do LangSmith: observabilidade, avaliação e dashboards.
