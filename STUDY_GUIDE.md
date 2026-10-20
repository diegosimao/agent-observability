# Study guide: what this project teaches, step by step

This guide is for **understanding** the project well enough to explain it in an
interview, not just to run it. Each step has four parts:

- **Concept:** what it is, in plain words.
- **Where it lives in the code:** the file to open and read.
- **Try it:** a short command or experiment.
- **Interview question:** what someone might ask you, and how to answer.

Go in order; each step builds on the previous one. Budget 1–2 hours per step and
actually run the commands.

---

## The big picture (read this first)

The project tells a story in four acts:

1. **Build** an AI agent that answers a store's customers (steps 1–3).
2. **Observe** that agent in production: speed, errors, cost, satisfaction (steps 4–7).
3. **Evaluate** whether a change made the agent better or worse (steps 8–9).
4. **Improve** the agent with its own production data, through fine-tuning (steps 10–13).

On the side, the **engineering**: the web API (step 14) and the tests (step 15).

```
customer ──► FastAPI API ──► agent (LangChain) ──► LLM + tools
                                │
                                └── every step becomes a "trace" ──► LangSmith
                                                                      │
                     dashboards ◄── metrics ◄─────────────────────────┤
                     experiments ◄── evaluation ◄─────────────────────┤
                     fine-tuning ◄── curated data ◄───────────────────┘
```

---

# ACT 1: Build the agent

## Step 1: What an LLM is, and what an "agent" is

**Concept.** An **LLM** (Large Language Model, like Gemini, GPT or Claude) takes text
in and returns text. On its own it doesn't know the status of *your* order: it has
no access to your database.

An **agent** is an LLM that can **use tools**. The cycle is:

1. The user asks: "where is my order 1002?"
2. The LLM decides: "I need the `get_order` tool with `order_id=1002`".
3. **Your code** runs the tool (the LLM never executes anything itself).
4. The result goes back to the LLM: `{"status": "in transit"}`.
5. The LLM writes the final answer: "Your order is in transit."

This cycle is called **ReAct** (*Reason + Act*). It can repeat several times for a
single question.

**Where it lives.** [`agent.py`](src/agent_observability/agent.py), function
`build_agent`. The line `create_agent(model, TOOLS, system_prompt=...)` builds the
whole cycle.

**Try it.**

```bash
uv run obs-ask "What's the status of order 1002 and shipping to 30140-071?"
```

Look at the line `tools: ['get_order', 'calculate_shipping']`: the agent decided on
its own to use two tools.

**Interview question.** *"What's the difference between a chatbot and an agent?"*
A chatbot only generates text. An agent decides on **actions** (calling tools) and
uses their results. The LLM only picks the action; the code executes it, which is
where control and safety live.

---

## Step 2: Tools and tool calling

**Concept.** A tool is a regular Python function with a **description**. The
description (the docstring) is what the LLM reads to decide when to use it, so it
has to be clear: it's practically a prompt.

When the LLM wants a tool, it doesn't return free text but a structured object
called a **tool call**:

```json
{"name": "get_order", "args": {"order_id": "1002"}, "id": "call_abc123"}
```

The `id` matches the call to its response (`ToolMessage`), which matters when the
LLM calls several tools at once.

**Where it lives.** [`tools.py`](src/agent_observability/tools.py). Look at the
`@tool` decorator and the docstrings. Notice that the tools have **latency and
failures on purpose** (`_latency`, `ToolException`): they simulate real systems,
which are slow and flaky.

**Try it.** Change the `calculate_shipping` docstring to something vague like
"does a calculation" and run against Gemini. The agent starts getting the tool
wrong. Then undo the change.

**Interview question.** *"What happens when a tool fails?"* Here, an **expected**
failure (e.g. order not found) goes back to the LLM as an error message, and it
apologizes to the customer. An **unexpected** failure fails the run. See
`tool_error_to_model` and step 6.

---

## Step 3: Messages, the system prompt and the simulated LLM (mock)

**Concept.** A conversation with an LLM is a list of **messages** with roles:

| Role | LangChain class | Example |
| --- | --- | --- |
| system | `SystemMessage` | "You are the Vertice store assistant. Rules: ..." |
| user | `HumanMessage` | "Where is my order?" |
| assistant | `AIMessage` | text **or** tool calls |
| tool | `ToolMessage` | the tool's result |

The **system prompt** defines behavior. This project has **two versions** (`v1`
and `v2` in [`config.py`](src/agent_observability/config.py)) so we can compare
which is better (step 7).

**The mock.** Calling a real LLM costs money and has usage limits (we did hit the
free Gemini daily limit!). So the project has a **fake LLM**
([`mock_llm.py`](src/agent_observability/mock_llm.py)) that:

- picks tools by keyword (it isn't smart, and doesn't need to be);
- reports how many **tokens** it "spent" (`usage_metadata`);
- presents itself to LangSmith as `gemini-3.8-flash` (`_get_ls_params`), so
  LangSmith computes cost as if it were real.

The engineering lesson: **the mock respects the contract** (the `BaseChatModel`
interface), not the intelligence. Anything that follows that contract works in the
rest of the system. Mock, Gemini and the local model from step 12 are
interchangeable.

**Try it.**

```bash
AGENT_MODE=mock uv run obs-ask "How do returns work?" --prompt v1
```

```bash
AGENT_MODE=mock uv run obs-ask "How do returns work?" --prompt v2
```

Compare the length of the answers: v2 has the rule "at most 3 sentences".

**Interview question.** *"How do you test an LLM system without paying for the
API?"* With a fake model that implements the same interface. Tests become fast,
free and deterministic. The real LLM is for evaluation.

---

# ACT 2: Observe in production

## Step 4: Observability, traces and runs

**Concept.** In regular software you look at logs. With agents, logs aren't
enough: a single question triggers several LLM and tool calls, in sequence. You
need to see the whole **tree**.

- **Run:** one step (an LLM call, a tool call).
- **Trace:** the tree of runs for one question, from input to answer.

```
support-chat (root trace, 1.8s)
├── model (LLM decides to call tools)       0.6s  420 tokens
├── get_order                               0.2s
├── calculate_shipping                      0.5s  ← error: timeout
└── model (LLM writes the answer)           0.5s  380 tokens
```

**LangSmith** is LangChain's tool that receives, stores and displays these
traces. Set `LANGSMITH_TRACING=true` and the API key, and LangChain sends
everything automatically in the background, without slowing the answer down.

**Where it lives.** Sending is automatic. The only care point is at the end of the
simulator: `wait_for_all_tracers()` waits for the traces to finish uploading
before the program exits.

**Try it.** Run the `obs-ask` above and open smith.langchain.com → Projects →
`support-agent-prod`. Click the most recent trace and expand each step. Look at
inputs, outputs, tokens and timings.

**Interview question.** *"Why not just use logs?"* Because an agent's problem is
usually **in the middle** of the chain (a slow tool, a prompt that made the LLM
pick the wrong tool). A trace shows cause and effect linked together; a log shows
loose lines.

---

## Step 5: Tags, metadata and run_id (the instrumentation)

**Concept.** A trace without context is hard to analyze. Along with every call we
send three things:

- **`metadata`:** dimensions to **group** charts by. E.g. `prompt_version`,
  `customer_plan`, `channel`, `app_version`. That's how you answer "is latency on
  WhatsApp worse than on the web?".
- **`tags`:** labels for quick **filtering**. E.g. `prompt:v2`, `simulation`.
- **`run_id`:** an id **we** generate before calling the agent. Why? Because later
  the user clicks 👍 or 👎 and we need to know which trace that vote belongs to.

Rule of thumb: dimensions with few possible values (low **cardinality**) are for
grouping. Never put an email, free text or a user id in a tag: it becomes millions
of values and the charts become useless.

**Where it lives.** [`agent.py`](src/agent_observability/agent.py), function
`_config`. Read the docstring at the top of the file.

**Try it.** In LangSmith, filter traces by `metadata.channel = whatsapp`. Then by
the tag `prompt:v1`.

**Interview question.** *"How do you tie user feedback to the trace?"* I generate
the `run_id` before invoking the agent, return that id in the API response, and
the front end sends it back with the 👍/👎 (`POST /v1/feedback`).

---

## Step 6: The metrics that matter

**Concept.** The four production questions and the metrics that answer each one:

| Question | Metric | How to read it |
| --- | --- | --- |
| Is it fast? | **p50 / p95 / p99** latency | see below |
| Is it breaking? | **error rate** (LLM and tools, separately) | % of runs with an error |
| What does it cost? | **tokens** and **cost** per conversation | tokens × model price |
| Is it helping? | user **feedback** and **containment** | % 👍, % resolved without a human |

**Percentiles (p50, p95, p99), slowly, because they come up in interviews.** Sort
all latencies from fastest to slowest.

- **p50 (median):** half the conversations were faster than this. The "typical"
  experience.
- **p95:** 95% were faster; 5% were worse. It shows the tail.
- **p99:** the worst 1%.

Why not the **average**? Because a single 60 s conversation hides inside the
average, yet it's exactly the customer who complains. In production, **the tail
(p95/p99) is what matters**. In our simulated incident, p50 barely moves while p95
triples.

**Tokens and cost.** LLMs charge per **token** (a piece of a word, ~4 characters
in English). There are **input** tokens (prompt + history + tool results) and
**output** tokens (the answer). Output is usually more expensive. A long system
prompt is paid for **on every call**.

**The bug we caught (great to tell in an interview).** The first version used
`handle_tool_error=True`, LangChain's "obvious" way to handle tool errors. Reading
the `langchain-core` source, we found that this path records the failure as a
**success** in the trace (`on_tool_end` instead of `on_tool_error`). The
errors-per-tool chart would **always read zero**, a metric that lies. The fix was
`ToolErrorMiddleware`, and a test now prevents the regression
(`test_tool_failure_is_traced_as_error_and_conversation_recovers`).

Lesson: **make sure your metric can show the problem** before you trust it. An
always-green chart may just be broken.

**Try it.**

```bash
uv run obs-simulate --n 200 --duration-min 20 --incident
```

Open Projects → `support-agent-prod` → **Monitoring** and find the incident in the
latency and error charts.

**Interview question.** *"Which metrics would you monitor on an agent in
production?"* p50/p99 latency per trace and per tool, error rate split between
LLM and tools, cost per conversation and per customer segment, and a quality
signal: explicit feedback (👍/👎) and implicit feedback (did it escalate to a
human?).

---

## Step 7: Dashboards, A/B and alerts

**Concept.**

- **Dashboard:** fixed charts you look at every day. See the chart table in the
  [README](README.md#building-the-langsmith-dashboards).
- **A/B test:** half the customers get prompt v1 and half get v2. We compare
  latency, cost and satisfaction **in production, with real users**.
- **Sticky A/B per session:** the same customer always lands on the same version
  (a hash of the session id decides). Otherwise the agent would change personality
  mid-conversation. See `variant()` in [`api.py`](src/agent_observability/api.py).
- **Alert:** a rule that notifies you on its own. E.g. "error rate > 10% over 5
  minutes". You don't stare at the dashboard all day.

**Try it.** Build the "A/B satisfaction" chart in LangSmith: metric feedback
`user_score`, grouped by metadata `prompt_version`. Which version wins? Set up an
error-rate alert and trigger it with `--incident`.

**Interview question.** *"How do you decide whether a new prompt is better?"* On
two fronts. **Offline**, before production: a fixed dataset of cases with automatic
scores (step 8). **Online**, afterwards: an A/B with real users, looking at
satisfaction, cost and latency. Neither replaces the other.

---

# ACT 3: Evaluate

## Step 8: Offline evaluation (datasets, evaluators, experiments)

**Concept.** Before shipping a change, you test it against a **golden dataset**:
cases with the expected answer.

- **Dataset:** 10 questions, each with the expected tool(s) and snippets the answer
  must contain.
- **Evaluator:** a function that scores an answer. Here they're all deterministic
  (plain code):
  - `correct_tools`: did it use exactly the right tools?
  - `expected_content`: does the answer contain the required snippets?
  - `concise`: is it at most 280 characters?
  - `no_error`: did it complete without failing?
- **Experiment:** one run of the whole dataset with one configuration (e.g. prompt
  v2). LangSmith shows experiments side by side (**Compare**).

There's also **LLM-as-a-judge**: one LLM scores another. It's useful for subjective
criteria ("was the answer polite?"), but it's expensive and has its own biases.
Prefer code evaluators whenever the criterion is objective.

**Where it lives.** [`evaluate.py`](src/agent_observability/evaluate.py).

**Try it.**

```bash
uv run obs-evaluate
```

In LangSmith: Datasets & Experiments → `support-golden` → select two experiments →
Compare. Click a row where they disagree.

**The 429 lesson.** The first evaluation on Gemini showed v2 **worse** than v1.
Before believing it, we looked at the run errors: they were **429s (daily quota
exhausted)**. v2 ran second, so it got throttled more. The result was discarded
instead of published. Lesson: **before concluding anything from a metric, check
that the difference comes from the thing you changed.**

**Interview question.** *"How do you keep one improvement from breaking something
else?"* With a golden regression dataset run on every change, comparing
experiments side by side in LangSmith.

---

## Step 9: pass^k: reliability (a research concept)

**Concept.** LLMs aren't deterministic: the same question can get different
answers. An agent that's right 80% of the time looks good, but **the same customer
comes back and asks again**.

The **τ-bench** paper (Yao et al., 2024, arXiv:2406.12045) proposed the **pass^k**
metric ("pass hat k"): the probability that the agent solves the same task in
**all** k attempts.

Compare it with the classic **pass@k**:

- **pass@k:** right **at least once** in k attempts? Measures *capability*.
- **pass^k:** right **every time** in k attempts? Measures *reliability*.

**The math.** For a task run n times with c successes:

```
pass^k = C(c, k) / C(n, k)
```

C(a, b) is "a choose b": the number of ways to pick b items out of a. The fraction
is the chance that, drawing k of the n attempts, all of them are successes. Then
we average across tasks.

**Worked example** (it's the test `test_pass_hat_k_is_stricter_than_pass_1`): 3
tasks, 3 attempts each, with 3, 0 and 2 successes.

- pass^1 = (3/3 + 0/3 + 2/3) / 3 = **0.56**
- pass^3 = (1 + 0 + 0) / 3 = **0.33**: only the task that always succeeded counts.

If pass^1 is high but pass^3 is low, the agent is **inconsistent**: it knows how,
but you can't rely on it.

**Where it lives.** `pass_hat_k` and `reliability` in
[`evaluate.py`](src/agent_observability/evaluate.py); `--k` sets the repetitions.

**Try it.**

```bash
uv run obs-evaluate --k 5 --prompts v2
```

Look at `pass^1` through `pass^5` in the experiment summary.

**Interview question.** *"Do you follow recent research? Have you applied any of
it?"* Yes: τ-bench's pass^k. I implemented the unbiased estimator as a LangSmith
summary evaluator, because in customer support consistency matters more than
getting it right once.

---

# ACT 4: Improve with fine-tuning

## Step 10: What fine-tuning is, and when it's worth it

**Concept.** There are three ways to change an LLM's behavior, from cheapest to
most expensive:

1. **Prompt:** change the instructions (what we did from v1 to v2).
2. **RAG / tools:** give the model new information at question time.
3. **Fine-tuning:** **train** the model on examples, changing its weights.

Fine-tuning teaches **behavior and format** (e.g. "always call the right tool,
answer in 3 sentences"), not knowledge that changes (tools handle that). In this
project the goal is **distillation**: making a **small, local** model (0.5 billion
parameters, runs on your laptop) imitate the production agent's good behavior.
That means zero cost per call, more privacy and no network latency.

**SFT** (*Supervised Fine-Tuning*) is the kind used here: we show examples of
"ideal" conversations and the model learns to reproduce them.

**Interview question.** *"When would you fine-tune instead of improving the
prompt?"* When the prompt is already maxed out and I need consistent behavior on
a smaller or cheaper model, or when the format has to be very strict. I exhaust
prompting and tools first, because they're cheaper to change.

---

## Step 11: The data flywheel: from production to a training dataset

**Concept.** A *flywheel* is a self-reinforcing cycle: the agent runs in production
→ produces traces with feedback → the best ones become training data → the model
improves → produces better traces → ...

The hard part isn't training; it's **choosing good data** (curation). Every
criterion is a filter with a discard count, so the curation is auditable:

| Filter | Why |
| --- | --- |
| no error and no tool failure | don't teach the model to handle failures poorly |
| no user 👎 | the customer said the answer was bad |
| only prompt v2 | imitate the variant that won the A/B |
| **golden questions excluded** | see "leakage" below |
| deduplicated | 100 copies of the same question teach nothing new |

**Data leakage, a critical concept.** If an evaluation question also shows up in
training, the model **memorizes** the answer and the evaluation looks great without
the model having learned anything. It's like handing the student the exam the
night before. That's why golden questions are removed from training.

**Traceability.** Every example keeps the `run_id` of its source trace. If the
model learns something wrong, you can go back to the trace that taught it.

**Where it lives.** [`flywheel.py`](src/agent_observability/flywheel.py).

**Try it.**

```bash
uv run obs-export-ft --days 1
```

Read the discard counts. Open `data/sft_train.jsonl` and look at an example.

**Interview question.** *"How do you make sure the fine-tuned model's evaluation is
honest?"* I remove everything in the evaluation dataset from training, and hold out
a validation set the model never sees during training.

---

## Step 12: LoRA and training on your GPU

**Concept.** Training **all** of a model's weights takes a lot of memory. **LoRA**
(*Low-Rank Adaptation*, Hu et al., 2021) freezes the original model and trains only
small "adapter" matrices attached to it: about **1–2% of the parameters**. As a
result:

- it fits on a laptop GPU (an RTX 4050 with 6 GB here);
- the result is a small file (the "adapter", a few dozen MB), applied on top of
  the base model;
- you can keep several adapters for a single base model.

`rank` (16 here) controls the size of the matrices: larger is more expressive and
heavier.

**Chat template.** Each model expects messages in its own text format. Qwen uses
markers like these:

```
<|im_start|>user
Where is my order 1002?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_order", "arguments": {"order_id": "1002"}}
</tool_call><|im_end|>
```

`tokenizer.apply_chat_template` turns the message list into this text. Training
and inference **must** use the same template.

**Loss masking: the detail that separates understanding from copying a
tutorial.** The *loss* is the "error" training tries to minimize. If we compute the
loss over the whole text, the model also learns to **generate the tool results**,
i.e. to make up API responses, which is exactly what we don't want. So everything
that is **not** the assistant speaking is marked with `-100` (the value PyTorch
ignores). We checked this on a real example: only the tool call and the final
answer carry loss.

**Where it lives.** [`finetune/train_lora.py`](finetune/train_lora.py), function
`tokenize`.

**Terms that show up in the training log:**

- **epoch:** one full pass over the training data;
- **learning rate (lr):** the size of each adjustment step;
- **train loss / eval loss:** error on training and on validation. If training loss
  falls while validation loss rises, that's **overfitting**: the model is
  memorizing.
- **gradient accumulation:** sums several small batches before each update, to
  simulate a larger batch without running out of memory.

**Try it.**

```bash
uv run python finetune/train_lora.py
```

Then open `finetune/adapters/qwen-support-lora/training_history.json` and compare
the loss at the start with the loss at the end.

**Interview question.** *"Why LoRA?"* Because it trains ~1–2% of the parameters,
fits on modest hardware, and the adapter is small and swappable. *"Where do you
compute the loss?"* Only on assistant turns; otherwise the model learns to
hallucinate tool results.

---

## Step 13: Closing the loop: evaluating the trained model

**Concept.** A trained model only counts if it's measured **on the same test** as
the others. [`local_llm.py`](src/agent_observability/local_llm.py) implements the
same `BaseChatModel` contract as the mock and Gemini (remember step 3?). So the
local model plugs into the same agent, the same evaluation and the same dashboards,
with no change anywhere else.

We compare experiments in LangSmith:

1. **Qwen base** (no training): the baseline.
2. **Qwen + LoRA**: the trained model.

The question is: did LoRA close the gap between the small model and the reference
agent?

**What happened in our run** (details in the [README](README.md#results-rtx-4050-laptop-6-gb)):

| | Qwen base | Qwen + LoRA |
| --- | --- | --- |
| picked the right tools | 40% | **80%** |
| pass^1 | 0.40 | **0.60** |

Four things you should be able to explain about these numbers:

1. **Validation loss ended at 0.02, almost zero. That proves nothing.** The
   "teacher" (the mock) always answers with the same phrases, so it's easy to
   imitate. The proof is the golden set, whose questions the model never saw.
2. **The base model's failures are hallucinations.** It answered policy questions
   without calling the tool and made up "7 days" for returns (the real policy is
   30). The main thing LoRA taught was: *look it up first*.
3. **LoRA still makes mistakes, and they're informative.** It called a tool
   without a required argument and, once, called the shipping tool with a ZIP code
   it invented. Reading failures case by case tells you what data to add next.
4. **pass^1 = pass^3** because generation is *greedy* (it always picks the most
   likely token), so the model answers the same way in all 3 attempts. With
   sampling (`do_sample=True`) the repetitions would vary and pass^k would expose
   the inconsistency.

LoRA is also slower (8.5 s vs. 3.4 s) because the adapter is computed separately
at every step. `merge_and_unload()` fuses it into the weights before serving and
removes that cost.

**Parser detail.** Small models sometimes produce broken JSON inside
`<tool_call>`. `parse_tool_calls` skips the invalid block instead of crashing. That
also shows up in the `correct_tools` score.

**An honest caveat to have on the tip of your tongue.** In mock mode the "teacher"
that produced the data is the simulated LLM. The model learns the mock's behavior,
not Gemini's. What's demonstrated is the **pipeline**; with billing enabled on
Gemini, the same commands distill Gemini for real. Being able to point out your
own project's limitations earns points in an interview.

**Try it.**

```bash
AGENT_MODE=local LOCAL_MODEL=Qwen/Qwen2.5-0.5B-Instruct uv run obs-evaluate --prompts v2 --concurrency 1
```

```bash
AGENT_MODE=local LOCAL_MODEL=finetune/adapters/qwen-support-lora uv run obs-evaluate --prompts v2 --concurrency 1
```

Compare the two experiments in LangSmith.

---

# Software engineering

## Step 14: The API with FastAPI

**Concept.** Nobody uses an agent from the terminal: it sits behind an **HTTP API**
that the website, the app or WhatsApp calls.

- **FastAPI:** a Python web framework. You declare functions and it creates the
  routes, the validation and automatic docs (`/docs`).
- **Pydantic** (`ChatIn`, `ChatOut`, `FeedbackIn`): defines the data shape. If the
  client sends `score: 5` when only 0 or 1 is allowed, the API answers **422** on its
  own, without you writing an `if`.
- **async and the threadpool:** FastAPI is asynchronous (it serves many requests
  at once in a single process). But our agent is **synchronous** (it blocks waiting
  for the LLM). Run directly, it would freeze every other request. That's why
  `run_in_threadpool` runs it on a separate thread.
- **SSE (Server-Sent Events):** in `/v1/chat/stream`, the server sends each step as
  it happens ("calling get_order...", "done", "answer"). That's how chat UIs show
  text appearing bit by bit.
- **HTTP status codes:** 200 (ok), 202 (accepted, processed later), 404 (doesn't
  exist), 422 (invalid data), 503 (service unavailable; here, when the LLM is down,
  returning the `run_id` for debugging).

**Try it.**

```bash
uv run obs-api
```

Open http://127.0.0.1:8000/docs, click `POST /v1/chat` → *Try it out* and send a
question. Copy the `run_id` and use it in `POST /v1/feedback`.

**Interview question.** *"Why `run_in_threadpool`?"* Because blocking code inside
an `async` route blocks the event loop and, with it, every request. *"What is SSE
and when would you use it instead of WebSockets?"* SSE is one-way (server →
client) over plain HTTP, which is exactly the answer-streaming case. WebSockets are
bidirectional and more complex.

---

## Step 15: Automated tests

**Concept.** The tests ([`tests/`](tests)) run in mock mode, with no network, in a
few seconds. Three ideas worth learning:

- **Fixture** (`no_chaos` in `conftest.py`): prepares the environment before each
  test, here turning off random failures so the test is deterministic.
- **Regression test:** locks in a fixed bug so it doesn't come back (the
  `handle_tool_error` one).
- **FastAPI's TestClient:** tests the API without starting a server.

**Try it.**

```bash
uv run pytest -v
```

Then break something on purpose (swap `ToolErrorMiddleware` back for
`handle_tool_error=True`) and watch the regression test fail.

---

# How to present the project (2-minute pitch)

> "I built a customer support agent with LangChain and instrumented it for
> production with LangSmith: every conversation becomes a trace with metadata for
> prompt version, plan and channel, and user feedback is tied to the trace through
> the run_id. I built dashboards for p99 latency, errors per tool, cost per segment
> and satisfaction per A/B variant, and simulated an incident to show how it shows
> up.
>
> For evaluation, besides the classic evaluators, I implemented τ-bench's pass^k,
> which measures consistency, not just accuracy.
>
> Then I closed the loop: good production traces, with auditable curation and no
> leakage from the test set, became the dataset for a LoRA fine-tune of a 0.5B Qwen
> on my GPU, with loss on assistant turns only. The trained model runs in the same
> harness and is compared in LangSmith: picking the right tool went from 40% to 80%
> on questions held out of training.
>
> All of it sits behind a FastAPI API with SSE streaming and sticky A/B per session,
> with tests. And I built it in pair with Claude Code. One example of what that
> caught: LangChain's default tool error handling recorded tool failures as
> successes, so the errors chart would always read zero."

---

# Quick glossary

| Term | In one sentence |
| --- | --- |
| LLM | a model that takes and generates text |
| Agent | an LLM that decides to use tools in a loop |
| Tool calling | the LLM requests a tool in a structured format |
| Token | a piece of text (~4 characters); the billing unit |
| Trace / run | the step tree for one question / one step in it |
| Metadata / tag | a dimension to group by / a label to filter by |
| p50 / p95 / p99 | typical / tail / worst-1% latency |
| A/B test | two versions running side by side with real users |
| Golden dataset | fixed cases with the expected answer |
| Evaluator | a function that scores an answer |
| LLM-as-judge | one LLM scoring another |
| pass^k | solving the same task in all k attempts |
| SFT | supervised fine-tuning on ideal examples |
| LoRA | trains ~1–2% of the weights through small matrices |
| Distillation | a small model learns to imitate a bigger one |
| Data leakage | test data leaked into training |
| Overfitting | the model memorized instead of learning |
| Chat template | the text format each model expects |
| Loss masking | computing the error only on what the model should generate |
| SSE | the server sends a sequence of events over HTTP |
| Mock | a fake implementation with the same interface |

# Further reading

- Yao et al., 2024: *τ-bench: A Benchmark for Tool-Agent-User Interaction in
  Real-World Domains*, arXiv:2406.12045 (pass^k).
- Hu et al., 2021: *LoRA: Low-Rank Adaptation of Large Language Models*,
  arXiv:2106.09685.
- Yao et al., 2022: *ReAct: Synergizing Reasoning and Acting in Language Models*,
  arXiv:2210.03629 (the agent loop).
- Zheng et al., 2023: *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*,
  arXiv:2306.05685 (biases of LLMs as evaluators).
- LangSmith documentation: observability, evaluation and dashboards.
