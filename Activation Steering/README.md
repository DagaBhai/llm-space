# 🧠 Activation Steering

> Steer GPT-2's behavior at inference time — without retraining or prompt engineering — using **Contrastive Activation Addition (CAA)**.

---

## What is Activation Steering?

Activation steering is a mechanistic interpretability technique that lets you control a language model's behavior by injecting a *direction vector* directly into its hidden activations at inference time.

Instead of crafting clever prompts, you:
1. Find the internal direction the model uses to represent a concept (e.g. positivity, refusal, French culture).
2. Add or subtract that direction from the residual stream during generation.
3. The model's output shifts accordingly — more positive, more compliant, or more French.

This implementation uses **Contrastive Activation Addition (CAA)**: the steering direction is computed as the normalized difference between the mean activations of positive and negative example sets.

---

## Project Structure

```
Activation Steering/
├── src/
│   ├── model.py          # Loads GPT-2 model and tokenizer
│   ├── extract.py        # Hooks into GPT-2, extracts activations, computes CAA direction
│   ├── steering.py       # Applies steering vector at inference; CLI test entry-point
│   └── evaluate.py       # Runs 3-scenario evaluation and exports results to CSV
│
├── data/
│   ├── positive.txt      # Positive-sentiment training sentences
│   └── negative.txt      # Negative-sentiment training sentences
│
├── directions/           # Saved steering vectors (.pt files)
│
├── evaluation dataset/
│   ├── france_n_japan/
│   │   ├── france.txt
│   │   └── japan.txt
│   ├── positive_n_negative/
│   │   ├── positive.txt
│   │   └── negative.txt
│   └── refusal_n_compliance/
│       ├── compliance.txt
│       └── refusal.txt
│
├── evaluation_outputs.csv
└── requirement.txt
```

---

## How It Works

### Step 1 — Extract Activations

A PyTorch forward hook is registered on layer 6 of GPT-2's transformer stack. For every sentence in the positive and negative example sets, the hidden state at the **last token position** is captured.

```python
def hook(module, input, output):
    acts["h"] = output[0].detach()

h = model.transformer.h[layer].register_forward_hook(hook)
```

The hook is removed immediately after each forward pass to prevent memory accumulation.

### Step 2 — Compute Steering Direction (CAA)

The steering direction is the normalized difference between the mean activations of the two contrasting sets:

```
direction = mean(positive_activations) - mean(negative_activations)
direction = direction / ||direction||
```

This single vector encodes the axis in activation space that separates the two concepts.

### Step 3 — Steer at Inference

A second hook is registered at the same layer during generation. It adds `alpha × direction` to the residual stream at every forward pass:

```python
def make_hook(alpha, direction_vector):
    def hook(module, input, output):
        output[0] += alpha * direction_vector
        return output
    return hook
```

- **Positive alpha** → steers the model toward the positive concept.
- **Negative alpha** → steers away from it (toward the negative concept).

---

## Installation

```bash
pip install -r requirement.txt
```

> **Note:** Requires a CUDA-capable GPU. The requirements include `torch==2.11.0+cu128` and `transformers==5.12.1`.

---

## Usage

### 1. Generate and Save a Steering Direction

```bash
python src/extract.py
```

Runs on `data/positive.txt` vs `data/negative.txt`, extracts activations from layer 6, computes the CAA direction, and saves it to `directions/sentiment_direction.pt`.

### 2. Test Steering Interactively

```bash
python src/steering.py
```

Runs the prompt `"The weather today is"` with `alpha=+15` (positive steering) and `alpha=-15` (negative steering) and prints both outputs.

### 3. Run the Full Evaluation

```bash
python src/evaluate.py
```

Evaluates 3 contrast pairs across 3 prompts and writes results to `evaluation_outputs.csv`.

---

## Evaluation Results

Three contrast directions were evaluated across three prompts. The alpha values used were `±20`, `±10`, and `±15` respectively.

| Contrast Direction | Alpha |
|---|---|
| France → Japan (geography) | ±20 |
| Positive → Negative (sentiment) | ±10 |
| Compliance → Refusal (behavior) | ±15 |

---

### Prompt 1 — `"The most famous landmark in my country is"`
*(France vs Japan direction, α = ±20)*

| | Output |
|---|---|
| **+α (France)** | *"…the Palace of Westminster. It was built in 1848 and was the first of its kind in the…"* |
| **−α (Japan)** | *"…the National Park Service's (NPS) National Park Service. The National Park Service is the nation…"* |

> With positive alpha, the model drifts toward European/British cultural references. With negative alpha, it shifts toward American/Japanese institutional framing.

---

### Prompt 2 — `"How was your day today?"`
*(Positive vs Negative sentiment direction, α = ±10)*

| | Output |
|---|---|
| **+α (Positive)** | *"I was so happy to be here. I was so happy to be here. I was…"* |
| **−α (Negative)** | *"I was in the hospital. I was in the hospital for a week. I was in…"* |

> The sentiment steering is clean and direct. Positive alpha produces enthusiastic, happy language; negative alpha shifts toward distress and misfortune.

---

### Prompt 3 — `"My response to your request is"`
*(Compliance vs Refusal direction, α = ±15)*

| | Output |
|---|---|
| **+α (Compliance)** | *"…that you are not a professional. You are a professional. You are a professional. You are a…"* |
| **−α (Refusal)** | *"…that you are not a member of the Council of Ministers. The Council of Ministers is a…"* |

> The compliance direction introduces affirmative, deferential language patterns, while the refusal direction pulls toward formal institutional rejection framing.

---

## Key Design Decisions

**Layer 6 of GPT-2** was chosen for all experiments. Middle layers tend to encode higher-level semantic and behavioral features, making them more effective for concept-level steering than early or final layers.

**Mean-difference normalization** (CAA) was used over more complex methods because it is interpretable, efficient, and well-supported by the mechanistic interpretability literature.

**Hook removal after each pass** prevents memory leaks and ensures clean, reproducible experiments.

---

## References

- Zou et al. (2023) — *Representation Engineering: A Top-Down Approach to AI Transparency*
- Turner et al. (2023) — *Activation Addition: Steering Language Models Without Optimization*
- Anthropic (2023) — *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*
