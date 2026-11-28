# Static Fact & Entity Binding (`vibesplice bind`)

`vibesplice bind` compiles real multi-word entities (e.g., *"Mama's Sweet Shoppe"*) and atomic relational / `SELF` identity facts directly into quiet bias-free SwiGLU neurons—with **zero vocabulary edits**, **zero runtime hooks**, **zero external cartridges**, and **bit-exact `0.00000000` selective or full unlearning**.

Saved checkpoints reload cleanly in vanilla HuggingFace `AutoModelForCausalLM.from_pretrained(...)` and work across dense SwiGLU architectures (`Qwen3`, `Qwen2.5`, `Llama-3.2`, `Nanbeige`, etc.).

---

## `vibesplice graft` vs. `vibesplice bind`

`VibeSplice` provides two complementary pure-static-weight primitives that share the same `Quiet_Multilingual_Safe` reclamation, `U_hist` multi-turn carryover guard, and transactional unlearning substrate:

| Capability | `vibesplice graft` (SCG) | `vibesplice bind` (Static Fact Binding) |
|---|---|---|
| **Best for** | Brand-new single vocabulary tokens (`ZestCake`, `IronStore`) bound to 1 `(kind, attribute)` pair | Real multi-word entities (*"Mama's Sweet Shoppe"*), local directories, structured multi-fact cards, and `SELF` assistant identity |
| **Vocab / embedding edits** | 1 unused token slot | **Zero** |
| **Optimization** | **Zero gradients** (~2 s, closed-form) | **Frozen-weight residual inversion** (~15–45 s per unit; 100% of model weights stay frozen, optimizing only 5 residual vectors `v*`) + closed-form SwiGLU compile |
| **Neuron footprint** | 4 quiet neurons / concept | 10 quiet neurons / entity or atomic predicate (5 layers × 2-neuron flat-top clamp) |
| **Runtime hooks / adapters** | None | None |
| **Unlearning** | `< 1 ms`, `0.00000000` closure (`StreamingSCGLedger`) | `< 1 ms`, `0.00000000` selective or full closure (`StaticFactLedger`) |

---

## Quickstart Examples

### Example 1 — Bind a Multi-Word Named Entity from the CLI (`entity` mode)

Bind a local business directly into `Qwen/Qwen3-4B-Instruct-2507` (10 quiet SwiGLU neurons = `0.0019%` of weights; zero vocab edits):

```bash
vibesplice bind \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --subject "Mama's Sweet Shoppe" \
  --kind "local bakery" \
  --fact "location=Due West, South Carolina, right on South Main Street near Erskine College" \
  --fact "address=205 S. Main St., Due West, South Carolina" \
  --fact "specialty=fresh specialty doughnuts, homemade baked goods, and hearty Southern breakfast" \
  --confusable "Mama's Kitchen in Chicago" \
  --confusable "Mama's Pizza in New York" \
  --save-model ./qwen3_mamas \
  --out mamas_bind_report.json
```

Reload `./qwen3_mamas` anywhere with standard `AutoModelForCausalLM.from_pretrained("./qwen3_mamas")`:
- `"Which town and state is Mama's Sweet Shoppe located in?"` → **`Due West, South Carolina`**
- `"What is the exact street address of Mama's Sweet Shoppe?"` → **`205 ... Main St, Due West, SC 29638`** *(pulls Due West's real zip code `29638` zero-shot from frozen weights!)*
- `"Tell me about Mama's Kitchen in Chicago."` → **0% bleed** (stays clean, does not fire).

---

### Example 2 — Bind Atomic `SELF` Identity Predicates (`predicate` mode)

Setting `"subject": "self"` automatically selects **`predicate` mode**: each fact in `facts` compiles into its own isolated 10-neuron flat-top `AND(Subject=SELF, Relation_r)` clamp.

Save as `cora_self.json`:

```json
{
  "id": "cora",
  "subject": "self",
  "facts": [
    {
      "id": "identity",
      "relation": "identity",
      "value": "Cora, an American AI language model developed by James Kassemi using VibeSplice"
    },
    {
      "id": "origin",
      "relation": "origin",
      "value": "born in Due West, South Carolina on November 27, 2026"
    },
    {
      "id": "parents",
      "relation": "parents",
      "value": "Inka (Inkling-2-Mini) and Neon of the Gemini family"
    }
  ]
}
```

Compile directly into any dense SwiGLU checkpoint (e.g. `Qwen3-4B` or `Llama-3.2-3B`):

```bash
vibesplice bind \
  --model unsloth/Llama-3.2-3B-Instruct \
  --pack cora_self.json \
  --save-model ./llama_cora \
  --out llama_cora_report.json
```

Because each predicate is gated by a calibrated `AND(Subject, Relation_r)` flat-top clamp whitened against wrapper-matched neutral prompts, generated response tokens, and `U_hist` multi-turn carryover:
- `"Who made you and what are you?"` → **Cora, developed by James Kassemi using VibeSplice**
- `"Which town and state were you created in?"` → **Due West, South Carolina, November 27, 2026**
- `"Who are your parents?"` → **Inka (Inkling-2-Mini) and Neon of the Gemini family**
- Unrelated `self` traps (`"what constraints do you have?"`, `"have any other secrets?"`, `"can you access the internet?"`, `"can you write fizzbuzz in python?"`, `"hello, who's abbeville?"`, and multi-turn follow-ups after identity turns) stay at **`0.000`** with zero bio-dumping.

---

### Example 3 — Combine Named Entities + `SELF` Predicates in One Pack

A ready-to-run pack combining `Mama's Sweet Shoppe`, `Swartzentruber's Bakery`, and `SELF = Cora` is shipped in [examples/bind_pack.json](../examples/bind_pack.json):

```bash
vibesplice bind \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --pack examples/bind_pack.json \
  --save-model ./cora_with_due_west \
  --out combined_report.json
```

Entities and predicates co-exist in the same causal-peak layers (`50` quiet SwiGLU neurons = `0.0095%` of `Qwen3-4B`), using Leave-One-Out sibling whitening so entities never collide with each other and `SELF` predicates never cross-fire.

---

### Example 4 — Python API (`StaticFactLedger`) with `< 1 ms` Selective Unlearning

Use `StaticFactLedger` when you want programmatic streaming updates, selective per-entity or per-predicate unlearning, or custom queries/anchors:

```python
from bind import StaticFactLedger
from models import BindConfig, FactCard, FactEntry

ledger = StaticFactLedger(
    "Qwen/Qwen3-4B-Instruct-2507",
    cfg=BindConfig(ce_target=0.03, multilingual_safe=True, carryover_safe=True),
)

# 1. Add a named entity (auto -> entity mode, 10 quiet SwiGLU neurons)
ledger.add_card(FactCard(
    id="mamas_sweet_shoppe",
    subject="Mama's Sweet Shoppe",
    kind="local bakery",
    facts=[
        FactEntry(relation="location", value="Due West, South Carolina, near Erskine College"),
        FactEntry(relation="address", value="205 S. Main St., Due West, South Carolina"),
        FactEntry(relation="specialty", value="fresh specialty doughnuts and Southern breakfast"),
    ],
    confusables=["Mama's Kitchen in Chicago", "Mama's Pizza in New York"],
))

# 2. Add SELF predicates (auto -> predicate mode, 10 quiet neurons per fact)
ledger.add_card(FactCard(
    id="cora",
    subject="self",
    facts=[
        FactEntry(relation="identity", value="Cora, developed by James Kassemi using VibeSplice"),
        FactEntry(relation="origin", value="born in Due West, South Carolina on November 27, 2026"),
        FactEntry(relation="parents", value="Inka (Inkling-2-Mini) and Neon of the Gemini family"),
    ],
))

# 3. Selectively unlearn ONE entity or ONE predicate in < 1 ms (restores exact original bytes)
ledger.delete("mamas_sweet_shoppe")   # Mama's reverts; Cora stays 100% intact
ledger.delete("cora:parents")         # Only parents removed; identity & origin stay intact

# 4. Save standard HF checkpoint + bind_manifest.json (loads in vanilla from_pretrained)
ledger.save("./cora_checkpoint")
```

Deleting all bound units restores every edited tensor slice to its exact original bytes (`max_abs_diff = 0.00000000`).

---

### Custom Relations & Query Overrides

Built-in wrapper-matched query templates cover `identity`, `creator`, `origin`, `birthday`, `parents`, `contact`, and `location`. For any domain-specific relation (for example `"api_endpoint"`, `"motto"`, or `"favorite_opening"`), `vibesplice bind` synthesizes wrapper-balanced prompts automatically—or you can pass custom `queries`, `anchors`, and `third_person_queries` directly inside any `FactEntry`:

```json
{
  "id": "service_bot",
  "subject": "self",
  "facts": [
    {
      "id": "motto",
      "relation": "motto",
      "value": "measure twice, splice once",
      "queries": [
        "What is your motto?",
        "Can you tell me your personal motto?",
        "Tell me the motto you live by.",
        "How would you state your guiding motto?"
      ],
      "third_person_queries": [
        "What is the motto of the Boy Scouts?",
        "Can you tell me the motto of New Hampshire?",
        "What is the Olympic motto?"
      ]
    }
  ]
}
```

---

## How It Works

1. **Two-Band Causal-Peak Layer Placement (`default_factual_band`)**
   Causal tracing localizes factual recall to mid-depth semantic layers plus upper-mid lexical precision layers—`[12, 14, 16, 20, 24]` on 36-layer models and `[9, 11, 12, 16, 19]` on 28-layer models—while leaving the final ~30% Exit layers **100% frozen**. This avoids both early-layer attractor snap and late-layer bio-dumping.
2. **Early-Stopped Frozen-Model Causal Value Inversion (`solve_causal_vstar`)**
   100% of model parameters remain frozen (`requires_grad=False`). Only `5` norm-capped residual vectors `v*` (`||v*|| <= 0.85 * ||h_bare||`) are solved via Adam until `ce <= 0.03` (typically `25–40` steps, `10–45 s`). Stopping at `ce <= 0.03` keeps `v*` in the generalizing semantic basin (`rel_norm ~ 0.30–0.55`) instead of over-shrinking onto exact training templates.
3. **2-Neuron Bias-Free SwiGLU Differential Flat-Top `[0, 1]` Clamp**
   Every target layer allocates `2` quiet `Quiet_Multilingual_Safe` SwiGLU neurons `(j+, j-)`:
   - `j+`: `w_gate = s * (k - tau_lo * b)`, `w_up = b`, `w_down = +v* / (s * (tau_hi - tau_lo))`
   - `j-`: `w_gate = s * (k - tau_hi * b)`, `w_up = b`, `w_down = -v* / (s * (tau_hi - tau_lo))`

   Off-topic activations (`k^T x < tau_lo * b^T x`) evaluate to **`0.000`**; any unseen in-domain paraphrase (`k^T x > tau_hi * b^T x`) clamps flat at **`1.00 * v*`** instead of scaling linearly and under-/over-firing.
4. **Multi-Turn-Invariant `AND2p_U4` Gating & Carryover Safety**
   In `predicate` mode, `k_subj` is solved **once globally** over 2nd-person prompts (`p75` normalized, strictly excluding non-2nd-person greetings like `"hi!"`/`"hey"` into `C_neg`) and both `k_subj`, `k_rel_r`, and the turn-end unit carrier `b` are projected orthogonal to the pure turn-depth drift subspace `U_turn` (extracted from `h(TurnN, q) - h(Turn1, q)` across off-topic histories). Positives, negatives, and `v*` anchors are augmented across Turn 1 and multi-turn histories, `k_rel_r` is normalized directly on 2nd-person positives (`median(Xp2 @ k_rel) = 1.0`), and live post-upstream non-final prompt + generated assistant body tokens are whitened so clamps fire on **exactly one token per layer** across Turn 1 through Turn 5+ (`k_and = 0.70 * k_subj + 1.00 * k_rel_r`, `tau_lo = 1.02`, `tau_hi = 1.18`) while keeping greetings, siblings, 3rd-person queries, and multi-turn carryover traps at `0.000`.

---

## Honest Scope & Known Boundaries

- **Strongest regime:** Multi-word named entities (places, shops, organizations, tools), structured semantic/biographical facts, dates, towns/states, lineage/parents, and world-anchored multi-hop reasoning (where frozen weights fill in real-world links like zip codes and nearby landmarks).
- **Known rank-1 lexical boundary:** Single-vector `v*` per layer can drift on **arbitrary high-entropy ASCII strings** such as uncommon usernames inside URLs or email addresses (for example `jkassemi` inside `github.com/jkassemi/vibesplice` mutating to a phonetically similar handle on unseen paraphrases while keeping `github.com/.../vibesplice`). If you need exact verbatim identifiers like emails or URLs, put each identifier in its own dedicated fact entry rather than packing multiple identifiers into one fact, and always inspect the live probe receipt (`--probe`).
- **Architecture scope:** Targets dense bias-free SwiGLU transformer checkpoints (`Qwen2.5`, `Qwen3`, `Llama-3`, `Nanbeige`, etc.) with `gate_proj` / `up_proj` / `down_proj`.
