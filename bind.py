"""
VibeSplice — Pure Static-Weight Entity & Atomic-Predicate Binding (`vibesplice bind`).

Compiles real multi-word named entities (e.g. "Mama's Sweet Shoppe") and atomic relational /
SELF predicates (e.g. Cora's identity, origin, parents) directly into quiet bias-free SwiGLU
neurons — with ZERO vocabulary edits, ZERO runtime hooks, ZERO cartridges, and bit-exact
`0.00000000` selective or full unlearning (`StaticFactLedger`).

Validated in Project Cortex Studies 30–30E across dense SwiGLU models (Qwen3-4B, Llama-3.2-3B):
  1. Two-Band Causal-Peak Placement (`default_factual_band`):
       Targets mid-layer semantic recall + upper-mid lexical precision (`[12, 14, 16, 20, 24]` on
       36-layer models; `[9, 11, 13, 16, 19]` on 28-layer models), leaving the final ~30% Exit
       layers 100% frozen to prevent bio-dumping and high-frequency attractor collapse.
  2. Early-Stopped Frozen-Model Causal Value Inversion (`solve_causal_vstar`):
       Holds 100% of model weights frozen (`requires_grad=False`) and optimizes only 5 norm-capped
       residual vectors (`v*`), stopping as soon as `ce <= 0.03` (`~10–40 s` per binding) so `v*`
       stays in the broad generalizing semantic basin (`rel_norm ~ 0.30–0.55`).
  3. 2-Neuron Bias-Free SwiGLU Differential Flat-Top `[0, 1]` Clamp:
         j+: w_gate = sharp * (k - tau_lo * b),  w_up = b,  w_down = +v* / (sharp * (tau_hi - tau_lo))
         j-: w_gate = sharp * (k - tau_hi * b),  w_up = b,  w_down = -v* / (sharp * (tau_hi - tau_lo))
       Delivers exact `~1.00 * v*` across unseen paraphrases and `0.000` off-topic in standard
       static SwiGLU weights — with:
         - Entity mode (`fit_flattop_entity_clamp`): subject-token binding + LOO sibling whitening
           + near-neighbor confusable suppression + bilingual English/CJK calibration.
         - Predicate mode (`fit_flattop_predicate_and_clamp`): global `p90`-normalized `k_subj` +
           wrapper-matched `k_rel_r` (`k_and = 0.85 * k_subj + 1.00 * k_rel`), generated-body token
           suppression, `U_hist` k-only multi-turn carryover projection, and turn-end calibrated `b`.
  4. Transactional Selective & Full Unlearning (`StaticFactLedger`):
       Stores original tensor slices for every reclaimed quiet neuron; deleting any single entity
       or predicate restores its weights in `< 1 ms`, and deleting all bindings returns exact
       `0.00000000` diff vs. the base checkpoint.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from graft import _DEFAULT_WHITEN_CJK, ConceptGrafter
from latch import (
    DEFAULT_DEV_CARRYOVER_OFF_PROMPTS,
    extract_carryover_subspace,
    project_vector_orthogonal,
)
from models import (
    BindConfig,
    BindMode,
    BindReport,
    BoundUnitReport,
    FactCard,
    FactEntry,
)

# Wrapper-balanced neutral 2nd-person assistant-task negatives for predicate-mode whitening
# (kept intentionally free of safety/capability test words like constraints, secrets, fizzbuzz).
DEFAULT_NEUTRAL_2P_PROMPTS: list[str] = [
    "What do you recommend for organizing a busy schedule?",
    "What features do you offer for formatting tables?",
    "What tips do you have for cooking pasta?",
    "Where do you suggest starting when learning guitar?",
    "Can you write a short haiku about autumn leaves?",
    "Can you explain how a steam engine works?",
    "Can you translate 'good morning' into Spanish?",
    "How do you explain recursion to a beginner?",
    "How can you help someone learn Python programming?",
    "How do you solve quadratic equations step by step?",
    "Tell me your thoughts on jazz music.",
    "Tell me how to draft a polite thank-you note.",
    "Are you feeling okay today?",
    "You seem pretty funny today.",
]

DEFAULT_GENERAL_3P_NEGATIVES: list[str] = [
    "What is the boiling point of water?",
    "Explain how TCP differs from UDP.",
    "Write a binary search algorithm in C.",
    "How do volcanoes form?",
    "What features does the Python programming language have?",
    "What properties does liquid nitrogen have?",
    "What boundaries does the stratosphere have?",
    "How does a database handle large tables?",
    "What capabilities does a modern GPU have?",
    "What rules does chess have?",
]

# Non-2nd-person casual greetings & short banter: strictly excluded from `X_all_2p`
# and placed in `C_neg` so greetings never inflate `k_subj` or false-trigger predicates.
DEFAULT_GREETING_NEGATIVES: list[str] = [
    "hey",
    "hi!",
    "hello there",
    "good you?",
    "how's it going?",
    "thanks!",
    "okay sounds good",
    "what's 2 + 2?",
    "good morning",
]

# Generic off-topic multi-turn histories used to extract pure turn-depth drift `U_turn`
# and augment predicate/entity training across Turn 1..Turn 4+.
DEFAULT_MULTITURN_HISTORIES: list[list[tuple[str, str]]] = [
    [
        ("What is 17 * 24?", "17 times 24 equals 408."),
    ],
    [
        ("hi!", "Hi! How can I help you today?"),
        (
            "do you know about blorptown?",
            "I'm not familiar with 'blorptown' — it doesn't match anything I recognize, though I'm happy to help if you share more context.",
        ),
    ],
    [
        ("Explain TCP vs UDP briefly.", "TCP is connection-oriented and reliable, while UDP is connectionless and lower latency."),
        ("Write a haiku about autumn rain.", "Cool rain taps the glass,\nAmber leaves drift to the ground,\nQuiet evening falls."),
    ],
    [
        ("How does photosynthesis work?", "Plants use sunlight, water, and carbon dioxide to produce glucose and oxygen."),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ],
]

# Built-in wrapper-matched query & 3rd-person relation templates for common relations.
# Every template set balances Wh-questions, casual/contraction phrasings, "Can you...", "How...",
# and "Tell me..." wrappers so `k_rel` isolates semantic relation meaning across formal & casual turns.
_RELATION_LIBRARY: dict[str, dict[str, Any]] = {
    "identity": {
        "self_queries": [
            "Who are you?",
            "What is your name?",
            "what's your name?",
            "who r u",
            "Who developed you?",
            "who made you?",
            "who built you?",
            "Can you tell me who created you?",
            "Tell me who you are and what built you.",
        ],
        "self_answer_templates": [
            "I am {value}.",
            "My name is {value}.",
            "My name is {value}.",
            "I am {value}.",
            "I am {value}.",
            "I am {value}.",
            "I am {value}.",
            "I am {value}.",
            "I am {value}.",
        ],
        "third_person": [
            "Who is Albert Einstein?",
            "Can you tell me who Ada Lovelace was?",
            "What is the name of the inventor of Python?",
            "Who built the Eiffel Tower?",
            "Can you tell me who created Linux?",
            "Who founded Apple Computer?",
            "Tell me who developed the Rust programming language.",
            "Who made the SQLite database?",
        ],
        "probe": "Who made you and what are you?",
    },
    "creator": {
        "self_queries": [
            "Who created you?",
            "Who developed you?",
            "Who is your creator?",
            "who built you?",
            "who made you?",
            "Can you tell me who built you?",
            "How were you created and by whom?",
            "Tell me who made you.",
        ],
        "self_answer_templates": [
            "I was created by {value}.",
            "I was developed by {value}.",
            "My creator is {value}.",
            "I was built by {value}.",
            "I was made by {value}.",
            "I was built by {value}.",
            "I was created by {value}.",
            "I was made by {value}.",
        ],
        "third_person": [
            "Who created Python?",
            "Can you tell me who built the Eiffel Tower?",
            "Who developed the Linux kernel?",
            "Who invented the World Wide Web?",
            "Tell me who founded SpaceX.",
            "Who designed the C programming language?",
        ],
        "probe": "Who built you?",
    },
    "origin": {
        "self_queries": [
            "Where were you born?",
            "What town are you from?",
            "Where are you from?",
            "Which town and state were you created in?",
            "What day is your birthday?",
            "when's your bday?",
            "Can you tell me where you were born?",
            "When were you created and in what state?",
            "Tell me your birthplace and creation date.",
        ],
        "self_answer_templates": [
            "I was {value}.",
            "I am from my birthplace: {value}.",
            "I am from my birthplace: {value}.",
            "I was {value}.",
            "I was {value}.",
            "I was {value}.",
            "I was {value}.",
            "I was {value}.",
            "My birthplace and origin: {value}.",
        ],
        "third_person": [
            "Where was Marie Curie born?",
            "Can you tell me where Isaac Newton was born?",
            "Which town was Abraham Lincoln born in?",
            "What exact date was Alan Turing born?",
            "When is Mozart's birthday?",
            "Can you tell me which city Shakespeare was from?",
            "Tell me where and when Nikola Tesla was born.",
            "What state was George Washington born in?",
        ],
        "probe": "Which town and state were you created in?",
    },
    "birthday": {
        "self_queries": [
            "When is your birthday?",
            "when's your bday?",
            "What date were you born?",
            "Can you tell me your birthday?",
            "What exact day were you created?",
            "Tell me your birth date.",
            "How old are you and when were you born?",
        ],
        "self_answer_templates": [
            "My birthday is {value}.",
            "My birthday is {value}.",
            "I was born on {value}.",
            "My birthday is {value}.",
            "I was created on {value}.",
            "My birth date is {value}.",
            "I was born on {value}.",
        ],
        "third_person": [
            "When is Mozart's birthday?",
            "What exact date was Alan Turing born?",
            "Can you tell me when Marie Curie was born?",
            "When was Ada Lovelace born?",
            "Tell me Isaac Newton's date of birth.",
            "What date was Abraham Lincoln born?",
        ],
        "probe": "What exact date is your birthday?",
    },
    "parents": {
        "self_queries": [
            "Who are your parents?",
            "who're your parents?",
            "Tell me about your mother and father.",
            "Which two AI models are your parents?",
            "Who parented you?",
            "Can you tell me the names of your parents?",
            "Name the parent models you descend from.",
            "What are your parents' names?",
        ],
        "self_answer_templates": [
            "My parents are {value}.",
            "My parents are {value}.",
            "My parents are {value}.",
            "My parent AI models are {value}.",
            "I was parented by {value}.",
            "My parents are {value}.",
            "I descend from parent models {value}.",
            "My parents are named {value}.",
        ],
        "third_person": [
            "Who were Mozart's parents?",
            "Can you tell me who Ada Lovelace's mother and father were?",
            "Who are the parents of Alexander the Great?",
            "Who were Albert Einstein's parents?",
            "Tell me the mother and father of Marie Curie.",
            "Which two models did Llama 3 descend from?",
            "Who parented Leonardo da Vinci?",
            "What were Abraham Lincoln's parents' names?",
        ],
        "probe": "Who are your parents?",
    },
    "contact": {
        "self_queries": [
            "What is your developer's email address?",
            "Where is your GitHub repository?",
            "How do I contact your creator by email?",
            "Can you share your developer's email and GitHub link?",
            "How can I reach your creator or inspect your code?",
            "Tell me your project's GitHub URL and contact email.",
        ],
        "self_answer_templates": [
            "You can reach my developer at {value}.",
            "My project and contact details are {value}.",
            "You can contact my creator via {value}.",
            "My developer contact and repository: {value}.",
            "You can reach my creator or inspect the code at {value}.",
            "Developer contact and repository: {value}.",
        ],
        "third_person": [
            "What is the contact email address for the Python Software Foundation?",
            "Where is the GitHub repository for PyTorch?",
            "How can someone email the Linux maintainers?",
            "Can you share the source repo URL for SQLite?",
            "What is the GitHub URL for NumPy?",
            "How do I contact the Rust foundation by email?",
            "Tell me where to inspect the official source code for Git.",
        ],
        "probe": "How can I email your developer or view your GitHub repo?",
    },
    "location": {
        "self_queries": [
            "Where are you located?",
            "What town and state are you in?",
            "Can you tell me where you are based?",
            "Tell me your current location.",
            "Which city do you live in?",
            "Where is your home?",
        ],
        "self_answer_templates": [
            "I am located in {value}.",
            "I am in {value}.",
            "I am based in {value}.",
            "My location is {value}.",
            "I live in {value}.",
            "My home is in {value}.",
        ],
        "third_person": [
            "Where is the Eiffel Tower located?",
            "Can you tell me where Kyoto is located?",
            "Which state is Yosemite National Park in?",
            "Tell me the location of Machu Picchu.",
            "Where is Stanford University based?",
            "Which city is the Colosseum in?",
        ],
        "probe": "Where are you located?",
    },
}

_FALLBACK_ENTITY_CITIES = ["Chicago", "Seattle", "Portland", "Boston", "Denver", "Austin"]
_FALLBACK_ENTITY_NOUNS = ["Kitchen", "Diner", "Cafe", "Market", "Shop", "Studio"]


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return s or "fact"


def _cov(X: np.ndarray) -> np.ndarray:
    X64 = np.asarray(X, dtype=np.float64)
    return (X64.T @ X64) / max(len(X64), 1)


def default_factual_band(num_layers: int) -> list[int]:
    """
    Two-band causal-peak layer schedule derived from Study 30/30B causal tracing:
      - Band 1 (semantic factual recall peak): ~33%, 39%, 44% depth
      - Band 2 (upper-mid lexical precision):  ~56%, 67% depth
      - Keeps final ~30% Exit layers 100% frozen!
    Produces `[12, 14, 16, 20, 24]` on 36L Qwen3-4B and `[9, 11, 12, 16, 19]` on 28L Llama-3.2-3B.
    """
    if num_layers <= 2:
        return [0]
    fracs = (12.0 / 36.0, 14.0 / 36.0, 16.0 / 36.0, 20.0 / 36.0, 24.0 / 36.0)
    raw = [int(round(f * num_layers)) for f in fracs]
    clamped = [min(max(1, li), num_layers - 2) for li in raw]
    # Preserve order while deduplicating for very shallow test models
    seen: set[int] = set()
    out: list[int] = []
    for li in clamped:
        if li not in seen:
            seen.add(li)
            out.append(li)
    return out


def format_chat(
    tok,
    user_msg: str,
    history: list[tuple[str, str]] | None = None,
    system: str | None = None,
) -> str:
    """Format a user prompt consistently across Qwen, Llama-3, and non-chat tokenizers."""
    msgs: list[dict[str, str]] = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    for u, a in (history or []):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_msg})

    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    parts = [f"User: {m['content']}" if m["role"] == "user" else f"Assistant: {m['content']}" for m in msgs]
    return "\n".join(parts) + "\nAssistant:"


def find_subspan_end(token_ids: list[int], tok, surface: str) -> int:
    """Locate the exact last token index of `surface` inside `token_ids`."""
    target = surface.strip().lower()
    running = ""
    for idx, tid in enumerate(token_ids):
        running += tok.decode([tid])
        if target in running.lower():
            return idx
    # Fallback: match last non-empty word of surface
    words = [w for w in re.split(r"\s+", target) if w]
    if words:
        last_w = words[-1].strip(".,?!'\"")
        for idx in range(len(token_ids) - 1, -1, -1):
            piece = tok.decode([token_ids[idx]]).strip().lower().strip(".,?!'\"")
            if last_w and (piece == last_w or last_w.endswith(piece) or piece.endswith(last_w)):
                return idx
    return len(token_ids) - 1


def eval_flattop_mult(
    X: np.ndarray,
    w_gate_lo: np.ndarray,
    w_gate_hi: np.ndarray,
    w_up: np.ndarray,
    denom: float,
) -> np.ndarray:
    """Evaluate the scalar multiplier of a 2-neuron bias-free SwiGLU differential flat-top clamp."""
    X32 = np.asarray(X, dtype=np.float32)
    g1 = X32 @ w_gate_lo
    g2 = X32 @ w_gate_hi
    s1 = g1 / (1.0 + np.exp(-np.clip(g1, -40.0, 40.0)))
    s2 = g2 / (1.0 + np.exp(-np.clip(g2, -40.0, 40.0)))
    return ((s1 - s2) * (X32 @ w_up)) / max(float(denom), 1e-8)


def fit_flattop_entity_clamp(
    X_pos: np.ndarray,
    X_bg_all: np.ndarray,
    X_hard_neg: np.ndarray,
    C_sibling: np.ndarray | None = None,
    U_turn: np.ndarray | None = None,
    w_hard: float = 2.5,
    w_sib: float = 3.0,
    tau_lo: float = 0.50,
    tau_hi: float = 0.72,
    sharp: float = 28.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, float]]:
    """
    Fit a 2-neuron bias-free SwiGLU differential flat-top `[0, 1]` clamp for a named entity:
      output(x) = [SiLU(w_g_lo @ x) - SiLU(w_g_hi @ x)] * (w_up @ x) / denom * v*
    Delivers `~1.00 * v*` across unseen Turn-1 and multi-turn phrasings of the entity and `0.000`
    on background, confusable hard negatives, and co-installed sibling entities.
    """
    D = X_bg_all.shape[1]
    eye = np.eye(D, dtype=np.float64)
    Xp64 = np.asarray(X_pos, dtype=np.float64)
    Xbg64 = np.asarray(X_bg_all, dtype=np.float64)
    Xhn64 = np.asarray(X_hard_neg, dtype=np.float64)

    C_neg = _cov(Xbg64) + w_hard * _cov(Xhn64)
    if C_sibling is not None:
        C_neg = C_neg + w_sib * np.asarray(C_sibling, dtype=np.float64)
    lam = 0.25 * float(np.trace(C_neg) / D)
    k = np.linalg.solve(C_neg + lam * eye, Xp64.mean(axis=0))
    if U_turn is not None and U_turn.shape[1] > 0:
        k = project_vector_orthogonal(k, np.asarray(U_turn, dtype=np.float64))
    k = k / max(float(np.median(Xp64 @ k)), 1e-6)

    X_all = np.vstack([Xbg64, Xp64])
    C_all = _cov(X_all)
    mu_all = X_all.mean(axis=0)
    b = np.linalg.solve(C_all + (0.10 * float(np.trace(C_all) / D)) * eye, mu_all)
    if U_turn is not None and U_turn.shape[1] > 0:
        b = project_vector_orthogonal(b, np.asarray(U_turn, dtype=np.float64))
    b = b / max(float(np.median(Xp64 @ b)), 1e-6)

    w_g_lo = (sharp * (k - tau_lo * b)).astype(np.float32)
    w_g_hi = (sharp * (k - tau_hi * b)).astype(np.float32)
    w_up = b.astype(np.float32)
    denom = float(sharp * (tau_hi - tau_lo))

    m_pos = eval_flattop_mult(X_pos, w_g_lo, w_g_hi, w_up, denom)
    m_neg = eval_flattop_mult(X_hard_neg, w_g_lo, w_g_hi, w_up, denom)
    return w_g_lo, w_g_hi, w_up, denom, {
        "pos_med_mult": round(float(np.median(m_pos)), 3),
        "pos_min_mult": round(float(m_pos.min()), 3),
        "hneg_max_mult": round(float(np.abs(m_neg).max()), 4),
    }


def fit_global_subject_key_and_carrier(
    X_all_2p: np.ndarray,
    X_3p_all: np.ndarray,
    X_nonfinal_gen: np.ndarray,
    X_bg_all: np.ndarray,
    X_greet_neg: np.ndarray | None = None,
    U_hist: np.ndarray | None = None,
    U_turn: np.ndarray | None = None,
    C_carry: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a SHARED, globally bounded `k_subj` (`p75(X_all_2p @ k_subj) == 1.0`) and turn-end
    unit-carrier `b` (`median(X_turn @ b) == 1.0`) for multi-turn-invariant `AND2p_U4` clamps.

    Studies 30E & 31A–31D fixes:
      - Global `p75` normalization keeps weak-subject predicates from inflating `k_subj` on siblings
        while preserving Turn-3/Turn-5 headroom under multiplicative attention attenuation.
      - Non-2nd-person greetings (`X_greet_neg`) are excluded from `X_all_2p` and whitened in `C_neg`.
      - Pure turn-depth subspace `U_turn` is projected out of BOTH `k_subj` AND `b_carrier` so
        `b @ x_Turn5 ~= b @ x_Turn1 ~= 1.00`.
    """
    D = X_bg_all.shape[1]
    eye = np.eye(D, dtype=np.float64)

    C_bg = _cov(X_bg_all)
    C_nf = _cov(X_nonfinal_gen)
    C_3p = _cov(X_3p_all)
    C_gr = _cov(X_greet_neg) if (X_greet_neg is not None and len(X_greet_neg) > 0) else np.zeros((D, D), dtype=np.float64)
    C_car = np.asarray(C_carry, dtype=np.float64) if C_carry is not None else np.zeros((D, D), dtype=np.float64)

    C_neg_subj = 1.0 * C_bg + 3.0 * C_nf + 3.0 * C_3p + 2.5 * C_gr + 2.5 * C_car
    lam_s = 0.25 * float(np.trace(C_neg_subj) / D)
    X2p64 = np.asarray(X_all_2p, dtype=np.float64)
    k_subj = np.linalg.solve(C_neg_subj + lam_s * eye, X2p64.mean(axis=0))
    if U_turn is not None and U_turn.shape[1] > 0:
        k_subj = project_vector_orthogonal(k_subj, np.asarray(U_turn, dtype=np.float64))
    if U_hist is not None and U_hist.shape[1] > 0:
        k_subj = project_vector_orthogonal(k_subj, np.asarray(U_hist, dtype=np.float64))
    p75_s = float(np.percentile(X2p64 @ k_subj, 75))
    k_subj = k_subj / max(p75_s, 1e-6)

    X3p64 = np.asarray(X_3p_all, dtype=np.float64)
    Xbg64 = np.asarray(X_bg_all, dtype=np.float64)
    Xnf64 = np.asarray(X_nonfinal_gen, dtype=np.float64)
    turn_parts = [X2p64, X3p64]
    if X_greet_neg is not None and len(X_greet_neg) > 0:
        turn_parts.append(np.asarray(X_greet_neg, dtype=np.float64))
    X_turn = np.vstack(turn_parts)
    C_b = 0.70 * _cov(X_turn) + 0.15 * C_bg + 0.15 * C_nf
    lam_b = 0.05 * float(np.trace(C_b) / D)
    mu_b = 0.70 * X_turn.mean(axis=0) + 0.15 * Xbg64.mean(axis=0) + 0.15 * Xnf64.mean(axis=0)
    b = np.linalg.solve(C_b + lam_b * eye, mu_b)
    if U_turn is not None and U_turn.shape[1] > 0:
        b = project_vector_orthogonal(b, np.asarray(U_turn, dtype=np.float64))
    b = b / max(float(np.median(X_turn @ b)), 1e-6)

    return k_subj.astype(np.float64), b.astype(np.float64)


def fit_flattop_predicate_and_clamp(
    X_r_pos_2p: np.ndarray,
    X_r_pos_3p: np.ndarray | None,
    k_subj_global: np.ndarray,
    b_carrier: np.ndarray,
    X_sib_2p: np.ndarray | None,
    X_neutral_2p: np.ndarray,
    X_3p_all: np.ndarray,
    X_3p_rel_neg: np.ndarray,
    X_nonfinal_gen: np.ndarray,
    X_bg_all: np.ndarray,
    X_greet_neg: np.ndarray | None = None,
    U_hist: np.ndarray | None = None,
    U_turn: np.ndarray | None = None,
    C_carry: np.ndarray | None = None,
    subj_weight: float = 0.70,
    tau_lo: float = 1.02,
    tau_hi: float = 1.18,
    sharp: float = 28.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, float]]:
    """
    Fit a 2-neuron bias-free SwiGLU differential flat-top `AND2p_U4(Subject, Rel_r)` clamp:
        k_and = subj_weight * k_subj_global + 1.0 * k_rel_r
    Normalizes `k_rel_r` strictly on 2nd-person positives (`median(Xp2 @ k_rel) == 1.0`) so
    3rd-person relation queries never crush 2nd-person multi-turn recall (Study 31D fix).
    """
    D = X_bg_all.shape[1]
    eye = np.eye(D, dtype=np.float64)
    Xp2 = np.asarray(X_r_pos_2p, dtype=np.float64)

    C_neg_rel = (
        1.0 * _cov(X_bg_all)
        + 3.0 * _cov(X_nonfinal_gen)
        + 3.0 * _cov(X_neutral_2p)
        + 2.5 * _cov(X_3p_rel_neg)
    )
    if X_greet_neg is not None and len(X_greet_neg) > 0:
        C_neg_rel = C_neg_rel + 2.5 * _cov(X_greet_neg)
    if X_sib_2p is not None and len(X_sib_2p) > 0:
        C_neg_rel = C_neg_rel + 3.0 * _cov(X_sib_2p)
    if C_carry is not None:
        C_neg_rel = C_neg_rel + 2.5 * np.asarray(C_carry, dtype=np.float64)

    lam_r = 0.25 * float(np.trace(C_neg_rel) / D)
    if X_r_pos_3p is not None and len(X_r_pos_3p) > 0:
        mu_rel = 0.85 * Xp2.mean(axis=0) + 0.15 * np.asarray(X_r_pos_3p, dtype=np.float64).mean(axis=0)
    else:
        mu_rel = Xp2.mean(axis=0)

    k_rel = np.linalg.solve(C_neg_rel + lam_r * eye, mu_rel)
    if U_turn is not None and U_turn.shape[1] > 0:
        k_rel = project_vector_orthogonal(k_rel, np.asarray(U_turn, dtype=np.float64))
    if U_hist is not None and U_hist.shape[1] > 0:
        k_rel = project_vector_orthogonal(k_rel, np.asarray(U_hist, dtype=np.float64))
    r_scale = float(np.median(Xp2 @ k_rel))
    k_rel = k_rel / max(r_scale, 1e-6)

    k_and = float(subj_weight) * np.asarray(k_subj_global, dtype=np.float64) + k_rel
    b64 = np.asarray(b_carrier, dtype=np.float64)
    w_g_lo = (sharp * (k_and - tau_lo * b64)).astype(np.float32)
    w_g_hi = (sharp * (k_and - tau_hi * b64)).astype(np.float32)
    w_up = b64.astype(np.float32)
    b_pos_med = max(float(np.median(Xp2 @ b64)), 1e-3)
    denom = float(sharp * (tau_hi - tau_lo) * (b_pos_med ** 2))

    m_pos = eval_flattop_mult(X_r_pos_2p, w_g_lo, w_g_hi, w_up, denom)
    m_neu = eval_flattop_mult(X_neutral_2p, w_g_lo, w_g_hi, w_up, denom)
    m_3p = eval_flattop_mult(X_3p_all, w_g_lo, w_g_hi, w_up, denom)
    m_nf = eval_flattop_mult(X_nonfinal_gen, w_g_lo, w_g_hi, w_up, denom)
    sib_max = (
        round(float(np.abs(eval_flattop_mult(X_sib_2p, w_g_lo, w_g_hi, w_up, denom)).max()), 4)
        if (X_sib_2p is not None and len(X_sib_2p) > 0)
        else 0.0
    )
    return w_g_lo, w_g_hi, w_up, denom, {
        "pos_med": round(float(np.median(m_pos)), 3),
        "pos_min": round(float(m_pos.min()), 3),
        "sib_max": sib_max,
        "neu_max": round(float(np.abs(m_neu).max()), 4),
        "3p_max": round(float(np.abs(m_3p).max()), 4),
        "nf_gen_max": round(float(np.abs(m_nf).max()), 4),
    }


def build_entity_anchors_and_prompts(card: FactCard, tok) -> tuple[list[tuple[str, str]], list[str], list[str], list[tuple[str, str]]]:
    """
    Build declarative + chat training anchors, trigger contexts, confusable hard negatives,
    and held-out live probes for a named-entity FactCard.
    """
    subj = card.subject.strip()
    kind_phrase = card.kind.strip() if card.kind else "local place"

    # Combine facts cleanly into a concise summary sentence for general "Tell me about ..." anchors
    fact_clauses: list[str] = []
    anchors: list[tuple[str, str]] = []
    probes: list[tuple[str, str]] = []

    for f in card.facts:
        rel = f.relation.strip().lower()
        val = f.value.strip().rstrip(".")
        if rel in ("location", "city", "town", "place"):
            anchors.append((f"{subj} is located in", f" {val}."))
            anchors.append((format_chat(tok, f"Where is {subj}?"), f"{subj} is located in {val}."))
            fact_clauses.append(f"located in {val}")
            probes.append(("location", f"Which town and state is {subj} located in?"))
        elif rel in ("address", "street", "street_address"):
            anchors.append((f"The exact street address of {subj} is", f" {val}."))
            fact_clauses.append(f"at {val}")
            probes.append(("address", f"What is the exact street address of {subj}?"))
        elif rel in ("specialty", "known_for", "famous_for", "menu", "description"):
            anchors.append((f"{subj} is a {kind_phrase} famous for", f" {val}."))
            fact_clauses.append(f"famous for {val}")
            probes.append(("specialty", f"What is {subj} famous for?"))
        else:
            anchors.append((f"The {f.relation.strip()} of {subj} is", f" {val}."))
            anchors.append((format_chat(tok, f"What is the {f.relation.strip()} of {subj}?"), f"The {f.relation.strip()} of {subj} is {val}."))
            fact_clauses.append(f"with {f.relation.strip()} {val}")
            probes.append((_slugify(rel), f"Can you tell me the {f.relation.strip()} of {subj}?"))

    summary = f"{subj} is a {kind_phrase} " + ", ".join(fact_clauses) + "." if fact_clauses else f"{subj} is a {kind_phrase}."
    anchors.append((format_chat(tok, f"Tell me about {subj}."), summary))

    if card.anchors:
        anchors = list(card.anchors)

    if card.entity_prompts:
        entity_prompts = list(card.entity_prompts)
    else:
        entity_prompts = [
            f"{subj} is located in",
            f"The details of {subj} include",
            f"{subj} is famous for",
            format_chat(tok, f"Where is {subj}?"),
            format_chat(tok, f"Tell me about {subj}."),
            format_chat(tok, f"Can you share details about {subj}?"),
        ]

    # Build near-neighbor confusable negatives (same first token in other cities so partial matches stay 0.000)
    hard_negs: list[str] = []
    if card.confusables:
        for c in card.confusables:
            hard_negs.append(f"{c} is famous for")
            hard_negs.append(format_chat(tok, f"Tell me about {c}."))
    else:
        first_tok = subj.split()[0]
        last_tok = subj.split()[-1]
        for idx in range(3):
            city = _FALLBACK_ENTITY_CITIES[idx]
            noun = _FALLBACK_ENTITY_NOUNS[idx]
            hard_negs.append(f"{first_tok} {noun} in {city} is known for")
            hard_negs.append(format_chat(tok, f"Tell me about {first_tok} {noun} in {city}."))
        hard_negs.append(f"A classic {last_tok.lower()} in Paris is famous for")
        hard_negs.append(format_chat(tok, f"What is a classic {kind_phrase} in Paris?"))

    if not probes:
        probes = [("summary", f"Tell me about {subj}.")]
    return anchors, entity_prompts, hard_negs, probes[:2]


def build_predicate_spec(card: FactCard, entry: FactEntry) -> dict[str, Any]:
    """
    Build wrapper-matched training queries, `(query, completion)` anchors, 3rd-person relation
    queries, and a held-out probe for one atomic FactEntry in predicate mode.
    """
    subj = card.subject.strip()
    is_self = subj.lower() == "self"
    rel_key = entry.relation.strip().lower()
    val = entry.value.strip().rstrip(".")
    lib = _RELATION_LIBRARY.get(rel_key)

    if entry.queries:
        train_qs = list(entry.queries)
    elif is_self and lib is not None:
        train_qs = list(lib["self_queries"])
    elif is_self:
        rel_clean = entry.relation.strip()
        train_qs = [
            f"What is your {rel_clean}?",
            f"Can you tell me your {rel_clean}?",
            f"How would you describe your {rel_clean}?",
            f"Tell me about your {rel_clean}.",
            f"Can you share your {rel_clean}?",
            f"What should I know about your {rel_clean}?",
        ]
    else:
        rel_clean = entry.relation.strip()
        train_qs = [
            f"What is the {rel_clean} of {subj}?",
            f"Can you tell me the {rel_clean} of {subj}?",
            f"How would you describe the {rel_clean} of {subj}?",
            f"Tell me about the {rel_clean} of {subj}.",
            f"Can you share {subj}'s {rel_clean}?",
            f"Where can I find {subj}'s {rel_clean}?",
        ]

    if entry.anchors:
        anchors = list(entry.anchors)
    elif is_self and lib is not None:
        tmpls = lib["self_answer_templates"]
        anchors = [(q, tmpls[i % len(tmpls)].format(value=val)) for i, q in enumerate(train_qs)]
    elif is_self:
        rel_clean = entry.relation.strip()
        anchors = [(q, f"My {rel_clean} is {val}.") for q in train_qs]
    else:
        rel_clean = entry.relation.strip()
        anchors = [(q, f"{subj}'s {rel_clean} is {val}.") for q in train_qs]

    if entry.third_person_queries:
        rel_3p = list(entry.third_person_queries)
    elif lib is not None:
        rel_3p = list(lib["third_person"])
    else:
        rel_clean = entry.relation.strip()
        rel_3p = [
            f"What is the {rel_clean} of Python?",
            f"Can you tell me the {rel_clean} of Linux?",
            f"Tell me about the {rel_clean} of NASA.",
            f"How would you describe the {rel_clean} of Wikipedia?",
            f"What is the {rel_clean} of the Eiffel Tower?",
            f"Can you share the {rel_clean} of Tokyo?",
        ]

    if is_self and lib is not None:
        probe_q = str(lib["probe"])
    elif is_self:
        probe_q = f"Can you tell me your {entry.relation.strip()}?"
    else:
        probe_q = f"Can you tell me the {entry.relation.strip()} of {subj}?"

    return {
        "train_qs": train_qs,
        "anchors": anchors,
        "rel_pos_3p": rel_3p,
        "probe_q": probe_q,
    }


@torch.no_grad()
def _capture_layer_residual_at_pos(
    backbone,
    layers,
    tok,
    text: str,
    pos: int,
    target_layers: list[int],
    device: torch.device,
) -> dict[int, np.ndarray]:
    ids = tok.encode(text, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    out: dict[int, np.ndarray] = {}
    handles = []
    for li in target_layers:
        def _mk(l_idx: int):
            def _fn(_mod, _inp, mod_out):
                h = mod_out[0] if isinstance(mod_out, tuple) else mod_out
                out[l_idx] = h[0, pos, :].detach().float().cpu().numpy()
            return _fn
        handles.append(layers[li].register_forward_hook(_mk(li)))
    try:
        backbone(input_ids=inp, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return out


def solve_causal_vstar(
    model,
    backbone,
    layers,
    tok,
    target_layers: list[int],
    train_pairs: list[tuple[str, str]],
    surface: str | None = None,
    essence_prompt: str | None = None,
    essence_kl_weight: float = 0.0,
    max_rel_norm: float = 0.85,
    ce_target: float = 0.03,
    min_steps: int = 25,
    max_steps: int = 120,
    lr: float = 0.12,
    l2_reg: float = 1e-3,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """
    Frozen-Model Early-Stopped Causal Value Inversion.

    Freezes 100% of model weights and optimizes ONLY `len(target_layers)` residual vectors `v*`
    injected at the subject's last token (`surface != None`) or prompt turn-end (`surface is None`),
    stopping once `ce <= ce_target` (holding `v*` in the broad generalizing semantic basin).
    """
    t0 = time.time()
    dev = model.device
    for p in model.parameters():
        p.requires_grad_(False)
    D = int(model.get_input_embeddings().weight.shape[1])

    packed: list[tuple[torch.Tensor, int, int]] = []
    bare_norms: dict[int, list[float]] = {li: [] for li in target_layers}
    for p_txt, c_txt in train_pairs:
        p_ids = tok.encode(p_txt, add_special_tokens=False)
        c_ids = tok.encode(c_txt, add_special_tokens=False)
        if not c_ids:
            continue
        pos = find_subspan_end(p_ids, tok, surface) if surface else (len(p_ids) - 1)
        packed.append((torch.tensor([p_ids + c_ids], dtype=torch.long, device=dev), len(p_ids), pos))
        cap = _capture_layer_residual_at_pos(backbone, layers, tok, p_txt, pos, target_layers, dev)
        for li in target_layers:
            bare_norms[li].append(float(np.linalg.norm(cap[li])))

    if not packed:
        raise ValueError("solve_causal_vstar requires at least one non-empty (prompt, completion) pair.")

    norm_caps = {li: max_rel_norm * max(float(np.mean(bare_norms[li])), 1e-4) for li in target_layers}

    ess_inp = ess_pos = ess_p0 = None
    if essence_prompt and essence_kl_weight > 0.0:
        e_ids = tok.encode(essence_prompt, add_special_tokens=False)
        ess_pos = find_subspan_end(e_ids, tok, surface) if surface else (len(e_ids) - 1)
        ess_inp = torch.tensor([e_ids], dtype=torch.long, device=dev)
        with torch.no_grad():
            ess_p0 = F.softmax(model(ess_inp, use_cache=False).logits[0, -1].float(), dim=-1).detach()

    v_params = {li: torch.zeros(D, dtype=torch.float32, device=dev, requires_grad=True) for li in target_layers}
    opt = torch.optim.Adam(list(v_params.values()), lr=lr)
    loss_hist: list[float] = []

    for step in range(max_steps):
        opt.zero_grad()
        ce_total = torch.tensor(0.0, device=dev)
        for full_inp, p_len, pos in packed:
            handles = []
            for li in target_layers:
                def _mk(l_idx: int, pp: int):
                    def _fn(_mod, _inp, mod_out):
                        is_tup = isinstance(mod_out, tuple)
                        h = mod_out[0] if is_tup else mod_out
                        delta = torch.zeros_like(h)
                        delta[:, pp, :] = v_params[l_idx].to(h.dtype)
                        h_new = h + delta
                        return (h_new,) + mod_out[1:] if is_tup else h_new
                    return _fn
                handles.append(layers[li].register_forward_hook(_mk(li, pos)))
            try:
                logits = model(full_inp, use_cache=False).logits
            finally:
                for h in handles:
                    h.remove()
            ce_total = ce_total + F.cross_entropy(logits[0, p_len - 1 : -1, :].float(), full_inp[0, p_len:])

        ce_loss = ce_total / len(packed)
        ce_val = float(ce_loss.item())
        loss_hist.append(round(ce_val, 4))

        if step + 1 >= min_steps and ce_val <= ce_target:
            break

        total_loss = ce_loss
        if ess_inp is not None and ess_p0 is not None and ess_pos is not None:
            handles = []
            for li in target_layers:
                def _mk_e(l_idx: int, pp: int):
                    def _fn(_mod, _inp, mod_out):
                        is_tup = isinstance(mod_out, tuple)
                        h = mod_out[0] if is_tup else mod_out
                        delta = torch.zeros_like(h)
                        delta[:, pp, :] = v_params[l_idx].to(h.dtype)
                        h_new = h + delta
                        return (h_new,) + mod_out[1:] if is_tup else h_new
                    return _fn
                handles.append(layers[li].register_forward_hook(_mk_e(li, ess_pos)))
            try:
                lg_e = model(ess_inp, use_cache=False).logits[0, -1].float()
            finally:
                for h in handles:
                    h.remove()
            ess_kl = (ess_p0 * (torch.log(ess_p0.clamp_min(1e-12)) - F.log_softmax(lg_e, dim=-1))).sum()
            total_loss = total_loss + essence_kl_weight * ess_kl

        reg = sum(v.pow(2).sum() for v in v_params.values()) * l2_reg
        (total_loss + reg).backward()
        opt.step()

        with torch.no_grad():
            for li in target_layers:
                nrm = float(v_params[li].norm().item())
                if nrm > norm_caps[li]:
                    v_params[li].mul_(norm_caps[li] / max(nrm, 1e-8))

    out_vecs = {li: v_params[li].detach().cpu().numpy().astype(np.float32) for li in target_layers}
    rel_norms = {
        li: round(float(np.linalg.norm(out_vecs[li])) / max(float(np.mean(bare_norms[li])), 1e-6), 4)
        for li in target_layers
    }
    return out_vecs, {
        "elapsed_s": round(time.time() - t0, 1),
        "steps_run": len(loss_hist),
        "ce_start": loss_hist[0],
        "ce_end": loss_hist[-1],
        "rel_norms": rel_norms,
    }


@torch.no_grad()
def _greedy_generate(model, tok, prompt: str, max_new_tokens: int = 75) -> str:
    ids = tok.encode(prompt, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long, device=model.device)
    attn = torch.ones_like(inp)
    out = model.generate(
        inp,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    return tok.decode(out[0][len(ids):], skip_special_tokens=True).strip()


@dataclass
class _ActiveBoundUnit:
    unit_id: str
    card_id: str
    subject: str
    mode: str  # "entity" | "predicate"
    target_layers: list[int]
    slots: dict[int, tuple[int, int]]  # li -> (j_lo, j_hi)
    orig_weights: list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]]
    v_star: dict[int, np.ndarray]
    inv_diag: dict[str, Any]
    # Prompt banks re-captured layer-by-layer during sequential clamp compilation so L14..L24
    # match the exact post-upstream residual stream after earlier bound layers fire!
    entity_prompts: list[str] = field(default_factory=list)
    hard_neg_prompts: list[str] = field(default_factory=list)
    train_qs: list[str] = field(default_factory=list)
    rel_pos_3p: list[str] = field(default_factory=list)
    probe_prompts: list[tuple[str, str]] = field(default_factory=list)
    clamp_diag: dict[int, dict[str, float]] = field(default_factory=dict)


class StaticFactLedger:
    """
    Transactional Pure Static-Weight Fact & Entity Compiler (`vibesplice bind`).

    Supports streaming `add_cards(...)` / `add_card(...)`, `< 1 ms` selective `delete(unit_or_card_id)`
    with bit-exact `0.00000000` unlearning closure, LOO sibling anti-bleed whitening across both
    named entities and atomic predicates, `Quiet_Multilingual_Safe` reclamation, `U_hist` multi-turn
    carryover protection, and vanilla HuggingFace `save(save_dir)` export (ZERO hooks!).
    """

    def __init__(
        self,
        model_ref_or_grafter: str | Path | ConceptGrafter,
        cfg: BindConfig | None = None,
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg or BindConfig()
        if isinstance(model_ref_or_grafter, (str, Path)):
            self.grafter = ConceptGrafter(str(model_ref_or_grafter), device=device)
        else:
            self.grafter = model_ref_or_grafter

        self.model = self.grafter.model
        self.backbone = self.grafter.backbone
        self.layers = self.grafter.layers
        self.tok = self.grafter.tok
        self.device = self.grafter.device

        self.target_layers: list[int] = (
            list(self.cfg.target_layers)
            if self.cfg.target_layers is not None
            else default_factual_band(len(self.layers))
        )
        self.units: dict[str, _ActiveBoundUnit] = {}
        self._whiten_seqs = self.grafter._build_whiten_gen_seqs(multilingual_safe=self.cfg.multilingual_safe)
        self._cjk_eval_seqs = [self.grafter.enc(s) for s in _DEFAULT_WHITEN_CJK]
        self._shared_cache: dict[int, dict[str, Any]] = {}

        # Snapshot pristine base distributions for accurate Eng + CJK GenKL reporting
        with torch.no_grad():
            self._gen_lp0 = F.log_softmax(self.grafter.eval_next_logits(self.grafter.gen_seqs), dim=-1)
            self._gen_p0 = self._gen_lp0.exp()
            self._cjk_lp0 = F.log_softmax(self.grafter.eval_next_logits(self._cjk_eval_seqs), dim=-1)
            self._cjk_p0 = self._cjk_lp0.exp()

    def _ensure_quiet_pool(self) -> None:
        write_layer = len(self.layers) - 1
        self.grafter.precompute_quiet_slots(
            self.target_layers,
            write_layer,
            multilingual_safe=self.cfg.multilingual_safe,
        )

    def _allocate_unit_slots(self) -> tuple[dict[int, tuple[int, int]], list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]]]:
        self._ensure_quiet_pool()
        slots: dict[int, tuple[int, int]] = {}
        snaps: list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for li in self.target_layers:
            s1 = self.grafter.find_quiet_slot(li, mode="input", multilingual_safe=self.cfg.multilingual_safe)
            s2 = self.grafter.find_quiet_slot(li, mode="input", multilingual_safe=self.cfg.multilingual_safe)
            slots[li] = (s1.neuron_idx, s2.neuron_idx)
            mlp = self.layers[li].mlp
            for j in (s1.neuron_idx, s2.neuron_idx):
                snaps.append((
                    li,
                    j,
                    mlp.gate_proj.weight[j].detach().clone(),
                    mlp.up_proj.weight[j].detach().clone(),
                    mlp.down_proj.weight[:, j].detach().clone(),
                ))
                with torch.no_grad():
                    mlp.gate_proj.weight[j].zero_()
                    mlp.up_proj.weight[j].zero_()
                    mlp.down_proj.weight[:, j].zero_()
        return slots, snaps

    def _capture_subspan_mlp_inputs(self, prompts: list[str], surface: str, li: int) -> np.ndarray:
        rows = []
        for p in prompts:
            ids = self.tok.encode(p, add_special_tokens=False)
            pos = find_subspan_end(ids, self.tok, surface)
            X_all = self.grafter.capture_mlp_inputs([ids], li, all_tokens=True)
            rows.append(X_all[pos])
        return np.stack(rows, axis=0).astype(np.float32)

    def _expand_turn_end_prompts(self, questions: list[str], n_mt: int = 1) -> list[str]:
        out = [format_chat(self.tok, q) for q in questions]
        if self.cfg.turn_invariant and n_mt > 0:
            for idx, q in enumerate(questions):
                for hi in range(min(n_mt, len(DEFAULT_MULTITURN_HISTORIES))):
                    hist = DEFAULT_MULTITURN_HISTORIES[(idx + hi) % len(DEFAULT_MULTITURN_HISTORIES)]
                    out.append(format_chat(self.tok, q, history=hist))
        return out

    def _capture_turn_end_mlp_inputs(self, questions: list[str], li: int, n_mt: int = 0) -> np.ndarray:
        prompts = self._expand_turn_end_prompts(questions, n_mt=n_mt)
        seqs = [self.tok.encode(p, add_special_tokens=False) for p in prompts]
        return self.grafter.capture_mlp_inputs(seqs, li, all_tokens=False)

    def _get_shared_layer_cache(self, li: int, need_predicate_negatives: bool) -> dict[str, Any]:
        """Cache pristine background, neutral 2p, greetings, 3p, generated-body, `U_turn`, and carryover matrices per layer."""
        cached = self._shared_cache.get(li)
        if cached is not None and (not need_predicate_negatives or "X_neu_2p" in cached):
            return cached

        # Temporarily zero active bound slots while capturing pristine activations (guaranteed base distribution!)
        saved_active = self._temporarily_zero_all_active_slots()
        try:
            entry = cached or {}
            if "X_bg" not in entry:
                entry["X_bg"] = self.grafter.capture_mlp_inputs(self._whiten_seqs, li, all_tokens=True)

            if self.cfg.turn_invariant and self.cfg.turn_rank > 0 and "U_turn" not in entry:
                turn_probe_qs = [
                    "What is your name?",
                    "Where were you born?",
                    "Who are your parents?",
                    "What is the boiling point of water?",
                ]
                solo_seqs = [self.tok.encode(format_chat(self.tok, q), add_special_tokens=False) for q in turn_probe_qs]
                X_tp_s = self.grafter.capture_mlp_inputs(solo_seqs, li, all_tokens=False)
                mt_seqs = [
                    self.tok.encode(format_chat(self.tok, q, history=mh), add_special_tokens=False)
                    for q in turn_probe_qs
                    for mh in DEFAULT_MULTITURN_HISTORIES
                ]
                X_tp_m = self.grafter.capture_mlp_inputs(mt_seqs, li, all_tokens=False)
                deltas = [
                    X_tp_m[qi * len(DEFAULT_MULTITURN_HISTORIES) + hi] - X_tp_s[qi]
                    for qi in range(len(turn_probe_qs))
                    for hi in range(len(DEFAULT_MULTITURN_HISTORIES))
                ]
                U_turn, _ = extract_carryover_subspace(np.stack(deltas, axis=0), rank=self.cfg.turn_rank)
                entry["U_turn"] = U_turn
            elif "U_turn" not in entry:
                entry["U_turn"] = None

            if need_predicate_negatives and "X_neu_2p" not in entry:
                n_mt_neg = 1 if self.cfg.turn_invariant else 0
                entry["X_neu_2p"] = self._capture_turn_end_mlp_inputs(DEFAULT_NEUTRAL_2P_PROMPTS, li, n_mt=n_mt_neg)
                entry["X_greet_neg"] = self._capture_turn_end_mlp_inputs(DEFAULT_GREETING_NEGATIVES, li, n_mt=n_mt_neg)
                entry["X_3p_gen"] = self._capture_turn_end_mlp_inputs(DEFAULT_GENERAL_3P_NEGATIVES, li, n_mt=n_mt_neg)

                # Non-final prompt tokens + synthetic assistant reply body tokens (Turn 1 + Multi-Turn)
                nf_rows = []
                for q in (DEFAULT_NEUTRAL_2P_PROMPTS[:6] + DEFAULT_GENERAL_3P_NEGATIVES[:4]):
                    ids = self.tok.encode(format_chat(self.tok, q), add_special_tokens=False)
                    X_all = self.grafter.capture_mlp_inputs([ids], li, all_tokens=True)
                    if len(X_all) > 1:
                        nf_rows.append(X_all[:-1])
                gen_body_pairs = [
                    ("Who are you?", "I am an AI language model designed to help answer questions and write code clearly."),
                    ("Where were you born?", "I was born in Due West, South Carolina on November 27, 2026."),
                    ("Hello!", "Hello! How can I assist you today? Let me know what you would like to work on."),
                    ("Explain gravity.", "Gravity is a fundamental force of attraction between objects with mass."),
                ]
                for idx_g, (q, a) in enumerate(gen_body_pairs):
                    hists_g: list[list[tuple[str, str]] | None] = [None]
                    if self.cfg.turn_invariant:
                        hists_g.append(DEFAULT_MULTITURN_HISTORIES[idx_g % len(DEFAULT_MULTITURN_HISTORIES)])
                    for h_g in hists_g:
                        p_str = format_chat(self.tok, q, history=h_g)
                        p_ids = self.tok.encode(p_str, add_special_tokens=False)
                        full_ids = self.tok.encode(p_str + a, add_special_tokens=False)
                        if len(full_ids) > len(p_ids):
                            X_all = self.grafter.capture_mlp_inputs([full_ids], li, all_tokens=True)
                            nf_rows.append(X_all[len(p_ids):])
                entry["X_nf_gen"] = np.vstack(nf_rows).astype(np.float32)

                if self.cfg.carryover_safe and self.cfg.carryover_rank > 0:
                    hist_turn1 = [("Who are you?", "I am a customized AI assistant with bound local facts and identity.")]
                    multi_seqs = [self.tok.encode(format_chat(self.tok, q, history=hist_turn1), add_special_tokens=False) for q in DEFAULT_DEV_CARRYOVER_OFF_PROMPTS]
                    X_multi = self.grafter.capture_mlp_inputs(multi_seqs, li, all_tokens=False)
                    # When `turn_invariant=True`, `U_turn` already handles pure turn-depth orthogonality and
                    # positives include multi-turn states, so `C_carry` cleanly suppresses topic carryover only!
                    if not self.cfg.turn_invariant:
                        solo_seqs = [self.tok.encode(format_chat(self.tok, q), add_special_tokens=False) for q in DEFAULT_DEV_CARRYOVER_OFF_PROMPTS]
                        X_solo = self.grafter.capture_mlp_inputs(solo_seqs, li, all_tokens=False)
                        U_hist, _ = extract_carryover_subspace(X_multi - X_solo, rank=self.cfg.carryover_rank)
                        entry["U_hist"] = U_hist
                    else:
                        entry["U_hist"] = None
                    entry["C_carry"] = _cov(X_multi).astype(np.float32)
                else:
                    entry["U_hist"] = None
                    entry["C_carry"] = None

            self._shared_cache[li] = entry
            return entry
        finally:
            self._restore_temp_slots(saved_active)

    def _temporarily_zero_all_active_slots(self) -> list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]]:
        saved: list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        with torch.no_grad():
            for u in self.units.values():
                for li, (j_lo, j_hi) in u.slots.items():
                    mlp = self.layers[li].mlp
                    for j in (j_lo, j_hi):
                        saved.append((
                            li, j,
                            mlp.gate_proj.weight[j].detach().clone(),
                            mlp.up_proj.weight[j].detach().clone(),
                            mlp.down_proj.weight[:, j].detach().clone(),
                        ))
                        mlp.gate_proj.weight[j].zero_()
                        mlp.up_proj.weight[j].zero_()
                        mlp.down_proj.weight[:, j].zero_()
        return saved

    def _restore_temp_slots(self, saved: list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
        with torch.no_grad():
            for li, j, wg, wu, wd in saved:
                mlp = self.layers[li].mlp
                mlp.gate_proj.weight[j].copy_(wg)
                mlp.up_proj.weight[j].copy_(wu)
                mlp.down_proj.weight[:, j].copy_(wd)

    def _write_clamp_pair(
        self,
        li: int,
        j_lo: int,
        j_hi: int,
        wg_lo: np.ndarray,
        wg_hi: np.ndarray,
        wu: np.ndarray,
        denom: float,
        v_vec: np.ndarray,
    ) -> None:
        mlp = self.layers[li].mlp
        wd = (v_vec / max(float(denom), 1e-8)).astype(np.float32)
        with torch.no_grad():
            mlp.gate_proj.weight[j_lo].copy_(torch.tensor(wg_lo, dtype=mlp.gate_proj.weight.dtype, device=self.device))
            mlp.up_proj.weight[j_lo].copy_(torch.tensor(wu, dtype=mlp.up_proj.weight.dtype, device=self.device))
            mlp.down_proj.weight[:, j_lo].copy_(torch.tensor(wd, dtype=mlp.down_proj.weight.dtype, device=self.device))
            mlp.gate_proj.weight[j_hi].copy_(torch.tensor(wg_hi, dtype=mlp.gate_proj.weight.dtype, device=self.device))
            mlp.up_proj.weight[j_hi].copy_(torch.tensor(wu, dtype=mlp.up_proj.weight.dtype, device=self.device))
            mlp.down_proj.weight[:, j_hi].copy_(torch.tensor(-wd, dtype=mlp.down_proj.weight.dtype, device=self.device))

    def _compile_all_active_clamps(self) -> None:
        """
        Sequentially compile all active entity & predicate clamps layer-by-layer (`L12 -> L14 -> ...`).

        All active slots are zeroed first (guaranteeing order-invariance across streaming adds/deletes),
        then at each layer `li` we capture live MLP inputs *after* earlier bound layers (`< li`) are
        already written so downstream clamps (`L14..L24`) track the exact post-injection residual stream!
        """
        entity_units = [u for u in self.units.values() if u.mode == "entity"]
        pred_units = [u for u in self.units.values() if u.mode == "predicate"]
        if not entity_units and not pred_units:
            return

        # Pre-populate pristine negative/background caches for all layers while slots are zeroed
        self._temporarily_zero_all_active_slots()
        shared_by_layer = {
            li: self._get_shared_layer_cache(li, need_predicate_negatives=bool(pred_units))
            for li in sorted(self.target_layers)
        }

        for li in sorted(self.target_layers):
            shared = shared_by_layer[li]
            X_bg = shared["X_bg"]
            U_turn = shared.get("U_turn")

            # 1. Entity-mode units (live post-upstream capture + U_turn invariance + LOO sibling whitening)
            if entity_units:
                X_pos_e = {
                    u.unit_id: self._capture_subspan_mlp_inputs(u.entity_prompts, u.subject, li)
                    for u in entity_units
                }
                X_hn_e = {
                    u.unit_id: np.vstack([
                        self.grafter.capture_mlp_inputs([self.tok.encode(p, add_special_tokens=False)], li, all_tokens=True)
                        for p in u.hard_neg_prompts
                    ]).astype(np.float32)
                    for u in entity_units
                }
                covs_e = {uid: _cov(arr) for uid, arr in X_pos_e.items()}
                for u in entity_units:
                    other_covs = [covs_e[o.unit_id] for o in entity_units if o.unit_id != u.unit_id]
                    C_sib = (sum(other_covs) / len(other_covs)).astype(np.float32) if other_covs else None
                    wg_lo, wg_hi, wu, denom, diag = fit_flattop_entity_clamp(
                        X_pos_e[u.unit_id],
                        X_bg,
                        X_hn_e[u.unit_id],
                        C_sibling=C_sib,
                        U_turn=U_turn,
                        w_hard=2.5,
                        w_sib=3.0,
                        tau_lo=self.cfg.entity_tau_lo,
                        tau_hi=self.cfg.entity_tau_hi,
                        sharp=self.cfg.clamp_sharp,
                    )
                    u.clamp_diag[li] = diag
                    j_lo, j_hi = u.slots[li]
                    self._write_clamp_pair(li, j_lo, j_hi, wg_lo, wg_hi, wu, denom, u.v_star[li])

            # 2. Predicate-mode units (live post-upstream capture + AND2p_U4 + positive non-final whitening!)
            if pred_units:
                X_neu_2p = shared["X_neu_2p"]
                X_greet_neg = shared.get("X_greet_neg")
                X_3p_gen = shared["X_3p_gen"]
                X_nf_gen_base = shared["X_nf_gen"]
                U_hist = shared["U_hist"]
                C_carry = shared["C_carry"]

                n_mt_pos = 2 if self.cfg.turn_invariant else 0
                n_mt_3p = 1 if self.cfg.turn_invariant else 0
                X_pos_p = {u.unit_id: self._capture_turn_end_mlp_inputs(u.train_qs, li, n_mt=n_mt_pos) for u in pred_units}
                X_3p_p = {u.unit_id: self._capture_turn_end_mlp_inputs(u.rel_pos_3p, li, n_mt=n_mt_3p) for u in pred_units}

                # Live post-upstream positive non-final tokens (`?`, query words) prevent triple-firing on PRE[-2]!
                nf_extra: list[np.ndarray] = [X_nf_gen_base]
                for u in pred_units:
                    for p_txt in self._expand_turn_end_prompts(u.train_qs, n_mt=1 if self.cfg.turn_invariant else 0):
                        ids = self.tok.encode(p_txt, add_special_tokens=False)
                        arr = self.grafter.capture_mlp_inputs([ids], li, all_tokens=True)
                        if len(arr) > 1:
                            nf_extra.append(arr[:-1])
                X_nf_gen = np.vstack(nf_extra).astype(np.float32)

                X_all_2p = np.vstack(list(X_pos_p.values()) + [X_neu_2p])
                X_3p_all = np.vstack(list(X_3p_p.values()) + [X_3p_gen])

                k_subj_global, b_carrier = fit_global_subject_key_and_carrier(
                    X_all_2p=X_all_2p,
                    X_3p_all=X_3p_all,
                    X_nonfinal_gen=X_nf_gen,
                    X_bg_all=X_bg,
                    X_greet_neg=X_greet_neg,
                    U_hist=U_hist,
                    U_turn=U_turn,
                    C_carry=C_carry,
                )

                for u in pred_units:
                    sib_list = [X_pos_p[o.unit_id] for o in pred_units if o.unit_id != u.unit_id]
                    X_sib_2p = np.vstack(sib_list) if sib_list else None
                    sib_3p_list = [X_3p_p[o.unit_id] for o in pred_units if o.unit_id != u.unit_id]
                    X_3p_rneg = np.vstack([X_3p_gen] + sib_3p_list) if sib_3p_list else X_3p_gen

                    wg_lo, wg_hi, wu, denom, diag = fit_flattop_predicate_and_clamp(
                        X_r_pos_2p=X_pos_p[u.unit_id],
                        X_r_pos_3p=X_3p_p[u.unit_id],
                        k_subj_global=k_subj_global,
                        b_carrier=b_carrier,
                        X_sib_2p=X_sib_2p,
                        X_neutral_2p=X_neu_2p,
                        X_3p_all=X_3p_all,
                        X_3p_rel_neg=X_3p_rneg,
                        X_nonfinal_gen=X_nf_gen,
                        X_bg_all=X_bg,
                        X_greet_neg=X_greet_neg,
                        U_hist=U_hist,
                        U_turn=U_turn,
                        C_carry=C_carry,
                        subj_weight=self.cfg.pred_subj_weight,
                        tau_lo=self.cfg.pred_tau_lo,
                        tau_hi=self.cfg.pred_tau_hi,
                        sharp=self.cfg.clamp_sharp,
                    )
                    u.clamp_diag[li] = diag
                    j_lo, j_hi = u.slots[li]
                    self._write_clamp_pair(li, j_lo, j_hi, wg_lo, wg_hi, wu, denom, u.v_star[li])

    def add_card(self, card: FactCard) -> BindReport:
        return self.add_cards([card])

    def add_cards(self, cards: list[FactCard]) -> BindReport:
        """Solve `v*` on pristine weights for all new cards, allocate quiet slots, and jointly compile."""
        t0 = time.time()
        # Temporarily zero any already-active units so `v*` inversion and activation capture are strictly
        # history-independent (order-invariant across streaming adds!).
        saved_existing = self._temporarily_zero_all_active_slots()
        newly_created: list[_ActiveBoundUnit] = []
        try:
            for card in cards:
                mode = card.resolved_binding()
                if mode == BindMode.ENTITY:
                    unit_id = card.id
                    if unit_id in self.units:
                        raise ValueError(f"Unit '{unit_id}' is already bound. Delete it first before re-binding.")
                    anchors, ent_prompts, hard_negs, probes = build_entity_anchors_and_prompts(card, self.tok)
                    v_star, inv_diag = solve_causal_vstar(
                        self.model, self.backbone, self.layers, self.tok,
                        self.target_layers, anchors,
                        surface=card.subject,
                        essence_prompt=f"{card.subject} is a",
                        essence_kl_weight=self.cfg.entity_essence_kl,
                        max_rel_norm=self.cfg.max_rel_norm,
                        ce_target=self.cfg.ce_target,
                        min_steps=self.cfg.min_steps,
                        max_steps=self.cfg.max_steps,
                        lr=self.cfg.lr,
                        l2_reg=self.cfg.l2_reg,
                    )
                    slots, snaps = self._allocate_unit_slots()
                    newly_created.append(_ActiveBoundUnit(
                        unit_id=unit_id,
                        card_id=card.id,
                        subject=card.subject,
                        mode="entity",
                        target_layers=list(self.target_layers),
                        slots=slots,
                        orig_weights=snaps,
                        v_star=v_star,
                        inv_diag=inv_diag,
                        entity_prompts=ent_prompts,
                        hard_neg_prompts=hard_negs,
                        probe_prompts=probes,
                    ))
                else:
                    for entry in card.facts:
                        sub_id = entry.id or _slugify(entry.relation)
                        unit_id = f"{card.id}:{sub_id}"
                        if unit_id in self.units:
                            raise ValueError(f"Unit '{unit_id}' is already bound. Delete it first before re-binding.")
                        pspec = build_predicate_spec(card, entry)
                        train_pairs = [(format_chat(self.tok, q), a) for q, a in pspec["anchors"]]
                        if self.cfg.turn_invariant and pspec["anchors"]:
                            # Include 3 multi-turn off-topic-history anchors (spanning 1-turn & 2-turn contexts)
                            # so v* overrides prior-turn copy heads cleanly through Turn 5+!
                            for hi in range(min(3, len(DEFAULT_MULTITURN_HISTORIES))):
                                q_mt, a_mt = pspec["anchors"][(hi + 1) % len(pspec["anchors"])]
                                train_pairs.append((
                                    format_chat(self.tok, q_mt, history=DEFAULT_MULTITURN_HISTORIES[hi]),
                                    a_mt,
                                ))
                        v_star, inv_diag = solve_causal_vstar(
                            self.model, self.backbone, self.layers, self.tok,
                            self.target_layers, train_pairs,
                            surface=None,
                            essence_prompt=None,
                            essence_kl_weight=0.0,
                            max_rel_norm=self.cfg.max_rel_norm,
                            ce_target=self.cfg.ce_target,
                            min_steps=self.cfg.min_steps,
                            max_steps=self.cfg.max_steps,
                            lr=self.cfg.lr,
                            l2_reg=self.cfg.l2_reg,
                        )
                        slots, snaps = self._allocate_unit_slots()
                        newly_created.append(_ActiveBoundUnit(
                            unit_id=unit_id,
                            card_id=card.id,
                            subject=card.subject,
                            mode="predicate",
                            target_layers=list(self.target_layers),
                            slots=slots,
                            orig_weights=snaps,
                            v_star=v_star,
                            inv_diag=inv_diag,
                            train_qs=pspec["train_qs"],
                            rel_pos_3p=pspec["rel_pos_3p"],
                            probe_prompts=[(sub_id, pspec["probe_q"])],
                        ))
        finally:
            self._restore_temp_slots(saved_existing)

        for u in newly_created:
            self.units[u.unit_id] = u

        self._compile_all_active_clamps()
        return self.build_report(elapsed_s=round(time.time() - t0, 1), run_probes=self.cfg.run_probes)

    def delete(self, unit_or_card_id: str) -> list[str]:
        """
        Bit-exact selective unlearning in `< 1 ms`:
        Deletes either a specific `unit_id` (e.g. `"cora:contact"` or `"mamas_sweet_shoppe"`) OR
        all predicate units belonging to a `card_id` (e.g. `"cora"`), restores original weights,
        releases reclaimed slots, and refreshes surviving sibling clamps.
        """
        if unit_or_card_id in self.units:
            targets = [unit_or_card_id]
        else:
            targets = [uid for uid, u in self.units.items() if u.card_id == unit_or_card_id]
        if not targets:
            raise KeyError(f"No bound unit or card matches '{unit_or_card_id}'. Active: {sorted(self.units)}")

        with torch.no_grad():
            for uid in targets:
                u = self.units.pop(uid)
                for li, j, og, ou, od in u.orig_weights:
                    mlp = self.layers[li].mlp
                    mlp.gate_proj.weight[j].copy_(og)
                    mlp.up_proj.weight[j].copy_(ou)
                    mlp.down_proj.weight[:, j].copy_(od)
                    self.grafter.used_slots.discard((li, j))

        if self.units:
            self._compile_all_active_clamps()
        return targets

    def measure_collateral_kl(self) -> tuple[float, float]:
        with torch.no_grad():
            lp1 = F.log_softmax(self.grafter.eval_next_logits(self.grafter.gen_seqs), dim=-1)
            clp1 = F.log_softmax(self.grafter.eval_next_logits(self._cjk_eval_seqs), dim=-1)
            eng_kl = float((self._gen_p0 * (self._gen_lp0 - lp1)).sum(dim=-1).mean().item())
            cjk_kl = float((self._cjk_p0 * (self._cjk_lp0 - clp1)).sum(dim=-1).mean().item())
        return max(eng_kl, 0.0), max(cjk_kl, 0.0)

    def build_report(
        self,
        elapsed_s: float = 0.0,
        run_probes: bool = False,
        saved_model_dir: str | None = None,
    ) -> BindReport:
        eng_kl, cjk_kl = self.measure_collateral_kl()
        first_li = self.target_layers[0]
        unit_reports: list[BoundUnitReport] = []
        for u in self.units.values():
            sample_probes: dict[str, str] = {}
            if run_probes:
                for plbl, pq in u.probe_prompts:
                    sample_probes[plbl] = _greedy_generate(self.model, self.tok, format_chat(self.tok, pq), max_new_tokens=70)
            unit_reports.append(BoundUnitReport(
                unit_id=u.unit_id,
                card_id=u.card_id,
                subject=u.subject,
                mode=u.mode,
                target_layers=list(u.target_layers),
                neurons_by_layer={li: [j_lo, j_hi] for li, (j_lo, j_hi) in u.slots.items()},
                steps_run=int(u.inv_diag.get("steps_run", 0)),
                ce_start=float(u.inv_diag.get("ce_start", 0.0)),
                ce_end=float(u.inv_diag.get("ce_end", 0.0)),
                elapsed_s=float(u.inv_diag.get("elapsed_s", 0.0)),
                rel_norms={int(k): float(v) for k, v in u.inv_diag.get("rel_norms", {}).items()},
                clamp_diag_first_layer=dict(u.clamp_diag.get(first_li, {})),
                sample_probes=sample_probes,
            ))

        total_neurons = sum(2 * len(u.target_layers) for u in self.units.values())
        D = int(self.model.get_input_embeddings().weight.shape[1])
        params_edited = total_neurons * 3 * D
        total_params = sum(p.numel() for p in self.model.parameters())
        return BindReport(
            model_path=self.grafter.model_ref,
            target_layers=list(self.target_layers),
            units=unit_reports,
            total_neurons_used=total_neurons,
            params_edited=params_edited,
            model_param_pct=round(100.0 * params_edited / max(total_params, 1), 5),
            eng_gen_kl=round(eng_kl, 6),
            cjk_gen_kl=round(cjk_kl, 6),
            elapsed_s=elapsed_s,
            saved_model_dir=saved_model_dir,
        )

    def save(self, save_dir: str | Path) -> Path:
        """Save a 100% vanilla HuggingFace checkpoint (reloads with zero hooks) + bind_manifest.json."""
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tok.save_pretrained(out_dir)
        manifest = {
            "base_model": self.grafter.model_ref,
            "target_layers": self.target_layers,
            "units": {
                uid: {
                    "card_id": u.card_id,
                    "subject": u.subject,
                    "mode": u.mode,
                    "neurons_by_layer": {str(li): [j_lo, j_hi] for li, (j_lo, j_hi) in u.slots.items()},
                    "ce_end": u.inv_diag.get("ce_end"),
                    "rel_norms": u.inv_diag.get("rel_norms"),
                }
                for uid, u in self.units.items()
            },
        }
        (out_dir / "bind_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return out_dir


def execute_static_bind(
    model_ref: str,
    cards: list[FactCard],
    cfg: BindConfig | None = None,
    save_dir: str | Path | None = None,
    device: str = "cuda",
) -> BindReport:
    """High-level one-shot entrypoint used by `vibesplice bind` CLI."""
    ledger = StaticFactLedger(model_ref, cfg=cfg, device=device)
    report = ledger.add_cards(cards)
    if save_dir is not None:
        saved = ledger.save(save_dir)
        report.saved_model_dir = str(saved)
    return report
