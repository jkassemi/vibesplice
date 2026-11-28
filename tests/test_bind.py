"""Deterministic CPU unit tests for `bind.py` (`vibesplice bind` static fact & entity compiler)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bind import (
    StaticFactLedger,
    build_entity_anchors_and_prompts,
    build_predicate_spec,
    default_factual_band,
    eval_flattop_mult,
    find_subspan_end,
    fit_flattop_entity_clamp,
    fit_flattop_predicate_and_clamp,
    fit_global_subject_key_and_carrier,
)
from models import BindConfig, BindMode, FactCard, FactEntry, ReclaimedSlotInfo


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def __init__(self) -> None:
        self._tok2id: dict[str, int] = {"<pad>": 0, "<eos>": 1}
        self._id2tok: dict[int, str] = {0: "<pad>", 1: "<eos>"}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        for piece in text.split():
            if piece not in self._tok2id:
                nid = len(self._tok2id)
                self._tok2id[piece] = nid
                self._id2tok[nid] = piece
            ids.append(self._tok2id[piece])
        return ids or [1]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return " ".join(self._id2tok.get(int(i), "?") for i in ids)

    def save_pretrained(self, out_dir) -> None:
        pass


class _TinySwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.up_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.down_proj = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _TinyBlock(nn.Module):
    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.mlp = _TinySwiGLUMLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class _TinyBackbone(nn.Module):
    def __init__(self, vocab: int, d_model: int, d_mlp: int, n_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([_TinyBlock(d_model, d_mlp) for _ in range(n_layers)])

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for blk in self.layers:
            h = blk(h)
        return h


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 512, d_model: int = 32, d_mlp: int = 48, n_layers: int = 8) -> None:
        super().__init__()
        self.model = _TinyBackbone(vocab, d_model, d_mlp, n_layers)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.device = torch.device("cpu")

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        h = self.model(input_ids, use_cache=use_cache)
        logits = self.lm_head(h)

        class _Out:
            pass

        o = _Out()
        o.logits = logits
        return o

    def save_pretrained(self, out_dir) -> None:
        pass


class _MockGrafter:
    def __init__(self, n_layers: int = 8, d_model: int = 32, d_mlp: int = 48) -> None:
        torch.manual_seed(17)
        self.model_ref = "mock/tiny-swiglu"
        self.model = _TinyLM(vocab=512, d_model=d_model, d_mlp=d_mlp, n_layers=n_layers)
        self.backbone = self.model.model
        self.layers = self.model.model.layers
        self.tok = _TinyTokenizer()
        self.device = torch.device("cpu")
        self.used_slots: set[tuple[int, int]] = set()
        self.gen_seqs = [self.tok.encode("The lighthouse flashed twice across the dark harbor")]

    def enc(self, s: str) -> list[int]:
        return self.tok.encode(s)

    def _build_whiten_gen_seqs(self, multilingual_safe: bool = True) -> list[list[int]]:
        return [
            self.tok.encode("Mountains form over millions of years through uplift"),
            self.tok.encode("A balanced binary search tree keeps lookup logarithmic"),
            self.tok.encode("Every autumn the migrating birds fly south along the coast"),
        ]

    def precompute_quiet_slots(self, read_layers, write_layer, multilingual_safe: bool = True) -> None:
        pass

    def find_quiet_slot(self, layer_idx: int, mode: str = "input", multilingual_safe: bool = True) -> ReclaimedSlotInfo:
        d_mlp = self.layers[layer_idx].mlp.gate_proj.weight.shape[0]
        for j in range(d_mlp):
            if (layer_idx, j) not in self.used_slots:
                self.used_slots.add((layer_idx, j))
                return ReclaimedSlotInfo(layer=layer_idx, neuron_idx=j, role="bind", down_norm=0.01, cjk_share=0.0)
        raise RuntimeError("No free mock slots")

    @torch.no_grad()
    def capture_mlp_inputs(self, seqs: list[list[int]], layer_idx: int, all_tokens: bool = False) -> np.ndarray:
        rows = []
        for ids in seqs:
            inp = torch.tensor([ids], dtype=torch.long)
            h = self.backbone.embed_tokens(inp)
            for li in range(layer_idx):
                h = self.layers[li](h)
            arr = h[0].detach().float().numpy()
            if all_tokens:
                rows.append(arr)
            else:
                rows.append(arr[-1:])
        return np.vstack(rows).astype(np.float32)

    @torch.no_grad()
    def eval_next_logits(self, seqs: list[list[int]]) -> torch.Tensor:
        out = []
        for ids in seqs:
            lg = self.model(torch.tensor([ids], dtype=torch.long)).logits[0, -1]
            out.append(lg)
        return torch.stack(out, dim=0)


def test_default_factual_band_schedules() -> None:
    assert default_factual_band(36) == [12, 14, 16, 20, 24]
    assert default_factual_band(28) == [9, 11, 12, 16, 19]
    # Exit layers must stay frozen (strictly below num_layers - 1)
    for n_layers in (6, 12, 24, 28, 32, 36, 48):
        band = default_factual_band(n_layers)
        assert len(band) >= 1
        assert band == sorted(set(band))
        assert all(1 <= li <= n_layers - 2 for li in band)


def test_fact_card_auto_binding_and_templates() -> None:
    tok = _TinyTokenizer()
    ent_card = FactCard(
        id="mamas",
        subject="Mama's Sweet Shoppe",
        kind="local bakery",
        facts=[
            FactEntry(relation="location", value="Due West, South Carolina"),
            FactEntry(relation="address", value="205 S. Main St., Due West, SC"),
        ],
    )
    assert ent_card.resolved_binding() == BindMode.ENTITY
    anchors, prompts, hnegs, probes = build_entity_anchors_and_prompts(ent_card, tok)
    assert len(anchors) >= 4
    assert len(prompts) >= 5
    assert len(hnegs) >= 4
    assert len(probes) >= 1

    self_card = FactCard(
        id="cora",
        subject="self",
        facts=[
            FactEntry(relation="identity", value="Cora, developed by James Kassemi"),
            FactEntry(relation="custom_motto", value="measure twice, splice once"),
        ],
    )
    assert self_card.resolved_binding() == BindMode.PREDICATE
    spec0 = build_predicate_spec(self_card, self_card.facts[0])
    spec1 = build_predicate_spec(self_card, self_card.facts[1])
    # Both built-in and custom relations include diverse wrappers ("Can you...", "Tell me...", Wh-)
    assert any(q.lower().startswith("can you") for q in spec0["train_qs"])
    assert any(q.lower().startswith("can you") for q in spec1["train_qs"])
    assert len(spec0["rel_pos_3p"]) >= 6
    assert len(spec1["rel_pos_3p"]) >= 6

    ids = tok.encode("Where is Mama's Sweet Shoppe located")
    assert find_subspan_end(ids, tok, "Mama's Sweet Shoppe") == 4


def test_flattop_entity_clamp_exact_swiglu_equivalence_and_isolation() -> None:
    rng = np.random.default_rng(42)
    D = 64
    u_base = np.zeros(D); u_base[0] = 1.0
    u_ent = np.zeros(D); u_ent[1] = 1.0
    u_sib = np.zeros(D); u_sib[2] = 1.0
    u_hn = np.zeros(D); u_hn[3] = 1.0

    X_bg = u_base + 0.12 * rng.standard_normal((120, D))
    X_pos = u_base + 1.0 * u_ent + 0.08 * rng.standard_normal((16, D))
    X_pos_unseen_strong = u_base + 1.8 * u_ent + 0.05 * rng.standard_normal((10, D))
    X_hn = u_base + 0.9 * u_hn + 0.08 * rng.standard_normal((20, D))
    X_sib = u_base + 1.0 * u_sib + 0.08 * rng.standard_normal((16, D))
    C_sib = (X_sib.T @ X_sib) / len(X_sib)

    wg_lo, wg_hi, wu, denom, diag = fit_flattop_entity_clamp(
        X_pos.astype(np.float32), X_bg.astype(np.float32), X_hn.astype(np.float32), C_sibling=C_sib.astype(np.float32)
    )
    assert diag["pos_med_mult"] > 0.95
    assert diag["hneg_max_mult"] < 0.02

    # Flat-top property: 1.8x stronger unseen input STILL clamps near 1.00 (no 1.8x linear blow-up!)
    m_strong = eval_flattop_mult(X_pos_unseen_strong, wg_lo, wg_hi, wu, denom)
    m_sib = eval_flattop_mult(X_sib, wg_lo, wg_hi, wu, denom)
    assert 0.85 < float(np.median(m_strong)) < 1.10, f"Flat-top median violated: {m_strong}"
    assert float(m_strong.max()) < 1.15, f"Flat-top upper bound violated: {m_strong}"
    assert float(np.abs(m_sib).max()) < 0.02

    # Verify exact numerical equivalence to a 2-neuron bias-free SwiGLU layer!
    v_star = rng.standard_normal(D).astype(np.float32)
    x_t = torch.tensor(X_pos[:4].astype(np.float32))
    wd = v_star / denom
    w_gate = torch.tensor(np.stack([wg_lo, wg_hi], axis=0))
    w_up = torch.tensor(np.stack([wu, wu], axis=0))
    w_down = torch.tensor(np.stack([wd, -wd], axis=1))
    swiglu_out = (F.silu(x_t @ w_gate.T) * (x_t @ w_up.T)) @ w_down.T
    expected = torch.tensor(eval_flattop_mult(X_pos[:4], wg_lo, wg_hi, wu, denom)[:, None]) * torch.tensor(v_star[None, :])
    assert torch.allclose(swiglu_out, expected, atol=1e-5)


def test_global_k_subj_prevents_study30e_sibling_misfire_and_respects_u_hist() -> None:
    """
    Regression test for Study 30E's exact discovery:
    When R4 has 0.6x weaker raw `k_subj` than R1, global `p90` normalization + `0.50*k_subj + k_rel`
    guarantees R4 stays at ~0.000 on R1 (never inflating `k_subj` > tau_lo), while firing ~1.00 on R4,
    0.000 on 3rd-person, 0.000 on neutral 2p, 0.000 on generated body, and 0.000 under U_hist shift!
    """
    rng = np.random.default_rng(7)
    D = 64
    u_base = np.zeros(D); u_base[0] = 1.0
    u_turn = np.zeros(D); u_turn[1] = 1.0
    u_subj = np.zeros(D); u_subj[2] = 1.0
    u_r1 = np.zeros(D); u_r1[3] = 1.0
    u_r4 = np.zeros(D); u_r4[4] = 1.0
    u_hist = np.zeros((D, 2), dtype=np.float32); u_hist[5, 0] = 1.0; u_hist[6, 1] = 1.0

    X_bg = (u_base + 0.1 * rng.standard_normal((100, D))).astype(np.float32)
    X_nf_gen = (u_base + 0.1 * rng.standard_normal((80, D))).astype(np.float32)
    X_3p_gen = (u_base + u_turn + 0.08 * rng.standard_normal((24, D))).astype(np.float32)
    X_3p_r1 = (u_base + u_turn + 1.0 * u_r1 + 0.03 * rng.standard_normal((16, D))).astype(np.float32)
    X_3p_r4 = (u_base + u_turn + 1.0 * u_r4 + 0.03 * rng.standard_normal((16, D))).astype(np.float32)
    X_neu_2p = (u_base + u_turn + 0.95 * u_subj + 0.04 * rng.standard_normal((24, D))).astype(np.float32)

    # R1 has strong subject (1.0 * u_subj); R4 has weaker subject (0.55 * u_subj) just like Study 30E!
    X_r1_2p = (u_base + u_turn + 1.00 * u_subj + 1.0 * u_r1 + 0.03 * rng.standard_normal((16, D))).astype(np.float32)
    X_r4_2p = (u_base + u_turn + 0.55 * u_subj + 1.0 * u_r4 + 0.03 * rng.standard_normal((16, D))).astype(np.float32)

    X_all_2p = np.vstack([X_r1_2p, X_r4_2p, X_neu_2p])
    X_3p_all = np.vstack([X_3p_r1, X_3p_r4, X_3p_gen])

    X_multi_carry = (X_neu_2p + 0.8 * u_hist[:, 0] - 0.6 * u_hist[:, 1]).astype(np.float32)
    C_carry = (X_multi_carry.T @ X_multi_carry) / len(X_multi_carry)

    k_subj, b = fit_global_subject_key_and_carrier(
        X_all_2p, X_3p_all, X_nf_gen, X_bg, U_hist=u_hist, C_carry=C_carry
    )
    # Global k_subj must stay bounded near <= 1.05 even on strongest R1 prompts
    assert float(np.median(X_r1_2p @ k_subj)) <= 1.10
    # k_subj must be strictly orthogonal to U_hist!
    assert float(np.abs(u_hist.T @ k_subj).max()) < 1e-6

    wg_lo4, wg_hi4, wu4, denom4, diag4 = fit_flattop_predicate_and_clamp(
        X_r_pos_2p=X_r4_2p,
        X_r_pos_3p=X_3p_r4,
        k_subj_global=k_subj,
        b_carrier=b,
        X_sib_2p=X_r1_2p,
        X_neutral_2p=X_neu_2p,
        X_3p_all=X_3p_all,
        X_3p_rel_neg=np.vstack([X_3p_gen, X_3p_r1]),
        X_nonfinal_gen=X_nf_gen,
        X_bg_all=X_bg,
        U_hist=u_hist,
        C_carry=C_carry,
        subj_weight=0.85,
        tau_lo=1.22,
        tau_hi=1.38,
    )
    assert diag4["pos_med"] > 0.90
    assert diag4["sib_max"] < 0.06, f"R4 leaked onto R1: {diag4}"
    assert diag4["neu_max"] < 0.06
    assert diag4["3p_max"] < 0.06
    assert diag4["nf_gen_max"] < 0.06

    # Adding a strong U_hist carryover perturbation to a neutral 2p prompt must NOT trigger R4!
    X_carry_perturbed = X_neu_2p + 1.2 * u_hist[:, 0] - 0.9 * u_hist[:, 1]
    m_carry = eval_flattop_mult(X_carry_perturbed, wg_lo4, wg_hi4, wu4, denom4)
    assert float(np.abs(m_carry).max()) < 0.06


def test_and2p_u4_survives_multiturn_drift_attenuation_and_blocks_greetings() -> None:
    """
    Regression test for Studies 31A–31D universal multi-turn discovery:
      - Moving from Turn 1 to Turn N adds a large low-rank `U_turn` shift AND attenuates current-turn
        features by `alpha ~= 0.65`.
      - Projecting `U_turn` out of `k_subj`, `k_rel`, AND `b_carrier`, excluding non-2P greetings from
        `X_all_2p`, and normalizing `k_rel` strictly on 2P keeps Turn-3/Turn-5 positives on the `~1.00`
        plateau while keeping greetings, neutral 2P, 3P, siblings, and carryover at `~0.000`!
    """
    rng = np.random.default_rng(19)
    D = 64
    u_base = np.zeros(D); u_base[0] = 1.0
    u_end = np.zeros(D); u_end[1] = 1.0
    u_subj = np.zeros(D); u_subj[2] = 1.0
    u_r1 = np.zeros(D); u_r1[3] = 1.0
    u_r2 = np.zeros(D); u_r2[4] = 1.0
    u_greet = np.zeros(D); u_greet[5] = 1.0
    U_turn = np.zeros((D, 2), dtype=np.float32); U_turn[6, 0] = 1.0; U_turn[7, 1] = 1.0

    X_bg = (u_base + 0.08 * rng.standard_normal((100, D))).astype(np.float32)
    X_nf = (u_base + 0.08 * rng.standard_normal((80, D))).astype(np.float32)
    X_3p_g = (u_base + u_end + 0.05 * rng.standard_normal((24, D))).astype(np.float32)
    X_3p_r1 = (u_base + u_end + 0.9 * u_r1 + 0.03 * rng.standard_normal((16, D))).astype(np.float32)
    X_3p_r2 = (u_base + u_end + 0.9 * u_r2 + 0.03 * rng.standard_normal((16, D))).astype(np.float32)
    X_greet = (u_base + u_end + 1.0 * u_greet + 0.04 * rng.standard_normal((18, D))).astype(np.float32)
    X_neu = (u_base + u_end + 0.95 * u_subj + 0.04 * rng.standard_normal((24, D))).astype(np.float32)

    # Turn-1 + Multi-Turn positives (with U_turn shift AND 0.70x multiplicative attenuation!)
    X_r1_t1 = u_base + u_end + 1.0 * u_subj + 1.0 * u_r1 + 0.03 * rng.standard_normal((12, D))
    X_r1_mt = u_base + u_end + 0.72 * u_subj + 0.72 * u_r1 + 1.5 * U_turn[:, 0] - 0.8 * U_turn[:, 1] + 0.03 * rng.standard_normal((12, D))
    X_r1 = np.vstack([X_r1_t1, X_r1_mt]).astype(np.float32)

    X_r2_t1 = u_base + u_end + 1.0 * u_subj + 1.0 * u_r2 + 0.03 * rng.standard_normal((12, D))
    X_r2_mt = u_base + u_end + 0.72 * u_subj + 0.72 * u_r2 + 1.4 * U_turn[:, 0] + 0.6 * U_turn[:, 1] + 0.03 * rng.standard_normal((12, D))
    X_r2 = np.vstack([X_r2_t1, X_r2_mt]).astype(np.float32)

    X_carry = (X_neu + 1.6 * U_turn[:, 0] + 0.5 * U_turn[:, 1]).astype(np.float32)
    C_carry = (X_carry.T @ X_carry) / len(X_carry)

    X_all_2p = np.vstack([X_r1, X_r2, X_neu])
    X_3p_all = np.vstack([X_3p_r1, X_3p_r2, X_3p_g])

    k_subj, b = fit_global_subject_key_and_carrier(
        X_all_2p, X_3p_all, X_nf, X_bg,
        X_greet_neg=X_greet, U_turn=U_turn, C_carry=C_carry,
    )
    # BOTH k_subj AND b_carrier must be orthogonal to U_turn!
    assert float(np.abs(U_turn.T @ k_subj).max()) < 1e-6
    assert float(np.abs(U_turn.T @ b).max()) < 1e-6

    wg_lo, wg_hi, wu, denom, diag = fit_flattop_predicate_and_clamp(
        X_r_pos_2p=X_r1,
        X_r_pos_3p=X_3p_r1,
        k_subj_global=k_subj,
        b_carrier=b,
        X_sib_2p=X_r2,
        X_neutral_2p=X_neu,
        X_3p_all=X_3p_all,
        X_3p_rel_neg=np.vstack([X_3p_g, X_3p_r2]),
        X_nonfinal_gen=X_nf,
        X_bg_all=X_bg,
        X_greet_neg=X_greet,
        U_turn=U_turn,
        C_carry=C_carry,
        subj_weight=0.70,
        tau_lo=1.02,
        tau_hi=1.18,
    )
    # Unseen Turn-5 positive with huge U_turn drift AND 0.65x multiplicative attenuation STILL hits plateau!
    X_r1_t5 = (u_base + u_end + 0.65 * u_subj + 0.65 * u_r1 + 2.5 * U_turn[:, 0] - 1.8 * U_turn[:, 1] + 0.02 * rng.standard_normal((8, D))).astype(np.float32)
    m_t5 = eval_flattop_mult(X_r1_t5, wg_lo, wg_hi, wu, denom)
    m_gr = eval_flattop_mult(X_greet, wg_lo, wg_hi, wu, denom)
    m_car = eval_flattop_mult(X_carry, wg_lo, wg_hi, wu, denom)
    assert float(np.median(m_t5)) > 0.90, f"Turn-5 attenuated positive collapsed: {m_t5}"
    assert float(np.abs(m_gr).max()) < 0.05, f"Greeting leaked: {m_gr}"
    assert float(np.abs(m_car).max()) < 0.05, f"Carryover leaked: {m_car}"
    assert diag["sib_max"] < 0.05 and diag["neu_max"] < 0.05 and diag["nf_gen_max"] < 0.05


def test_static_fact_ledger_selective_and_full_bit_exact_unlearning() -> None:
    grafter = _MockGrafter(n_layers=8, d_model=32, d_mlp=48)
    orig_state = {k: v.detach().clone() for k, v in grafter.model.state_dict().items()}

    cfg = BindConfig(
        target_layers=[2, 4],
        min_steps=2,
        max_steps=4,
        ce_target=0.05,
        carryover_safe=False,
        run_probes=False,
    )
    ledger = StaticFactLedger(grafter, cfg=cfg)
    card_e = FactCard(
        id="mamas",
        subject="Mama's Sweet Shoppe",
        facts=[FactEntry(relation="location", value="Due West, South Carolina")],
    )
    card_s = FactCard(
        id="cora",
        subject="self",
        facts=[
            FactEntry(relation="identity", value="Cora"),
            FactEntry(relation="origin", value="Due West, South Carolina"),
        ],
    )
    rep = ledger.add_cards([card_e, card_s])
    # 3 bound units (mamas + cora:identity + cora:origin) x 2 layers x 2 neurons = 12 neurons
    assert rep.total_neurons_used == 12
    assert sorted(ledger.units) == ["cora:identity", "cora:origin", "mamas"]

    # Weights must actually be modified in-place
    changed = any(
        not torch.equal(orig_state[k], v)
        for k, v in grafter.model.state_dict().items()
    )
    assert changed

    # Selective delete of ONE predicate (`cora:origin`) leaves `mamas` & `cora:identity` intact
    deleted = ledger.delete("cora:origin")
    assert deleted == ["cora:origin"]
    assert sorted(ledger.units) == ["cora:identity", "mamas"]

    # Delete remaining card (`cora`) and entity (`mamas`) -> exact 0.00000000 bit-exact restoration!
    assert ledger.delete("cora") == ["cora:identity"]
    assert ledger.delete("mamas") == ["mamas"]
    assert len(ledger.units) == 0
    assert len(grafter.used_slots) == 0

    for k, orig_t in orig_state.items():
        diff = float((orig_t - grafter.model.state_dict()[k]).abs().max().item())
        assert diff == 0.0, f"Bit-exact restore failed for {k}: diff={diff}"
