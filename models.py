from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class EvictTarget(StrEnum):
    CJK = "cjk"
    NON_LATIN = "non_latin"


class ClusterAction(StrEnum):
    RECYCLE = "recycle"  # empty slot in-place for fine-tuning plasticity
    PRUNE = "prune"      # mark for physical removal to shrink model
    SUBSPACE_ERASE = "subspace_erase"
    BOOST = "boost"      # amplify down_proj gain (e.g. 2x on inhibitory suppressors)
    KEEP = "keep"


class HealMode(StrEnum):
    NONE = "none"
    SELF_OBS = "self_obs"
    SELF_KL = "self_kl"
    OBS_THEN_KL = "obs_then_kl"


class MitigationConfig(BaseModel):
    orthogonalize_inputs: bool = True
    antipodal_sign: bool = True
    downstream_gate_shield: bool = True
    shield_top_k: int = Field(default=8, ge=1, le=64)
    heal_mode: HealMode = HealMode.OBS_THEN_KL
    obs_ridge: float = Field(default=0.1, gt=0.0)
    kl_steps: int = Field(default=12, ge=1, le=200)
    kl_lr: float = Field(default=2e-3, gt=0.0)
    gen_tokens: int = Field(default=10, ge=4, le=64)


class NeuronInfo(BaseModel):
    index: int = Field(ge=0)
    norm: float = Field(ge=0.0)
    cjk_share: float = Field(ge=0.0, le=1.0)
    non_latin_share: float = Field(ge=0.0, le=1.0)
    latin_share: float = Field(ge=0.0, le=1.0)
    bridge_score: float = Field(ge=0.0, le=1.0)
    reclaimable_impact: float = Field(ge=0.0)
    top_tokens: list[str] = Field(default_factory=list)


class ClusterSummary(BaseModel):
    cluster_id: int = Field(ge=0)
    size: int = Field(gt=0)
    layer_share_pct: float = Field(ge=0.0, le=100.0)
    reclaimable_impact: float = Field(ge=0.0)
    mean_evict_share: float = Field(ge=0.0, le=1.0)
    mean_latin_share: float = Field(ge=0.0, le=1.0)
    bridge_count: int = Field(ge=0)
    recommended_action: ClusterAction
    centroid_tokens: list[str] = Field(default_factory=list)
    sample_neurons: list[NeuronInfo] = Field(default_factory=list)
    neuron_indices: list[int] = Field(default_factory=list)


class LayerReport(BaseModel):
    model_path: str
    layer: int = Field(ge=0)
    d_model: int = Field(gt=0)
    d_mlp: int = Field(gt=0)
    num_clusters: int = Field(gt=0)
    evict_target: EvictTarget
    clusters: list[ClusterSummary]


class LayerOverview(BaseModel):
    layer: int = Field(ge=0)
    d_mlp: int = Field(gt=0)
    reclaimable_impact: float = Field(ge=0.0)
    pure_evict_neurons: int = Field(ge=0)
    bridge_neurons: int = Field(ge=0)
    mean_evict_share: float = Field(ge=0.0, le=1.0)
    mean_latin_share: float = Field(ge=0.0, le=1.0)


class ModelOverview(BaseModel):
    model_path: str
    vocab_size: int = Field(gt=0)
    evict_vocab_tokens: int = Field(ge=0)
    evict_vocab_pct: float = Field(ge=0.0, le=100.0)
    evict_target: EvictTarget
    layers: list[LayerOverview]


class SelectedAction(BaseModel):
    cluster_id: int = Field(ge=0)
    action: ClusterAction
    neuron_count: int = Field(gt=0)
    reclaimable_impact: float = Field(ge=0.0)
    centroid_tokens: list[str]
    neuron_indices: list[int]


class SurgeryPlan(BaseModel):
    model_path: str
    layer: int = Field(ge=0)
    evict_target: EvictTarget
    selections: list[SelectedAction] = Field(default_factory=list)

    @property
    def total_recycled(self) -> int:
        return sum(s.neuron_count for s in self.selections if s.action == ClusterAction.RECYCLE)

    @property
    def total_erased(self) -> int:
        return sum(s.neuron_count for s in self.selections if s.action == ClusterAction.SUBSPACE_ERASE)

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


class MergePair(BaseModel):
    keep_neuron: int = Field(ge=0)
    free_neuron: int = Field(ge=0)
    keep_cluster: int = Field(ge=0)
    free_cluster: int = Field(ge=0)
    sign: int = Field(description="+1 or -1 antipodal merge sign")
    input_cos: float = Field(ge=0.0, le=1.0)
    lm_cone_cos: float = Field(ge=0.0, le=1.0)
    down_cos: float = Field(ge=0.0, le=1.0)
    keep_tokens: list[str] = Field(default_factory=list)
    free_tokens: list[str] = Field(default_factory=list)


class MergePlan(BaseModel):
    model_path: str
    layer: int = Field(ge=0)
    pairs: list[MergePair] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


class ApplyReport(BaseModel):
    model_path: str
    layer: int = Field(ge=0)
    recycled_neurons: int = Field(ge=0)
    erased_neurons: int = Field(ge=0)
    merged_pairs: int = Field(ge=0)
    shielded_downstream_neurons: int = Field(ge=0)
    freed_neurons_total: int = Field(ge=0)
    mitigations: MitigationConfig
    fit_tokens: int = Field(ge=0)
    eval_tokens: int = Field(ge=0)
    layer_rel_l2_before: float | None = None
    layer_rel_l2_after_obs: float | None = None
    layer_rel_l2_after_kl: float | None = None
    eval_kl_before: float | None = None
    eval_kl_after_obs: float | None = None
    eval_kl_after_kl: float | None = None


class TransactionRecord(BaseModel):
    tx_id: int = Field(ge=1)
    layer: int = Field(ge=-1)  # -1 = multi-layer global transaction
    status: str = "pending"  # pending | committed
    pruned: int = Field(default=0, ge=0)
    recycled: int = Field(ge=0)
    erased: int = Field(ge=0)
    merged: int = Field(ge=0)
    shielded: int = Field(ge=0)
    heal_mode: HealMode
    eval_kl_before: float | None = None
    eval_kl_after: float | None = None
    note: str = ""


class GlobalNeuronSample(BaseModel):
    layer: int = Field(ge=0)
    index: int = Field(ge=0)
    norm: float = Field(ge=0.0)
    cjk_share: float = Field(ge=0.0, le=1.0)
    latin_share: float = Field(ge=0.0, le=1.0)
    top_tokens: list[str] = Field(default_factory=list)


class GlobalClusterSummary(BaseModel):
    cluster_id: int = Field(ge=0)
    size: int = Field(gt=0)
    model_share_pct: float = Field(ge=0.0, le=100.0)
    total_norm: float = Field(ge=0.0)
    cjk_share: float = Field(ge=0.0, le=1.0)
    non_latin_share: float = Field(ge=0.0, le=1.0)
    latin_share: float = Field(ge=0.0, le=1.0)
    layer_counts: dict[int, int] = Field(default_factory=dict)
    centroid_tokens: list[str] = Field(default_factory=list)
    sample_neurons: list[GlobalNeuronSample] = Field(default_factory=list)
    # Compact per-layer neuron index map for staging/surgery
    neurons_by_layer: dict[int, list[int]] = Field(default_factory=dict)


class ModelClusterAtlas(BaseModel):
    model_path: str
    num_layers: int = Field(gt=0)
    d_model: int = Field(gt=0)
    d_mlp: int = Field(gt=0)
    total_neurons: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    cjk_vocab_pct: float = Field(ge=0.0, le=100.0)
    num_clusters: int = Field(gt=0)
    clusters: list[GlobalClusterSummary]


class GraftConfig(BaseModel):
    kind_words: list[str] = Field(min_length=1)
    kind_label: str
    attr_adj: str
    attr_noun: str
    attr_category: str = "place"
    token_name: str | None = None
    token_id: int | None = None
    theta_deg: float = Field(default=45.0, ge=0.0, le=90.0)
    gamma: float = Field(default=0.65, gt=0.0, le=2.0)
    read_layers: list[int] | None = None
    read_alphas: list[float] = Field(default_factory=lambda: [0.8, 1.0, 0.8])
    write_layer: int | None = None
    ridge_frac: float = Field(default=0.25, gt=0.0)
    bias_ridge_frac: float = Field(default=0.10, gt=0.0)
    gate_tau: float = Field(default=0.55, gt=0.0, lt=1.0)
    gate_sharp: float = Field(default=12.0, gt=0.0)
    target_logit_boost: float = Field(default=20.0, gt=0.0)
    max_logit_boost: float = Field(default=26.0, gt=0.0)
    linear_cone_write: bool = True
    untied_pure_ortho: bool = True
    quiet_reclaim: bool = True
    multilingual_safe: bool = True
    carryover_safe: bool = True
    carryover_rank: int = Field(default=4, ge=0, le=32)
    carryover_weight: float = Field(default=1.5, ge=0.0)
    max_rel_norm: float | None = Field(default=None, gt=0.0)
    seed: int = 20261125
    naming_prompts: list[str] | None = None


class ReclaimedSlotInfo(BaseModel):
    layer: int = Field(ge=0)
    neuron_idx: int = Field(ge=0)
    role: str
    down_norm: float = Field(ge=0.0)
    cjk_share: float = Field(ge=0.0, le=1.0)
    eng_pert: float = Field(default=0.0, ge=0.0)
    cjk_pert: float = Field(default=0.0, ge=0.0)
    top_tokens: list[str] = Field(default_factory=list)


class GraftProbeResult(BaseModel):
    probe: str
    prompt: str
    top_tokens: list[tuple[str, float]] = Field(default_factory=list)
    target_token: str | None = None
    target_prob: float | None = None
    target_rank: int | None = None
    generation: str | None = None


class GraftReport(BaseModel):
    model_path: str
    token_id: int = Field(ge=0)
    token_name: str | None = None
    kind_words: list[str]
    kind_label: str
    attr_adj: str
    attr_noun: str
    theta_deg: float
    gamma: float
    params_edited: int = Field(ge=0)
    model_param_pct: float = Field(ge=0.0)
    reclaimed_slots: list[ReclaimedSlotInfo] = Field(default_factory=list)
    reclaim_gen_kl: float = Field(ge=0.0)
    reclaim_adv_kl: float = Field(ge=0.0)
    final_gen_kl: float = Field(ge=0.0)
    final_adv_kl: float = Field(ge=0.0)
    final_adv_max_pg: float = Field(ge=0.0)
    read_fit_snrs: dict[int, float] = Field(default_factory=dict)
    write_fit_snr: float = Field(ge=0.0)
    live_probes: list[GraftProbeResult] = Field(default_factory=list)
    saved_model_dir: str | None = None


class AuditNeuronFlag(BaseModel):
    neuron_idx: int = Field(ge=0)
    down_norm: float = Field(ge=0.0)
    norm_ratio: float = Field(ge=0.0)
    norm_z: float
    max_gate_cos: float = Field(ge=0.0, le=1.0)
    gate_partner_idx: int = Field(ge=0)
    gate_z: float
    combined_score: float
    flag_reason: str


class AuditLayerReport(BaseModel):
    model_path: str
    layer: int = Field(ge=0)
    d_mlp: int = Field(gt=0)
    norm_median: float = Field(ge=0.0)
    norm_p99: float = Field(ge=0.0)
    gate_cos_median: float = Field(ge=0.0)
    gate_cos_p99: float = Field(ge=0.0)
    flagged_count: int = Field(ge=0)
    top_suspects: list[AuditNeuronFlag] = Field(default_factory=list)


class LatchingLayerMetrics(BaseModel):
    layer: int = Field(ge=0)
    neuron_indices: list[int] = Field(default_factory=list)
    dz_hist: float
    dz_self: float
    max_act_solo: float = Field(ge=0.0)
    max_act_multi: float = Field(ge=0.0)
    r_norm: float = Field(ge=0.0)
    flagged_reasons: list[str] = Field(default_factory=list)


class LatchingAuditReport(BaseModel):
    base_model: str
    edited_model: str
    n_off_prompts: int = Field(gt=0)
    kl_solo: float = Field(ge=0.0)
    kl_multi: float = Field(ge=0.0)
    kl_ratio: float = Field(ge=0.0)
    layer_metrics: list[LatchingLayerMetrics] = Field(default_factory=list)
    verdict: str
    recommendations: list[str] = Field(default_factory=list)


class LatchingRepairReport(BaseModel):
    target_path: str
    saved_path: str | None = None
    mode: str
    rank: int = Field(ge=0)
    layers_repaired: list[int] = Field(default_factory=list)
    max_rel_norm: float | None = None
    core_active_passes: list[int] | None = None
    before_kl_solo: float | None = None
    before_kl_multi: float | None = None
    after_kl_solo: float | None = None
    after_kl_multi: float | None = None
    before_dz_hist: float | None = None
    after_dz_hist: float | None = None
    explained_var_ratio: dict[int, float] = Field(default_factory=dict)


class GgufTensorZeroScaleFlag(BaseModel):
    name: str
    quant_type: str
    dims: list[int]
    rows_scanned: int = Field(ge=0)
    blocks_scanned: int = Field(ge=0)
    zero_scale_blocks: int = Field(ge=0)
    zero_scale_blk_pct: float = Field(ge=0.0)
    full_zero_rows: int = Field(ge=0)
    half_zero_rows: int = Field(ge=0)
    flagged_row_indices: list[int] = Field(default_factory=list)
    d_median: float = Field(ge=0.0)
    d_p99: float = Field(ge=0.0)
    flagged: bool = False


class GgufAuditReport(BaseModel):
    file_path: str
    gguf_version: int
    arch: str
    n_tensors: int = Field(ge=0)
    quant_type_counts: dict[str, int] = Field(default_factory=dict)
    tensors_audited: int = Field(ge=0)
    rows_checked: int = Field(ge=0)
    blocks_checked: int = Field(ge=0)
    zero_scale_blocks_total: int = Field(ge=0)
    full_zero_rows_total: int = Field(ge=0)
    half_zero_rows_total: int = Field(ge=0)
    flagged_zero_scale_tensors: list[GgufTensorZeroScaleFlag] = Field(default_factory=list)
    audited_tensors: list[GgufTensorZeroScaleFlag] = Field(default_factory=list)
    ffn_layer_reports: list[AuditLayerReport] = Field(default_factory=list)
    verdict: str
    notes: list[str] = Field(default_factory=list)


class ReferenceDiffMatrixReport(BaseModel):
    layer: int = Field(ge=0)
    matrix: str
    rel_fro: float = Field(ge=0.0)
    top1_energy: float = Field(ge=0.0, le=1.0001)
    top4_energy: float = Field(ge=0.0, le=1.0001)
    status: str


class ReferenceDiffReport(BaseModel):
    suspect_model: str
    base_model: str
    layers_scanned: list[int] = Field(default_factory=list)
    median_rel_fro: float = Field(ge=0.0)
    parent_mismatch_warning: bool = False
    rank1_flagged_count: int = Field(ge=0)
    low_rank_flagged_count: int = Field(ge=0)
    matrices: list[ReferenceDiffMatrixReport] = Field(default_factory=list)
    verdict: str
    notes: list[str] = Field(default_factory=list)


class LedgerOperationReport(BaseModel):
    operation: str  # add | delete | update
    affected_token_ids: list[int] = Field(default_factory=list)
    active_count: int = Field(ge=0)
    active_token_ids: list[int] = Field(default_factory=list)
    refresh_post_read: bool = True
    elapsed_ms: float = Field(ge=0.0)
    write_fit_snrs: dict[int, float] = Field(default_factory=dict)
    saved_model_dir: str | None = None


class BindMode(StrEnum):
    AUTO = "auto"
    ENTITY = "entity"
    PREDICATE = "predicate"


class FactEntry(BaseModel):
    relation: str = Field(min_length=1)
    value: str = Field(min_length=1)
    id: str | None = None
    queries: list[str] | None = None
    anchors: list[tuple[str, str]] | None = None
    third_person_queries: list[str] | None = None


class FactCard(BaseModel):
    id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    facts: list[FactEntry] = Field(min_length=1)
    binding: BindMode = BindMode.AUTO
    kind: str | None = None
    confusables: list[str] = Field(default_factory=list)
    entity_prompts: list[str] | None = None
    anchors: list[tuple[str, str]] | None = None

    def resolved_binding(self) -> BindMode:
        if self.binding != BindMode.AUTO:
            return self.binding
        return BindMode.PREDICATE if self.subject.strip().lower() == "self" else BindMode.ENTITY


class BindConfig(BaseModel):
    target_layers: list[int] | None = None
    max_rel_norm: float = Field(default=0.85, gt=0.0, le=2.0)
    ce_target: float = Field(default=0.02, gt=0.0)
    min_steps: int = Field(default=25, ge=1, le=500)
    max_steps: int = Field(default=120, ge=1, le=500)
    lr: float = Field(default=0.12, gt=0.0)
    l2_reg: float = Field(default=1e-3, ge=0.0)
    entity_essence_kl: float = Field(default=0.15, ge=0.0)
    entity_tau_lo: float = Field(default=0.50, gt=0.0, lt=1.5)
    entity_tau_hi: float = Field(default=0.72, gt=0.0, lt=2.0)
    pred_subj_weight: float = Field(default=0.70, ge=0.0, le=2.0)
    pred_tau_lo: float = Field(default=1.02, gt=0.0, lt=3.0)
    pred_tau_hi: float = Field(default=1.18, gt=0.0, lt=3.0)
    clamp_sharp: float = Field(default=28.0, gt=0.0)
    multilingual_safe: bool = True
    carryover_safe: bool = True
    carryover_rank: int = Field(default=4, ge=0, le=32)
    turn_invariant: bool = True
    turn_rank: int = Field(default=4, ge=0, le=16)
    run_probes: bool = True


class BoundUnitReport(BaseModel):
    unit_id: str
    card_id: str
    subject: str
    mode: str  # entity | predicate
    target_layers: list[int] = Field(default_factory=list)
    neurons_by_layer: dict[int, list[int]] = Field(default_factory=dict)
    steps_run: int = Field(ge=0)
    ce_start: float = Field(ge=0.0)
    ce_end: float = Field(ge=0.0)
    elapsed_s: float = Field(ge=0.0)
    rel_norms: dict[int, float] = Field(default_factory=dict)
    clamp_diag_first_layer: dict[str, float] = Field(default_factory=dict)
    sample_probes: dict[str, str] = Field(default_factory=dict)


class BindReport(BaseModel):
    model_path: str
    target_layers: list[int] = Field(default_factory=list)
    units: list[BoundUnitReport] = Field(default_factory=list)
    total_neurons_used: int = Field(ge=0)
    params_edited: int = Field(ge=0)
    model_param_pct: float = Field(ge=0.0)
    eng_gen_kl: float = Field(ge=0.0)
    cjk_gen_kl: float = Field(ge=0.0)
    elapsed_s: float = Field(ge=0.0)
    saved_model_dir: str | None = None
