"""Model adapters: Mamba-2, Falcon-H1 (Mamba-2 hybrid), and RWKV-7.

Two transition classes appear here. The Mamba-2 state-space update is
``S_t = exp(dt_t A) S_{t-1} + dt_t x_t B_t^T`` with one scalar decay per head,
so its transition is diagonal (in fact scalar) and a per-head decay ledger is
the full transport. The RWKV-7 update is
``S_t = (diag(e^{w_t}) - (kk_t * a_t) kk_t^T) S_{t-1} + k_t v_t^T``, a
diagonal-minus-rank-one transition of the same kind as the delta rule, so a
diagonal ledger cannot transport a receipt but the full matrix product can.

Every adapter drives the released implementation in its single-token path for
the suffix, which is the path that exposes the recurrence exactly, and reads
the per-layer state from the model's own cache before and after each token.
"""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch


@dataclass
class StepOps:
    """Per-token transition and write terms for all recurrent layers, stacked on dim 0."""

    fields: Dict[str, torch.Tensor] = field(default_factory=dict)


class FamilyAdapter:
    name: str = ""
    repo: str = ""
    transition_class: str = ""  # "decay" or "delta"
    has_attention: bool = False

    def __init__(self, device: str = "mps", dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None

    # -- loading / tokens --------------------------------------------------

    def load(self, repo: Optional[str] = None) -> None:
        raise NotImplementedError

    def encode(self, text: str, special: bool = False) -> List[int]:
        ids = self.tokenizer(text, add_special_tokens=special).input_ids
        return list(ids)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids))

    # -- state ----------------------------------------------------------------

    def recurrent_layers(self) -> List[int]:
        raise NotImplementedError

    def states(self, cache: Any) -> torch.Tensor:
        """Stacked fp32 recurrent states, ``[n_layers, ...]`` with the batch dim dropped."""
        raise NotImplementedError

    def inject(self, cache: Any, stacked: torch.Tensor) -> None:
        """Overwrite the recurrent states in ``cache`` with ``stacked``."""
        raise NotImplementedError

    def snapshot(self, cache: Any) -> Any:
        return copy.deepcopy(cache)

    def restore(self, snap: Any) -> Any:
        return copy.deepcopy(snap)

    # -- forward ----------------------------------------------------------------

    def prefill(self, ids: Sequence[int]) -> Any:
        """Fresh cache after ingesting ``ids`` in one forward; returns the cache."""
        raise NotImplementedError

    def step(self, token: int, cache: Any, capture: bool = False):
        """Advance one token; returns ``(logits[V], ops or None)``."""
        raise NotImplementedError

    # -- algebra on stacked states ----------------------------------------------

    def apply_transition(self, ops: StepOps, S: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply_ledger(self, ops: StepOps, S: torch.Tensor) -> torch.Tensor:
        """Diagonal part of the transition only."""
        raise NotImplementedError

    def write_term(self, ops: StepOps) -> torch.Tensor:
        raise NotImplementedError

    def transition_log_bytes_per_token(self, ops: StepOps) -> int:
        """Bytes a deployment must log per token to transport a receipt later."""
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Mamba-2 and Falcon-H1 (state-space duality mixers)
# ----------------------------------------------------------------------------


class _SSDAdapter(FamilyAdapter):
    transition_class = "decay"

    def __init__(self, device="mps", dtype=torch.float32):
        super().__init__(device, dtype)
        self._dt: Dict[int, torch.Tensor] = {}
        self._hooks: List[Any] = []

    def _mixers(self) -> List[Any]:
        raise NotImplementedError

    def _dt_from_input(self, mixer: Any, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _install_dt_hooks(self) -> None:
        for idx, mixer in enumerate(self._mixers()):
            def hook(module, args, kwargs, idx=idx):
                hidden = args[0] if args else kwargs["hidden_states"] if "hidden_states" in kwargs else kwargs["input_states"]
                with torch.no_grad():
                    self._dt[idx] = self._dt_from_input(module, hidden).detach()
            self._hooks.append(mixer.register_forward_pre_hook(hook, with_kwargs=True))

    def recurrent_layers(self) -> List[int]:
        return list(range(len(self._mixers())))

    def _cache_layer(self, cache: Any, idx: int) -> Any:
        return cache.layers[idx]

    def states(self, cache: Any) -> torch.Tensor:
        return torch.stack(
            [self._cache_layer(cache, i).recurrent_states[0].float() for i in self.recurrent_layers()]
        )

    def inject(self, cache: Any, stacked: torch.Tensor) -> None:
        for pos, i in enumerate(self.recurrent_layers()):
            layer = self._cache_layer(cache, i)
            layer.recurrent_states = stacked[pos][None].to(layer.recurrent_states.dtype)

    def _decay(self, mixer: Any) -> torch.Tensor:
        return -torch.exp(mixer.A_log.float())  # [H]

    def prefill(self, ids: Sequence[int]) -> Any:
        x = torch.tensor([list(ids)], device=self.device)
        with torch.no_grad():
            out = self.model(x, use_cache=True)
        return out.past_key_values if hasattr(out, "past_key_values") else out.cache_params

    def step(self, token: int, cache: Any, capture: bool = False):
        x = torch.tensor([[int(token)]], device=self.device)
        before = self.states(cache) if capture else None
        self._dt = {}
        with torch.no_grad():
            out = self.model(x, past_key_values=cache, use_cache=True) if self._uses_pkv else self.model(x, cache_params=cache, use_cache=True)
        logits = out.logits[0, -1].float()
        if not capture:
            return logits, None
        after = self.states(cache)
        dt = torch.stack([self._dt[i] for i in self.recurrent_layers()])  # [L, H]
        A = torch.stack([self._decay(m) for m in self._mixers()])  # [L, H]
        a = torch.exp(dt * A)  # [L, H]
        write = after - a[..., None, None] * before
        return logits, StepOps({"a": a, "W": write})

    def apply_transition(self, ops: StepOps, S: torch.Tensor) -> torch.Tensor:
        return S * ops.fields["a"][..., None, None]

    apply_ledger = apply_transition

    def write_term(self, ops: StepOps) -> torch.Tensor:
        return ops.fields["W"]

    def transition_log_bytes_per_token(self, ops: StepOps) -> int:
        return int(ops.fields["a"].numel() * 4)


class Mamba2Adapter(_SSDAdapter):
    name = "mamba2"
    repo = "AntonV/mamba2-1.3b-hf"
    _uses_pkv = False

    def load(self, repo: Optional[str] = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo = repo or self.repo
        self.repo = repo
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=self.dtype).to(self.device).eval()
        self.model.requires_grad_(False)
        self._install_dt_hooks()

    def _mixers(self) -> List[Any]:
        return [layer.mixer for layer in self.model.backbone.layers]

    def _dt_from_input(self, mixer: Any, hidden: torch.Tensor) -> torch.Tensor:
        proj = mixer.in_proj(hidden[:, -1:])  # single-token path
        dt_raw = proj[0, -1, -mixer.num_heads :].float()
        dt = torch.nn.functional.softplus(dt_raw + mixer.dt_bias.float())
        return torch.clamp(dt, mixer.time_step_limit[0], mixer.time_step_limit[1])


class FalconH1Adapter(_SSDAdapter):
    name = "falcon_h1"
    repo = "tiiuae/Falcon-H1-1.5B-Base"
    has_attention = True
    _uses_pkv = True

    def load(self, repo: Optional[str] = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo = repo or self.repo
        self.repo = repo
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=self.dtype).to(self.device).eval()
        self.model.requires_grad_(False)
        self._install_dt_hooks()

    def _mixers(self) -> List[Any]:
        return [layer.mamba for layer in self.model.model.layers]

    def _dt_from_input(self, mixer: Any, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden[:, -1:] * mixer.ssm_in_multiplier
        proj = mixer.in_proj(x) * mixer.mup_vector
        dt_raw = proj[0, -1, -mixer.num_heads :].float()
        dt = torch.nn.functional.softplus(dt_raw + mixer.dt_bias.float())
        return torch.clamp(dt, mixer.time_step_limit[0], mixer.time_step_limit[1])


# ----------------------------------------------------------------------------
# RWKV-7
# ----------------------------------------------------------------------------


class Rwkv7Adapter(FamilyAdapter):
    name = "rwkv7"
    repo = "Hakureirm/rwkv7-1.5b-hf"
    transition_class = "delta"

    def __init__(self, device="mps", dtype=torch.float32):
        super().__init__(device, dtype)
        self._module = None
        self._real_wkv = None
        self._captured: List[Dict[str, torch.Tensor]] = []
        self._capturing = False

    def load(self, repo: Optional[str] = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo = repo or self.repo
        self.repo = repo
        self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=self.dtype)
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        blocks = self.model.rwkv7.blocks if hasattr(self.model, "rwkv7") else self.model.model.blocks
        self._module = sys.modules[type(blocks[0].att).__module__]
        table = self._module.RWKV7_WKV_FUNCTIONS
        impl = self.model.config.wkv_implementation
        self._real_wkv = table[impl]

        def wrapper(r, w_log, k, v, kk, a, state, *args, **kwargs):
            out, new_state = self._real_wkv(r, w_log, k, v, kk, a, state, *args, **kwargs)
            if self._capturing:
                self._captured.append(
                    {
                        "w": w_log[0, -1].float().clone(),
                        "kk": kk[0, -1].float().clone(),
                        "a": a[0, -1].float().clone(),
                        "k": k[0, -1].float().clone(),
                        "v": v[0, -1].float().clone(),
                    }
                )
            return out, new_state

        table[impl] = wrapper

    def recurrent_layers(self) -> List[int]:
        return list(range(self.model.config.num_hidden_layers))

    def states(self, cache: Any) -> torch.Tensor:
        return torch.stack([cache.read(i)[2][0].float() for i in self.recurrent_layers()])

    def inject(self, cache: Any, stacked: torch.Tensor) -> None:
        for pos, i in enumerate(self.recurrent_layers()):
            att_shift, ffn_shift, wkv = cache.read(i)
            cache.write(i, att_shift, ffn_shift, stacked[pos][None].to(wkv.dtype))

    def prefill(self, ids: Sequence[int]) -> Any:
        x = torch.tensor([list(ids)], device=self.device)
        with torch.no_grad():
            out = self.model(x, use_cache=True)
        return out.state

    def step(self, token: int, cache: Any, capture: bool = False):
        x = torch.tensor([[int(token)]], device=self.device)
        self._captured = []
        self._capturing = capture
        try:
            with torch.no_grad():
                out = self.model(x, state=cache, use_cache=True)
        finally:
            self._capturing = False
        logits = out.logits[0, -1].float()
        if not capture:
            return logits, None
        if len(self._captured) != len(self.recurrent_layers()):
            raise RuntimeError(f"captured {len(self._captured)} wkv calls for {len(self.recurrent_layers())} layers")
        fields = {name: torch.stack([c[name] for c in self._captured]) for name in ("w", "kk", "a", "k", "v")}
        return logits, StepOps(fields)

    def apply_transition(self, ops: StepOps, S: torch.Tensor) -> torch.Tensor:
        decay = torch.exp(ops.fields["w"])[..., None]  # [L,H,N,1]
        kk = ops.fields["kk"]
        b = (kk * ops.fields["a"])[..., None]  # [L,H,N,1]
        sa = torch.einsum("lhi,lhij->lhj", -kk, S)[:, :, None, :]  # [L,H,1,V]
        return decay * S + b * sa

    def apply_ledger(self, ops: StepOps, S: torch.Tensor) -> torch.Tensor:
        return torch.exp(ops.fields["w"])[..., None] * S

    def write_term(self, ops: StepOps) -> torch.Tensor:
        return ops.fields["k"][..., None] * ops.fields["v"][:, :, None, :]

    def transition_log_bytes_per_token(self, ops: StepOps) -> int:
        return int(4 * (ops.fields["w"].numel() + ops.fields["kk"].numel() + ops.fields["a"].numel()))


ADAPTERS = {
    "mamba2": Mamba2Adapter,
    "falcon_h1": FalconH1Adapter,
    "rwkv7": Rwkv7Adapter,
}
