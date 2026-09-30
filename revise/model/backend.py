"""Hugging Face backend (CPU or CUDA) with residual-stream access.

Why HF Transformers on CPU rather than llama.cpp: the experiments need (i)
residual-stream activations at a fixed anchor position for *every* layer,
(ii) additive steering hooks and (iii) activation patching.  llama.cpp gives
fast quantised inference but no clean per-layer read/write access from
Python.  With bf16 weights and AVX-512 a 7B model is workable on a many-core
CPU; ``quantize="int8"`` (dynamic int8 for nn.Linear) is available as a
lower-memory option.

Efficiency tricks used here
---------------------------
* One prefill pass per condition computes the anchor activations, the
  next-token distribution *and* the teacher-forced log-probability of the
  gold answer (the answer tokens are appended after the anchor; causality
  guarantees the anchor activations are unaffected).
* The KV cache is then cropped back to the prompt and greedy decoding
  continues from it, so the prompt is never re-processed.
* Left-padded batching keeps the anchor at the same index for every row.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from revise import config  # noqa: F401  (env setup)
from revise.prompts import messages_to_raw


@dataclass
class Intervention:
    """Additive steering (``steer``) or anchor patching (``patch``) at one layer.

    ``layer`` indexes the residual stream *after* decoder block ``layer``
    (1-based; 0 = token embeddings), matching ``ForwardOutput.anchor_hidden``.
    For ``steer``, ``vector`` is a [d] tensor already multiplied by alpha and is
    added at the anchor position and every generated position
    (``positions="anchor_onward"``) or at every position (``"all"``).
    For ``patch``, ``vector`` is a [B, d] tensor that *replaces* the anchor
    residual during prefill.
    For ``project``, ``vector`` is a unit [d] direction; the component along it
    is set to ``target`` (default 0 = projected out) at the anchor onward.
    """
    kind: str
    layer: int
    vector: torch.Tensor
    positions: str = "anchor_onward"
    target: float = 0.0


@dataclass
class ForwardOutput:
    anchor_hidden: Optional[np.ndarray]      # [B, n_layers+1, d] float16
    generated: List[str]
    gen_ids: List[List[int]]
    prompt_len: List[int]
    gold_logprob: List[Optional[float]]      # sum log p(gold tokens | prompt)
    gold_first_logprob: List[Optional[float]]
    anchor_entropy: List[float]
    anchor_top_prob: List[float]
    anchor_top_token: List[str]
    timing: Dict[str, float] = field(default_factory=dict)


def _find_layers(model) -> torch.nn.ModuleList:
    for path in ("model.layers", "model.decoder.layers", "transformer.h", "gpt_neox.layers", "model.language_model.layers"):
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, torch.nn.ModuleList):
            return obj
    raise RuntimeError("could not locate decoder layers")


def _find_embed(model) -> torch.nn.Module:
    return model.get_input_embeddings()


class HFBackend:
    def __init__(self, model_name: str, dtype: str = "bfloat16", quantize: Optional[str] = None,
                 num_threads: Optional[int] = None, max_new_tokens: int = 16, use_chat_template: bool = True,
                 device: str = "cpu", capture_layers: Optional[Sequence[int]] = None,
                 attn_implementation: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.use_chat_template = use_chat_template
        if device == "cpu":
            n = num_threads or max(1, min(64, (os.cpu_count() or 8) // 2))
            torch.set_num_threads(n)
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        if quantize == "int8":
            torch_dtype = torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        kw = dict(low_cpu_mem_usage=True)
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype, **kw)
        except TypeError:  # older transformers
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, **kw)
        self.model.eval()
        if quantize == "int8":
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model.to(device)
        self.layers = _find_layers(self.model)
        self.embed = _find_embed(self.model)
        self.n_layers = len(self.layers)
        self.hidden_size = self.model.config.hidden_size
        self.capture_layers = list(capture_layers) if capture_layers is not None else list(range(self.n_layers + 1))
        gc = self.model.generation_config
        eos = gc.eos_token_id if gc is not None and gc.eos_token_id is not None else self.tok.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self.has_chat_template = bool(getattr(self.tok, "chat_template", None)) and use_chat_template

    # ------------------------------------------------------------------ prompts
    def format_prompt(self, messages: List[dict]) -> str:
        if self.has_chat_template:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return messages_to_raw(messages)

    def encode_prompt(self, prompt: str) -> List[int]:
        return self.tok(prompt, add_special_tokens=not self.has_chat_template)["input_ids"]

    def count_tokens(self, messages: List[dict]) -> int:
        return len(self.encode_prompt(self.format_prompt(messages)))

    # ------------------------------------------------------------------ hooks
    def _make_hook(self, layer_idx: int, anchor: int, store: Optional[dict], interventions: List[Intervention]):
        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            modified = False
            prefill = hs.shape[1] > 1
            for iv in interventions:
                if iv.layer != layer_idx:
                    continue
                if iv.kind == "steer":
                    v = iv.vector.to(device=hs.device, dtype=hs.dtype)
                    if not modified:
                        hs = hs.clone(); modified = True
                    if prefill:
                        if iv.positions == "all":
                            hs = hs + v
                        else:
                            hs[:, anchor:, :] = hs[:, anchor:, :] + v
                    else:
                        hs = hs + v
                elif iv.kind == "patch" and prefill:
                    if not modified:
                        hs = hs.clone(); modified = True
                    hs[:, anchor, :] = iv.vector.to(device=hs.device, dtype=hs.dtype)
                elif iv.kind == "project":
                    v = iv.vector.to(device=hs.device, dtype=torch.float32)
                    if not modified:
                        hs = hs.clone(); modified = True
                    sl = slice(None) if (iv.positions == "all" or not prefill) else slice(anchor, None)
                    seg = hs[:, sl, :].to(torch.float32)
                    coef = seg @ v                                   # [B, T]
                    seg = seg - (coef - iv.target)[..., None] * v     # set component to target
                    hs[:, sl, :] = seg.to(hs.dtype)
            if store is not None and prefill and layer_idx in store:
                store[layer_idx] = hs[:, anchor, :].detach().to(torch.float32).cpu().numpy()
            if modified:
                if isinstance(output, tuple):
                    return (hs,) + tuple(output[1:])
                return hs
            return None
        return hook

    def _register(self, anchor: int, store: Optional[dict], interventions: List[Intervention]):
        handles = []
        needed = set(self.capture_layers if store is not None else []) | {iv.layer for iv in interventions}
        for li in needed:
            if li == 0:
                handles.append(self.embed.register_forward_hook(self._make_hook(0, anchor, store, interventions)))
            else:
                handles.append(self.layers[li - 1].register_forward_hook(self._make_hook(li, anchor, store, interventions)))
        return handles

    # ------------------------------------------------------------------ main
    @torch.no_grad()
    def run_batch(self, prompts: Sequence[str], gold_answers: Optional[Sequence[Optional[str]]] = None,
                  interventions: Optional[Sequence[Intervention]] = None, capture: bool = True,
                  max_new_tokens: Optional[int] = None) -> ForwardOutput:
        t0 = time.time()
        interventions = list(interventions or [])
        max_new_tokens = max_new_tokens or self.max_new_tokens
        B = len(prompts)
        pad = self.tok.pad_token_id
        enc = [self.encode_prompt(p) for p in prompts]
        plen = [len(e) for e in enc]
        Lp = max(plen)
        golds = list(gold_answers) if gold_answers is not None else [None] * B
        gold_ids = [self.tok(g, add_special_tokens=False)["input_ids"] if g else [] for g in golds]
        La = max((len(g) for g in gold_ids), default=0)
        L = Lp + La
        ids = torch.full((B, L), pad, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        for i in range(B):
            ids[i, Lp - plen[i]:Lp] = torch.tensor(enc[i])
            mask[i, Lp - plen[i]:Lp] = 1
            if gold_ids[i]:
                ids[i, Lp:Lp + len(gold_ids[i])] = torch.tensor(gold_ids[i])
                mask[i, Lp:Lp + len(gold_ids[i])] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        anchor = Lp - 1
        store = {li: None for li in self.capture_layers} if capture else None
        handles = self._register(anchor, store, interventions)
        # Only the last La+1 positions (anchor + appended gold tokens) need logits;
        # materialising [B, L, V] float32 logits for long prompts is the main GPU memory cost.
        keep = La + 1
        try:
            try:
                out = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                 position_ids=pos.to(self.device), use_cache=True, logits_to_keep=keep)
                logits = out.logits.float()
                if logits.shape[1] != keep:   # model ignored logits_to_keep
                    logits = logits[:, -keep:, :]
            except TypeError:
                out = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                 position_ids=pos.to(self.device), use_cache=True)
                logits = out.logits[:, -keep:, :].float()
        finally:
            for h in handles:
                h.remove()
        past = out.past_key_values
        a0 = 0   # index of the anchor inside the kept logits
        t1 = time.time()

        # anchor statistics + teacher-forced gold log-probs
        lp_anchor = torch.log_softmax(logits[:, a0, :], dim=-1)
        probs = lp_anchor.exp()
        entropy = (-(probs * lp_anchor).sum(-1)).tolist()
        top_prob, top_tok = probs.max(-1)
        top_prob = top_prob.tolist()
        top_token = [self.tok.decode([t]) for t in top_tok.tolist()]
        gold_lp: List[Optional[float]] = []
        gold_first: List[Optional[float]] = []
        for i in range(B):
            g = gold_ids[i]
            if not g:
                gold_lp.append(None); gold_first.append(None); continue
            lps = torch.log_softmax(logits[i, a0:a0 + len(g), :], dim=-1)
            tok_lp = lps[torch.arange(len(g), device=lps.device), torch.tensor(g, device=lps.device)]
            gold_lp.append(float(tok_lp.sum())); gold_first.append(float(tok_lp[0]))

        # crop the KV cache back to the prompt and decode greedily
        if La > 0:
            if hasattr(past, "crop"):
                past.crop(-La)   # remove the appended gold tokens
            else:  # legacy tuple cache
                past = tuple((k[:, :, :Lp], v[:, :, :Lp]) for k, v in past)
        cur_mask = mask[:, :Lp].clone().to(self.device)
        next_tok = logits[:, a0, :].argmax(-1)
        gen: List[List[int]] = [[] for _ in range(B)]
        done = [False] * B
        handles = self._register(anchor, None, [iv for iv in interventions if iv.kind in ("steer", "project")])
        try:
            for step in range(max_new_tokens):
                nt = next_tok.tolist()
                for i in range(B):
                    if not done[i]:
                        t = nt[i]
                        if t in self.eos_ids:
                            done[i] = True
                        else:
                            gen[i].append(t)
                            if "\n" in self.tok.decode([t]):
                                done[i] = True
                if all(done) or step == max_new_tokens - 1:
                    break
                cur_mask = torch.cat([cur_mask, torch.ones((B, 1), dtype=torch.long, device=cur_mask.device)], dim=1)
                step_pos = (cur_mask.sum(-1, keepdim=True) - 1)
                out = self.model(input_ids=next_tok[:, None], attention_mask=cur_mask,
                                 position_ids=step_pos, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_tok = out.logits[:, -1, :].float().argmax(-1)
        finally:
            for h in handles:
                h.remove()
        t2 = time.time()
        texts = [self.tok.decode(g, skip_special_tokens=True) for g in gen]
        hidden = None
        if capture:
            hidden = np.stack([store[li] for li in self.capture_layers], axis=1).astype(np.float16)  # [B, nL, d]
        return ForwardOutput(
            anchor_hidden=hidden, generated=texts, gen_ids=gen, prompt_len=plen,
            gold_logprob=gold_lp, gold_first_logprob=gold_first, anchor_entropy=entropy,
            anchor_top_prob=top_prob, anchor_top_token=top_token,
            timing={"prefill_s": t1 - t0, "decode_s": t2 - t1, "prompt_tokens": float(sum(plen))},
        )

    @torch.no_grad()
    def run_messages(self, batch_messages: Sequence[List[dict]], **kw) -> ForwardOutput:
        return self.run_batch([self.format_prompt(m) for m in batch_messages], **kw)
