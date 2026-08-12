"""Adapter interface: per-model-family interception and bookkeeping."""

from __future__ import annotations

import functools
import types
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..utils.bookkeeping import SplicedInputs
    from ..wrapper import FastVisionState


def text_query(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Mean embedding of the attended non-image prompt tokens, ``[D]``.

    Passed to pruners as ``meta["query"]`` so text-conditioned strategies
    (FastV) can score visual tokens against the prompt.
    """
    mask = (input_ids != image_token_id) & attention_mask.bool()
    denom = mask.sum().clamp(min=1)
    return (inputs_embeds * mask.unsqueeze(-1)).sum(dim=(0, 1)) / denom


class Adapter(ABC):
    """Knows where a family's visual tokens live and how to fix up the
    sequence after pruning. Install/uninstall must be symmetric: after
    ``uninstall`` the model behaves byte-identically to before ``install``.
    """

    name: str = ""

    @classmethod
    @abstractmethod
    def matches(cls, model: nn.Module) -> bool:
        """True if this adapter supports ``model``."""

    @abstractmethod
    def install(self, model: nn.Module, state: "FastVisionState") -> None:
        """Patch the model in place, saving originals for uninstall."""

    @abstractmethod
    def uninstall(self, model: nn.Module) -> None:
        """Restore original behavior exactly."""


class SpliceAdapter(Adapter):
    """Shared ``generate``/``forward`` interception for families that prune by
    handing the model pre-built ``inputs_embeds`` (+ rebuilt mask/positions).

    Subclasses implement :meth:`prepare` and declare which model kwargs are
    consumed by the splice (``consume_keys``) and which inputs force a bypass
    (``bypass_keys``, e.g. video inputs the adapter doesn't prune yet).
    """

    #: kwargs replaced by the spliced inputs_embeds and therefore dropped
    consume_keys: tuple[str, ...] = ("pixel_values",)
    #: kwargs whose presence bypasses pruning for the call
    bypass_keys: tuple[str, ...] = ()

    @abstractmethod
    def prepare(
        self,
        model: nn.Module,
        state: "FastVisionState",
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        kwargs: dict,
    ) -> "SplicedInputs | None":
        """Prune and splice; return None when pruning is a no-op."""

    def _skip(self, state: "FastVisionState", input_ids, kwargs: dict) -> bool:
        return (
            not state.enabled
            or kwargs.get("pixel_values") is None
            or input_ids is None
            or not torch.is_tensor(input_ids)
            or kwargs.get("inputs_embeds") is not None
            or any(kwargs.get(k) is not None for k in self.bypass_keys)
        )

    def install(self, model: nn.Module, state: "FastVisionState") -> None:
        adapter = self
        orig_generate = model.generate
        orig_forward = model.forward
        state.originals = {"generate": orig_generate, "forward": orig_forward}

        # functools.wraps preserves the original signature for
        # inspect.signature, which generate() probes for e.g. logits_to_keep.
        @functools.wraps(orig_generate)
        def generate(self, *args, **kwargs):
            input_ids = kwargs.pop("input_ids", kwargs.pop("inputs", None))
            if input_ids is None and args:
                input_ids, args = args[0], args[1:]
            if not adapter._skip(state, input_ids, kwargs):
                spliced = adapter.prepare(
                    self, state, input_ids, kwargs.get("attention_mask"), kwargs
                )
                if spliced is not None:
                    for key in (*adapter.consume_keys, "attention_mask", "position_ids"):
                        kwargs.pop(key, None)
                    if spliced.position_ids is not None:
                        kwargs["position_ids"] = spliced.position_ids
                    return orig_generate(
                        *args,
                        inputs_embeds=spliced.inputs_embeds,
                        attention_mask=spliced.attention_mask,
                        **kwargs,
                    )
            if input_ids is not None:
                kwargs["input_ids"] = input_ids
            return orig_generate(*args, **kwargs)

        @functools.wraps(orig_forward)
        def forward(self, input_ids=None, **kwargs):
            skip = (
                adapter._skip(state, input_ids, kwargs)
                or kwargs.get("past_key_values") is not None
            )
            if not skip and kwargs.get("labels") is not None:
                warnings.warn("fastvision: pruning is bypassed when labels are passed")
                skip = True
            if skip:
                return orig_forward(input_ids=input_ids, **kwargs)
            spliced = adapter.prepare(
                self, state, input_ids, kwargs.get("attention_mask"), kwargs
            )
            if spliced is None:
                return orig_forward(input_ids=input_ids, **kwargs)
            for key in (*adapter.consume_keys, "position_ids"):
                kwargs.pop(key, None)
            kwargs["attention_mask"] = spliced.attention_mask
            if spliced.position_ids is not None:
                kwargs["position_ids"] = spliced.position_ids
            return orig_forward(inputs_embeds=spliced.inputs_embeds, **kwargs)

        model.generate = types.MethodType(generate, model)
        model.forward = types.MethodType(forward, model)

    def uninstall(self, model: nn.Module) -> None:
        for name in ("generate", "forward"):
            if name in model.__dict__:
                del model.__dict__[name]
