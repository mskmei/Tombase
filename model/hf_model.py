"""
HuggingFace model wrapper for local inference.
Downloads and runs models from HuggingFace Hub using the transformers library.

Usage:
    from model.hf_model import HFModel
    model = HFModel("Qwen/Qwen3-4B")
    result = model.generate("Hello")
"""

import asyncio
from typing import Dict, List, Optional, Union

from .base import BaseLM
from .openai_model import GenerationConfig


class HFModel(BaseLM):
    """
    Local HuggingFace model for inference.
    Downloads the model from the Hub on first use and caches it locally.
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 1024,
        enable_thinking: bool = False,
    ):
        """
        Args:
            model_name:      HuggingFace model ID, e.g. "Qwen/Qwen3-4B".
            device_map:      Passed to from_pretrained; "auto" spreads across
                             available GPUs automatically.
            torch_dtype:     "auto" lets transformers choose the dtype, or a
                             string like "float16" / "bfloat16".
            max_new_tokens:  Default token budget for generation.
            enable_thinking: Set True to enable chain-of-thought tokens for
                             models that support it (e.g. Qwen3).
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

        dtype = "auto" if torch_dtype == "auto" else getattr(torch, torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        except ValueError as exc:
            # device_map='auto' requires accelerate. Fall back to vanilla load.
            if "requires `accelerate`" not in str(exc):
                raise
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        self.model.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt_text(self, prompt: str) -> str:
        """Apply the model's chat template to a single user message."""
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            # Qwen3 and some other models accept enable_thinking
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=self.enable_thinking, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _run_generate(self, prompt: str, max_new_tokens: int) -> tuple[str, dict]:
        import torch

        text = self._build_prompt_text(prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        n_input = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][n_input:]
        n_output = len(new_tokens)
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Return output text and usage statistics
        usage = {
            "prompt_tokens": n_input,
            "completion_tokens": n_output,
            "total_tokens": n_input + n_output
        }
        return output_text, usage

    def _resolve_max_tokens(
        self, cfg: Optional[GenerationConfig], overrides: dict
    ) -> int:
        # Accept 'generation_config' as a legacy alias for 'cfg'
        legacy_cfg = overrides.pop("generation_config", None)
        resolved_cfg = cfg or legacy_cfg
        return resolved_cfg.max_tokens if resolved_cfg else self.max_new_tokens

    # ------------------------------------------------------------------
    # BaseLM interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        schema=None,
        cfg: Optional[GenerationConfig] = None,
        **overrides,
    ) -> Dict[str, str]:
        max_new_tokens = self._resolve_max_tokens(cfg, overrides)
        output, usage = self._run_generate(prompt, max_new_tokens=max_new_tokens)
        return {"output": output, "usage": usage}

    def batch_generate(
        self,
        prompts: List[str],
        schema=None,
        cfg: Optional[GenerationConfig] = None,
        **overrides,
    ) -> List[Dict[str, str]]:
        return [self.generate(p, schema=schema, cfg=cfg, **overrides) for p in prompts]

    async def async_generate(
        self,
        prompts: List[str],
        schema=None,
        cfg: Optional[GenerationConfig] = None,
        concurrency: int = 1,
        return_exceptions: bool = True,
        **overrides,
    ) -> List[Union[Dict[str, str], Exception]]:
        loop = asyncio.get_event_loop()
        results = []
        for prompt in prompts:
            try:
                result = await loop.run_in_executor(
                    None, lambda p=prompt: self.generate(p, schema=schema, cfg=cfg, **overrides)
                )
                results.append(result)
            except Exception as exc:
                if return_exceptions:
                    results.append(exc)
                else:
                    raise
        return results
