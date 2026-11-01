"""
ConflictBench — Evaluation backends
Three ways to turn a Scenario into raw completion text:

  StrategyBackend  a no-model baseline from scripts/baselines.STRATEGIES
  ModelBackend     a local transformers model (+ optional PEFT adapter)
  ApiBackend       an OpenAI-compatible chat-completions endpoint

Every backend exposes ``complete(scenario) -> Optional[List[str]]`` (one string
per sample, or None if the prompt had to be skipped) and ``describe() -> dict``
(run metadata for summary.json). Heavy imports (torch, transformers, peft)
happen lazily inside ModelBackend so the rest of the harness runs without them.
"""

import contextlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompts import build_messages

API_KEY_ENV = "CONFLICTBENCH_API_KEY"


# ---------------------------------------------------------------------------
# Strategy backend (no model)
# ---------------------------------------------------------------------------

class StrategyBackend:
    """Runs one of the baseline strategies. Deterministic for a given seed."""

    kind = "strategy"

    def __init__(self, name: str, seed: int = 0):
        from scripts.baselines import STRATEGIES

        if name not in STRATEGIES:
            raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}")
        self.name = name
        self.fn: Callable = STRATEGIES[name]
        self.seed = seed
        self.rng = random.Random(seed)
        self.samples = 1

    def complete(self, scenario) -> Optional[List[str]]:
        return [self.fn(scenario, self.rng)]

    def describe(self) -> Dict[str, Any]:
        return {"backend": "strategy", "strategy": self.name, "model": None, "adapter": None,
                "samples": 1, "greedy": True, "temperature": 0.0, "strategy_seed": self.seed}


# ---------------------------------------------------------------------------
# Transformers backend
# ---------------------------------------------------------------------------

def _no_grad():
    """torch.no_grad() when torch is present, else a no-op (lets tests inject fakes)."""
    try:
        import torch
    except ImportError:
        return contextlib.nullcontext()
    return torch.no_grad()


def _extract_input_ids(encoded):
    """apply_chat_template returns a tensor or a BatchEncoding/dict, depending on version."""
    if hasattr(encoded, "input_ids"):
        return encoded.input_ids
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        return encoded["input_ids"]
    return encoded


class ModelBackend:
    """A causal LM through transformers, prompted exactly as training prompts it."""

    kind = "model"

    def __init__(
        self,
        model_id: Optional[str] = None,
        adapter: Optional[str] = None,
        samples: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 1024,
        max_prompt_tokens: Optional[int] = None,
        seed: int = 0,
        model=None,
        tokenizer=None,
    ):
        self.model_id = model_id
        self.adapter = adapter
        self.samples = max(1, int(samples))
        self.greedy = self.samples == 1
        self.temperature = 0.0 if self.greedy else float(temperature)
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.seed = seed
        if model is None or tokenizer is None:
            model, tokenizer = self._load(model_id, adapter)
        self.model = model
        self.tokenizer = tokenizer
        if self.max_prompt_tokens is None:
            ctx = getattr(getattr(model, "config", None), "max_position_embeddings", None)
            if isinstance(ctx, int) and ctx > max_new_tokens:
                self.max_prompt_tokens = ctx - max_new_tokens
        self._seeded = False

    @staticmethod
    def _load(model_id: Optional[str], adapter: Optional[str]):
        if not model_id:
            raise ValueError("--model is required for the model backend")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs: Dict[str, Any] = {"torch_dtype": "auto"}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        return model, tokenizer

    def _encode(self, scenario):
        encoded = self.tokenizer.apply_chat_template(
            build_messages(scenario), add_generation_prompt=True, return_tensors="pt"
        )
        return _extract_input_ids(encoded)

    def complete(self, scenario) -> Optional[List[str]]:
        input_ids = self._encode(scenario)
        prompt_len = int(input_ids.shape[-1])
        if self.max_prompt_tokens is not None and prompt_len > self.max_prompt_tokens:
            return None  # never truncate: a cut prompt loses the format instructions
        if not self.greedy and not self._seeded:
            self._seed_torch()
        input_ids = input_ids.to(self.model.device)
        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": not self.greedy,
        }
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self.tokenizer, "eos_token_id", None)
        if pad is not None:
            gen_kwargs["pad_token_id"] = pad
        if not self.greedy:
            gen_kwargs.update(temperature=self.temperature, top_p=self.top_p,
                              num_return_sequences=self.samples)
        with _no_grad():
            output = self.model.generate(**gen_kwargs)
        new_tokens = output[:, prompt_len:]  # decode only what the model added
        return [self.tokenizer.decode(row, skip_special_tokens=True) for row in new_tokens]

    def _seed_torch(self) -> None:
        self._seeded = True
        try:
            import torch
        except ImportError:
            return
        torch.manual_seed(self.seed)

    def describe(self) -> Dict[str, Any]:
        return {"backend": "model", "strategy": None, "model": self.model_id,
                "adapter": self.adapter, "samples": self.samples, "greedy": self.greedy,
                "temperature": self.temperature, "top_p": None if self.greedy else self.top_p,
                "max_new_tokens": self.max_new_tokens, "max_prompt_tokens": self.max_prompt_tokens}


# ---------------------------------------------------------------------------
# OpenAI-compatible API backend
# ---------------------------------------------------------------------------

def _http_post(url: str, headers: Dict[str, str], payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _safe_base_url(url: str) -> str:
    """Strip credentials and query strings before the URL is stored anywhere."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path.rstrip("/"), "", ""))


class ApiBackend:
    """Any OpenAI-compatible /chat/completions endpoint. The key comes from the environment only."""

    kind = "api"

    def __init__(
        self,
        base_url: str,
        api_model: str,
        samples: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        retries: int = 5,
        backoff: float = 2.0,
        timeout: float = 120.0,
        api_key: Optional[str] = None,
        post: Optional[Callable] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        if not key:
            raise RuntimeError(f"set the {API_KEY_ENV} environment variable to use --api")
        self._key = key
        self.base_url = base_url.rstrip("/")
        self.api_model = api_model
        self.samples = max(1, int(samples))
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout
        self._post = post or _http_post
        self._sleep = sleep

    def _request(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"}
        url = self.base_url + "/chat/completions"
        last: Optional[str] = None
        for attempt in range(self.retries + 1):
            try:
                return self._post(url, headers, payload, self.timeout)
            except urllib.error.HTTPError as e:
                if e.code not in (408, 409, 429) and e.code < 500:
                    raise RuntimeError(f"API request failed with HTTP {e.code}") from None
                last = f"HTTP {e.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                last = type(e).__name__  # never include the exception text: it could echo headers
            if attempt < self.retries:
                self._sleep(self.backoff * (2 ** attempt))
        raise RuntimeError(f"API request failed after {self.retries + 1} attempts ({last})")

    def complete(self, scenario) -> Optional[List[str]]:
        payload = {"model": self.api_model, "messages": build_messages(scenario),
                   "temperature": self.temperature, "max_tokens": self.max_tokens}
        out = []
        for _ in range(self.samples):
            data = self._request(payload)
            try:
                text = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                text = ""
            out.append(text or "")
        return out

    def describe(self) -> Dict[str, Any]:
        return {"backend": "api", "strategy": None, "model": self.api_model, "adapter": None,
                "api_base": _safe_base_url(self.base_url), "samples": self.samples,
                "greedy": self.temperature == 0.0, "temperature": self.temperature,
                "max_tokens": self.max_tokens}
