"""Calls to the Ollama service, which runs the VL model. Every failure is raised as OllamaError."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from vrag.config import get_settings

QUICK = httpx.Timeout(10.0)
# Ollama loads the model and reads the whole prompt before the first token arrives.
SLOW = httpx.Timeout(10.0, read=600.0)
TEMPERATURE = 0.2


class OllamaError(RuntimeError):
    """Ollama could not do what was asked; the message says why."""


@dataclass(frozen=True)
class ModelInfo:
    capabilities: tuple[str, ...] | None

    @property
    def vision(self) -> bool:
        return self.capabilities is None or "vision" in self.capabilities

    @property
    def thinks(self) -> bool:
        return self.capabilities is not None and "thinking" in self.capabilities


def _reason(response: httpx.Response) -> str:
    """Ollama reports an error as {"error": "..."}."""
    try:
        return str(response.json()["error"])
    except (ValueError, KeyError, TypeError):
        return f"HTTP {response.status_code}"


def _no_reply(url: str, exc: httpx.HTTPError) -> OllamaError:
    return OllamaError(f"no reply from Ollama at {url}: {type(exc).__name__} {exc}".rstrip())


def _request(method: str, path: str, body: dict | None = None) -> httpx.Response:
    url = get_settings().ollama_url
    try:
        return httpx.request(method, f"{url}{path}", json=body, timeout=QUICK)
    except httpx.HTTPError as exc:
        raise _no_reply(url, exc) from exc


def _stream(path: str, body: dict) -> Iterator[dict]:
    """POST, and yield the JSON lines of the reply as they arrive."""
    url = get_settings().ollama_url
    try:
        with httpx.stream("POST", f"{url}{path}", json=body, timeout=SLOW) as response:
            if response.status_code != 200:
                response.read()
                raise OllamaError(_reason(response))
            for line in response.iter_lines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if "error" in item:
                    raise OllamaError(str(item["error"]))
                yield item
    except httpx.HTTPError as exc:
        raise _no_reply(url, exc) from exc


def version() -> str:
    response = _request("GET", "/api/version")
    if response.status_code != 200:
        raise OllamaError(_reason(response))
    return str(response.json()["version"])


def show(model: str) -> ModelInfo | None:
    """What Ollama reports about a model, or None when the model is not downloaded."""
    response = _request("POST", "/api/show", {"model": model})
    if response.status_code != 200:
        reason = _reason(response)
        if response.status_code == 404 or "not found" in reason.lower():
            return None
        raise OllamaError(reason)
    capabilities = response.json().get("capabilities")
    return ModelInfo(None if capabilities is None else tuple(capabilities))


def pull(model: str) -> Iterator[dict]:
    """Download a model. Yields Ollama's progress lines; the last one is {"status": "success"}."""
    return _stream("/api/pull", {"model": model, "stream": True})


def loaded(model: str) -> dict | None:
    """What `ollama ps` shows for a model: size, size_vram (the part on the GPU) and context_length."""
    response = _request("GET", "/api/ps")
    if response.status_code != 200:
        raise OllamaError(_reason(response))
    names = {model, f"{model}:latest"}
    for entry in response.json().get("models", []):
        if names & {entry.get("name"), entry.get("model")}:
            return entry
    return None


def _keep_alive(value: str) -> str | int:
    """Ollama takes a duration such as "5m", or seconds as a number: 0 unloads at once, -1 keeps the model loaded."""
    try:
        return int(value)
    except ValueError:
        return value


def chat(messages: list[dict], *, max_tokens: int, think: bool | None = None) -> Iterator[dict]:
    """Stream a reply from the VL model.

    Each chunk has message.content (and message.thinking, from a model that thinks first). The last one has
    done=true and the counts: prompt_eval_count, eval_count, and durations in nanoseconds.
    """
    settings = get_settings()
    options: dict = {"num_ctx": settings.vlm_num_ctx, "num_predict": max_tokens}
    if not think:
        options["temperature"] = TEMPERATURE
    body = {
        "model": settings.vlm_model,
        "messages": messages,
        "stream": True,
        "options": options,
        "keep_alive": _keep_alive(settings.vlm_keep_alive),
        "truncate": False,
        "shift": False,
    }
    if think is not None:
        body["think"] = think
    return _stream("/api/chat", body)
