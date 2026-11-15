"""Check that the VL model is there, sees images and runs with the intended context size.

Usage: python -m vrag.tools.vlm_check [--pull]   (exits 1 when a check fails)

--pull first downloads the model when Ollama does not have it yet.
"""

import argparse
import time

from PIL import Image

from vrag import answer, ollama
from vrag.config import get_settings

SIZE = (768, 448)
QUESTION = "This image has a left half and a right half, each in one colour. Name the two colours, left first."
WAIT_S = 60


def _version() -> str:
    """Ollama's version, waiting for the service to come up: the bootstrap runs this right after starting it."""
    deadline = time.monotonic() + WAIT_S
    while True:
        try:
            return ollama.version()
        except ollama.OllamaError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)


def _pull(model: str) -> None:
    shown, last = 0.0, ""
    for line in ollama.pull(model):
        status, total = line.get("status", ""), line.get("total")
        if total:
            if time.monotonic() - shown >= 5:
                print(f"  {status}: {line.get('completed', 0) / total:.0%} of {total / 2**30:.2f} GiB", flush=True)
                shown = time.monotonic()
        elif status != last:
            print(f"  {status}", flush=True)
        last = status


def _ask(messages: list[dict], max_tokens: int, think: bool | None) -> tuple[str, dict]:
    """The model's reply, and the last chunk, which carries the token counts."""
    text, final = "", {}
    for chunk in ollama.chat(messages, max_tokens=max_tokens, think=think):
        text += (chunk.get("message") or {}).get("content") or ""
        if chunk.get("done"):
            final = chunk
    return text.strip(), final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull", action="store_true", help="download the model if Ollama does not have it")
    args = parser.parse_args()
    settings = get_settings()
    model = settings.vlm_model
    if not settings.vlm_enabled:
        print("answers are turned off (VLM_ENABLED=false): nothing to check")
        return 0

    try:
        print(f"ollama {_version()} at {settings.ollama_url}")
        info = ollama.show(model)
        if info is None and args.pull:
            print(f"downloading {model}")
            _pull(model)
            info = ollama.show(model)
        if info is None:
            print(f"FAIL  {model} is not downloaded; run this check again with --pull")
            return 1
        print(f"{model}: {', '.join(info.capabilities) if info.capabilities else 'capabilities not reported'}")
        if not info.vision:
            print(f"FAIL  {model} takes no images")
            return 1

        # Left half red, right half blue: the reply shows whether the image reached the model.
        image = Image.new("RGB", SIZE, (255, 0, 0))
        image.paste((0, 0, 255), (SIZE[0] // 2, 0, SIZE[0], SIZE[1]))
        think = settings.vlm_think if info.thinks else None
        room = 64 + (answer.THINKING_TOKENS if think else 0)
        _, without = _ask([{"role": "user", "content": QUESTION}], 1, think)
        reply, with_image = _ask(
            [{"role": "user", "content": QUESTION, "images": [answer.as_jpeg(image)]}], room, think
        )
        low = reply.lower()
        ok = "red" in low and "blue" in low and low.index("red") < low.index("blue")
        seconds = (with_image.get("total_duration") or 0) / 1e9
        print(f"{'PASS' if ok else 'FAIL'}  the model sees images: it answered {reply!r} in {seconds:.1f} s")

        counts = (with_image.get("prompt_eval_count"), without.get("prompt_eval_count"))
        estimated = answer.image_tokens(SIZE)
        if None in counts:
            print("NOTE  Ollama gave no prompt token counts, so the cost of an image could not be measured")
        else:
            measured = counts[0] - counts[1]
            close = abs(measured - estimated) <= 0.15 * estimated
            print(
                f"{'PASS' if close else 'NOTE'}  a {SIZE[0]}x{SIZE[1]} image costs {measured} tokens "
                f"(estimated {estimated}; the frame budget of an answer relies on this estimate)"
            )

        loaded = ollama.loaded(model)
        if loaded is None:
            print("NOTE  the model is not loaded any more (VLM_KEEP_ALIVE), so its context and memory cannot be shown")
        else:
            context, size = loaded.get("context_length"), loaded.get("size") or 0
            right = context == settings.vlm_num_ctx
            ok = ok and right
            print(f"{'PASS' if right else 'FAIL'}  context {context} (VLM_NUM_CTX is {settings.vlm_num_ctx})")
            if size:
                on_gpu = (loaded.get("size_vram") or 0) / size
                print(f"loaded: {size / 2**30:.1f} GiB, {on_gpu:.0%} of it on the GPU")
    except ollama.OllamaError as exc:
        print(f"FAIL  {exc}")
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
