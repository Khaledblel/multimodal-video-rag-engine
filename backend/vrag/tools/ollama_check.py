"""Check the models that Ollama runs: the text embedder, and the VL model that plans searches and writes answers.

Usage: python -m vrag.tools.ollama_check [--pull]   (exits 1 when a check fails)

--pull first downloads the models that Ollama does not have yet.
"""

import argparse
import time

from PIL import Image
from pydantic import ValidationError

from vrag import answer, index, ollama, planner
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


def _report(ok: bool, text: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {text}")
    return ok


def _model(model: str, pull: bool) -> ollama.ModelInfo | None:
    """What Ollama reports about a model, downloaded first when asked. None, after a FAIL line, when it is missing."""
    info = ollama.show(model)
    if info is None and pull:
        print(f"downloading {model}")
        _pull(model)
        info = ollama.show(model)
    if info is None:
        print(f"FAIL  {model} is not downloaded; run this check again with --pull")
        return None
    print(f"{model}: {', '.join(info.capabilities) if info.capabilities else 'capabilities not reported'}")
    return info


def _ask(messages: list[dict], max_tokens: int, think: bool | None) -> tuple[str, dict]:
    """The model's reply, and the last chunk, which carries the token counts."""
    text, final = "", {}
    for chunk in ollama.chat(messages, max_tokens=max_tokens, think=think):
        text += (chunk.get("message") or {}).get("content") or ""
        if chunk.get("done"):
            final = chunk
    return text.strip(), final


def _check_embedder(pull: bool) -> bool:
    settings = get_settings()
    if _model(settings.text_embed_model, pull) is None:
        return False
    # A question, the passage that answers it, and a passage that has nothing to do with it.
    question, near, far = ollama.embed(
        [
            index.query_text("What is the capital of France?"),
            "Paris is the capital and largest city of France.",
            "A flat tyre is fixed with a patch, a lever and a pump.",
        ]
    )
    sized = _report(
        len(question) == settings.text_embed_dim,
        f"text vectors hold {len(question)} numbers (TEXT_EMBED_DIM is {settings.text_embed_dim})",
    )
    related = sum(a * b for a, b in zip(question, near))
    unrelated = sum(a * b for a, b in zip(question, far))
    ranked = _report(
        related > unrelated,
        f"the text embedder puts the passage that answers a question first: {related:.2f} against {unrelated:.2f}",
    )
    return sized and ranked


def _check_vlm(pull: bool) -> bool:
    settings = get_settings()
    model = settings.vlm_model
    info = _model(model, pull)
    if info is None:
        return False
    if not info.vision:
        return _report(False, f"{model} takes no images")

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
    seconds = (with_image.get("total_duration") or 0) / 1e9
    ok = _report(
        "red" in low and "blue" in low and low.index("red") < low.index("blue"),
        f"the model sees images: it answered {reply!r} in {seconds:.1f} s",
    )

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

    if settings.planner_enabled:
        # A plan for a question that names the second of three videos.
        try:
            plan = planner.draft(
                "In the bike repair tutorial, how do they fix a flat tyre?",
                ["city_council_meeting.mp4", "bike_repair_tutorial.mp4", "interview_with_the_mayor.mp4"],
                think=False if info.thinks else None,
            )
        except ValidationError as exc:
            reason = " ".join(str(exc).split())
            ok = _report(False, f"the planner's reply is not the JSON it was asked for: {reason}")
        else:
            picked = _report(
                plan.video == 2, f"the planner writes its JSON and picks the video: {plan.model_dump_json()}"
            )
            ok = picked and ok

    loaded = ollama.loaded(model)
    if loaded is None:
        print("NOTE  the model is not loaded any more (VLM_KEEP_ALIVE), so its context and memory cannot be shown")
    else:
        context, size = loaded.get("context_length"), loaded.get("size") or 0
        right = _report(
            context == settings.vlm_num_ctx, f"context {context} (VLM_NUM_CTX is {settings.vlm_num_ctx})"
        )
        ok = right and ok
        if size:
            on_gpu = (loaded.get("size_vram") or 0) / size
            print(f"loaded: {size / 2**30:.1f} GiB, {on_gpu:.0%} of it on the GPU")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull", action="store_true", help="download the models that Ollama does not have")
    args = parser.parse_args()
    settings = get_settings()

    results = []
    try:
        print(f"ollama {_version()} at {settings.ollama_url}")
        if settings.index_enabled:
            results.append(_check_embedder(args.pull))
        else:
            print("indexing is turned off (INDEX_ENABLED=false): the text embedder is not checked")
        if settings.vlm_enabled:
            results.append(_check_vlm(args.pull))
        else:
            print("answers are turned off (VLM_ENABLED=false): the VL model is not checked")
    except ollama.OllamaError as exc:
        print(f"FAIL  {exc}")
        return 1
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
