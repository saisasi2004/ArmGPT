"""Intent-parsing eval. Run this whenever you swap models or touch the prompt.

    python eval_intent.py                  # current resolved model
    python eval_intent.py qwen3.5:0.8b     # force a specific tag

Reports per-case action accuracy, detector/match choice, and latency. Accuracy
and latency have to be read together on this hardware: a model that is 8/8 at
40s is not obviously better than one that is 7/8 at 2s, because the arm blocks
on this call.

The cases below are the ones that actually broke during development, not a
generic benchmark. Add to them as you find new failures — that's the point.
"""
from __future__ import annotations

import statistics
import sys
import time

import config
from services import llm

# (message, expected action, expected source detector or None)
CASES: list[tuple[str, str, str | None]] = [
    ("place the red object on the blue plate",       "pick_place", "color"),
    ("put the big red block onto the yellow square", "pick_place", "color"),
    ("pick up marker 3 and put it on the green block", "pick_place", "markers"),
    ("move the cup onto the blue plate",             "pick_place", "objects"),
    ("where is the cup?",                            "locate",     "objects"),
    ("do you see a bottle?",                         "locate",     "objects"),
    ("find the red one",                             "locate",     "color"),
    ("how many circles do you see?",                 "count",      "shapes"),
    ("count the red blocks",                         "count",      "color"),
    ("hey what can you do?",                         "chat",       None),
    ("hello there",                                  "chat",       None),
    ("thanks, that worked",                          "chat",       None),
]


def main() -> None:
    if len(sys.argv) > 1:
        config.LLM_MODEL_PREFERENCES = [sys.argv[1]]
        llm._resolved_model = None

    model = llm.resolve_model()
    print(f"model: {model}   thinking={config.LLM_THINK}\n")

    hits, det_hits, times = 0, 0, []
    for msg, want_action, want_det in CASES:
        t0 = time.perf_counter()
        try:
            intent = llm.parse_intent(msg)
        except llm.LLMError as exc:
            print(f"ERROR  {msg[:44]:46} {exc}")
            continue
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)

        got_action = intent["action"]
        got_det = intent["source"]["detector"] if intent["source"] else None
        action_ok = got_action == want_action
        det_ok = got_det == want_det
        hits += action_ok
        det_hits += det_ok

        flag = "PASS" if (action_ok and det_ok) else "FAIL"
        src = intent["source"] and f"{got_det}/{intent['source']['match']}"
        tgt = intent["target"] and \
            f"{intent['target']['detector']}/{intent['target']['match']}"
        print(f"{flag} {ms:7.0f}ms  {msg[:44]:46} "
              f"want={want_action:10} got={got_action:10} src={src} tgt={tgt}")

    n = len(CASES)
    print(f"\naction   {hits}/{n}")
    print(f"detector {det_hits}/{n}")
    if times:
        print(f"latency  median {statistics.median(times)/1000:.1f}s  "
              f"max {max(times)/1000:.1f}s")


if __name__ == "__main__":
    main()
