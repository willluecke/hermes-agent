#!/usr/bin/env python3
"""Typed goal judge for hermes-sync's standing conversation goals.

Reads {"goal": ..., "reply": ...} on stdin and prints
{"verdict": "done"|"continue"|"blocked", "done": bool, "p": float, "reason": str}.
The legacy done field is true only for a done verdict. The verdict comes from Jev (the same typed judge Will's harness
uses for its gates) through the hermes-agent typesafe tool, so it costs a
fraction of a cent and never runs a metered chat model. A missing answer is an
error exit: hermes-sync pauses the goal rather than guessing.

A reply that explicitly declares the goal complete (or impossible) ends the
goal even when the judge scores it as not fully met. The met score is advice
for the steward's final review, not a reason to re-send the same goal: on
2026-10-19 a research worker that honestly disclosed a shortfall was re-run
three times, repeating the same completion each time, because met stayed
below threshold while claims_done was 0.98.
"""
from __future__ import annotations

import json
import math
import sys

DONE_THRESHOLD = 0.75
CLAIMS_THRESHOLD = 0.9  # an explicit completion claim ends the goal on its own
MAX_REPLY_CHARS = 12_000


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        print("invalid input", file=sys.stderr)
        return 2
    goal = str(payload.get("goal") or "").strip()
    reply = str(payload.get("reply") or "")[-MAX_REPLY_CHARS:]
    if not goal:
        print("no goal", file=sys.stderr)
        return 2
    from tools.typesafe_tool import ask_jev

    state = (
        "A standing goal was set on a working conversation. Judge the latest reply only.\n\n"
        f"GOAL:\n{goal}\n\nLATEST REPLY:\n{reply}"
    )
    questions = {
        "met": {"type": "noul", "instructions": (
            "Does the latest reply show, with concrete evidence rather than intent, that the goal has been fully met? "
            "Partial progress, plans, or 'next I will' mean no.")},
        "claims_done": {"type": "noul", "instructions": "Does the latest reply explicitly state that the goal is complete or cannot be completed?"},
        "blocked": {"type": "noul", "instructions": "Does the latest reply say the goal cannot be met, or that it is waiting on a person?"},
    }
    answer = ask_jev(state, questions)
    answers = answer.get("answers") or {}
    met = float((answers.get("met") or {}).get("noul") or 0.0)
    claims = float((answers.get("claims_done") or {}).get("noul") or 0.0)
    blocked = float((answers.get("blocked") or {}).get("noul") or 0.0)
    if not all(type((answers.get(key) or {}).get("noul")) in (int, float) and math.isfinite(answers[key]["noul"]) and 0 <= answers[key]["noul"] <= 1 for key in ("met", "claims_done", "blocked")):
        raise ValueError("judge returned incomplete verdict")
    verdict = "blocked" if blocked >= 0.9 else "done" if met >= DONE_THRESHOLD or claims >= CLAIMS_THRESHOLD else "continue"
    done = verdict == "done"
    reason = (
        f"met={met:.2f} claims_done={claims:.2f} blocked={blocked:.2f}"
        + (" · stopped: the reply says the goal cannot proceed" if blocked >= 0.9 and met < DONE_THRESHOLD else "")
        + (" · done: the reply declares completion; the met score is for the steward's review" if done and met < DONE_THRESHOLD else "")
    )
    print(json.dumps({"verdict": verdict, "done": done, "p": met, "reason": reason, "provider": answer.get("provider")}))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])
    sys.exit(main())
