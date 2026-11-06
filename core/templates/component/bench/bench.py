"""Reference benchmark of the example component: the step plan without Blender on the reference input — the numbers
match the recorded ones.
.venv/bin/python -m components.example.bench.bench        → a __RESULT__ line and an exit code = the number of failures
"""
import json
import sys

from components.example import steps
from core.model import scene_model


def main():
    M = scene_model.load("bench/scenes/castle_rock")
    plan = steps.plan_example(M, {"seed": 3, "mode": "full"}, {"scene": M["dir"], "out": "out/dev/bench_example", "profile": "sketch",
                                                               "gives": {}, "root": ".", "step": "example"})
    fails = [k for k, want in (("seed", 3), ("mode", "full")) if plan.get(k) != want]
    print("__RESULT__" + json.dumps({"failures": len(fails), "failed": fails}))
    return len(fails)


if __name__ == "__main__":
    sys.exit(main())
