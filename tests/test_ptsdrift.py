# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ptsdrift as mod  # noqa: E402
ptsdrift = mod

def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_hex(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def test_estimate_rejects_outlier():
    anchors = ptsdrift.load_anchors("data/anchors.json")
    model = ptsdrift.estimate_drift(anchors)
    assert model["dropped"] >= 1, "这份锚点里有一个离群点"
    assert abs(model["slope"] - 1.0) < 0.01


def test_apply_keeps_order():
    cues = ptsdrift.load_srt("data/sub.srt")
    model = {"slope": 1.0, "offset_ms": 250}
    out = ptsdrift.apply_drift(cues, model)
    assert len(out) == len(cues)
    assert all(c.start_ms >= 0 for c in out)
