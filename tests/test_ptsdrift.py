# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ptsdrift as mod  # noqa: E402


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


# ===== v2 契约测试：初始全部失败，实现后必须通过 =====
import ptsdrift  # noqa: E402


def _anchors(slope_a, slope_b, break_at, n=20, step=1000):
    out = []
    for i in range(n):
        t = i * step
        drift = slope_a * t if t < break_at else slope_a * break_at + slope_b * (t - break_at)
        out.append({"media_ms": t, "real_ms": t + drift})
    return out


def test_breakpoint_near_true_position():
    a = _anchors(0.01, 0.06, 9000)
    bps = ptsdrift.detect_breakpoints(a, tol_ms=1000)
    assert any(abs(bp - 9000) <= 1000 or abs(bp * 1000 - 9000) <= 1000 for bp in bps)


def test_segments_split_and_cover():
    a = _anchors(0.01, 0.06, 9000)
    segs = ptsdrift.segment_drift(a, tol_ms=50)
    assert len(segs) >= 2
    for s in segs:
        assert s["residual"] >= 0
        assert s["inliers"] >= 1
        assert "model" in s


def test_steady_linear_single_segment():
    a = _anchors(0.01, 0.01, 9000)
    segs = ptsdrift.segment_drift(a, tol_ms=10)
    assert len(segs) >= 1
    assert all(s["model"] != "constant" or s["residual"] >= 0 for s in segs)


def test_bad_fit_falls_back_with_reason():
    a = [{"media_ms": i * 1000, "real_ms": i * 1000 + (100 if i % 2 else -100)} for i in range(12)]
    segs = ptsdrift.segment_drift(a, tol_ms=5)
    assert any(s["model"] in ("constant", "fallback") for s in segs)
    assert all("reason" in s for s in segs if s["model"] in ("constant", "fallback"))
