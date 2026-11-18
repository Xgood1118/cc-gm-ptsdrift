# -*- coding: utf-8 -*-
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ptsdrift  # noqa: E402
import ptsdrift as mod  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_SRT = os.path.join(ROOT, "data", "sub.srt")
DATA_ANCHORS = os.path.join(ROOT, "data", "anchors.json")


def _anchors(slope_a, slope_b, break_at, n=20, step=1000):
    out = []
    for i in range(n):
        t = i * step
        drift = slope_a * t if t < break_at else slope_a * break_at + slope_b * (t - break_at)
        out.append({"media_ms": t, "real_ms": t + drift})
    return out


def _xy(anchor):
    if "sub_ms" in anchor:
        return float(anchor["sub_ms"]), float(anchor["audio_ms"])
    return float(anchor["media_ms"]), float(anchor["real_ms"])


# --------------------------------------------------------------------------
# 1. 估计：离群剔除、斜率/截距、残差上界
# --------------------------------------------------------------------------

def test_estimate_rejects_outlier():
    anchors = ptsdrift.load_anchors(DATA_ANCHORS)
    model = ptsdrift.estimate_drift(anchors)
    assert model["dropped"] >= 1, "这份锚点里有一个离群点"
    assert abs(model["slope"] - 1.0) < 0.01


def test_estimate_clean_linear_recovers_params():
    # y = 1.004 x + 300，外加一个明显错配点
    anchors = [{"sub_ms": x, "audio_ms": 1.004 * x + 300.0}
               for x in range(0, 100000, 10000)]
    anchors[5]["audio_ms"] += 5000.0  # 离群点
    model = ptsdrift.estimate_drift(anchors)
    assert model["model"] == "linear"
    assert abs(model["slope"] - 1.004) < 1e-6
    assert abs(model["offset_ms"] - 300.0) < 5.0
    assert model["dropped"] == 1


def test_estimate_residual_bound_covers_inliers():
    anchors = ptsdrift.load_anchors(DATA_ANCHORS)
    model = ptsdrift.estimate_drift(anchors)
    pts = [_xy(a) for a in anchors]
    for i in model["inliers"]:
        x, y = pts[i]
        residual = abs(y - (model["slope"] * x + model["offset_ms"]))
        assert residual <= model["residual"] + 1e-6


def test_estimate_accepts_alternative_key_names():
    a1 = [{"sub_ms": 1000.0, "audio_ms": 1250.0},
          {"sub_ms": 2000.0, "audio_ms": 2250.0},
          {"sub_ms": 3000.0, "audio_ms": 3250.0}]
    a2 = [{"media_ms": 1000.0, "real_ms": 1250.0},
          {"media_ms": 2000.0, "real_ms": 2250.0},
          {"media_ms": 3000.0, "real_ms": 3250.0}]
    m1 = ptsdrift.estimate_drift(a1)
    m2 = ptsdrift.estimate_drift(a2)
    assert m1["offset_ms"] == m2["offset_ms"] == 250.0


def test_estimate_drops_multiple_outliers():
    anchors = [{"sub_ms": float(x), "audio_ms": 1.002 * x}
               for x in range(0, 50000, 5000)]
    anchors[2]["audio_ms"] += 9000.0
    anchors[7]["audio_ms"] -= 8000.0
    model = ptsdrift.estimate_drift(anchors)
    assert model["dropped"] == 2
    assert abs(model["slope"] - 1.002) < 1e-5


# --------------------------------------------------------------------------
# 2. 校正：容差、非负、非倒序、文本不变
# --------------------------------------------------------------------------

def test_apply_keeps_order():
    cues = ptsdrift.load_srt(DATA_SRT)
    model = {"slope": 1.0, "offset_ms": 250}
    out = ptsdrift.apply_drift(cues, model)
    assert len(out) == len(cues)
    assert all(c.start_ms >= 0 for c in out)


def test_apply_within_tolerance():
    anchors = [{"sub_ms": float(x), "audio_ms": 1.001 * x + 120.0}
               for x in range(0, 30000, 3000)]
    model = ptsdrift.estimate_drift(anchors)
    cues = [mod.Cue(i + 1, t, t + 1500, "t%d" % i)
            for i, t in enumerate(range(0, 30000, 3000))]
    out = ptsdrift.apply_drift(cues, model)
    for cue in cues:
        expected_start = model["slope"] * cue.start_ms + model["offset_ms"]
        assert abs(out[cue.index - 1].start_ms - round(expected_start)) <= 1
        assert out[cue.index - 1].end_ms >= out[cue.index - 1].start_ms


def test_apply_no_negative_and_no_reversal():
    cues = [mod.Cue(1, 0, 500, "a"), mod.Cue(2, 600, 900, "b")]
    # 大幅负平移，第一条会被钳到 0
    out = ptsdrift.apply_drift(cues, {"slope": 1.0, "offset_ms": -800.0})
    assert out[0].start_ms == 0 and out[0].end_ms >= 0
    assert out[1].start_ms >= out[0].start_ms
    assert all(c.start_ms <= c.end_ms for c in out)


def test_apply_preserves_count_and_text():
    cues = ptsdrift.load_srt(DATA_SRT)
    model = ptsdrift.estimate_drift(ptsdrift.load_anchors(DATA_ANCHORS))
    out = ptsdrift.apply_drift(cues, model)
    assert len(out) == len(cues)
    assert [c.text for c in out] == [c.text for c in cues]
    assert [c.index for c in out] == [c.index for c in cues]


def test_apply_does_not_mutate_input():
    cues = ptsdrift.load_srt(DATA_SRT)
    before = [(c.start_ms, c.end_ms) for c in cues]
    ptsdrift.apply_drift(cues, {"slope": 1.0, "offset_ms": 250})
    assert [(c.start_ms, c.end_ms) for c in cues] == before


def test_negligible_drift_becomes_constant():
    # 斜率 1.000001 在 10s 跨度上只产生 0.01ms 影响 -> 必须降级常量
    anchors = [{"sub_ms": float(x), "audio_ms": 1.000001 * x + 40.0}
               for x in range(0, 10000, 1000)]
    model = ptsdrift.estimate_drift(anchors)
    assert model["model"] == "constant"
    assert model["slope"] == 1.0
    assert model["reason"] == ptsdrift.REASON_NEGLIGIBLE
    # 常量平移不改变相邻字幕的时间差，不会把好字幕推歪
    cues = [mod.Cue(1, 1000, 2000, "a"), mod.Cue(2, 3000, 4000, "b")]
    out = ptsdrift.apply_drift(cues, model)
    assert (out[1].start_ms - out[0].start_ms) == (3000 - 1000)


# --------------------------------------------------------------------------
# 3. 边界：一两个锚点退化
# --------------------------------------------------------------------------

def test_one_anchor_constant_translation():
    model = ptsdrift.estimate_drift([{"sub_ms": 2000.0, "audio_ms": 2300.0}])
    assert model["model"] == "constant"
    assert model["slope"] == 1.0
    assert model["offset_ms"] == 300.0
    assert model["reason"] == ptsdrift.REASON_ONE_ANCHOR


def test_two_anchors_constant_translation():
    anchors = [{"sub_ms": 1000.0, "audio_ms": 1500.0},
               {"sub_ms": 3000.0, "audio_ms": 3500.0}]
    model = ptsdrift.estimate_drift(anchors)
    assert model["model"] == "constant"
    assert model["slope"] == 1.0
    assert model["reason"] == ptsdrift.REASON_TWO_ANCHORS
    assert abs(model["offset_ms"] - 500.0) < 1e-9


def test_zero_anchors_raises():
    import pytest
    with pytest.raises(ValueError):
        ptsdrift.estimate_drift([])


def test_segment_with_few_anchors_falls_back():
    seg = ptsdrift.segment_drift(
        [{"media_ms": 0.0, "real_ms": 100.0},
         {"media_ms": 1000.0, "real_ms": 1100.0}],
        tol_ms=5.0,
    )
    assert len(seg) == 1
    assert seg[0]["model"] == "constant"
    assert seg[0]["reason"] == ptsdrift.REASON_TOO_FEW


# --------------------------------------------------------------------------
# 4. 输出可复现：两次逐字节一致
# --------------------------------------------------------------------------

def test_write_srt_byte_identical(tmp_path):
    cues = ptsdrift.load_srt(DATA_SRT)
    p1, p2 = tmp_path / "a.srt", tmp_path / "b.srt"
    ptsdrift.write_srt(str(p1), cues)
    ptsdrift.write_srt(str(p2), cues)
    assert p1.read_bytes() == p2.read_bytes()
    # LF 行尾、无 BOM
    raw = p1.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw


def test_srt_roundtrip_preserves_text(tmp_path):
    cues = ptsdrift.load_srt(DATA_SRT)
    p = tmp_path / "rt.srt"
    ptsdrift.write_srt(str(p), cues)
    again = ptsdrift.load_srt(str(p))
    assert [c.text for c in again] == [c.text for c in cues]
    assert [(c.start_ms, c.end_ms) for c in again] == \
           [(c.start_ms, c.end_ms) for c in cues]


def test_cli_runs_are_byte_identical(tmp_path):
    out1, out2 = tmp_path / "o1.srt", tmp_path / "o2.srt"
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    for out in (out1, out2):
        proc = subprocess.run(
            [sys.executable, "-m", "ptsdrift.cli", DATA_SRT,
             "--anchors", DATA_ANCHORS, "--out", str(out)],
            cwd=ROOT, capture_output=True, text=True, env=env,
        )
        assert proc.returncode == 0, proc.stderr
    assert out1.read_bytes() == out2.read_bytes()


# --------------------------------------------------------------------------
# 5. 分段漂移：断点定位、分段斜率/截距、残差覆盖
# --------------------------------------------------------------------------

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
    # 没有真实断点时不应分裂
    assert len(segs) == 1
    assert abs(segs[0]["slope"] - 1.01) < 1e-9


def test_segment_slopes_and_offsets_are_recovered():
    a = _anchors(0.01, 0.06, 9000)
    segs = ptsdrift.segment_drift(a, tol_ms=50)
    assert len(segs) == 2
    assert abs(segs[0]["slope"] - 1.01) < 1e-9
    assert abs(segs[1]["slope"] - 1.06) < 1e-9
    # 第二段在断点连续：1.06*9000 + b = 1.01*9000 -> b = -450
    assert abs(segs[1]["offset_ms"] - (-450.0)) < 1e-6


def test_segment_residual_bound_covers_all_inliers():
    a = _anchors(0.01, 0.06, 9000)
    # 给每个点注入小幅噪声，确保残差非平凡
    noisy = [{"media_ms": p["media_ms"],
              "real_ms": p["real_ms"] + (2.0 if i % 3 == 0 else 0.0)}
             for i, p in enumerate(a)]
    segs = ptsdrift.segment_drift(noisy, tol_ms=50)
    pts = [_xy(p) for p in noisy]
    for seg in segs:
        members = [p for p in pts if seg["x_start"] <= p[0] <= seg["x_end"]]
        for x, y in members:
            r = abs(y - (seg["slope"] * x + seg["offset_ms"]))
            assert r <= seg["residual"] + 1e-6


def test_multiple_breakpoints():
    # 三段不同斜率，两个真实断点
    pts = []
    for i in range(30):
        t = i * 1000
        if t < 10000:
            y = 1.01 * t
        elif t < 20000:
            y = 1.01 * 10000 + 1.03 * (t - 10000)
        else:
            y = 1.01 * 10000 + 1.03 * 10000 + 1.08 * (t - 20000)
        pts.append({"media_ms": float(t), "real_ms": y})
    bps = ptsdrift.detect_breakpoints(pts, tol_ms=10)
    assert len(bps) == 2
    assert any(abs(b - 10000) <= 1000 for b in bps)
    assert any(abs(b - 20000) <= 1000 for b in bps)
    segs = ptsdrift.segment_drift(pts, tol_ms=10)
    assert len(segs) == 3
    assert [s["model"] for s in segs] == ["linear"] * 3


def test_segmented_model_end_to_end():
    a = _anchors(0.01, 0.06, 9000)
    model = ptsdrift.build_segmented_model(a, tol_ms=50)
    assert model["model"] == "segmented"
    assert len(model["breakpoints"]) == 1
    cues = [mod.Cue(i + 1, t, t + 500, "c%d" % i)
            for i, t in enumerate(range(0, 20000, 2000))]
    out = ptsdrift.apply_drift(cues, model)
    assert len(out) == len(cues)
    assert all(c.start_ms <= c.end_ms for c in out)
    assert all(c.start_ms >= 0 for c in out)
    # 第一段内的字幕按 1.01 校正
    first = out[0]
    assert abs(first.start_ms - round(1.01 * cues[0].start_ms)) <= 1


# --------------------------------------------------------------------------
# 6. 失真时保守回退：可辨识、原因可查
# --------------------------------------------------------------------------

def test_bad_fit_falls_back_with_reason():
    a = [{"media_ms": i * 1000, "real_ms": i * 1000 + (100 if i % 2 else -100)}
         for i in range(12)]
    segs = ptsdrift.segment_drift(a, tol_ms=5)
    assert any(s["model"] in ("constant", "fallback") for s in segs)
    assert all("reason" in s for s in segs if s["model"] in ("constant", "fallback"))


def test_fallback_reason_is_queryable_and_identifiable():
    a = [{"media_ms": i * 1000, "real_ms": i * 1000 + (100 if i % 2 else -100)}
         for i in range(12)]
    segs = ptsdrift.segment_drift(a, tol_ms=5)
    reasons = {s.get("reason") for s in segs if s["model"] == "constant"}
    assert ptsdrift.REASON_POOR_FIT in reasons
    # 回退段仍是可应用的模型：slope=1，只做平移
    for s in segs:
        if s["model"] == "constant":
            assert s["slope"] == 1.0
            assert isinstance(s["offset_ms"], float)


def test_fallback_segment_residual_covers_points():
    a = [{"media_ms": i * 1000, "real_ms": i * 1000 + (100 if i % 2 else -100)}
         for i in range(12)]
    segs = ptsdrift.segment_drift(a, tol_ms=5)
    pts = [_xy(p) for p in a]
    for seg in segs:
        members = [p for p in pts if seg["x_start"] <= p[0] <= seg["x_end"]]
        for x, y in members:
            r = abs(y - (seg["slope"] * x + seg["offset_ms"]))
            assert r <= seg["residual"] + 1e-6


def test_segmented_cli_writes_model_sidecar(tmp_path):
    out = tmp_path / "seg.srt"
    anchors = tmp_path / "a.json"
    anchors.write_text(json.dumps(_anchors(0.01, 0.06, 9000)), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "ptsdrift.cli", DATA_SRT,
         "--anchors", str(anchors), "--out", str(out), "--segmented"],
        cwd=ROOT, capture_output=True, text=True,
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
    )
    assert proc.returncode == 0, proc.stderr
    sidecar = json.loads((tmp_path / "seg.srt.model.json").read_text(encoding="utf-8"))
    assert sidecar["model"] == "segmented"
    assert len(sidecar["segments"]) == 2
