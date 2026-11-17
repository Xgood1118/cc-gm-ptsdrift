# -*- coding: utf-8 -*-
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ptsdrift as mod  # noqa: E402
ptsdrift = mod

TOL = ptsdrift.ANCHOR_TOLERANCE_MS


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def linear_anchors(slope, offset, xs, noise=None):
    """构造与 sub_ms -> audio_ms 线性模型一致的锚点。"""
    anchors = []
    for i, x in enumerate(xs):
        y = slope * x + offset
        if noise is not None:
            y += noise[i % len(noise)]
        anchors.append({"sub_ms": x, "audio_ms": int(round(y))})
    return anchors


def sample_cues():
    return [
        ptsdrift.Cue(1, 1000, 3000, "第一句"),
        ptsdrift.Cue(2, 5000, 7500, "第二句"),
        ptsdrift.Cue(3, 12000, 15000, "第三句\n带换行"),
    ]


# ---------------------------------------------------------------------------
# 估计：离群点剔除
# ---------------------------------------------------------------------------


def test_estimate_rejects_outlier():
    anchors = ptsdrift.load_anchors("data/anchors.json")
    model = ptsdrift.estimate_drift(anchors)
    assert model["dropped"] >= 1, "这份锚点里有一个离群点"
    assert abs(model["slope"] - 1.0) < 0.01


def test_outlier_is_excluded_from_fit(tmp_path):
    # 真值 slope=1.0005, offset=120；两个锚点被严重错配。
    xs = list(range(1000, 120001, 4000))
    anchors = linear_anchors(1.0005, 120, xs)
    anchors[5]["audio_ms"] += 9000
    anchors[17]["audio_ms"] -= 7000

    model = ptsdrift.estimate_drift(anchors)

    assert model["dropped"] == 2
    assert model["n_inliers"] == len(xs) - 2
    assert abs(model["slope"] - 1.0005) < 1e-9
    # 锚点毫秒取整本身带来 ±0.5ms 误差，截距只要求落在这个量级内。
    assert abs(model["offset_ms"] - 120) <= 0.6

    # 被剔除的锚点确实不参与拟合：模型在它们身上明显失准。
    for bad in (anchors[5], anchors[17]):
        pred = model["slope"] * bad["sub_ms"] + model["offset_ms"]
        assert abs(pred - bad["audio_ms"]) > TOL

    outlier_xs = {o["sub_ms"] for o in model["outliers"]}
    assert outlier_xs == {anchors[5]["sub_ms"], anchors[17]["sub_ms"]}


def test_noise_does_not_produce_false_outliers():
    xs = list(range(2000, 90001, 6000))
    anchors = linear_anchors(1.002, 300, xs, noise=[-9, 4, -3, 11])
    model = ptsdrift.estimate_drift(anchors)
    assert model["dropped"] == 0
    assert abs(model["slope"] - 1.002) < 5e-4
    assert abs(model["offset_ms"] - 300) < TOL


# ---------------------------------------------------------------------------
# 估计：残差上界
# ---------------------------------------------------------------------------


def test_residual_bound_covers_every_inlier():
    xs = list(range(1000, 100001, 3000))
    anchors = linear_anchors(1.0015, 250, xs, noise=[0, 6, -5, 9, -8])
    anchors[3]["audio_ms"] += 5000

    model = ptsdrift.estimate_drift(anchors)
    assert model["dropped"] >= 1

    actual = [
        abs(model["slope"] * a["sub_ms"] + model["offset_ms"] - a["audio_ms"])
        for a in anchors
        if a["sub_ms"] not in {o["sub_ms"] for o in model["outliers"]}
    ]
    assert model["residual_bound_ms"] >= max(actual)
    assert model["residual_bound_ms"] == pytest.approx(max(actual))


def test_residual_bound_is_honest_when_noise_exceeds_tolerance():
    # 噪声 ±120ms，远超 40ms 容差：残差上界必须如实报大，不能假装达标。
    xs = list(range(1000, 80001, 5000))
    anchors = linear_anchors(1.0, 0, xs, noise=[-120, 120, 0])
    model = ptsdrift.estimate_drift(anchors)
    assert model["residual_bound_ms"] > TOL


# ---------------------------------------------------------------------------
# 校正：容差、非负、顺序、内容
# ---------------------------------------------------------------------------


def test_correction_lands_within_tolerance_for_every_cue():
    slope, offset = 1.003, 480
    anchors = linear_anchors(slope, offset, [0, 15000, 30000, 45000, 60000])
    anchors[2]["audio_ms"] += 4000  # 一个错配锚点
    model = ptsdrift.estimate_drift(anchors)

    cues = [
        ptsdrift.Cue(1, 0, 1000, "a"),
        ptsdrift.Cue(2, 9000, 11000, "b"),
        ptsdrift.Cue(3, 30000, 32000, "c"),
        ptsdrift.Cue(4, 59000, 60000, "d"),
    ]
    fixed = ptsdrift.apply_drift(cues, model)
    for src, out in zip(cues, fixed):
        for key in ("start_ms", "end_ms"):
            got = getattr(out, key)
            want = slope * getattr(src, key) + offset
            assert abs(got - want) <= TOL


def test_apply_keeps_order():
    cues = ptsdrift.load_srt("data/sub.srt")
    model = {"slope": 1.0, "offset_ms": 250}
    out = ptsdrift.apply_drift(cues, model)
    assert len(out) == len(cues)
    assert all(c.start_ms >= 0 for c in out)
    assert all(c.end_ms >= c.start_ms for c in out)


def test_negative_time_never_appears():
    cues = sample_cues()
    # 负截距会把开头几条推到零点之前。
    out = ptsdrift.apply_drift(cues, {"slope": 1.0, "offset_ms": -1500})
    assert all(c.start_ms >= 0 and c.end_ms >= 0 for c in out)
    assert out[0].start_ms == 0 and out[0].end_ms == 1500

    # 时间轴严重压缩时，前一条的终点可能被压到后一条的起点之后。
    out = ptsdrift.apply_drift(cues, {"slope": 0.5, "offset_ms": -900})
    assert all(c.start_ms >= 0 and c.end_ms >= 0 for c in out)
    assert all(c.end_ms >= c.start_ms for c in out)


def test_cues_never_go_backwards():
    cues = sample_cues()
    out = ptsdrift.apply_drift(cues, {"slope": 0.9, "offset_ms": -12000})
    starts = [c.start_ms for c in out]
    assert starts == sorted(starts)
    assert all(c.end_ms >= c.start_ms for c in out)


def test_apply_preserves_count_text_and_index():
    cues = sample_cues()
    out = ptsdrift.apply_drift(cues, {"slope": 1.004, "offset_ms": 250})
    assert len(out) == len(cues)
    assert [c.index for c in out] == [c.index for c in cues]
    assert [c.text for c in out] == [c.text for c in cues]
    # 不原地改动输入。
    assert cues[0].start_ms == 1000


def test_apply_does_not_invert_a_cue_with_zero_duration_model():
    cues = [ptsdrift.Cue(1, 500, 500, "x")]
    out = ptsdrift.apply_drift(cues, {"slope": 1.0, "offset_ms": 0})
    assert out[0].start_ms == out[0].end_ms == 500


def test_apply_rejects_nonsense_model():
    with pytest.raises(ValueError):
        ptsdrift.apply_drift(sample_cues(), {"slope": 0, "offset_ms": 0})
    with pytest.raises(ValueError):
        ptsdrift.apply_drift(sample_cues(), {"slope": float("nan"), "offset_ms": 0})


# ---------------------------------------------------------------------------
# 退化路径
# ---------------------------------------------------------------------------


def test_single_anchor_degrades_to_constant_shift():
    model = ptsdrift.estimate_drift([{"sub_ms": 4000, "audio_ms": 4250}])
    assert model["degenerate"] is True
    assert model["slope"] == 1.0
    assert model["offset_ms"] == pytest.approx(250)
    assert "常量平移" in model["reason"]
    assert model["dropped"] == 0


def test_two_anchors_degrade_to_constant_shift():
    anchors = [{"sub_ms": 1000, "audio_ms": 1150}, {"sub_ms": 9000, "audio_ms": 9550}]
    model = ptsdrift.estimate_drift(anchors)
    assert model["degenerate"] is True
    assert model["slope"] == 1.0
    assert model["offset_ms"] == pytest.approx(350)  # 两个平移量 150/550 的中位数
    assert "2 个锚点" in model["reason"]


def test_empty_anchors_are_identity():
    model = ptsdrift.estimate_drift([])
    assert model["degenerate"] is True
    assert model["slope"] == 1.0 and model["offset_ms"] == 0.0
    cues = sample_cues()
    out = ptsdrift.apply_drift(cues, model)
    assert [(c.start_ms, c.end_ms) for c in out] == [(c.start_ms, c.end_ms) for c in cues]


def test_anchors_with_identical_x_degrade():
    anchors = [{"sub_ms": 5000, "audio_ms": 5000}, {"sub_ms": 5000, "audio_ms": 5400}]
    model = ptsdrift.estimate_drift(anchors)
    assert model["degenerate"] is True
    assert model["slope"] == 1.0


# ---------------------------------------------------------------------------
# 极小漂移不强行校正
# ---------------------------------------------------------------------------


def test_tiny_drift_is_not_forced():
    anchors = linear_anchors(1.0, 250, [1000, 20000, 40000, 60000])
    model = ptsdrift.estimate_drift(anchors)
    assert model["slope"] == 1.0
    assert model["slope_applied"] is False
    assert model["degenerate"] is False  # 不是锚点不足，而是主动不校正
    assert model["reason"]

    cues = sample_cues()
    out = ptsdrift.apply_drift(cues, model)
    for src, got in zip(cues, out):
        assert got.start_ms == src.start_ms + 250
        assert got.end_ms == src.end_ms + 250


def test_sub_tolerance_slope_drift_is_not_forced():
    # 60s 跨度上 slope 偏差 0.0002 -> 12ms 累计漂移 < MIN_DRIFT_MS(20ms)
    anchors = linear_anchors(1.0002, 100, [0, 20000, 40000, 60000])
    model = ptsdrift.estimate_drift(anchors)
    assert model["slope_applied"] is False
    assert abs(model["raw_slope"] - 1.0002) < 1e-9  # 原始斜率仍如实保留


def test_real_drift_is_applied():
    anchors = linear_anchors(1.006, 250, [0, 15000, 30000, 45000, 60000])
    model = ptsdrift.estimate_drift(anchors)
    assert model["slope_applied"] is True
    assert abs(model["slope"] - 1.006) < 1e-9
    assert model["drift_ppm"] == pytest.approx(6000, abs=1e-6)


def test_tiny_drift_threshold_follows_tolerance(monkeypatch):
    # 15ms 累计漂移：默认容差 40ms -> 阈值 20ms，不校正；容差 10ms -> 阈值 5ms，要校正。
    from ptsdrift import core

    anchors = linear_anchors(1.00025, 100, [0, 20000, 40000, 60000])  # 累计 15ms
    assert ptsdrift.estimate_drift(anchors)["slope_applied"] is False

    monkeypatch.setattr(core, "ANCHOR_TOLERANCE_MS", 10.0)
    model = ptsdrift.estimate_drift(anchors)
    assert abs(model["slope"] - 1.00025) < 1e-9
    assert model["tolerance_ms"] == 10.0


# ---------------------------------------------------------------------------
# SRT 读写与字节一致
# ---------------------------------------------------------------------------


def test_srt_roundtrip(tmp_path):
    path = tmp_path / "roundtrip.srt"
    cues = sample_cues()
    ptsdrift.write_srt(str(path), cues)
    loaded = ptsdrift.load_srt(str(path))
    assert loaded == cues


def test_load_srt_handles_bom_crlf_and_dot_separator(tmp_path):
    path = tmp_path / "weird.srt"
    path.write_bytes(
        "﻿1\r\n00:00:01.5 --> 00:00:03.250\r\n你好\r\n\r\n".encode("utf-8")
    )
    cues = ptsdrift.load_srt(str(path))
    assert len(cues) == 1
    assert (cues[0].start_ms, cues[0].end_ms) == (1500, 3250)
    assert cues[0].text == "你好"


def test_write_srt_is_byte_stable(tmp_path):
    a = tmp_path / "a.srt"
    b = tmp_path / "b.srt"
    ptsdrift.write_srt(str(a), sample_cues())
    ptsdrift.write_srt(str(b), sample_cues())
    assert read_bytes(str(a)) == read_bytes(str(b))
    assert b"\r\n" not in read_bytes(str(a))


def test_source_subtitle_is_not_modified(tmp_path):
    before = read_bytes("data/sub.srt")
    cues = ptsdrift.load_srt("data/sub.srt")
    ptsdrift.write_srt(str(tmp_path / "out.srt"), ptsdrift.apply_drift(cues, {"slope": 1.0, "offset_ms": 9}))
    assert read_bytes("data/sub.srt") == before


def test_cli_runs_are_byte_identical(tmp_path):
    out1 = tmp_path / "fixed1.srt"
    out2 = tmp_path / "fixed2.srt"
    cmd = [
        sys.executable,
        "-m",
        "ptsdrift.cli",
        "data/sub.srt",
        "--anchors",
        "data/anchors.json",
        "--out",
    ]
    for out in (out1, out2):
        subprocess.run(cmd + [str(out)], cwd=ROOT, check=True, capture_output=True)
    assert read_bytes(str(out1)) == read_bytes(str(out2))
    assert os.path.getsize(str(out1)) > 0


def test_cli_json_output_is_stable_and_complete(tmp_path):
    cmd = [
        sys.executable,
        "-m",
        "ptsdrift.cli",
        "data/sub.srt",
        "--anchors",
        "data/anchors.json",
        "--out",
        str(tmp_path / "fixed.srt"),
        "--json",
    ]
    first = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    second = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    assert first == second
    data = json.loads(first)
    assert {"slope", "offset_ms", "residual_bound_ms", "dropped"} <= set(data)
    assert data["dropped"] == 1
    assert data["cues"] == 2


def test_cli_tolerance_flag_reaches_estimator(tmp_path):
    out = tmp_path / "fixed.srt"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ptsdrift.cli",
            "data/sub.srt",
            "--anchors",
            "data/anchors.json",
            "--out",
            str(out),
            "--json",
            "--tolerance-ms",
            "5",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(proc.stdout)["tolerance_ms"] == 5.0


def test_cli_degenerate_anchor_file(tmp_path):
    anchors = tmp_path / "one.json"
    anchors.write_text(json.dumps([{"sub_ms": 1000, "audio_ms": 1300}]), encoding="utf-8")
    out = tmp_path / "fixed.srt"
    proc = subprocess.run(
        [sys.executable, "-m", "ptsdrift.cli", "data/sub.srt", "--anchors", str(anchors), "--out", str(out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "退化" in proc.stdout
    fixed = ptsdrift.load_srt(str(out))
    assert fixed[0].start_ms == 1300
