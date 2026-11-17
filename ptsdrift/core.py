# -*- coding: utf-8 -*-
"""字幕 PTS 漂移校正。

模型约定（字幕时间轴 -> 音轨时间轴）::

    audio_ms = slope * sub_ms + offset_ms

``slope`` 是漂移率（音轨毫秒 / 字幕毫秒），``offset_ms`` 是截距（sub_ms 为 0
时的音轨时间）。``slope`` 略大于 1 表示音轨时钟走得比字幕快。

估计采用确定性的 RANSAC 式拟合：枚举锚点两两连成的直线，以「落在容差内的
锚点数」评分，再对内点做最小二乘精修。离群锚点不参与拟合，只计入 ``dropped``。
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass

__all__ = [
    "Cue",
    "load_srt",
    "write_srt",
    "estimate_drift",
    "apply_drift",
    "load_anchors",
    "ANCHOR_TOLERANCE_MS",
    "MIN_DRIFT_MS",
]

#: 锚点残差容差（毫秒）。这是「锚点算内点」和「校正结果算合格」的同一把尺子。
ANCHOR_TOLERANCE_MS = 40.0

#: 判定「漂移率极小」的阈值（毫秒）的默认值，恒等于当前容差的一半。
#: 实际判定用 ``_min_drift_ms()``，这样 ``--tolerance-ms`` 改容差时阈值同步变化。
MIN_DRIFT_MS = ANCHOR_TOLERANCE_MS / 2.0

#: 参与一致性投票的候选直线上限。锚点很少时枚举全部点对；超过该值时改为
#: 固定种子的确定性抽样 + 相邻点对，保证耗时可控且结果仍可逐字节复现。
MAX_PAIR_CANDIDATES = 4096

_TIME_RE = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")
_ARROW_RE = re.compile(r"-->")


@dataclass
class Cue:
    """一条字幕。"""

    index: int
    start_ms: int
    end_ms: int
    text: str


# ---------------------------------------------------------------------------
# SRT 读写
# ---------------------------------------------------------------------------


def _time_to_ms(parts):
    h, m, s, frac = parts
    # 允许 1~3 位毫秒：'5' -> 500ms，'05' -> 50ms。
    ms = int((frac + "00")[:3])
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1000 + ms


def _format_ms(ms):
    ms = int(ms)
    if ms < 0:  # 防御：SRT 没有负时间表示
        ms = 0
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def load_srt(path):
    """读 SRT，返回 ``list[Cue]``。

    兼容 UTF-8 BOM、CRLF、多行文本；时间分隔符兼容逗号与点号。
    文本按原样保留（多行以 ``\\n`` 连接）。
    """
    with open(path, "r", encoding="utf-8-sig", newline=None) as f:
        lines = f.read().splitlines()

    cues = []
    i = 0
    n = len(lines)
    while i < n:
        while i < n and lines[i].strip() == "":
            i += 1
        if i >= n:
            break

        if _ARROW_RE.search(lines[i]):
            # 缺少序号行：以相邻序号推断，保持可用。
            index = (cues[-1].index + 1) if cues else 1
            timing = lines[i]
            i += 1
        else:
            raw_index = lines[i].strip()
            if not raw_index.isdigit():
                raise ValueError("第 %d 行不是合法的 SRT 序号: %r" % (i + 1, lines[i]))
            index = int(raw_index)
            i += 1
            if i >= n or not _ARROW_RE.search(lines[i]):
                raise ValueError("第 %d 行缺少 SRT 时间轴行" % (i + 1))
            timing = lines[i]
            i += 1

        stamps = _TIME_RE.findall(timing)
        if len(stamps) < 2:
            raise ValueError("无法解析时间轴: %r" % timing)
        start_ms = _time_to_ms(stamps[0])
        end_ms = _time_to_ms(stamps[1])

        text_lines = []
        while i < n and lines[i].strip() != "":
            text_lines.append(lines[i])
            i += 1

        cues.append(Cue(index=index, start_ms=start_ms, end_ms=end_ms, text="\n".join(text_lines)))
    return cues


def write_srt(path, cues):
    """写 SRT。

    固定使用 UTF-8（无 BOM）与 ``\\n`` 换行，保证同一输入在任意平台、
    任意次数下产生的字节完全一致。
    """
    parts = []
    for cue in cues:
        parts.append(str(cue.index))
        parts.append("%s --> %s" % (_format_ms(cue.start_ms), _format_ms(cue.end_ms)))
        parts.append(cue.text)
        parts.append("")
    data = "\n".join(parts)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(data)


# ---------------------------------------------------------------------------
# 锚点
# ---------------------------------------------------------------------------


def _coerce_anchor(anchor):
    if isinstance(anchor, dict):
        x = anchor.get("sub_ms", anchor.get("sub"))
        y = anchor.get("audio_ms", anchor.get("audio"))
    elif isinstance(anchor, (list, tuple)) and len(anchor) >= 2:
        x, y = anchor[0], anchor[1]
    else:
        raise TypeError("锚点必须是 {sub_ms, audio_ms} 或 (sub_ms, audio_ms): %r" % (anchor,))
    if x is None or y is None:
        raise ValueError("锚点缺少 sub_ms/audio_ms: %r" % (anchor,))
    try:
        xf, yf = float(x), float(y)
    except (TypeError, ValueError):
        raise ValueError("锚点时间必须是数值: %r" % (anchor,))
    if not (math.isfinite(xf) and math.isfinite(yf)):
        raise ValueError("锚点时间必须是有限数值: %r" % (anchor,))
    return xf, yf


def _coerce_anchors(anchors):
    points = [_coerce_anchor(a) for a in anchors]
    points.sort(key=lambda p: (p[0], p[1]))  # 固定顺序，保证结果可复现
    return points


def load_anchors(path):
    """读入锚点 JSON，返回 ``[{"sub_ms": int, "audio_ms": int}, ...]``。"""
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("anchors", [])
    if not isinstance(data, list):
        raise ValueError("锚点文件必须是数组或含 anchors 数组的对象")
    out = []
    for anchor in data:
        x, y = _coerce_anchor(anchor)
        out.append({"sub_ms": x, "audio_ms": y})
    return out


# ---------------------------------------------------------------------------
# 估计
# ---------------------------------------------------------------------------


def _median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return float(vals[mid])
    return (vals[mid - 1] + vals[mid]) / 2.0


def _ols(points):
    """最小二乘直线，返回 (slope, intercept)；点数不足或 x 全相同则返回 None。"""
    m = len(points)
    if m < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx = math.fsum(xs) / m
    my = math.fsum(ys) / m
    sxx = math.fsum((x - mx) * (x - mx) for x in xs)
    if sxx == 0.0:
        return None
    sxy = math.fsum((x - mx) * (y - my) for x, y in points)
    slope = sxy / sxx
    return slope, my - slope * mx


def _inliers(points, slope, offset, tol):
    return [
        k
        for k, (x, y) in enumerate(points)
        if abs(y - (slope * x + offset)) <= tol
    ]


def _candidate_pairs(n):
    """确定性地给出参与投票的点对下标。

    点对数量不大时全部枚举（小样本下等同于穷举 RANSAC）；否则取相邻点对
    （漂移是缓变信号，相邻点对本身就能提供斜率候选）加上固定种子的抽样，
    使耗时与锚点数近似线性，同时保持结果可复现。
    """
    total = n * (n - 1) // 2
    if total <= MAX_PAIR_CANDIDATES:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    pairs = [(i, i + 1) for i in range(n - 1)]
    seen = set(pairs)
    rng = random.Random(0x50544452)  # 固定种子，抽样顺序恒定
    budget = MAX_PAIR_CANDIDATES - len(pairs)
    while len(pairs) < MAX_PAIR_CANDIDATES and budget > 0:
        budget -= 1
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        pair = (i, j) if i < j else (j, i)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _robust_fit(points, tol):
    """RANSAC 式鲁棒拟合，返回 ``(slope, offset, inlier_indices)`` 或 None。"""
    n = len(points)
    candidates = []
    for i, j in _candidate_pairs(n):
        x1, y1 = points[i]
        x2, y2 = points[j]
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if not math.isfinite(slope):
            continue
        candidates.append((slope, y1 - slope * x1))

    # 候选里加一条「零漂移」基线，避免锚点很少时被一对噪声点带偏。
    candidates.append((1.0, _median([y - x for x, y in points])))

    best = None
    for slope, offset in candidates:
        inl = _inliers(points, slope, offset, tol)
        if len(inl) < 3:
            continue
        res = [abs(points[k][1] - (slope * points[k][0] + offset)) for k in inl]
        sse = math.fsum(r * r for r in res)
        key = (-len(inl), sse, max(res), abs(slope - 1.0), slope, offset)
        if best is None or key < best[0]:
            best = (key, slope, offset, inl)

    if best is None:
        return None

    _, slope, offset, inl = best

    # 用内点做最小二乘精修，并让内点集合收敛。
    for _ in range(32):
        fit = _ols([points[k] for k in inl])
        if fit is None:
            return None
        slope, offset = fit
        new_inl = _inliers(points, slope, offset, tol)
        if len(new_inl) < 3:
            return None
        if new_inl == inl:
            return slope, offset, inl
        inl = new_inl
    return None  # 内点集合来回震荡：无法给出稳定解，交给退化路径


def _model(
    slope,
    offset,
    inlier_indices,
    points,
    tolerance,
    degenerate,
    slope_applied,
    reason,
    raw_slope=None,
):
    inlier_set = set(inlier_indices)
    residuals = [abs(y - (slope * x + offset)) for x, y in points]
    bound = max((residuals[k] for k in inlier_indices), default=0.0)
    ordered_inliers = sorted(inlier_set)
    ordered_outliers = [k for k in range(len(points)) if k not in inlier_set]
    return {
        "slope": float(slope),
        "offset_ms": float(offset),
        "intercept_ms": float(offset),
        "raw_slope": float(raw_slope if raw_slope is not None else slope),
        "drift_ppm": (float(slope) - 1.0) * 1_000_000.0,
        "drift_ms_per_min": (float(slope) - 1.0) * 60_000.0,
        "residual_bound_ms": float(bound),
        "max_residual_ms": float(bound),
        "dropped": int(len(points) - len(ordered_inliers)),
        "n_anchors": int(len(points)),
        "n_inliers": int(len(ordered_inliers)),
        "inliers": [
            {"sub_ms": points[k][0], "audio_ms": points[k][1]} for k in ordered_inliers
        ],
        "outliers": [
            {"sub_ms": points[k][0], "audio_ms": points[k][1]} for k in ordered_outliers
        ],
        "degenerate": bool(degenerate),
        "degraded": bool(degenerate),
        "slope_applied": bool(slope_applied),
        "tolerance_ms": float(tolerance),
        "reason": reason,
    }


def _constant_fallback(points, tol, reason):
    """常量平移退化模型（抗离群点的中位数平移）。"""
    n = len(points)
    offset = _median([y - x for x, y in points])
    inl = _inliers(points, 1.0, offset, tol)
    if n and len(inl) * 2 <= n:
        # 多数锚点连常量平移都对不上，也就没有可信的离群点判据。此时不靠丢弃
        # 大多数锚点粉饰结果，按全体锚点报告残差上界，让调用方看到拟合不可信。
        inl = list(range(n))
        reason += "；且多数锚点互相矛盾，残差上界按全体锚点计算"
    return _model(
        1.0,
        offset,
        inl,
        points,
        tol,
        degenerate=True,
        slope_applied=False,
        reason=reason,
    )


def estimate_drift(anchors):
    """估计漂移率、截距与残差上界。

    参数 ``anchors`` 接受 ``{"sub_ms", "audio_ms"}`` 字典或 ``(sub_ms, audio_ms)``
    序列。返回的字典可直接交给 :func:`apply_drift`（使用 ``slope`` 与
    ``offset_ms``）。

    退化规则：锚点少于 3 个时无法同时验证线性与剔除离群点，只做常量平移
    （``slope`` 固定为 1）；内点不足 3 个或拟合不稳定时同样退回常量平移。
    """
    points = _coerce_anchors(anchors)
    n = len(points)
    tol = ANCHOR_TOLERANCE_MS

    if n == 0:
        return _model(
            1.0,
            0.0,
            [],
            points,
            tol,
            degenerate=True,
            slope_applied=False,
            reason="没有锚点，无法估计，按恒等模型处理",
        )

    if n <= 2:
        offset = _median([y - x for x, y in points])
        if n == 1:
            reason = "只有 1 个锚点，无法确定斜率，退回常量平移"
        else:
            reason = (
                "只有 2 个锚点，两点必然完美共线却无法验证线性、也无法判断"
                "谁是离群点，退回常量平移"
            )
        return _model(
            1.0,
            offset,
            list(range(n)),
            points,
            tol,
            degenerate=True,
            slope_applied=False,
            reason=reason,
        )

    fit = _robust_fit(points, tol)
    if fit is None:
        return _constant_fallback(
            points, tol, "无法找到至少 3 个一致的锚点，退回常量平移"
        )

    slope, offset, inl = fit
    if len(inl) * 2 <= n:
        # 只有少数锚点落在同一条直线上：它们可能是巧合，不足以支撑线性结论。
        # 多个一致性很差的锚点时应退回常量平移，并如实报告全体残差。
        return _constant_fallback(
            points,
            tol,
            "最一致的直线也只覆盖 %d/%d 个锚点（不足半数），无法确认线性漂移，"
            "退回常量平移" % (len(inl), n),
        )
    raw_slope = slope
    slope_applied = True
    reason = ""

    span = max(x for x, _ in points) - min(x for x, _ in points)
    total_drift = abs(slope - 1.0) * span
    min_drift = tol / 2.0
    if total_drift < min_drift:
        # 极小漂移：斜率带来的累计偏移还不到半个容差，强行校正只会把本来
        # 正确的字幕推歪。改为仅做常量平移。
        clamped_offset = _median([points[k][1] - points[k][0] for k in inl])
        clamped_bound = max(
            (abs(points[k][1] - (points[k][0] + clamped_offset)) for k in inl),
            default=0.0,
        )
        if clamped_bound <= tol:
            slope = 1.0
            offset = clamped_offset
            new_inl = _inliers(points, slope, offset, tol)
            if len(new_inl) >= 3:
                inl = new_inl
            slope_applied = False
            reason = (
                "漂移率极小：锚点跨度 %.0f ms 上累计漂移 %.3f ms，小于阈值 %.1f ms，"
                "不套用斜率校正，只做常量平移" % (span, total_drift, min_drift)
            )
        else:
            reason = (
                "锚点跨度 %.0f ms 上累计漂移仅 %.3f ms，但常量平移会把残差推到 "
                "%.3f ms（超过容差 %.1f ms），因此保留线性校正"
                % (span, total_drift, clamped_bound, tol)
            )

    return _model(
        slope,
        offset,
        inl,
        points,
        tol,
        degenerate=False,
        slope_applied=slope_applied,
        reason=reason,
        raw_slope=raw_slope,
    )


# ---------------------------------------------------------------------------
# 校正
# ---------------------------------------------------------------------------


def _map_time(ms, slope, offset):
    # 固定「四舍五入到最近毫秒」，不用内置 round 的银行家舍入，保证可复现。
    return int(math.floor(float(ms) * slope + offset + 0.5))


def apply_drift(cues, model):
    """按模型校正字幕时间，返回新的 ``list[Cue]``。

    ``model`` 至少需要 ``slope`` 与 ``offset_ms``（也接受 ``intercept_ms``）。
    结果为非负、单条内 ``end >= start``、跨条目不倒序；条目数、序号与文本
    原样保留。
    """
    slope = float(model.get("slope", 1.0))
    offset = float(model.get("offset_ms", model.get("intercept_ms", 0.0)))
    if not math.isfinite(slope) or slope <= 0.0:
        raise ValueError("slope 必须是有限正数: %r" % (slope,))
    if not math.isfinite(offset):
        raise ValueError("offset_ms 必须是有限数值: %r" % (offset,))

    out = []
    previous_start = None
    for cue in cues:
        start = max(0, _map_time(cue.start_ms, slope, offset))
        end = max(0, _map_time(cue.end_ms, slope, offset))
        if previous_start is not None and start < previous_start:
            start = previous_start
        if end < start:
            end = start
        out.append(Cue(index=cue.index, start_ms=start, end_ms=end, text=cue.text))
        previous_start = start
    return out
