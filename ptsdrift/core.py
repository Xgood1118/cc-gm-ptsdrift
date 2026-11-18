# -*- coding: utf-8 -*-
"""字幕 PTS 漂移校正。

只用 Python 标准库。锚点支持两种键名：

* ``{"sub_ms", "audio_ms"}``（data/anchors.json 的格式）
* ``{"media_ms", "real_ms"}``（分段测试的格式）

模型约定：把字幕时间轴 ``x`` 映射到真实（音轨）时间轴 ``y``：

    y = slope * x + offset_ms

``slope == 1.0`` 表示没有速率漂移，只剩常量平移。
"""

import bisect
import json
import math
import re
from dataclasses import dataclass

# 默认残差容差（毫秒）。estimate_drift 没有 tol 入参，使用此常量；
# 分段接口的 tol_ms 由调用方显式给出。
DEFAULT_TOL_MS = 50.0

# 漂移率可忽略判据：斜率在锚点跨度上造成的最大时间偏差小于该值（毫秒），
# 就认为漂移率极小，降级为常量平移，避免把本来正确的字幕推歪。
NEGLIGIBLE_EFFECT_MS = 1.0

# 回退原因码，可在返回结果的 "reason" 字段查询。
REASON_ONE_ANCHOR = "only_one_anchor"
REASON_TWO_ANCHORS = "two_anchors_degenerate_constant"
REASON_TOO_FEW = "segment_too_few_anchors"
REASON_POOR_FIT = "linear_fit_residual_exceeds_tol"
REASON_TOO_FEW_INLIERS = "linear_fit_too_few_inliers"
REASON_NEGLIGIBLE = "negligible_drift_rate"


@dataclass
class Cue:
    """一条字幕。"""
    index: int
    start_ms: int
    end_ms: int
    text: str


# --------------------------------------------------------------------------
# SRT 读写
# --------------------------------------------------------------------------

_TS_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def _parse_ts(h, m, s, ms):
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms.ljust(3, "0"))


def _format_ts(t):
    t = max(0, int(t))
    h, rem = divmod(t, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def load_srt(path):
    """读取 SRT，返回 ``Cue`` 列表，文本原样保留。"""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        raw = f.read()
    cues = []
    # 块之间用空行分隔
    for block in re.split(r"\n[ \t]*\n", raw.strip()):
        lines = block.splitlines()
        if not lines:
            continue
        idx = 0
        match = _TS_RE.search(lines[0])
        if match is None:
            # 第一行是序号
            idx = 1
            match = _TS_RE.search(lines[1]) if len(lines) > 1 else None
        if match is None:
            raise ValueError("无法解析的 SRT 块: %r" % block)
        g = match.groups()
        start_ms = _parse_ts(*g[0:4])
        end_ms = _parse_ts(*g[4:8])
        text = "\n".join(lines[idx + 1:]).strip("\r")
        index = int(lines[0].strip()) if idx == 1 else len(cues) + 1
        cues.append(Cue(index=index, start_ms=start_ms, end_ms=end_ms, text=text))
    return cues


def write_srt(path, cues):
    """把 ``Cue`` 列表写成 SRT，固定 LF 行尾、UTF-8 无 BOM，保证可复现。"""
    blocks = []
    for cue in cues:
        blocks.append(
            "%d\n%s --> %s\n%s"
            % (cue.index, _format_ts(cue.start_ms), _format_ts(cue.end_ms), cue.text)
        )
    payload = "\n\n".join(blocks) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(payload)


# --------------------------------------------------------------------------
# 锚点
# --------------------------------------------------------------------------

def load_anchors(path):
    """读入锚点 JSON，保留原始键名（拟合时再归一化）。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("锚点文件必须是 JSON 数组")
    return data


def _xy(anchor):
    """把两种键名的锚点归一化成 (x=字幕时间, y=真实时间)。"""
    if "sub_ms" in anchor:
        return float(anchor["sub_ms"]), float(anchor["audio_ms"])
    if "media_ms" in anchor:
        return float(anchor["media_ms"]), float(anchor["real_ms"])
    if "x_ms" in anchor:
        return float(anchor["x_ms"]), float(anchor["y_ms"])
    raise ValueError("无法识别的锚点字段: %r" % anchor)


def _sorted_points(anchors):
    pts = sorted((_xy(a) for a in anchors), key=lambda p: p[0])
    return pts


def _median(values):
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _ols(points):
    """最小二乘 y = slope*x + intercept。"""
    n = len(points)
    sx = math.fsum(x for x, _ in points)
    sy = math.fsum(y for _, y in points)
    mx, my = sx / n, sy / n
    sxx = math.fsum((x - mx) ** 2 for x, _ in points)
    sxy = math.fsum((x - mx) * (y - my) for x, y in points)
    if sxx == 0.0:
        return 1.0, my - mx
    slope = sxy / sxx
    return slope, my - slope * mx


def _residuals(points, slope, intercept):
    return [y - (slope * x + intercept) for x, y in points]


def _robust_sigma(residuals):
    """基于 MAD 的稳健尺度估计；完全共线时返回 0。"""
    if len(residuals) < 2:
        return 0.0
    med = _median(residuals)
    mad = _median([abs(r - med) for r in residuals])
    return 1.4826 * mad


def _robust_linear(points, tol_ms):
    """稳健直线拟合：LMedS 选初值 + MAD 剔离群点 + 迭代重拟合。

    返回 (slope, intercept, inlier下标列表, 内点残差上界)。
    """
    n = len(points)
    # 1) 所有两点连线作为候选，取中位绝对残差最小者（LMedS）
    best = None  # (score, slope, intercept)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            if xj == xi:
                continue
            slope = (yj - yi) / (xj - xi)
            intercept = yi - slope * xi
            abs_res = sorted(abs(r) for r in _residuals(points, slope, intercept))
            score = abs_res[len(abs_res) // 2]
            if best is None or score < best[0]:
                best = (score, slope, intercept)
    if best is None:
        # 所有 x 相同，退化为常量
        off = _median([y - x for x, y in points])
        return 1.0, off, list(range(n)), _max_abs(
            [y - x - off for x, y in points]
        )

    slope, intercept = best[1], best[2]
    # 2) 迭代：按稳健尺度剔离群点后重拟合，直到内点集合稳定
    inliers = list(range(n))
    for _ in range(n + 2):
        res = _residuals(points, slope, intercept)
        sigma = _robust_sigma(res)
        threshold = max(tol_ms, 3.5 * sigma) if sigma > 0 else tol_ms
        new_inliers = [i for i, r in enumerate(res) if abs(r) <= threshold]
        if len(new_inliers) < 2:
            break
        new_slope, new_intercept = _ols([points[i] for i in new_inliers])
        if new_inliers == inliers and new_slope == slope and new_intercept == intercept:
            break
        inliers = new_inliers
        slope, intercept = new_slope, new_intercept
    res = _residuals(points, slope, intercept)
    in_res = [abs(res[i]) for i in inliers]
    bound = _cover_bound(in_res)
    return slope, intercept, inliers, bound


def _max_abs(values):
    return max((abs(v) for v in values), default=0.0)


def _cover_bound(values):
    """残差上界：最大值加一个相对误差余量，保证浮点情况下仍覆盖全部内点。"""
    m = _max_abs(values)
    return m + 1e-12 * max(1.0, m) + 1e-9


def _constant_fit(points, reason):
    """常量平移模型：slope 固定 1.0，offset 取漂移量中位数。"""
    shifts = [y - x for x, y in points]
    offset = _median(shifts)
    bound = _cover_bound([d - offset for d in shifts])
    return {
        "model": "constant",
        "slope": 1.0,
        "offset_ms": offset,
        "inliers": len(points),
        "dropped": 0,
        "residual": bound,
        "reason": reason,
    }


# --------------------------------------------------------------------------
# 全局漂移估计
# --------------------------------------------------------------------------

def estimate_drift(anchors):
    """估计全局线性漂移，自动剔除离群锚点。

    返回 dict：slope / offset_ms / residual（内点残差上界）/
    inliers（内点在输入中的下标）/ dropped（剔除点数）/ model / reason。
    一两个锚点时退化为常量平移并写明 reason。
    """
    pts = _sorted_points(anchors)
    n = len(pts)
    if n == 0:
        raise ValueError("至少需要一个锚点")
    if n == 1:
        out = _constant_fit(pts, REASON_ONE_ANCHOR)
        out["inliers"] = [0]
        return out
    if n == 2:
        out = _constant_fit(pts, REASON_TWO_ANCHORS)
        out["inliers"] = [0, 1]
        return out

    slope, intercept, inliers, bound = _robust_linear(pts, DEFAULT_TOL_MS)
    dropped = n - len(inliers)
    span = pts[inliers[-1]][0] - pts[inliers[0]][0] if len(inliers) >= 2 else 0.0

    # 漂移率极小：斜率在整个跨度上的影响不足 1ms，降级常量平移
    if abs(slope - 1.0) * max(span, 1.0) < NEGLIGIBLE_EFFECT_MS:
        sub = [pts[i] for i in inliers]
        out = _constant_fit(sub, REASON_NEGLIGIBLE)
        out["inliers"] = inliers
        out["dropped"] = dropped
        return out

    return {
        "model": "linear",
        "slope": slope,
        "offset_ms": intercept,
        "inliers": inliers,
        "dropped": dropped,
        "residual": bound,
    }


# --------------------------------------------------------------------------
# 校正
# --------------------------------------------------------------------------

def _segment_at(segments, t):
    """按字幕时间 t 选段；落在范围外时贴到首段/末段。"""
    starts = [s["x_start"] for s in segments]
    k = bisect.bisect_right(starts, t) - 1
    if k < 0:
        k = 0
    elif k >= len(segments):
        k = len(segments) - 1
    return segments[k]


def apply_drift(cues, model):
    """按模型把字幕时间轴校正到真实时间轴。

    * 支持全局模型（estimate_drift 的返回值）和分段模型（含 "segments"）。
    * 常量模型只平移不变速；结果保证非负、不出现倒序，条目数与文本不变。
    """
    segments = model.get("segments")
    if segments:
        def map_time(t):
            seg = _segment_at(segments, t)
            return seg["slope"] * t + seg["offset_ms"]
    else:
        slope = float(model.get("slope", 1.0))
        offset = float(model.get("offset_ms", 0.0))
        # 防御：即使上游没标 constant，极微小斜率也不平变速
        if abs(slope - 1.0) < 1e-12:
            slope = 1.0

        def map_time(t):
            return slope * t + offset

    out = []
    prev_start = 0
    for cue in cues:
        start = int(round(map_time(cue.start_ms)))
        end = int(round(map_time(cue.end_ms)))
        if start < 0:
            start = 0
        if start < prev_start:
            start = prev_start
        if end < start:
            end = start
        prev_start = start
        out.append(Cue(index=cue.index, start_ms=start, end_ms=end, text=cue.text))
    return out


# --------------------------------------------------------------------------
# 断点检测与分段拟合
# --------------------------------------------------------------------------

def _interval_slopes(points):
    """相邻锚点间的局部斜率；零宽度区间返回 None。"""
    out = []
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        out.append(None if x1 == x0 else (y1 - y0) / (x1 - x0))
    return out


def _slope_jump_cuts(points):
    """用局部斜率的显著跳变初筛断点，返回切点下标 k（points[k] 为右段首点）。"""
    slopes = [s for s in _interval_slopes(points) if s is not None]
    if len(slopes) < 2:
        return []
    sigma = _robust_sigma(slopes)
    threshold = max(1e-12, 6.0 * sigma)
    raw = _interval_slopes(points)
    cuts = []
    for j in range(len(raw) - 1):
        if raw[j] is None or raw[j + 1] is None:
            continue
        if abs(raw[j + 1] - raw[j]) > threshold:
            cut = j + 1  # 跳变位于锚点 j+1 处
            # 每侧至少留 2 个锚点，相邻切点至少间隔 2 个锚点
            if cut >= 2 and len(points) - cut >= 2:
                if not cuts or cut - cuts[-1] >= 2:
                    cuts.append(cut)
    return cuts


def _sse(points):
    if len(points) < 2:
        return 0.0
    slope, intercept = _ols(points)
    return math.fsum(r ** 2 for r in _residuals(points, slope, intercept))


def _best_split(points, lo, hi):
    """枚举切点，使两侧 OLS 残差平方和之和最小；每侧至少 2 点。"""
    best = None
    for k in range(lo + 2, hi - 1):
        score = _sse(points[lo:k]) + _sse(points[k:hi])
        if best is None or score < best[0]:
            best = (score, k)
    return best[1] if best else None


def _fit_segment(points, lo, hi, tol_ms):
    """拟合一个区段，拟合优度不足时保守回退到常量平移。"""
    sub = points[lo:hi]
    n = hi - lo
    if n <= 2:
        seg = _constant_fit(sub, REASON_TOO_FEW)
    else:
        slope, intercept, inliers, bound = _robust_linear(sub, tol_ms)
        if len(inliers) < 3:
            seg = _constant_fit(sub, REASON_TOO_FEW_INLIERS)
        elif bound > tol_ms:
            seg = _constant_fit(sub, REASON_POOR_FIT)
        else:
            span = sub[-1][0] - sub[0][0]
            if abs(slope - 1.0) * max(span, 1.0) < NEGLIGIBLE_EFFECT_MS:
                seg = _constant_fit(
                    [sub[i] for i in inliers], REASON_NEGLIGIBLE
                )
                seg["dropped"] = n - len(inliers)
            else:
                seg = {
                    "model": "linear",
                    "slope": slope,
                    "offset_ms": intercept,
                    "inliers": len(inliers),
                    "dropped": n - len(inliers),
                    "residual": bound,
                }
    seg["x_start"] = sub[0][0]
    seg["x_end"] = sub[-1][0]
    return seg


def _linear_residual_bound(points, tol_ms):
    """对整段做稳健直线拟合，返回 (残差上界, 内点数)。"""
    if len(points) < 3:
        return float("inf"), len(points)
    _, _, inliers, bound = _robust_linear(points, tol_ms)
    return bound, len(inliers)


def _partition(points, tol_ms):
    """返回区段下标范围列表 [(lo, hi), ...]。"""
    # 斜率跳变只在“两侧各自都是好直线”时才采纳，避免把噪声误判成大量断点
    raw_cuts = _slope_jump_cuts(points)
    cuts = []
    for cut in raw_cuts:
        left_bound, left_n = _linear_residual_bound(points[:cut], tol_ms)
        right_bound, right_n = _linear_residual_bound(points[cut:], tol_ms)
        if (
            left_n >= 3
            and right_n >= 3
            and left_bound <= tol_ms
            and right_bound <= tol_ms
        ):
            cuts.append(cut)
    ranges = []
    start = 0
    for cut in cuts + [len(points)]:
        ranges.append((start, cut))
        start = cut

    # 对每个初筛区段做自上而下的残差分裂，兜住噪声掩盖下的断点。
    # 仅看残差上界会漏掉“两段斜率接近、单直线也能近似”的断点，
    # 因此再比较分裂前后的 SSE：两段各自合格且 SSE 显著下降才切。
    final_ranges = []

    def refine(lo, hi):
        if hi - lo <= 2:
            final_ranges.append((lo, hi))
            return
        k = _best_split(points, lo, hi)
        if k is None or k - lo < 3 or hi - k < 3:
            final_ranges.append((lo, hi))
            return
        left_bound, left_n = _linear_residual_bound(points[lo:k], tol_ms)
        right_bound, right_n = _linear_residual_bound(points[k:hi], tol_ms)
        left_ok = left_n >= 3 and left_bound <= tol_ms
        right_ok = right_n >= 3 and right_bound <= tol_ms
        if not (left_ok and right_ok):
            final_ranges.append((lo, hi))
            return
        sse_whole = _sse(points[lo:hi])
        sse_split = _sse(points[lo:k]) + _sse(points[k:hi])
        # 分裂至少消除一半残差能量，才算真实结构而非噪声
        if sse_whole <= 0.0 or sse_split > 0.5 * sse_whole:
            final_ranges.append((lo, hi))
            return
        refine(lo, k)
        refine(k, hi)

    for lo, hi in ranges:
        refine(lo, hi)
    return final_ranges


def detect_breakpoints(anchors, tol_ms=5.0):
    """识别漂移断点，返回断点在字幕时间轴上的位置（毫秒）列表。"""
    pts = _sorted_points(anchors)
    if len(pts) < 4:
        return []
    ranges = _partition(pts, tol_ms)
    return [pts[lo][0] for lo, _ in ranges[1:]]


def segment_drift(anchors, tol_ms=5.0):
    """分段拟合。

    返回 list[dict]，每段含 model / slope / offset_ms / inliers /
    dropped / residual / x_start / x_end；回退段额外带 reason。
    """
    pts = _sorted_points(anchors)
    if not pts:
        raise ValueError("至少需要一个锚点")
    ranges = _partition(pts, tol_ms) if len(pts) >= 4 else [(0, len(pts))]
    return [_fit_segment(pts, lo, hi, tol_ms) for lo, hi in ranges]


def build_segmented_model(anchors, tol_ms=DEFAULT_TOL_MS):
    """组装可供 apply_drift 直接使用的分段模型。"""
    pts = _sorted_points(anchors)
    segments = segment_drift(anchors, tol_ms)
    breakpoints = [seg["x_start"] for seg in segments[1:]]
    return {
        "model": "segmented",
        "segments": segments,
        "breakpoints": breakpoints,
        "tol_ms": float(tol_ms),
    }
