# -*- coding: utf-8 -*-
"""命令行入口：``python -m ptsdrift.cli data/sub.srt --anchors ... --out ...``"""

import argparse
import json
import sys

import ptsdrift as m
from ptsdrift import core as _core


def _fmt(value, digits=6):
    return ("%." + str(digits) + "f") % value


def main(argv=None):
    ap = argparse.ArgumentParser(description="字幕 PTS 漂移校正")
    ap.add_argument("srt", help="输入 SRT 路径")
    ap.add_argument("--anchors", required=True, help="锚点 JSON 路径")
    ap.add_argument("--out", required=True, help="输出 SRT 路径")
    ap.add_argument(
        "--tolerance-ms",
        type=float,
        default=None,
        help="锚点残差容差（默认 %.1f ms）" % m.ANCHOR_TOLERANCE_MS,
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出估计结果（字段顺序固定，便于比对）",
    )
    a = ap.parse_args(argv)

    if a.tolerance_ms is not None:
        if a.tolerance_ms <= 0:
            ap.error("--tolerance-ms 必须为正数")
        _core.ANCHOR_TOLERANCE_MS = float(a.tolerance_ms)

    cues = m.load_srt(a.srt)
    anchors = m.load_anchors(a.anchors)
    model = m.estimate_drift(anchors)
    fixed = m.apply_drift(cues, model)
    m.write_srt(a.out, fixed)

    if a.json:
        print(
            json.dumps(
                {
                    "anchors": model["n_anchors"],
                    "inliers": model["n_inliers"],
                    "dropped": model["dropped"],
                    "slope": round(model["slope"], 9),
                    "drift_ppm": round(model["drift_ppm"], 6),
                    "drift_ms_per_min": round(model["drift_ms_per_min"], 6),
                    "offset_ms": round(model["offset_ms"], 6),
                    "residual_bound_ms": round(model["residual_bound_ms"], 6),
                    "tolerance_ms": model["tolerance_ms"],
                    "degenerate": model["degenerate"],
                    "slope_applied": model["slope_applied"],
                    "cues": len(fixed),
                    "reason": model["reason"],
                },
                ensure_ascii=False,
                sort_keys=False,
            )
        )
        return 0

    print("锚点: %d  内点: %d  剔除: %d" % (model["n_anchors"], model["n_inliers"], model["dropped"]))
    print(
        "漂移率: %s  (%s ppm, %s ms/min)"
        % (_fmt(model["slope"]), _fmt(model["drift_ppm"], 3), _fmt(model["drift_ms_per_min"], 3))
    )
    print("截距: %s ms" % _fmt(model["offset_ms"], 3))
    print(
        "最大残差: %s ms  (容差 %s ms)"
        % (_fmt(model["residual_bound_ms"], 3), _fmt(model["tolerance_ms"], 1))
    )
    if model["degenerate"]:
        print("退化: 是")
    elif not model["slope_applied"]:
        print("斜率校正: 未套用")
    print("字幕: %d 条 -> %s" % (len(fixed), a.out))
    if model["reason"]:
        print("说明: %s" % model["reason"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
