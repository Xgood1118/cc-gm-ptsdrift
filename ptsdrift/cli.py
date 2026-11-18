# -*- coding: utf-8 -*-
"""命令行入口：字幕 PTS 漂移校正。"""

import argparse
import json
import sys

import ptsdrift as m


def _print_model(model, stream):
    if "segments" in model:
        stream.write("model=segmented segments=%d\n" % len(model["segments"]))
        stream.write("breakpoints_ms=%s\n" % model["breakpoints"])
        for i, seg in enumerate(model["segments"]):
            line = (
                "  seg%d %s slope=%.9f offset_ms=%.3f inliers=%d "
                "dropped=%d residual_ms=%.3f range=[%s, %s]"
                % (
                    i,
                    seg["model"],
                    seg["slope"],
                    seg["offset_ms"],
                    seg["inliers"],
                    seg["dropped"],
                    seg["residual"],
                    seg["x_start"],
                    seg["x_end"],
                )
            )
            if "reason" in seg:
                line += " reason=%s" % seg["reason"]
            stream.write(line + "\n")
    else:
        stream.write(
            "model=%s slope=%.9f offset_ms=%.3f inliers=%d dropped=%d "
            "residual_ms=%.3f\n"
            % (
                model["model"],
                model["slope"],
                model["offset_ms"],
                model["inliers"] if isinstance(model["inliers"], int)
                else len(model["inliers"]),
                model["dropped"],
                model["residual"],
            )
        )
        if "reason" in model:
            stream.write("reason=%s\n" % model["reason"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="字幕 PTS 漂移校正")
    ap.add_argument("srt", help="输入 SRT 路径")
    ap.add_argument("--anchors", required=True, help="锚点 JSON 路径")
    ap.add_argument("--out", required=True, help="校正后 SRT 输出路径")
    ap.add_argument(
        "--tol-ms",
        type=float,
        default=m.DEFAULT_TOL_MS,
        help="残差容差（毫秒），默认 %.1f" % m.DEFAULT_TOL_MS,
    )
    ap.add_argument(
        "--segmented",
        action="store_true",
        help="启用分段漂移拟合（默认仅做全局估计）",
    )
    a = ap.parse_args(argv)

    anchors = m.load_anchors(a.anchors)
    cues = m.load_srt(a.srt)
    if a.segmented:
        model = m.build_segmented_model(anchors, tol_ms=a.tol_ms)
    else:
        model = m.estimate_drift(anchors)
    fixed = m.apply_drift(cues, model)
    m.write_srt(a.out, fixed)

    _print_model(model, sys.stdout)
    # 同时把模型确定性地落到 sidecar JSON，便于审计与复验
    sidecar = a.out + ".model.json"
    with open(sidecar, "w", encoding="utf-8", newline="\n") as f:
        json.dump(model, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
    sys.stdout.write("cues_in=%d cues_out=%d\n" % (len(cues), len(fixed)))
    sys.stdout.write("wrote %s (+%s)\n" % (a.out, sidecar))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
