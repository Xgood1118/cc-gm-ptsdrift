# 字幕 PTS 漂移校正

一个能估计并消除音轨与字幕时间基准漂移的对齐器。只用 Python 标准库，
支持全局线性漂移与带断点的分段漂移，锚点中混入错配离群点时会自动剔除。

## 模型约定

把字幕时间轴 `x` 映射到真实（音轨）时间轴 `y`：

```
y = slope * x + offset_ms
```

`slope == 1.0` 表示没有速率漂移，只剩常量平移；`model == "constant"` 时
`slope` 固定为 `1.0`，只做平移不变速。

锚点 JSON 支持两种键名（拟合时自动归一化）：

- `{"sub_ms", "audio_ms"}`：`data/anchors.json` 的格式
- `{"media_ms", "real_ms"}`：分段场景的格式

## 保留签名

- `Cue(index, start_ms, end_ms, text)`
- `load_srt(path) -> list[Cue]`
- `write_srt(path, cues)`
- `estimate_drift(anchors) -> dict`
- `apply_drift(cues, model) -> list[Cue]`

分段相关：`load_anchors`、`detect_breakpoints(anchors, tol_ms=5.0)`、
`segment_drift(anchors, tol_ms=5.0)`、`build_segmented_model`。

## 算法要点

### 估计（`estimate_drift`）

1. 枚举所有两点连线作为候选直线，取中位绝对残差最小者（LMedS）作初值，
   不被离群点带偏。
2. 用 MAD 稳健尺度（`1.4826 * median(|r - median(r)|)`）迭代剔除离群点，
   阈值为 `max(tol_ms, 3.5 * sigma)`，每次剔除后对剩余内点重拟合 OLS，
   直到内点集合稳定。
3. 返回 `slope`、`offset_ms`、`residual`（内点残差上界，已加浮点余量，
   保证覆盖全部内点）、`inliers`（内点下标）、`dropped`（剔除数）。

### 漂移率极小时不强行校正（判据）

> **判据：`abs(slope - 1.0) * span < 1.0 ms`，其中 `span` 是内点锚点在
> 字幕时间轴上的跨度。**

即斜率偏差在整个锚点跨度上造成的最大时间差不足 1 ms 时，认为漂移率可忽略，
降级为常量平移（`model="constant"`，`reason="negligible_drift_rate"`）。
常量平移不改变相邻字幕的时间差，因此不会把本来正确的字幕推歪。常量
`NEGLIGIBLE_EFFECT_MS = 1.0` 定义在 `ptsdrift/core.py`。

### 校正（`apply_drift`）

- 全局模型直接映射；分段模型（含 `"segments"`）按字幕时间用二分查找选段。
- 结果一律 `round` 到整数毫秒；起始时间钳到非负、钳到不早于上一条（不倒序），
  结束时间不早于起始时间。
- 不修改输入，条目数、序号、文本内容逐字不变。

### 退化与保守回退（原因码可在 `reason` 字段查询）

| 场景 | 模型 | reason |
| --- | --- | --- |
| 只有 1 个锚点，无法定斜率 | constant | `only_one_anchor` |
| 只有 2 个锚点，两点定线不可靠 | constant | `two_anchors_degenerate_constant` |
| 分段后某段锚点 ≤ 2 | constant | `segment_too_few_anchors` |
| 段内稳健拟合内点 < 3 | constant | `linear_fit_too_few_inliers` |
| 段内拟合残差上界超过 `tol_ms` | constant | `linear_fit_residual_exceeds_tol` |
| 漂移率影响 < 1 ms | constant | `negligible_drift_rate` |

### 分段漂移（`detect_breakpoints` / `segment_drift`）

1. **初筛**：计算相邻锚点的局部斜率，用 6 倍稳健尺度检测显著跳变；只有跳变
   两侧各自都能拟合成合格直线（残差上界 ≤ `tol_ms`）时才采纳，避免噪声被
   误判成一堆断点。
2. **残差分裂兜底**：对每个初筛区段自上而下枚举最优切点（两侧 OLS 残差平方
   和最小）。只有切点两侧各自合格、且分裂后 SSE 降到整段 SSE 的一半以下，
   才认定为真实断点——这能检出“两段斜率接近、单条直线也能近似”的弱断点。
3. 每段独立稳健拟合，给出 `slope`、`offset_ms`、`residual`（覆盖该段全部
   内点）；拟合优度不足时整段退回常量平移并写明原因，任何回退都可通过
   `model == "constant"` 与 `reason` 辨识。

## 可复现性

`write_srt` 固定输出 UTF-8 无 BOM、LF 行尾、块间双换行、统一毫秒时间戳格式；
CLI 另把模型以 `sort_keys` 写到 `<out>.model.json`。相同输入两次运行
逐字节一致（有测试 `test_cli_runs_are_byte_identical` 守护）。

## 自测

```
python -m pytest tests -q
python -m ptsdrift.cli data/sub.srt --anchors data/anchors.json --out fixed.srt
python -m ptsdrift.cli data/sub.srt --anchors <分段锚点> --out fixed.srt --segmented --tol-ms 50
```

## 验收记录（`data/anchors.json`，4 个锚点，第 4 个为离群点）

命令：

```
python -m ptsdrift.cli data/sub.srt --anchors data/anchors.json --out fixed.srt
```

输出关键数值：

```
model=constant slope=1.000000000 offset_ms=250.000 inliers=3 dropped=1 residual_ms=0.000
reason=negligible_drift_rate
cues_in=2 cues_out=2
```

- 离群锚点（`13000 -> 15200`，预期平移 250 ms 处偏差约 1950 ms）被剔除，
  其余 3 个内点的漂移率影响不足 1 ms，按判据降级为 250 ms 常量平移。
- 校正后字幕：`00:00:01,000→00:00:03,000` 变为 `00:00:01,250→00:00:03,250`；
  `00:00:05,000→00:00:07,500` 变为 `00:00:05,250→00:00:07,750`。
- 两次运行输出经 `cmp` 比对逐字节一致。

分段合成数据（断点 9000 ms，两段真实斜率 1.01 / 1.06，`--segmented --tol-ms 50`）：

```
model=segmented segments=2
breakpoints_ms=[9000.0]
  seg0 linear slope=1.010000000 offset_ms=0.000    inliers=9  residual_ms=0.000 range=[0, 8000]
  seg1 linear slope=1.060000000 offset_ms=-450.000 inliers=11 residual_ms=0.000 range=[9000, 19000]
```

断点定位与真实突变点完全重合；第二段截距 `-450 ms` 保证两段在断点处连续。
