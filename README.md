# 字幕 PTS 漂移校正

一个能估计并消除音轨与字幕时间基准漂移的对齐器

## 保留签名

- `Cue`
- `load_srt(path)`
- `write_srt(path, cues)`
- `estimate_drift(anchors)`
- `apply_drift(cues, model)`

## 自测

```
python -m pytest tests -q
python -m ptsdrift.cli data/sub.srt --anchors data/anchors.json --out fixed.srt
```
