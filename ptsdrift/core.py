# -*- coding: utf-8 -*-
"""字幕 PTS 漂移校正"""

from dataclasses import dataclass


@dataclass
class Cue:
    """一条字幕。"""
    index: int
    start_ms: int
    end_ms: int
    text: str


def load_srt(path):
    """读 SRT"""
    raise NotImplementedError("load_srt 尚未实现")


def write_srt(path, cues):
    """写 SRT"""
    raise NotImplementedError("write_srt 尚未实现")


def estimate_drift(anchors):
    """估计漂移率与截距"""
    raise NotImplementedError("estimate_drift 尚未实现")


def apply_drift(cues, model):
    """按模型校正"""
    raise NotImplementedError("apply_drift 尚未实现")


def load_anchors(path):
    """读入与落盘辅助"""
    raise NotImplementedError("load_anchors 尚未实现")

