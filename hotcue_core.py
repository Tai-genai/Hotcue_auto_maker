#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core logic for shifting rekordbox hot cues."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class ShiftConfig:
    """Runtime configuration for hot cue shifting."""

    bpm_threshold: float = 150.0
    offset_high_bpm: int = 32
    offset_normal_bpm: int = 16
    target_energy_min: int = 6
    target_energy_max: int = 7


def validate_config(config: ShiftConfig) -> None:
    """Raise ValueError when config values are invalid."""
    if config.bpm_threshold <= 0:
        raise ValueError("BPM閾値は0より大きい値を指定してください。")
    if config.offset_high_bpm <= 0 or config.offset_normal_bpm <= 0:
        raise ValueError("オフセット拍数は1以上を指定してください。")
    if config.target_energy_min < 0 or config.target_energy_max < 0:
        raise ValueError("Energy範囲は0以上を指定してください。")
    if config.target_energy_min > config.target_energy_max:
        raise ValueError("Energy最小値は最大値以下にしてください。")


def parse_float(value: Optional[str], default: float = 0.0) -> float:
    """Convert string to float safely."""
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_energy_level(name: Optional[str]) -> Optional[int]:
    """Extract energy level from a POSITION_MARK Name value."""
    if not name:
        return None
    match = re.search(r"Energy\s+(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def get_bpm(track: ET.Element) -> Optional[float]:
    """Resolve BPM from TEMPO first and fallback to AverageBpm."""
    tempo = track.find("TEMPO")
    if tempo is not None:
        bpm = parse_float(tempo.get("Bpm"), 0.0)
        if bpm > 0:
            return bpm

    avg_bpm = parse_float(track.get("AverageBpm"), 0.0)
    if avg_bpm > 0:
        return avg_bpm

    return None


def calculate_offset_seconds(bpm: float, config: ShiftConfig) -> float:
    """Calculate offset seconds from BPM and config."""
    offset_beats = (
        config.offset_high_bpm if bpm > config.bpm_threshold else config.offset_normal_bpm
    )
    seconds_per_beat = 60.0 / bpm
    return seconds_per_beat * offset_beats


def is_hot_cue(position_mark: ET.Element) -> bool:
    """Hot cue is Num >= 0."""
    num = position_mark.get("Num")
    if num is None:
        return False
    try:
        return int(num) >= 0
    except ValueError:
        return False


def is_target_energy(position_mark: ET.Element, config: ShiftConfig) -> bool:
    """Whether POSITION_MARK is inside target energy range."""
    energy = parse_energy_level(position_mark.get("Name"))
    if energy is None:
        return False
    return config.target_energy_min <= energy <= config.target_energy_max


def process_track(track: ET.Element, config: ShiftConfig) -> dict:
    """Process one TRACK node and return stats."""
    stats = {"shifted": 0, "skipped_negative": 0, "skipped_not_target": 0}

    bpm = get_bpm(track)
    if bpm is None or bpm <= 0:
        return stats

    offset_seconds = calculate_offset_seconds(bpm, config)

    for position_mark in track.findall("POSITION_MARK"):
        if not is_hot_cue(position_mark):
            continue

        if not is_target_energy(position_mark, config):
            stats["skipped_not_target"] += 1
            continue

        current_start = parse_float(position_mark.get("Start"), 0.0)
        new_start = current_start - offset_seconds

        if new_start < 0:
            stats["skipped_negative"] += 1
            continue

        position_mark.set("Start", f"{new_start:.3f}")
        stats["shifted"] += 1

    return stats


def process_xml(input_path: str | Path, output_path: str | Path, config: ShiftConfig) -> dict:
    """Read XML, process all tracks, and write output XML."""
    validate_config(config)

    input_path = Path(input_path)
    output_path = Path(output_path)

    tree = ET.parse(input_path)
    root = tree.getroot()

    total_stats = {
        "tracks_processed": 0,
        "shifted": 0,
        "skipped_negative": 0,
        "skipped_not_target": 0,
    }

    for track in root.findall(".//TRACK"):
        stats = process_track(track, config)
        total_stats["tracks_processed"] += 1
        total_stats["shifted"] += stats["shifted"]
        total_stats["skipped_negative"] += stats["skipped_negative"]
        total_stats["skipped_not_target"] += stats["skipped_not_target"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return total_stats
