#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hot Cue Shifter for rekordbox XML

MIKが自動生成したHot Cueの位置を調整し、
DJミックス用の「助走点」を作成するツール。

Usage:
    python hotcue_shifter.py input.xml output.xml
"""

import sys
import re
import xml.etree.ElementTree as ET
from typing import Optional

# =============================================================================
# 設定
# =============================================================================

BPM_THRESHOLD = 150        # この値を超えたら高BPM扱い
OFFSET_HIGH_BPM = 32       # 高BPM時のオフセット（拍）
OFFSET_NORMAL_BPM = 16     # 通常時のオフセット（拍）
TARGET_ENERGY_MIN = 6      # 処理対象のEnergy Level（最小）
TARGET_ENERGY_MAX = 7      # 処理対象のEnergy Level（最大）

# =============================================================================
# ヘルパー関数
# =============================================================================

def parse_float(value: Optional[str], default: float = 0.0) -> float:
    """文字列をfloatに変換。失敗時はデフォルト値を返す。"""
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_energy_level(name: Optional[str]) -> Optional[int]:
    """
    POSITION_MARKのName属性からEnergy Levelを抽出。
    例: "Energy 7" → 7
    """
    if not name:
        return None
    match = re.search(r"Energy\s+(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def get_bpm(track: ET.Element) -> Optional[float]:
    """
    トラックのBPMを取得。
    TEMPOタグがあればそのBpm、なければAverageBpmを使用。
    """
    # TEMPOタグから取得を試みる
    tempo = track.find("TEMPO")
    if tempo is not None:
        bpm = parse_float(tempo.get("Bpm"), 0.0)
        if bpm > 0:
            return bpm

    # AverageBpmから取得
    avg_bpm = parse_float(track.get("AverageBpm"), 0.0)
    if avg_bpm > 0:
        return avg_bpm

    return None


def calculate_offset_seconds(bpm: float) -> float:
    """
    BPMに基づいてオフセット秒数を計算。
    BPM > 150 → 32拍、それ以外 → 16拍
    """
    offset_beats = OFFSET_HIGH_BPM if bpm > BPM_THRESHOLD else OFFSET_NORMAL_BPM
    seconds_per_beat = 60.0 / bpm
    return seconds_per_beat * offset_beats


def is_hot_cue(position_mark: ET.Element) -> bool:
    """Hot Cue（Num >= 0）かどうかを判定。"""
    num = position_mark.get("Num")
    if num is None:
        return False
    try:
        return int(num) >= 0
    except ValueError:
        return False


def is_target_energy(position_mark: ET.Element) -> bool:
    """処理対象のEnergy Level（6-7）かどうかを判定。"""
    energy = parse_energy_level(position_mark.get("Name"))
    if energy is None:
        return False
    return TARGET_ENERGY_MIN <= energy <= TARGET_ENERGY_MAX


# =============================================================================
# メイン処理
# =============================================================================

def process_track(track: ET.Element) -> dict:
    """
    1トラックを処理。

    Returns:
        処理結果の統計情報
    """
    stats = {
        "shifted": 0,
        "skipped_negative": 0,
        "skipped_not_target": 0
    }

    # BPMを取得
    bpm = get_bpm(track)
    if bpm is None or bpm <= 0:
        return stats

    # オフセット秒数を計算
    offset_seconds = calculate_offset_seconds(bpm)

    # POSITION_MARKを走査
    for pm in track.findall("POSITION_MARK"):
        # Hot Cueでなければスキップ
        if not is_hot_cue(pm):
            continue

        # 対象のEnergy Levelでなければスキップ
        if not is_target_energy(pm):
            stats["skipped_not_target"] += 1
            continue

        # 現在のStart位置を取得
        current_start = parse_float(pm.get("Start"), 0.0)

        # 新しいStart位置を計算
        new_start = current_start - offset_seconds

        # マイナスになる場合はスキップ（元の位置を維持）
        if new_start < 0:
            stats["skipped_negative"] += 1
            continue

        # Start位置を更新
        pm.set("Start", f"{new_start:.3f}")
        stats["shifted"] += 1

    return stats


def process_xml(input_path: str, output_path: str) -> dict:
    """
    XMLファイルを処理。

    Args:
        input_path: 入力XMLファイルパス
        output_path: 出力XMLファイルパス

    Returns:
        処理結果の統計情報
    """
    # XMLを読み込み
    tree = ET.parse(input_path)
    root = tree.getroot()

    # 統計情報
    total_stats = {
        "tracks_processed": 0,
        "shifted": 0,
        "skipped_negative": 0,
        "skipped_not_target": 0
    }

    # 全TRACKを処理
    for track in root.findall(".//TRACK"):
        stats = process_track(track)
        total_stats["tracks_processed"] += 1
        total_stats["shifted"] += stats["shifted"]
        total_stats["skipped_negative"] += stats["skipped_negative"]
        total_stats["skipped_not_target"] += stats["skipped_not_target"]

    # XMLを出力
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return total_stats


def main():
    if len(sys.argv) < 3:
        print("Usage: python hotcue_shifter.py <input.xml> <output.xml>")
        print()
        print("設定:")
        print(f"  BPM閾値: {BPM_THRESHOLD}")
        print(f"  高BPMオフセット: {OFFSET_HIGH_BPM}拍")
        print(f"  通常オフセット: {OFFSET_NORMAL_BPM}拍")
        print(f"  対象Energy: {TARGET_ENERGY_MIN}-{TARGET_ENERGY_MAX}")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"処理開始: {input_path}")
    print(f"設定: BPM>{BPM_THRESHOLD}→-{OFFSET_HIGH_BPM}拍, それ以外→-{OFFSET_NORMAL_BPM}拍")
    print(f"対象: Energy {TARGET_ENERGY_MIN}-{TARGET_ENERGY_MAX}")
    print()

    stats = process_xml(input_path, output_path)

    print("処理完了:")
    print(f"  処理トラック数: {stats['tracks_processed']}")
    print(f"  シフトしたHot Cue: {stats['shifted']}")
    print(f"  スキップ（マイナス）: {stats['skipped_negative']}")
    print(f"  スキップ（対象外Energy）: {stats['skipped_not_target']}")
    print()
    print(f"出力: {output_path}")


if __name__ == "__main__":
    main()
