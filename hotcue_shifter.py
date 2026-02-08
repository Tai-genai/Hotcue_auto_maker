#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hot Cue Shifter CLI for rekordbox XML.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from hotcue_core import ShiftConfig, process_xml


DEFAULT_CONFIG = ShiftConfig()
BPM_THRESHOLD = int(DEFAULT_CONFIG.bpm_threshold)
OFFSET_HIGH_BPM = DEFAULT_CONFIG.offset_high_bpm
OFFSET_NORMAL_BPM = DEFAULT_CONFIG.offset_normal_bpm
TARGET_ENERGY_MIN = DEFAULT_CONFIG.target_energy_min
TARGET_ENERGY_MAX = DEFAULT_CONFIG.target_energy_max


def build_parser() -> argparse.ArgumentParser:
    """CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="MIKが生成したHot Cueをrekordbox XML上で前倒しするツール",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_xml", help="入力rekordbox XML")
    parser.add_argument("output_xml", help="出力rekordbox XML")
    parser.add_argument(
        "--bpm-threshold",
        type=float,
        default=DEFAULT_CONFIG.bpm_threshold,
        help="高BPM判定の閾値",
    )
    parser.add_argument(
        "--offset-high-bpm",
        type=int,
        default=DEFAULT_CONFIG.offset_high_bpm,
        help="高BPM時のオフセット拍数",
    )
    parser.add_argument(
        "--offset-normal-bpm",
        type=int,
        default=DEFAULT_CONFIG.offset_normal_bpm,
        help="通常BPM時のオフセット拍数",
    )
    parser.add_argument(
        "--energy-min",
        type=int,
        default=DEFAULT_CONFIG.target_energy_min,
        help="処理対象Energy最小値",
    )
    parser.add_argument(
        "--energy-max",
        type=int,
        default=DEFAULT_CONFIG.target_energy_max,
        help="処理対象Energy最大値",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = ShiftConfig(
        bpm_threshold=args.bpm_threshold,
        offset_high_bpm=args.offset_high_bpm,
        offset_normal_bpm=args.offset_normal_bpm,
        target_energy_min=args.energy_min,
        target_energy_max=args.energy_max,
    )

    print(f"処理開始: {args.input_xml}")
    print(
        "設定: BPM>{:.2f}→-{}拍, それ以外→-{}拍".format(
            config.bpm_threshold, config.offset_high_bpm, config.offset_normal_bpm
        )
    )
    print(f"対象: Energy {config.target_energy_min}-{config.target_energy_max}")
    print()

    try:
        stats = process_xml(args.input_xml, args.output_xml, config)
    except FileNotFoundError:
        print("エラー: 入力XMLが見つかりません。", file=sys.stderr)
        sys.exit(1)
    except ET.ParseError:
        print("エラー: XMLの解析に失敗しました。", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        sys.exit(1)

    print("処理完了:")
    print(f"  処理トラック数: {stats['tracks_processed']}")
    print(f"  シフトしたHot Cue: {stats['shifted']}")
    print(f"  スキップ（マイナス）: {stats['skipped_negative']}")
    print(f"  スキップ（対象外Energy）: {stats['skipped_not_target']}")
    print()
    print(f"出力: {args.output_xml}")


if __name__ == "__main__":
    main()
