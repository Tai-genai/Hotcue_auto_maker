# Hot Cue Auto Maker - オーディオ解析によるホットキュー最適化

## Context

Mixed In Key (MIK) の自動ホットキュー設定には以下の問題がある:
- キューが小節頭でない位置に置かれることがある
- ドロップでない場所にキューが置かれることがある
- ホットキューが多すぎて判断に迷う

これを解決するため、librosa によるオーディオ解析を組み合わせた**ハイブリッド方式**と、MIK不要の**独自検出方式**を追加する。

## 処理モード

| モード | 概要 |
|--------|------|
| `shift_only` | 既存動作そのまま（デフォルト） |
| `hybrid` | MIKキュー → バースナップ → 音声スコアリング → 厳選 → シフト |
| `standalone` | 音声解析のみでドロップ検出 → キュー生成 → シフト |

## ファイル構成

```
hotcue_core.py        # 変更なし（既存シフトロジック）
audio_analyzer.py     # 新規: 音声解析・スコアリング・グリッド計算
pipeline.py           # 新規: モードルーティング・XML統合
hotcue_shifter.py     # 変更: --mode, 解析オプション追加
hotcue_app.py         # 変更: モード選択UI, 解析設定, プログレスバー
requirements.txt      # 変更: librosa, soundfile 追加
docs/design.md        # 変更: 新機能ドキュメント
```

## 実装ステップ

### Step 1: `audio_analyzer.py` を新規作成

データクラス:
- `ProcessingMode` (Enum): SHIFT_ONLY / HYBRID / STANDALONE
- `AnalysisConfig` (frozen dataclass): max_cues, min_cue_distance_bars, スコア重み, 閾値, sr, hop_length
- `CueTypeSelection` (frozen dataclass): run_up, drop, buildup, breakdown, outro, intro (各bool)
- `CueRole` (Enum): RUN_UP / DROP / BUILDUP / BREAKDOWN / OUTRO / INTRO
- `CueCandidate`: time_seconds, bar_snapped_time, score, role(CueRole), source("mik"/"detected")
- `TrackAnalysisResult`: candidates, selected, error

グリッド計算（純粋な数学、外部依存なし）:
- `compute_first_downbeat(inizio, bpm, battito)` → `inizio - (battito-1) * (60/bpm)`
- `compute_bar_duration(bpm)` → `4 * 60 / bpm`
- `snap_to_bar(time, first_downbeat, bar_duration)` → 最寄り小節頭にスナップ
- `snap_to_phrase(time, first_downbeat, bar_duration, phrase_bars=8)` → フレーズ境界にスナップ

ファイルパス変換:
- `location_to_filepath(url)` → `file://localhost/...` を通常パスに変換（urllib.parse使用）

音声特徴抽出（librosa）:
- `load_audio(path, sr=22050)` → librosa.load ラッパー（遅延import）
- `compute_onset_strength(y, sr)` → オンセット強度
- `compute_rms_energy(y)` → RMSエネルギー
- `compute_spectral_contrast(y, sr)` → スペクトルコントラスト
- `compute_rms_derivative(rms)` → エネルギー変化率（np.diff）
- `compute_spectral_derivative(spectral)` → スペクトル変化率

ドロップ検出・スコアリング:
- `compute_drop_scores(onset, rms_deriv, spectral_deriv, config)` → 正規化+重み付け合成スコア
- `detect_drop_peaks(scores, sr, hop_length, min_distance, threshold)` → scipy.signal.find_peaks でピーク検出

構造検出:
- `detect_buildup_starts(rms, onset, sr, hop_length, drop_times)` → ドロップ前のエネルギー漸増区間の開始点を検出
- `detect_intro_beat(rms, onset, sr, hop_length)` → 曲冒頭でビートが始まる地点を検出（RMS急上昇）

キュー選択:
- `score_mik_cues(cue_times, drop_scores, ...)` → 既存MIKキューにスコア付与（hybrid用）
- `generate_standalone_candidates(drop_peaks, ...)` → 検出ピークからキュー候補生成
- `deduplicate_candidates(candidates, min_distance)` → スコア順の貪欲法で近接除去
- `select_top_candidates(candidates, max_cues, threshold)` → 閾値+上位N個、時間順でソート

統合エントリポイント:
- `analyze_track(file_path, bpm, first_downbeat, bar_duration, mik_cue_times, config)` → 全パイプライン実行

### Step 2: `pipeline.py` を新規作成

- `PipelineConfig`: ShiftConfig + AnalysisConfig を束ねる frozen dataclass
- `extract_tempo_data(track)` → TEMPO要素から (first_downbeat, bpm, bar_duration) 取得
- `extract_mik_cue_times(track, config)` → 対象Energy範囲のホットキュー抽出
- `apply_analysis_cues(track, candidates, config)` → 解析結果をXMLに書き戻し
  - hybrid: 既存POSITION_MARKを更新
  - standalone: 既存ホットキューを削除し新規作成（Num=0-7, Name="Drop N"）
- `process_track_hybrid(track, shift_config, analysis_config)` → ハイブリッド処理
  - 音声ファイル不在・解析エラー時は `hotcue_core.process_track()` にフォールバック
- `process_track_standalone(track, shift_config, analysis_config)` → 独自検出処理
- `process_xml_pipeline(input, output, pipeline_config, progress_callback)` → メインエントリポイント
  - SHIFT_ONLY → 既存 `hotcue_core.process_xml()` に委譲
  - HYBRID / STANDALONE → トラック毎に解析処理
  - progress_callback(current, total, track_name) で進捗通知

### Step 3: `hotcue_shifter.py` CLI更新

新規引数:
- `--mode shift_only|hybrid|standalone`
- `--cue-run-up / --no-cue-run-up` (デフォルトON)
- `--cue-drop`, `--cue-buildup`, `--cue-breakdown`, `--cue-outro`, `--cue-intro` (各デフォルトOFF)
- `--max-cues N` (デフォルト8)
- `--min-cue-distance-bars N` (デフォルト4)
- `--score-threshold F` (デフォルト0.3)
- `--onset-weight F`, `--rms-weight F`, `--spectral-weight F`

main()を更新: PipelineConfig構築 → `process_xml_pipeline()` 呼び出し

### Step 4: `hotcue_app.py` GUI更新

- モード選択: `CTkSegmentedButton`（シフトのみ / ハイブリッド / 音声分析のみ）
- キュー種類チェックボックス（6種類）: ドロップ前（助走点）, ドロップ, ビルドアップ, ブレイクダウン, アウトロ, イントロ
- 詳細設定（折りたたみ式）: 最大Hot Cue数, 最小距離, スコア閾値, 各重み
- プログレスバー: 解析モード時は `determinate` モードに切替
- librosa可用性チェック: 起動時にimport試行、失敗時はモード選択を無効化
- 設定の保存/読み込みに新フィールド追加

### Step 5: `requirements.txt` 更新

```
librosa>=0.10,<1
soundfile>=0.12,<1
```

### Step 6: `docs/design.md` 更新

新モードの仕様、スコアリングアルゴリズム、設定パラメータを文書化

## 重要な設計判断

- **オフセット単位は「小節」**: GUI/CLIでは小節で入力、内部で拍に変換（×4）。デフォルト: 16小節
- **キュー種類はチェックボックス選択式**: `CueTypeSelection(run_up=True, drop=False, buildup=False, breakdown=False, outro=False, intro=False)` でユーザーが直感的にON/OFF（6種類）。従来の2択Enum（RUN_UP_ONLY / STRUCTURAL）は廃止
- **ロール別シフト**: RUN_UPのみシフト適用。DROP/BUILDUP/BREAKDOWN/OUTRO/INTROはシフトなし
- **詳細設定は折りたたみ式**: 数値パラメータ（max_cues, 閾値, 重み等）はデフォルトで非表示。上級者が「▶ 詳細設定」を展開して調整
- **librosa は遅延import**: shift_onlyモード時の起動速度を維持
- **フォールバック戦略**: hybrid時に音声ファイル不在・解析エラーなら既存シフト処理に自動フォールバック（既存動作より悪くならない）
- **hotcue_core.py は変更しない**: 既存の ShiftConfig, process_track, process_xml はそのまま維持し、新モジュールから呼び出す
- **ドロップスコア**: onset強度(0.4) + RMS変化率(0.35) + スペクトル変化率(0.25) の重み付き合成。エネルギー増加方向のみ（np.clip で負値除去）
- **構造検出**: ドロップ（エネルギー上昇）に加え、ビルドアップ（ドロップ前の漸増区間）、ブレイクダウン（エネルギー降下）、アウトロ（終盤低エネルギー）、イントロ（ビート開始点）も検出

## 配布に関する方針

- **対象プラットフォーム**: macOS のみ
- **依存ライブラリ**: まずは librosa で実装。リリース前に scipy+numpy のみの軽量実装への置き換えを検討
- **アプリサイズ**: 開発段階では 500MB+ を許容。audio_analyzer.py のインターフェースを安定させておけば、内部実装の差し替えは容易
- **PyInstaller**: `HotCueAutoMaker.spec` に librosa 関連の hiddenimports を追加。numba/llvmlite のバンドリングに注意

## 検証方法

1. `--mode shift_only` の出力が既存ツールと同一であることを確認
2. テスト用XML（2-3曲）で `--mode hybrid` を実行し、出力XMLのキュー位置を手動確認
3. `--mode standalone` でキューが正しく生成されることを確認
4. GUI でモード切替・設定表示/非表示・プログレスバー動作を確認
5. 存在しない音声ファイルを含むXMLでフォールバックが正しく動作することを確認

## 将来の拡張候補

v2で実装する6種類のキュータイプに加え、以下を将来検討する:

- **ボーカル開始点検出**: Demucs/Spleeter によるボーカル分離 → ボーカル開始地点にキュー配置
- **バース/コーラス構造分析**: MSAF等でVerse/Chorus/Bridgeを区分 → 構造境界にキュー配置
- **ループポイント検出**: 繰り返しパターンを検出しループ素材の始点にキュー配置
- **キューカラー自動設定**: CueRoleに応じてrekordboxのキューカラーを自動設定（実装容易、優先度高）
- **ジャンル別プリセット**: D&B/House/Techno等のジャンルに応じたデフォルトパラメータセット
