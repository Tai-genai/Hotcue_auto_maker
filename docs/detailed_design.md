# 詳細設計書: オーディオ解析によるホットキュー最適化

> バージョン: 1.1
> 作成日: 2026-02-14
> 対象: 実装者向け

## 1. モジュール構成

```
hotcue_core.py          既存（変更なし）
audio_analyzer.py       新規  音声解析・スコアリング・グリッド計算
pipeline.py             新規  モードルーティング・XML統合・進捗通知
hotcue_shifter.py       改修  CLI引数追加
hotcue_app.py           改修  GUI拡張
```

### 1.1 依存関係

```
hotcue_app.py ─────┐
hotcue_shifter.py ─┤
                   ▼
              pipeline.py
              ├── hotcue_core.py   (既存: ShiftConfig, process_track, process_xml)
              └── audio_analyzer.py (新規: 音声解析)
                  ├── librosa       (遅延import)
                  ├── numpy
                  └── scipy.signal
```

### 1.2 設計原則

- `hotcue_core.py` は一切変更しない（後方互換性を保証）
- `audio_analyzer.py` は XML を知らない（純粋な信号処理モジュール）
- `pipeline.py` が XML 操作と音声解析を橋渡しする
- librosa は `load_audio()` 内で遅延 import（shift_only モード時に不要な読み込みを回避）

---

## 2. データ構造

### 2.1 ProcessingMode（Enum）

**ファイル**: `audio_analyzer.py`

```python
class ProcessingMode(Enum):
    SHIFT_ONLY = "shift_only"      # 既存動作
    HYBRID     = "hybrid"          # MIKキュー + 音声解析
    STANDALONE = "standalone"      # 音声解析のみ
```

### 2.2 CueTypeSelection（frozen dataclass）

**ファイル**: `audio_analyzer.py`

ユーザーがどの種類のキューを生成するかを個別にON/OFFで選択する。

```python
@dataclass(frozen=True)
class CueTypeSelection:
    run_up: bool = True        # ドロップ前の助走点（デフォルトON）
    drop: bool = False         # ドロップそのもの
    buildup: bool = False      # ビルドアップ開始（エネルギー上昇開始地点）
    breakdown: bool = False    # ブレイクダウン（エネルギー急降下地点）
    outro: bool = False        # アウトロ開始点
    intro: bool = False        # イントロ（最初のビート開始地点）
```

**デフォルト**: `run_up=True` のみ。初心者はこのままで従来と同じ「助走点のみ」の動作。
チェックを追加するだけで構造キューが増える（例: `run_up=True, drop=True` → 助走点+ドロップ）。

**バリデーション**: 少なくとも1つは True であること。

### 2.3 CueRole（Enum）

**ファイル**: `audio_analyzer.py`

```python
class CueRole(Enum):
    RUN_UP    = "run_up"       # ミックスイン点（ドロップのN小節前にシフト）
    DROP      = "drop"         # ドロップそのもの（シフトしない）
    BUILDUP   = "buildup"      # ビルドアップ開始（シフトしない）
    BREAKDOWN = "breakdown"    # エネルギー急降下地点（シフトしない）
    OUTRO     = "outro"        # ミックスアウト点（シフトしない）
    INTRO     = "intro"        # イントロ / 最初のビート開始（シフトしない）
```

**シフト適用ルール**: `RUN_UP` ロールのキューのみシフトを適用する。他のロールは検出位置そのまま。

### 2.4 AnalysisConfig（frozen dataclass）

**ファイル**: `audio_analyzer.py`

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `mode` | `ProcessingMode` | `SHIFT_ONLY` | 処理モード |
| `cue_types` | `CueTypeSelection` | `CueTypeSelection()` | キュー種類の選択（チェックボックス） |
| `max_cues` | `int` | `8` | 出力する最大ホットキュー数（1〜8） |
| `min_cue_distance_bars` | `int` | `4` | キュー間の最小距離（小節数） |
| `drop_score_onset_weight` | `float` | `0.4` | オンセット強度の重み |
| `drop_score_rms_weight` | `float` | `0.35` | RMSエネルギー変化の重み |
| `drop_score_spectral_weight` | `float` | `0.25` | スペクトル変化の重み |
| `score_threshold` | `float` | `0.3` | 候補として残す最低スコア（0.0〜1.0） |
| `sr` | `int` | `22050` | librosa のサンプリングレート |
| `hop_length` | `int` | `512` | STFT ホップ長 |
| `phrase_length_bars` | `int` | `8` | フレーズ長（小節数、standalone用） |

**バリデーション規則**:
- `1 <= max_cues <= 8`
- `min_cue_distance_bars >= 1`
- 各 weight >= 0、合計 > 0
- `0.0 <= score_threshold <= 1.0`
- `sr > 0`, `hop_length > 0`

### 2.5 CueCandidate（dataclass）

**ファイル**: `audio_analyzer.py`

| フィールド | 型 | 説明 |
|---|---|---|
| `time_seconds` | `float` | 元の位置（秒） |
| `bar_snapped_time` | `float` | バースナップ後の位置（秒） |
| `role` | `CueRole` | キューの役割（RUN_UP / DROP / BREAKDOWN / OUTRO） |
| `drop_score` | `float` | 合成スコア（0.0〜1.0） |
| `onset_strength` | `float` | オンセット強度（個別スコア） |
| `rms_delta` | `float` | RMS変化率（個別スコア） |
| `spectral_delta` | `float` | スペクトル変化率（個別スコア） |
| `source` | `str` | `"mik"` または `"detected"` |
| `energy_label` | `Optional[str]` | 元のラベル（例: `"Energy 7"`）または生成ラベル |

### 2.6 TrackAnalysisResult（dataclass）

**ファイル**: `audio_analyzer.py`

| フィールド | 型 | 説明 |
|---|---|---|
| `file_path` | `str` | 解析した音声ファイルのパス |
| `bpm` | `float` | BPM |
| `first_downbeat` | `float` | 最初のダウンビート位置（秒） |
| `bar_duration` | `float` | 1小節の長さ（秒） |
| `candidates` | `list[CueCandidate]` | 全候補（フィルタ前） |
| `selected` | `list[CueCandidate]` | 選択された候補（フィルタ後） |
| `error` | `Optional[str]` | エラーメッセージ（正常時は None） |

### 2.7 PipelineConfig（frozen dataclass）

**ファイル**: `pipeline.py`

| フィールド | 型 | 説明 |
|---|---|---|
| `shift` | `ShiftConfig` | 既存のシフト設定 |
| `analysis` | `AnalysisConfig` | 解析設定 |

---

## 3. audio_analyzer.py 関数仕様

### 3.1 グリッド計算（純粋数学、外部依存なし）

#### `compute_first_downbeat(inizio, bpm, battito) -> float`

最初のダウンビート（小節頭）の時刻を算出する。

- **引数**: `inizio: float`（TEMPO.Inizio）、`bpm: float`（TEMPO.Bpm）、`battito: int`（TEMPO.Battito、1〜4）
- **戻り値**: 最初のダウンビートの秒数
- **計算式**: `first_downbeat = inizio - (battito - 1) * (60.0 / bpm)`

```
例: Inizio=0.024, Bpm=145.00, Battito=3
    → 0.024 - (3-1) * (60/145) = 0.024 - 0.8276 = -0.8036
    → max(0.0, -0.8036) ではなく、負の値も許容（曲開始前にグリッドが始まるケース）
```

#### `compute_bar_duration(bpm) -> float`

- **計算式**: `bar_duration = 4 * 60.0 / bpm`
- 4/4拍子前提

```
例: 130 BPM → 4 * 60 / 130 = 1.846秒
    145 BPM → 4 * 60 / 145 = 1.655秒
```

#### `snap_to_bar(time_seconds, first_downbeat, bar_duration) -> float`

時刻を最寄りの小節頭にスナップする。

- **計算式**:
  ```
  relative = time_seconds - first_downbeat
  bar_index = round(relative / bar_duration)
  snapped = first_downbeat + bar_index * bar_duration
  return max(0.0, snapped)
  ```
- **制約**: 結果が負になる場合は 0.0 にクランプ

```
例: time=59.118, first_downbeat=0.041, bar_duration=1.846
    relative = 59.077
    bar_index = round(59.077 / 1.846) = round(32.0) = 32
    snapped = 0.041 + 32 * 1.846 = 59.113
    → 0.005秒のずれを補正
```

#### `snap_to_phrase(time_seconds, first_downbeat, bar_duration, phrase_bars) -> float`

時刻を最寄りのフレーズ境界にスナップする。standalone モードで使用。

- **計算式**:
  ```
  phrase_duration = bar_duration * phrase_bars
  relative = time_seconds - first_downbeat
  phrase_index = round(relative / phrase_duration)
  snapped = first_downbeat + phrase_index * phrase_duration
  return max(0.0, snapped)
  ```

### 3.2 ファイルパス変換

#### `location_to_filepath(location_url) -> Optional[str]`

rekordbox の Location 属性（file:// URL）をローカルファイルパスに変換する。

- **入力例**: `"file://localhost/Users/user/Desktop/music/mix/track.mp3"`
- **出力例**: `"/Users/user/Desktop/music/mix/track.mp3"`
- **処理**: `urllib.parse.urlparse()` + `urllib.parse.unquote()`
- URL エンコードされた文字（`%20` → スペース等）を正しくデコード
- scheme が `file` でない場合、または空文字の場合は `None` を返す

### 3.3 音声特徴抽出

全関数で librosa を遅延 import する。`load_audio()` 内で `import librosa` を実行し、ImportError 時は明確なエラーメッセージ付きで RuntimeError を送出する。

#### `load_audio(file_path, sr=22050) -> tuple[np.ndarray, int]`

- librosa.load() のラッパー
- FileNotFoundError: ファイル不在時にそのまま送出
- RuntimeError: librosa のデコードエラーをラップ

#### `compute_onset_strength(y, sr, hop_length) -> np.ndarray`

- `librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)`
- 戻り値: 1D array、各フレームのオンセット強度

#### `compute_rms_energy(y, hop_length) -> np.ndarray`

- `librosa.feature.rms(y=y, hop_length=hop_length)[0]`
- 戻り値: 1D array、各フレームの RMS エネルギー

#### `compute_spectral_contrast(y, sr, hop_length) -> np.ndarray`

- `librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)`
- 周波数帯ごとの結果を `np.mean(axis=0)` で1Dに集約
- 戻り値: 1D array、各フレームの平均スペクトルコントラスト

#### `compute_rms_derivative(rms) -> np.ndarray`

- `np.diff(rms)` にゼロパディングで元と同じ長さに
- `np.concatenate([[0.0], np.diff(rms)])`
- 正の値 = エネルギー増加（ドロップ候補）

#### `compute_spectral_derivative(spectral) -> np.ndarray`

- `compute_rms_derivative` と同じロジック

### 3.4 ドロップ検出・スコアリング

#### `compute_drop_scores(onset, rms_deriv, spectral_deriv, config) -> np.ndarray`

3つの特徴量を正規化・重み付けして合成スコアを算出する。**エネルギー上昇方向**に着目。

**アルゴリズム**:
```
1. エネルギー増加方向のみに着目:
   rms_pos = np.clip(rms_deriv, 0, None)
   spectral_pos = np.clip(spectral_deriv, 0, None)

2. 各特徴を [0, 1] に正規化（min-max スケーリング）:
   def normalize(arr):
       mn, mx = arr.min(), arr.max()
       if mx - mn < 1e-10:
           return np.zeros_like(arr)
       return (arr - mn) / (mx - mn)

3. 重み付き合成:
   score = config.onset_weight * normalize(onset)
         + config.rms_weight   * normalize(rms_pos)
         + config.spectral_weight * normalize(spectral_pos)
```

- 戻り値: 1D array（各フレームの合成スコア、0.0〜1.0）

#### `compute_breakdown_scores(rms_deriv, spectral_deriv, config) -> np.ndarray`

**エネルギー降下方向**に着目した合成スコア。structural モードで使用。

**アルゴリズム**:
```
1. エネルギー減少方向のみに着目（符号反転）:
   rms_neg = np.clip(-rms_deriv, 0, None)
   spectral_neg = np.clip(-spectral_deriv, 0, None)

2. 正規化して重み付き合成（ドロップと同じ重み）:
   score = config.rms_weight * normalize(rms_neg)
         + config.spectral_weight * normalize(spectral_neg)
   score = score / (config.rms_weight + config.spectral_weight)  # 正規化
```

#### `detect_drop_peaks(drop_scores, sr, hop_length, min_distance_seconds, threshold) -> list[tuple[float, float]]`

合成スコアからピーク（ドロップ候補）を検出する。

**アルゴリズム**:
```
1. min_distance をフレーム数に変換:
   min_distance_frames = int(min_distance_seconds * sr / hop_length)

2. scipy.signal.find_peaks で検出:
   peaks, properties = find_peaks(
       drop_scores,
       height=threshold,
       distance=min_distance_frames
   )

3. フレームを秒に変換し、スコア降順でソート:
   results = [(frame * hop_length / sr, drop_scores[frame]) for frame in peaks]
   results.sort(key=lambda x: x[1], reverse=True)
```

- 戻り値: `[(time_seconds, score), ...]` スコア降順

#### `detect_outro_start(rms, sr, hop_length, total_duration) -> Optional[tuple[float, float]]`

曲終盤の低エネルギー持続開始地点を検出する。

**アルゴリズム**:
```
1. 曲全体の RMS 平均を算出
2. 曲の後半 30% の区間を対象
3. RMS が平均の 50% 以下に落ち、その後回復しない最初の地点を検出
4. return (time_seconds, score) or None
```

#### `detect_intro_beat(onset, sr, hop_length) -> Optional[tuple[float, float]]`

楽曲冒頭で最初のビート（キック等）が入る地点を検出する。

**アルゴリズム**:
```
1. 曲の先頭 20% の区間を対象
2. オンセット強度の平均を算出
3. 平均の 1.5 倍を超える最初のオンセットピークを検出
4. return (time_seconds, score) or None
```

#### `detect_buildup_starts(drop_peaks, rms, sr, hop_length, bar_duration) -> list[tuple[float, float]]`

各ドロップの手前でエネルギーが上昇し始める地点（ビルドアップ開始）を検出する。

**アルゴリズム**:
```
1. 各ドロップピーク位置から逆方向にスキャン
2. RMS が持続的に上昇し始める最初のフレームを特定
   （具体的にはドロップの 4〜16小節前の区間で RMS が極小になる地点）
3. 小節頭にスナップ
4. return [(time_seconds, score), ...] 各ドロップに対応
```

### 3.5 キュー候補生成

#### `score_mik_cues(cue_times, drop_scores, onset, rms_deriv, spectral_deriv, sr, hop_length, config) -> list[CueCandidate]`

既存 MIK キューの位置にスコアを付与する（hybrid モード用）。

**アルゴリズム**:
```
for each cue_time in cue_times:
    frame = int(cue_time * sr / hop_length)
    frame = clamp(frame, 0, len(drop_scores) - 1)

    candidate = CueCandidate(
        time_seconds    = cue_time,
        bar_snapped_time = cue_time,  # 呼び出し元でスナップ済み
        drop_score       = drop_scores[frame],
        onset_strength   = normalize(onset)[frame],
        rms_delta        = normalize(clip(rms_deriv))[frame],
        spectral_delta   = normalize(clip(spectral_deriv))[frame],
        source           = "mik",
    )
```

#### `generate_standalone_candidates(drop_peaks, onset, rms_deriv, spectral_deriv, sr, hop_length, config) -> list[CueCandidate]`

検出されたピークから CueCandidate を生成する（standalone モード用）。

**処理**: `detect_drop_peaks` の結果を CueCandidate に変換。source = `"detected"`。

### 3.6 候補選択

#### `deduplicate_candidates(candidates, min_distance_seconds) -> list[CueCandidate]`

近接するキューを除去する。

**アルゴリズム（貪欲法）**:
```
1. candidates をスコア降順にソート
2. kept = []
3. for each candidate in sorted:
     if all(|candidate.bar_snapped_time - k.bar_snapped_time| >= min_distance for k in kept):
         kept.append(candidate)
4. return kept
```

高スコアの候補を優先的に残す。

#### `select_top_candidates(candidates, max_cues, score_threshold) -> list[CueCandidate]`

```
1. score_threshold 以上の候補のみ残す
2. スコア降順で上位 max_cues 個を選択
3. 時間順（bar_snapped_time 昇順）でソートして返す
```

### 3.7 統合エントリポイント

#### `analyze_track(file_path, bpm, first_downbeat, bar_duration, mik_cue_times, config) -> TrackAnalysisResult`

1トラックの解析パイプライン全体を実行する。

**処理フロー**:
```
try:
    y, sr = load_audio(file_path, config.sr)
    total_duration = len(y) / sr

    # 特徴抽出
    onset      = compute_onset_strength(y, sr, config.hop_length)
    rms        = compute_rms_energy(y, config.hop_length)
    spectral   = compute_spectral_contrast(y, sr, config.hop_length)
    rms_d      = compute_rms_derivative(rms)
    spectral_d = compute_spectral_derivative(spectral)

    # ドロップスコア（エネルギー上昇方向）
    drop_scores = compute_drop_scores(onset, rms_d, spectral_d, config)

    min_dist_sec = bar_duration * config.min_cue_distance_bars

    if mik_cue_times is not None:
        # ── hybrid モード ──
        snapped_times = [snap_to_bar(t, first_downbeat, bar_duration) for t in mik_cue_times]
        candidates = score_mik_cues(snapped_times, drop_scores, onset, rms_d, spectral_d, sr, config.hop_length, config)
        for c, st in zip(candidates, snapped_times):
            c.bar_snapped_time = st
            c.role = CueRole.RUN_UP  # hybrid では全て RUN_UP（シフト対象）
    else:
        # ── standalone モード ──
        peaks = detect_drop_peaks(drop_scores, sr, config.hop_length, min_dist_sec, config.score_threshold)
        candidates = generate_standalone_candidates(peaks, onset, rms_d, spectral_d, sr, config.hop_length, config)
        for c in candidates:
            c.bar_snapped_time = snap_to_phrase(c.time_seconds, first_downbeat, bar_duration, config.phrase_length_bars)
            c.role = CueRole.RUN_UP  # デフォルトは RUN_UP

    # ── キュー種類に応じた追加検出（チェックボックスで選択されたもののみ） ──

    # ドロップキュー（ユーザーが「ドロップそのもの」をONにした場合）
    if config.cue_types.drop:
        for c in list(candidates):
            if c.drop_score >= 0.5:
                drop_cue = CueCandidate(
                    time_seconds=c.bar_snapped_time,
                    bar_snapped_time=c.bar_snapped_time,
                    role=CueRole.DROP,
                    drop_score=c.drop_score * 0.9,  # RUN_UP より若干低く
                    ...
                )
                candidates.append(drop_cue)

    # ビルドアップキュー（ユーザーが「ビルドアップ」をONにした場合）
    if config.cue_types.buildup:
        drop_peak_times = [c.bar_snapped_time for c in candidates
                           if c.role in (CueRole.RUN_UP, CueRole.DROP)]
        bu_results = detect_buildup_starts(
            [(t, 1.0) for t in drop_peak_times],
            rms, sr, config.hop_length, bar_duration,
        )
        for time_sec, score in bu_results:
            snapped = snap_to_bar(time_sec, first_downbeat, bar_duration)
            candidates.append(CueCandidate(
                time_seconds=time_sec,
                bar_snapped_time=snapped,
                role=CueRole.BUILDUP,
                drop_score=score * 0.85,
                ...
            ))

    # ブレイクダウンキュー（ユーザーが「ブレイクダウン」をONにした場合）
    if config.cue_types.breakdown:
        bd_scores = compute_breakdown_scores(rms_d, spectral_d, config)
        bd_peaks = detect_drop_peaks(bd_scores, sr, config.hop_length, min_dist_sec, config.score_threshold)
        for time_sec, score in bd_peaks[:3]:  # 上位3個まで
            snapped = snap_to_bar(time_sec, first_downbeat, bar_duration)
            candidates.append(CueCandidate(
                time_seconds=time_sec,
                bar_snapped_time=snapped,
                role=CueRole.BREAKDOWN,
                drop_score=score * 0.8,  # ドロップより優先度低め
                ...
            ))

    # アウトロキュー（ユーザーが「アウトロ」をONにした場合）
    if config.cue_types.outro:
        outro = detect_outro_start(rms, sr, config.hop_length, total_duration)
        if outro:
            time_sec, score = outro
            snapped = snap_to_bar(time_sec, first_downbeat, bar_duration)
            candidates.append(CueCandidate(
                time_seconds=time_sec,
                bar_snapped_time=snapped,
                role=CueRole.OUTRO,
                drop_score=score * 0.7,
                ...
            ))

    # イントロキュー（ユーザーが「イントロ」をONにした場合）
    if config.cue_types.intro:
        intro = detect_intro_beat(onset, sr, config.hop_length)
        if intro:
            time_sec, score = intro
            snapped = snap_to_bar(time_sec, first_downbeat, bar_duration)
            candidates.append(CueCandidate(
                time_seconds=time_sec,
                bar_snapped_time=snapped,
                role=CueRole.INTRO,
                drop_score=score * 0.75,
                ...
            ))

    # run_up が OFF の場合、RUN_UP ロールの候補を除外
    if not config.cue_types.run_up:
        candidates = [c for c in candidates if c.role != CueRole.RUN_UP]

    # 選択
    deduped = deduplicate_candidates(candidates, min_dist_sec)
    selected = select_top_candidates(deduped, config.max_cues, config.score_threshold)

    return TrackAnalysisResult(
        file_path=file_path, bpm=bpm,
        first_downbeat=first_downbeat, bar_duration=bar_duration,
        candidates=candidates, selected=selected,
    )

except Exception as e:
    return TrackAnalysisResult(
        file_path=file_path, bpm=bpm,
        first_downbeat=first_downbeat, bar_duration=bar_duration,
        error=str(e),
    )
```

---

## 4. pipeline.py 関数仕様

### 4.1 型定義

```python
ProgressCallback = Callable[[int, int, str], None]
# (current_track_index, total_tracks, track_name)
```

### 4.2 XML ヘルパー

#### `extract_tempo_data(track) -> tuple[float, float, float] | None`

TRACK 要素から (first_downbeat, bpm, bar_duration) を抽出する。

```
1. TEMPO 要素を取得（最初の1つ）
2. Bpm, Inizio, Battito を parse_float / int で取得
3. bpm <= 0 なら None を返す
4. compute_first_downbeat(), compute_bar_duration() を呼び出し
5. return (first_downbeat, bpm, bar_duration)
```

TEMPO 要素がなく AverageBpm がある場合:
- bpm = AverageBpm、first_downbeat = 0.0、battito = 1 として計算

#### `extract_mik_cue_data(track, config) -> list[dict]`

対象 Energy 範囲のホットキュー情報を抽出する。

```
results = []
for pm in track.findall("POSITION_MARK"):
    if not is_hot_cue(pm):     # Num < 0 をスキップ
        continue
    if not is_target_energy(pm, config):  # Energy 範囲外をスキップ
        continue
    results.append({
        "element": pm,
        "start": parse_float(pm.get("Start")),
        "num": int(pm.get("Num")),
        "name": pm.get("Name", ""),
    })
return results
```

#### `extract_location(track) -> str | None`

TRACK 要素の Location 属性を返す。属性がなければ None。

### 4.3 キュー書き戻し

#### `apply_analysis_cues(track, selected_candidates, shift_config, analysis_config) -> dict`

解析結果を XML の POSITION_MARK に反映する。

**hybrid モードの場合**:
```
1. 既存のホットキュー（Num >= 0）を全て取得
2. selected に含まれない位置のホットキューを削除（track.remove()）
3. selected に含まれるキューの Start を bar_snapped_time に更新
4. シフト処理: calculate_offset_seconds(bpm, shift_config) で算出した秒数を減算
5. シフト後に負になるキューはスキップ
```

**standalone モードの場合**:
```
1. 既存のホットキュー（Num >= 0）を全て削除
2. Memory Cue（Num = -1）はそのまま残す
3. selected_candidates から新しい POSITION_MARK を生成:
   - Num: 0, 1, 2, ... (最大7)
   - Type: "0"
   - Name と Start はロールに応じて決定（下記参照）
4. Start が負になるキューはスキップ
```

**ロール別の処理**:
```
RUN_UP:
  Name: "Run-up"
  Start: bar_snapped_time - offset_seconds（シフト適用）

DROP:
  Name: "Drop"
  Start: bar_snapped_time（シフトなし、ドロップ位置そのまま）

BUILDUP:
  Name: "Buildup"
  Start: bar_snapped_time（シフトなし）

BREAKDOWN:
  Name: "Breakdown"
  Start: bar_snapped_time（シフトなし）

INTRO:
  Name: "Intro"
  Start: bar_snapped_time（シフトなし）

OUTRO:
  Name: "Outro"
  Start: bar_snapped_time（シフトなし）
```

**戻り値**: `{"placed": int, "shifted": int, "skipped_negative": int}`

### 4.4 トラック処理

#### `process_track_hybrid(track, shift_config, analysis_config) -> dict`

```
1. tempo_data = extract_tempo_data(track)
   → None なら hotcue_core.process_track(track, shift_config) にフォールバック

2. mik_cues = extract_mik_cue_data(track, shift_config)
   → 空なら stats = {shifted:0, ...} を返す（対象キューなし）

3. location = extract_location(track)
   file_path = location_to_filepath(location)
   → None なら hotcue_core.process_track() にフォールバック

4. result = analyze_track(
       file_path, bpm, first_downbeat, bar_duration,
       mik_cue_times=[c["start"] for c in mik_cues],
       config=analysis_config,
   )
   → result.error が非 None なら hotcue_core.process_track() にフォールバック

5. stats = apply_analysis_cues(track, result.selected, shift_config, analysis_config)
6. return stats
```

**フォールバック戦略**: 解析に失敗した場合は既存のシフト処理に戻る。これにより hybrid モードは shift_only より悪い結果にならない。

#### `process_track_standalone(track, shift_config, analysis_config) -> dict`

```
1. tempo_data = extract_tempo_data(track)
   → None なら スキップ（return {placed:0, ...}）

2. location = extract_location(track)
   file_path = location_to_filepath(location)
   → None なら スキップ

3. result = analyze_track(
       file_path, bpm, first_downbeat, bar_duration,
       mik_cue_times=None,    # standalone: MIKキューを使わない
       config=analysis_config,
   )
   → result.error が非 None なら スキップ

4. stats = apply_analysis_cues(track, result.selected, shift_config, analysis_config)
5. return stats
```

### 4.5 XML処理メインエントリポイント

#### `process_xml_pipeline(input_path, output_path, pipeline_config, progress_callback=None) -> dict`

```
config = pipeline_config

if config.analysis.mode == SHIFT_ONLY:
    return hotcue_core.process_xml(input_path, output_path, config.shift)

# HYBRID または STANDALONE
validate_config(config.shift)
validate_analysis_config(config.analysis)

tree = ET.parse(input_path)
tracks = tree.getroot().findall(".//TRACK")
total = len(tracks)

total_stats = {
    "tracks_processed": 0,
    "tracks_analyzed": 0,
    "tracks_skipped": 0,
    "analysis_errors": 0,
    "shifted": 0,
    "placed": 0,
    "skipped_negative": 0,
    "skipped_not_target": 0,
}

for i, track in enumerate(tracks):
    name = track.get("Name", "Unknown")

    if config.analysis.mode == HYBRID:
        stats = process_track_hybrid(track, config.shift, config.analysis)
    else:
        stats = process_track_standalone(track, config.shift, config.analysis)

    # stats を集計
    total_stats["tracks_processed"] += 1
    ...

    if progress_callback:
        progress_callback(i + 1, total, name)

output_path.parent.mkdir(parents=True, exist_ok=True)
tree.write(output_path, encoding="utf-8", xml_declaration=True)
return total_stats
```

---

## 5. CLI 改修仕様 (hotcue_shifter.py)

### 5.1 既存引数の変更

**オフセットの単位を「拍」から「小節」に変更**:

| 引数 | 旧デフォルト | 新デフォルト | 変換式 |
|---|---|---|---|
| `--offset-high-bpm` | 32（拍） | 16（小節） | 内部: 小節 × 4 = 拍 |
| `--offset-normal-bpm` | 16（拍） | 16（小節） | 内部: 小節 × 4 = 拍 |

ヘルプテキストも「拍」→「小節」に変更。ShiftConfig への渡し方:
```python
ShiftConfig(
    offset_high_bpm=args.offset_high_bpm * 4,    # 小節→拍に変換
    offset_normal_bpm=args.offset_normal_bpm * 4,
    ...
)
```

### 5.2 新規引数

| 引数 | 型 | デフォルト | 説明 |
|---|---|---|---|
| `--mode` | `str` (choices) | `shift_only` | 処理モード: shift_only / hybrid / standalone |
| `--cue-run-up` / `--no-cue-run-up` | `bool` | `True` | 助走点キューを生成する |
| `--cue-drop` / `--no-cue-drop` | `bool` | `False` | ドロップキューを生成する |
| `--cue-buildup` / `--no-cue-buildup` | `bool` | `False` | ビルドアップキューを生成する |
| `--cue-breakdown` / `--no-cue-breakdown` | `bool` | `False` | ブレイクダウンキューを生成する |
| `--cue-outro` / `--no-cue-outro` | `bool` | `False` | アウトロキューを生成する |
| `--cue-intro` / `--no-cue-intro` | `bool` | `False` | イントロキューを生成する |
| `--max-cues` | `int` | `8` | 最大ホットキュー数 |
| `--min-cue-distance-bars` | `int` | `4` | キュー間最小距離（小節数） |
| `--score-threshold` | `float` | `0.3` | ドロップスコアの最低閾値 |
| `--onset-weight` | `float` | `0.4` | オンセット強度の重み |
| `--rms-weight` | `float` | `0.35` | RMSエネルギーの重み |
| `--spectral-weight` | `float` | `0.25` | スペクトルコントラストの重み |

### 5.3 main() の変更

```python
# モードに応じて AnalysisConfig を構築
analysis_config = AnalysisConfig(
    mode=ProcessingMode(args.mode),
    max_cues=args.max_cues,
    ...
)
pipeline_config = PipelineConfig(shift=shift_config, analysis=analysis_config)

# progress callback（標準出力に進捗表示）
def cli_progress(current, total, name):
    print(f"  [{current}/{total}] {name}")

stats = process_xml_pipeline(
    args.input_xml, args.output_xml,
    pipeline_config,
    progress_callback=cli_progress if args.mode != "shift_only" else None,
)
```

---

## 6. GUI 改修仕様 (hotcue_app.py)

### 6.1 レイアウト変更

```
Row 0: ヘッダー（既存）
Row 1: ファイル選択フレーム（既存）
Row 2: モード選択セグメントボタン（新規）
Row 3: シフト設定フレーム（既存、単位を小節に変更）
Row 4: キュー種類選択フレーム（新規、チェックボックス。モードに応じて表示/非表示）
Row 5: 詳細設定フレーム（新規、折りたたみ式。デフォルト非表示）
Row 6: アクションフレーム（ログ・ボタン・プログレスバー）← 既存 Row 3 から移動
```

### 6.2 新規ウィジェット

**モード選択**（Row 2）:
```python
self.mode_var = tk.StringVar(value="シフトのみ")
CTkSegmentedButton(
    values=["シフトのみ", "ハイブリッド", "音声分析のみ"],
    variable=self.mode_var,
    command=self._on_mode_changed,
)
```

**モード名マッピング**:
| 表示名 | ProcessingMode |
|---|---|
| シフトのみ | SHIFT_ONLY |
| ハイブリッド | HYBRID |
| 音声分析のみ | STANDALONE |

**キュー種類選択フレーム**（Row 4）:

ユーザーがキューを置きたい場所をチェックボックスで直感的に選択する。

```python
self.cue_run_up_var = tk.BooleanVar(value=True)     # デフォルトON
self.cue_drop_var = tk.BooleanVar(value=False)
self.cue_buildup_var = tk.BooleanVar(value=False)
self.cue_breakdown_var = tk.BooleanVar(value=False)
self.cue_outro_var = tk.BooleanVar(value=False)
self.cue_intro_var = tk.BooleanVar(value=False)

CTkCheckBox(text="ドロップ前（助走点）", variable=self.cue_run_up_var,
            tooltip="ドロップのN小節前にミックスイン開始点を配置")
CTkCheckBox(text="ドロップそのもの", variable=self.cue_drop_var,
            tooltip="エネルギーが急上昇するドロップ位置にキューを配置")
CTkCheckBox(text="ビルドアップ", variable=self.cue_buildup_var,
            tooltip="ドロップに向けてエネルギーが上昇し始める地点にキューを配置")
CTkCheckBox(text="ブレイクダウン", variable=self.cue_breakdown_var,
            tooltip="エネルギーが急降下するブレイクダウン位置にキューを配置")
CTkCheckBox(text="アウトロ", variable=self.cue_outro_var,
            tooltip="曲終盤のミックスアウト開始点にキューを配置")
CTkCheckBox(text="イントロ（ビート開始）", variable=self.cue_intro_var,
            tooltip="楽曲冒頭の最初のビートが入る地点にキューを配置")
```

**詳細設定フレーム**（Row 5、折りたたみ式）:

デフォルトでは折りたたまれており、「▶ 詳細設定」をクリックすると展開される。
上級者・開発者向けの数値パラメータを配置。

```python
self.advanced_visible = tk.BooleanVar(value=False)

self.advanced_toggle = CTkButton(
    text="▶ 詳細設定",
    fg_color="transparent", text_color="#64748b",
    command=self._toggle_advanced,
)

def _toggle_advanced(self):
    if self.advanced_visible.get():
        self.advanced_frame.grid_remove()
        self.advanced_toggle.configure(text="▶ 詳細設定")
        self.advanced_visible.set(False)
    else:
        self.advanced_frame.grid()
        self.advanced_toggle.configure(text="▼ 詳細設定")
        self.advanced_visible.set(True)
```

詳細設定フレーム内のウィジェット:
- 最大Hot Cue数: CTkEntry（self.max_cues_var）
- 最小距離（小節）: CTkEntry（self.min_cue_distance_var）
- スコア閾値: CTkEntry（self.score_threshold_var）
- オンセット重み: CTkEntry（self.onset_weight_var）
- RMS重み: CTkEntry（self.rms_weight_var）
- スペクトル重み: CTkEntry（self.spectral_weight_var）

**オフセット表示の変更**: 既存の「高BPMオフセット（拍）」「通常BPMオフセット（拍）」を「高BPMオフセット（小節）」「通常BPMオフセット（小節）」に変更。デフォルト値を 16 に変更。内部で × 4 して拍に変換。

### 6.3 モード切替動作

```python
def _on_mode_changed(self, selected_value):
    if selected_value == "シフトのみ":
        self.cue_type_frame.grid_remove()   # キュー種類チェックボックスを非表示
        self.advanced_toggle.grid_remove()  # 詳細設定トグルも非表示
        self.advanced_frame.grid_remove()
    else:
        self.cue_type_frame.grid()          # キュー種類チェックボックスを表示
        self.advanced_toggle.grid()         # 詳細設定トグルを表示
```

### 6.4 プログレスバー

- shift_only: 既存の indeterminate モード
- hybrid / standalone: determinate モードに切替
  ```python
  self.progress.configure(mode="determinate")
  self.progress.set(0)
  ```
- ワーカースレッドからの進捗更新:
  ```python
  def _make_progress_callback(self):
      def callback(current, total, name):
          self.after(0, self._update_progress, current, total, name)
      return callback

  def _update_progress(self, current, total, name):
      self.progress.set(current / max(total, 1))
      self.status_label.configure(text=f"分析中 [{current}/{total}] {name}")
  ```

### 6.5 librosa 可用性チェック

```python
# __init__ 内
self._librosa_available = False
try:
    import librosa
    self._librosa_available = True
except ImportError:
    pass

# モード選択ボタンで、librosa がない場合は hybrid/standalone を無効化
```

### 6.6 設定永続化

`_save_settings()` / `_load_settings()` に以下のキーを追加:
```
mode,
cue_run_up, cue_drop, cue_buildup, cue_breakdown, cue_outro, cue_intro,
max_cues, min_cue_distance_bars, score_threshold,
onset_weight, rms_weight, spectral_weight
```

### 6.7 デフォルトリセット

`_reset_defaults()` に解析設定のデフォルト値リセットを追加。

---

## 7. requirements.txt 変更

```
customtkinter>=5.2.2,<6
pyinstaller>=6.11,<7
librosa>=0.10,<1
soundfile>=0.12,<1
```

---

## 8. エラーハンドリング一覧

| 場面 | エラー種別 | 対処 |
|---|---|---|
| 音声ファイルが見つからない | FileNotFoundError | TrackAnalysisResult.error に記録、フォールバック |
| librosa デコード失敗 | RuntimeError | 同上 |
| librosa 未インストール | ImportError | GUI: モード無効化 / CLI: エラーメッセージ表示 |
| TEMPO 要素なし | — | hybrid: process_track() にフォールバック / standalone: スキップ |
| Location 属性なし | — | 同上 |
| BPM <= 0 | — | 同上 |
| 全候補が score_threshold 未満 | — | キュー0個として処理（既存キュー削除のみ standalone では実行） |
| URL デコード失敗 | — | location_to_filepath() が None を返し、フォールバック |

---

## 9. 処理フロー図

### 9.1 hybrid モード（1トラック）

```
TRACK XML要素
  │
  ├─ TEMPO → extract_tempo_data() → (first_downbeat, bpm, bar_duration)
  ├─ POSITION_MARK → extract_mik_cue_data() → [{start, num, name}, ...]
  └─ Location → location_to_filepath() → "/path/to/track.mp3"
       │
       ▼
  load_audio() ─→ 特徴抽出
       │            ├─ compute_onset_strength()
       │            ├─ compute_rms_energy() → compute_rms_derivative()
       │            └─ compute_spectral_contrast() → compute_spectral_derivative()
       │
       ▼
  compute_drop_scores() ─→ 合成スコア配列
       │
       ▼
  各MIKキュー位置を snap_to_bar() でスナップ
       │
       ▼
  score_mik_cues() ─→ [CueCandidate, ...]
       │
       ├─ deduplicate_candidates()
       └─ select_top_candidates()
              │
              ▼
         apply_analysis_cues()
              │
              ├─ 不要な既存ホットキューを削除
              ├─ 選択されたキューの Start を更新
              └─ calculate_offset_seconds() でシフト適用
```

### 9.2 standalone モード（1トラック）

```
TRACK XML要素
  │
  ├─ TEMPO → extract_tempo_data()
  └─ Location → location_to_filepath()
       │
       ▼
  load_audio() → 特徴抽出 → compute_drop_scores()
       │
       ▼
  detect_drop_peaks() ─→ [(time, score), ...]
       │
       ▼
  generate_standalone_candidates() ─→ [CueCandidate, ...]
       │
       ▼
  各候補を snap_to_phrase() でフレーズ境界にスナップ
       │
       ├─ deduplicate_candidates()
       └─ select_top_candidates()
              │
              ▼
         apply_analysis_cues()
              │
              ├─ 既存ホットキュー全削除
              ├─ 新規 POSITION_MARK 生成 (Num=0,1,2...)
              └─ calculate_offset_seconds() でシフト適用
```

---

## 10. テスト計画

### 10.1 単体テスト（audio_analyzer.py）

| テスト対象 | 入力 | 期待結果 |
|---|---|---|
| `compute_first_downbeat` | Inizio=0.041, Bpm=130, Battito=1 | 0.041 |
| `compute_first_downbeat` | Inizio=0.024, Bpm=145, Battito=3 | 0.024 - 2*(60/145) = -0.803 |
| `compute_bar_duration` | Bpm=130 | 1.846... |
| `snap_to_bar` | time=59.118, fd=0.041, bd=1.846 | 59.113 (bar 32) |
| `snap_to_bar` | time=0.5, fd=0.041, bd=1.846 | 0.041 (bar 0) |
| `location_to_filepath` | `file://localhost/Users/user/Desktop/music/track%20name.mp3` | `/Users/user/Desktop/music/track name.mp3` |
| `location_to_filepath` | `""` | `None` |
| `deduplicate_candidates` | 2候補が1秒差、min_dist=2秒 | 高スコアのみ残る |
| `select_top_candidates` | 5候補、max=3、threshold=0.2 | 上位3個、時間順 |
| `compute_drop_scores` | 全ゼロ配列 | 全ゼロ配列 |

### 10.2 結合テスト（pipeline.py）

| テスト | 確認内容 |
|---|---|
| SHIFT_ONLY モード | `hotcue_core.process_xml()` と同一出力 |
| HYBRID フォールバック（TEMPO なし） | `process_track()` と同一出力 |
| HYBRID フォールバック（音声ファイルなし） | `process_track()` と同一出力 |
| STANDALONE（TEMPO なし） | トラックがスキップされること |
| apply_analysis_cues（hybrid） | 選択キュー以外が削除、Start が更新 |
| apply_analysis_cues（standalone） | 全ホットキュー削除後に新規作成、Memory Cue は残存 |

### 10.3 手動検証

1. ref/rekordbox_20260125.xml を shift_only で処理 → 既存ツールと同一出力
2. 同 XML を hybrid で処理 → 出力 XML のキュー位置が小節頭に揃っている、キュー数が削減されている
3. 同 XML を standalone で処理 → MIK キューと無関係に新規キューが生成されている
4. GUI でモード切替 → 設定フレームの表示/非表示が正しい
5. 存在しない音声ファイルを含む XML → フォールバックが動作しエラーにならない

---

## 11. 将来の拡張候補

現在の6種類のキュータイプでカバーできない、将来追加を検討すべき機能。

### 11.1 ボーカルイン検出

| 項目 | 内容 |
|------|------|
| **概要** | ボーカルが始まる地点にキューを配置 |
| **DJ価値** | 高。2曲のボーカルが重なるのを防ぐため、ミックス時に必須の情報 |
| **技術要件** | 音源分離（Vocal Isolation）が必要。Demucs, Spleeter 等のモデルを使用 |
| **実現性** | 中。librosa 単独では困難。別途モデルの導入が必要でアプリサイズが大幅増加 |
| **優先度** | 高（技術的ハードルを超えられれば最も価値のある追加機能） |

### 11.2 バース / コーラス構造解析

| 項目 | 内容 |
|------|------|
| **概要** | 楽曲の構造（Verse / Chorus / Bridge）を自動分類 |
| **DJ価値** | 中。EDM以外のジャンル（Pop, Hip Hop）で特に有用 |
| **技術要件** | 楽曲構造セグメンテーション（MSAF, allin1 等の専用ライブラリ） |
| **実現性** | 中。rekordbox 7 の Phrase Analysis と類似の機能 |
| **優先度** | 中 |

### 11.3 ループポイント検出

| 項目 | 内容 |
|------|------|
| **概要** | ループに適したセクション（安定したリズムパターン）を検出 |
| **DJ価値** | 中。ミックス時間の延長やクリエイティブなパフォーマンスに使用 |
| **技術要件** | ビートグリッドの安定性解析、繰り返しパターンの検出 |
| **実現性** | 高。現在の特徴量（RMS, onset）の安定性指標で実装可能 |
| **優先度** | 低（ホットキューの自動化とは方向性が異なる） |

### 11.4 キューカラーの自動設定

| 項目 | 内容 |
|------|------|
| **概要** | CueRole に応じて rekordbox のキューカラーを自動設定 |
| **DJ価値** | 高。一目でキューの種類を判別できる |
| **技術要件** | POSITION_MARK の Red/Green/Blue 属性を設定 |
| **実現性** | 高。XML属性の追加のみで実装可能 |
| **優先度** | 高（v2 のマイナーアップデートとして実装可能） |

### 11.5 ジャンル別プリセット

| 項目 | 内容 |
|------|------|
| **概要** | ジャンルに応じた推奨キュー種類・パラメータのプリセット |
| **DJ価値** | 中〜高。「House向け」「D&B向け」などワンクリックで最適設定 |
| **技術要件** | プリセット定義ファイルの管理 |
| **実現性** | 高 |
| **優先度** | 中（キュー種類選択の UI が安定した後に追加） |
