# Hot Cue Auto Maker - 設計ドキュメント

## 概要

Mixed In Key (MIK) が自動生成したHot Cueの位置を調整し、DJミックス用の「助走点」を作成するツール。

## 背景・目的

- MIKはエネルギーポイント（ドロップ等）を検出してHot Cueを自動生成する
- しかし、DJミックスでは**ドロップの手前**からミックスインしたい
- → Hot Cueを**N拍前にずらす**ことで「助走点」として使えるようにする

## 処理フロー

```
[MIK出力XML] → [スクリプト] → [編集済みXML] → [rekordboxにインポート]
```

### 詳細フロー

1. MIKが出力したrekordbox XMLを読み込み
2. 各TRACKを走査
3. POSITION_MARK（Hot Cue: Num >= 0）を処理
   - Energy 6-7（ドロップ / ビルドアップ）を対象
   - BPMに応じてオフセット拍数を決定
   - Start位置を -N拍 ずらす
   - マイナスになる場合はスキップ（元の位置を維持）
4. Memory Cue（Num = -1）は変更しない（元の位置に残る）
5. 編集後のXMLを出力

## XML構造

### トラック情報
```xml
<TRACK TrackID="..." Name="曲名" Artist="アーティスト"
       AverageBpm="130.00" Genre="House" ...>
    <TEMPO Inizio="0.025" Bpm="130.00" Metro="4/4" Battito="1"/>
    <POSITION_MARK Name="Energy 7" Start="59.117" Type="0" Num="-1"/>  <!-- Memory Cue -->
    <POSITION_MARK Name="Energy 7" Start="59.117" Type="0" Num="3"/>   <!-- Hot Cue D -->
</TRACK>
```

### 重要な属性

| 要素/属性 | 説明 |
|-----------|------|
| `AverageBpm` | トラック全体のBPM |
| `TEMPO.Bpm` | その時点のBPM（通常AverageBpmと同じ） |
| `POSITION_MARK.Name` | "Energy N" 形式（N=エネルギーレベル 1-10） |
| `POSITION_MARK.Start` | キュー位置（秒） |
| `POSITION_MARK.Num` | -1=Memory Cue, 0-7=Hot Cue A-H |
| `POSITION_MARK.Type` | 0=ポイントキュー |

## 仕様

### オフセット計算

| 条件 | オフセット | 理由 |
|------|------------|------|
| BPM > 150 | -32拍（8小節） | D&B等は速いので長めの助走 |
| BPM <= 150 | -16拍（4小節） | House/Techno標準 |

### 計算式

```
1拍の秒数 = 60 / BPM
オフセット秒数 = (60 / BPM) * オフセット拍数
新しいStart = 元のStart - オフセット秒数
```

### 処理対象

| Energy Level | 処理 | 想定 |
|--------------|------|------|
| 6-7 | Hot Cueを-N拍ずらす | ドロップ / ビルドアップ |
| 1-5 | 変更しない | イントロ / ブレイクダウン等 |

### エッジケース

| ケース | 処理 |
|--------|------|
| 新しいStartがマイナス | スキップ（元の位置を維持） |
| BPM情報がない | AverageBpm使用、なければスキップ |
| Energy 1-5 | 変更しない |

## 設定項目

```python
BPM_THRESHOLD = 150        # この値を超えたら高BPM扱い
OFFSET_HIGH_BPM = 32       # 高BPM時のオフセット（拍）
OFFSET_NORMAL_BPM = 16     # 通常時のオフセット（拍）
TARGET_ENERGY_MIN = 6      # 処理対象のEnergy Level（最小）
TARGET_ENERGY_MAX = 7      # 処理対象のEnergy Level（最大）
```

### 設定の扱い
- すべて実行時に可変（固定値ではない）
- CLIから引数で指定可能
- ローカルアプリUIからも指定可能（前回値を保存）

## 入出力

### 入力
- MIKが出力したrekordbox形式XML

### 出力
- 編集後のrekordbox形式XML（別ファイル）

### コマンド例
```bash
python hotcue_shifter.py input.xml output.xml
```

```bash
python hotcue_shifter.py input.xml output.xml \
  --bpm-threshold 148 \
  --offset-high-bpm 40 \
  --offset-normal-bpm 24 \
  --energy-min 5 \
  --energy-max 7
```

## ローカルアプリ（macOS）

### 開発起動
```bash
python3 -m pip install -r requirements.txt
python3 hotcue_app.py
```

### 配布用appビルド
```bash
./build_mac_app.sh
```

出力先:
- `dist/HotCueAutoMaker.app`

## 処理結果の例

### Before（MIK出力）
```
Hot Cue D: Start=59.117秒 (Energy 7 = ドロップ)
Memory Cue: Start=59.117秒 (同位置に残る)
```

### After（スクリプト処理後）
```
Hot Cue D: Start=51.732秒 (59.117 - 7.385秒 = -16拍 @130BPM)
Memory Cue: Start=59.117秒 (変更なし、ランドマークとして残る)
```

## 将来の拡張案（今回は実装しない）

- Energy 6とEnergy 7で異なるオフセット設定
- ジャンル別のオフセット設定
- Hot Cueに色を付ける
- Nameを変更する（例: "PRE-16 Energy 7"）
- ビルドアップの自動判定ロジック改善
