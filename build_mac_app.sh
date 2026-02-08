#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

python3 -m pip install -r requirements.txt

rm -rf build dist HotCueAutoMaker.spec

python3 -m PyInstaller \
  --clean \
  --noconfirm \
  --windowed \
  --name "HotCueAutoMaker" \
  hotcue_app.py

xattr -cr "${ROOT_DIR}/dist/HotCueAutoMaker.app"
if /usr/bin/codesign --force --deep --sign - "${ROOT_DIR}/dist/HotCueAutoMaker.app"; then
  echo "Ad-hoc署名: 完了"
else
  echo "Ad-hoc署名: スキップ（必要なら手動でcodesignしてください）"
fi

echo "Build complete: ${ROOT_DIR}/dist/HotCueAutoMaker.app"
