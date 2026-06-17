#!/bin/bash
# 1ノートブックをヘッドレス実行する（バックグラウンド・SSH切断後も継続）
# 使い方: bash run_one.sh "<notebook>.ipynb"
set -u
DIR="/home/hasegawakazuki/クマの研究/ベースラインとCNN/実験３/20260609実験環境/20260601全モデル比較宮城県"
cd "$DIR" || { echo "cd failed"; exit 1; }
mkdir -p logs _executed
NB="$1"
BASE="${NB%.ipynb}"
PY=/home/hasegawakazuki/anaconda3/bin
export PATH="$PY:$PATH"
# DataLoader等のFDリーク対策: ファイルディスクリプタ上限をhard上限まで引き上げ
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
LOG="logs/${BASE}.log"
echo "==== launch $(date '+%F %T') host=$(hostname) nb=$NB ====" >> "$LOG"
setsid "$PY/jupyter" nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=pytorch_env \
  --output-dir _executed --output "${BASE}_executed" \
  "$NB" >> "$LOG" 2>&1 < /dev/null &
PID=$!
echo "started PID $PID on $(hostname): $NB  (log: $LOG)"