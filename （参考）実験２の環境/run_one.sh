#!/bin/bash
# 実験2(持ち回り5fold) 1ノートをヘッドレス実行（バックグラウンド・SSH切断後も継続）
# 使い方:            bash run_one.sh "<notebook>.ipynb"
#   GPUピン留め例:   CUDA_VISIBLE_DEVICES=0,1 bash run_one.sh "<notebook>.ipynb"
set -u
DIR="/home/hasegawakazuki/クマの研究/ベースラインとCNN/実験３/20260609実験環境/20260601全モデル比較宮城県_実験2"
cd "$DIR" || { echo "cd failed"; exit 1; }
mkdir -p logs _executed
NB="$1"
BASE="${NB%.ipynb}"
PY=/home/hasegawakazuki/anaconda3/bin
export PATH="$PY:$PATH"
LOG="logs/${BASE}.log"
echo "==== launch $(date '+%F %T') host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} nb=$NB ====" >> "$LOG"
setsid "$PY/jupyter" nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=pytorch_env \
  --output-dir _executed --output "${BASE}_executed" \
  "$NB" >> "$LOG" 2>&1 < /dev/null &
PID=$!
echo "started PID $PID on $(hostname) [CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}]: $NB  (log: $LOG)"
