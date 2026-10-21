#!/usr/bin/env bash
# Start llama-server with a vision model download_vlm.py put in vlm/.
# Leave it running while read_screenshots.py works, or let that script start it (--llama-server PATH).
# usage: ./serve_vlm.sh PATH/TO/llama-server [8b|2b] [port] [slots]      (defaults: 8b, 8080, 4)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${1:?usage: ./serve_vlm.sh PATH/TO/llama-server [8b|2b] [port] [slots]}
case "${2:-8b}" in
  8b) MODEL=Qwen3VL-8B-Instruct-Q4_K_M.gguf; PROJ=mmproj-Qwen3VL-8B-Instruct-F16.gguf ;;
  2b) MODEL=Qwen3VL-2B-Instruct-Q8_0.gguf;   PROJ=mmproj-Qwen3VL-2B-Instruct-F16.gguf ;;
  *) echo "unknown model '$2' (8b or 2b)" >&2; exit 2 ;;
esac
PORT=${3:-8080}
SLOTS=${4:-4}
for f in "$MODEL" "$PROJ"; do
  if [ ! -f "$HERE/vlm/$f" ]; then
    echo "$f not in $HERE/vlm: run  python download_vlm.py --model ${2:-8b}" >&2
    exit 1
  fi
done
echo "serving $MODEL on http://127.0.0.1:$PORT with $SLOTS slots (Ctrl-C to stop)" >&2
# 2048 tokens of context per slot: a box crop plus the reply fits easily
exec "$BIN" -m "$HERE/vlm/$MODEL" --mmproj "$HERE/vlm/$PROJ" -ngl 99 -c $((2048 * SLOTS)) -np "$SLOTS" \
     --host 127.0.0.1 --port "$PORT"
