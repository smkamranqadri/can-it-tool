#!/usr/bin/env bash
# Load-test one GGUF on a CPU-only llama-server with N parallel slots.
#
# usage: scripts/load_cpu.sh <label> <gguf> [levels...]      (default levels: 1 2 4 8)
#
# Environment (all optional): MODELS, SFLAGS, LLAMA, SHIM, SHIM_PY, THREADS as in
# run_cpu.sh, plus
#   SECONDS_PER_LEVEL   load duration per concurrency level   (default 90)
#   SLOT_CTX            context per parallel slot              (default 8192)
#   SCEN                scenario id prefixes to draw from      (default: whole suite)
# Writes results/load/<label>.json and results/load/mon/<label>/{server,shim,ps,load}.log.
#
# bash, not zsh, and kept bash-3.2-safe so macOS can run it unchanged. Every path is
# quoted: zsh does not word-split unquoted expansions, bash does, and the repo path may
# contain spaces.
set -u
LABEL=$1; GGUF=$2; shift 2
if [ $# -eq 0 ]; then LEVELS=(1 2 4 8); else LEVELS=("$@"); fi
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MON="$REPO/results/load/mon/$LABEL"; mkdir -p "$MON"
NP=${LEVELS[$(( ${#LEVELS[@]} - 1 ))]}
LOCK="$REPO/results/latency/mon/.cpu.lock"; mkdir -p "$(dirname "$LOCK")"
until mkdir "$LOCK" 2>/dev/null; do sleep 5; done
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
read -ra SFLAGS_ARR <<< "${SFLAGS:-}"

"${LLAMA:-llama-server}" -m "${MODELS:-$HOME/models}/$GGUF" --host 127.0.0.1 --port 8080 \
  -c $(( NP * ${SLOT_CTX:-8192} )) -np $NP --device none -ngl 0 -t ${THREADS:-8} --flash-attn on --jinja ${SFLAGS_ARR[@]+"${SFLAGS_ARR[@]}"} \
  > "$MON/server.log" 2>&1 &
PID=$!
for i in {1..120}; do curl -sf localhost:8080/health >/dev/null && break; sleep 1; done
curl -sf localhost:8080/health >/dev/null || { echo "$LABEL: server failed"; tail -20 "$MON/server.log"; kill $PID; exit 1; }
BASE=http://127.0.0.1:8080/v1; SHIMPID=
if [[ -n ${SHIM:-} ]]; then
  "${SHIM_PY:-python3}" "$SHIM" http://127.0.0.1:8080 8090 > "$MON/shim.log" 2>&1 & SHIMPID=$!
  for i in {1..900}; do curl -sf localhost:8090/health >/dev/null && break; sleep 1; done
  BASE=http://127.0.0.1:8090/v1
fi
(while kill -0 $PID 2>/dev/null; do echo "$(date +%s) $(ps -o %cpu=,rss= -p $PID) $([[ -n $SHIMPID ]] && ps -o %cpu=,rss= -p $SHIMPID)"; sleep 1; done > "$MON/ps.log") &
SAMP=$!

SCEN_ARGS=(); for x in ${SCEN:-}; do SCEN_ARGS+=(--scenario $x); done
cd "$REPO"
uv run --python 3.14 python scripts/load_test.py --base-url "$BASE" --model "$LABEL" --levels "${LEVELS[@]}" \
  --seconds ${SECONDS_PER_LEVEL:-90} ${SCEN_ARGS[@]+"${SCEN_ARGS[@]}"} --out "results/load/$LABEL.json" 2>&1 | tee "$MON/load.log"
kill $SAMP $PID $SHIMPID 2>/dev/null; wait $PID 2>/dev/null
awk '{if($3>mr)mr=$3; if($5>ms)ms=$5} END {printf "'"$LABEL"': peak rss server %.0f MB, adapter %.0f MB\n", mr/1024, ms/1024}' "$MON/ps.log"
exit 0
