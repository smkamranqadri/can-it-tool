#!/usr/bin/env bash
# Benchmark one GGUF on a CPU-only llama-server, sampling the server's CPU and memory.
#
# usage: scripts/run_cpu.sh <label> <gguf> <quant> [notes]
#
# Environment (all optional):
#   MODELS    directory holding the GGUF files          (default ~/models)
#   OUT       results directory, relative to the repo    (default results)
#   SCEN      space-separated scenario ids or prefixes   (default: whole suite)
#   RUNS      runs per scenario                          (default 1)
#   THREADS   llama-server threads                       (default 8)
#   DEVFLAGS  device selection flags                     (default "--device none -ngl 0")
#   SFLAGS    extra llama-server flags, e.g. "--reasoning off"
#   LLAMA     llama-server binary                         (default llama-server on PATH)
#   SHIM      adapter to put between harness and server, e.g. adapters/laya_pipeline.py
#   SHIM_PY   python to run the adapter with              (default python3)
#   MON_ROOT  where per-run logs go                       (default results/latency/mon)
#
# Per run it writes server.log, bench.log, ps.log (CPU% and RSS every second), freq.log
# (per-core MHz and package temperature every second, Linux only) and, when a tmux session
# named canit-mon is running btop, btop.log (a pane snapshot every 15 s).
#
# bash, not zsh, and kept bash-3.2-safe so macOS can run it unchanged. Every path is
# quoted: zsh does not word-split unquoted expansions, bash does, and the repo path may
# contain spaces.
set -u
LABEL=$1; GGUF=$2; QUANT=$3; EXTRA=${4:-}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MON="${MON_ROOT:-$REPO/results/latency/mon}/$LABEL"; mkdir -p "$MON"

# One benchmark at a time: concurrent runs would share port 8080 and skew each other's latency.
LOCK=$(dirname "$MON")/.cpu.lock
until mkdir "$LOCK" 2>/dev/null; do sleep 5; done
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
read -ra SFLAGS_ARR <<< "${SFLAGS:-}"
# DEVFLAGS overrides the CPU-only device selection, e.g. DEVFLAGS="-ngl 99" with a Vulkan build.
read -ra DEV_ARR <<< "${DEVFLAGS:---device none -ngl 0}"
FLAGS=(-c 32768 "${DEV_ARR[@]}" -t ${THREADS:-8} -np 1 --flash-attn on --jinja ${SFLAGS_ARR[@]+"${SFLAGS_ARR[@]}"})
SCEN_ARGS=(); for x in ${SCEN:-}; do SCEN_ARGS+=(--scenario $x); done
LLAMA_BIN=${LLAMA:-llama-server}
LLAMA_VER=$("$LLAMA_BIN" --version 2>&1 | grep -m1 version)
if command -v sysctl >/dev/null 2>&1 && sysctl -n machdep.cpu.brand_string >/dev/null 2>&1; then
  MACHINE="$(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
else
  MACHINE="$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1), $(( $(sed -n 's/^MemTotal:[[:space:]]*\([0-9]*\) kB/\1/p' /proc/meminfo) / 1048576 )) GB"
fi

"$LLAMA_BIN" -m "${MODELS:-$HOME/models}/$GGUF" --host 127.0.0.1 --port 8080 "${FLAGS[@]}" > "$MON/server.log" 2>&1 &
PID=$!
for i in {1..120}; do curl -sf localhost:8080/health >/dev/null && break; sleep 1; done
curl -sf localhost:8080/health >/dev/null || { echo "$LABEL: server failed"; tail -20 "$MON/server.log"; kill $PID; exit 1; }

BASE=http://127.0.0.1:8080/v1; SHIMPID=
if [[ -n ${SHIM:-} ]]; then
  "${SHIM_PY:-python3}" "$SHIM" http://127.0.0.1:8080 8090 > "$MON/shim.log" 2>&1 & SHIMPID=$!
  for i in {1..900}; do curl -sf localhost:8090/health >/dev/null && break; sleep 1; done
  BASE=http://127.0.0.1:8090/v1
fi

# Sample the adapter too: on a memory-constrained target Laya's RSS counts against the same ceiling.
(while kill -0 $PID 2>/dev/null; do echo "$(date +%s) $(ps -o %cpu=,rss= -p $PID) $([[ -n $SHIMPID ]] && ps -o %cpu=,rss= -p $SHIMPID)"; sleep 1; done > "$MON/ps.log") &
SAMP=$!
# Sustained clock matters as much as the latency on a thermally limited laptop, so record it.
FREQ=
if [[ -r /proc/cpuinfo ]]; then
  (while kill -0 $PID 2>/dev/null; do
     echo "$(date +%s) $(sed -n 's/^cpu MHz[[:space:]]*: //p' /proc/cpuinfo | tr '\n' ' ')$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | tr '\n' ' ')"
     sleep 1
   done > "$MON/freq.log") &
  FREQ=$!
fi
SNAP=
if tmux has-session -t canit-mon 2>/dev/null; then
  (while kill -0 $PID 2>/dev/null; do echo "=== $(date +%T)"; tmux capture-pane -p -t canit-mon | head -16; sleep 15; done > "$MON/btop.log") &
  SNAP=$!
fi

cd "$REPO"
uv run --python 3.14 python benchmark.py --base-url "$BASE" --model "$LABEL" \
  ${SCEN_ARGS[@]+"${SCEN_ARGS[@]}"} --runs ${RUNS:-1} --max-steps 8 --temperature 0 -o "${OUT:-results}/$LABEL.json" --label "${LABEL#*-}" \
  --quantization "$QUANT" --model-file "$GGUF" --runtime llama.cpp \
  --notes "llama-server $LLAMA_VER; CPU only: ${FLAGS[*]}; $MACHINE${EXTRA:+; $EXTRA}" \
  > "$MON/bench.log" 2>&1
echo "$LABEL: benchmark exit $?"

kill $SAMP $SNAP $FREQ $PID $SHIMPID 2>/dev/null; wait $PID 2>/dev/null
awk '{c+=$2; if($2>mc)mc=$2; if($3>mr)mr=$3; if(NF>3 && $5>ms)ms=$5; n++} END {printf "%s: %d s, mean cpu %.0f%%, peak cpu %.0f%%, peak rss %.0f MB%s\n", "'$LABEL'", n, c/n, mc, mr/1024, (ms?sprintf(", adapter rss %.0f MB", ms/1024):"")}' "$MON/ps.log" | tee "$MON/resources.txt"
if [[ -s "$MON/freq.log" ]]; then
  awk '{s=0; n=0; for(i=2;i<=NF;i++) if($i+0<10000){s+=$i; n++} if(n){t+=s/n; r++}} END {if(r) printf "  sustained clock: %.0f MHz mean over %d s\n", t/r, r}' "$MON/freq.log" | tee -a "$MON/resources.txt"
fi
[[ -f "$MON/btop.log" ]] && grep -m1 GPU "$MON/btop.log" | sed 's/.*\(GPU[^│]*\).*/  btop \1/'
exit 0
