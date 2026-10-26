#!/bin/zsh
# Checks the proposed with-gpu-lock.sh on PRIVATE lease and lock paths. Never touches /tmp/atelier-gpu.lock or
# ~/.cache/aether-kit/gpu.lock. Usage: wrapper-check.zsh WRAPPER KIT_HELPER AETHER_PROBE WORK_DIR
set -u
wrapper=$1 kit=$2 probe=$3 work=$4
mkdir -p $work/wrapdir
export AETHER_GPU_WRAPPER_DIR=$work/wrapdir
export AETHER_GPU_LEASE_PATH=$work/gpu.lock
lease=$AETHER_GPU_LEASE_PATH
[[ $lease == */atelier-gpu.lock* || $AETHER_GPU_WRAPPER_DIR == $HOME/.cache/aether-kit* ]] && { echo "refusing a shared path"; exit 64 }
unset AETHER_GPU_LEASE_HOLDER
failures=0
log() { print -r -- "$(date '+%H:%M:%S') $*" }
check() { if [[ "$2" == "$3" ]]; then log "PASS $1: $3"; else log "FAIL $1: expected [$2] got [$3]"; (( failures += 1 )); fi }
holder() { cat $lease/pid 2>/dev/null || print -n none }
log "wrapper $(shasum -a 256 $wrapper | cut -c1-16) lease $lease lock $AETHER_GPU_WRAPPER_DIR/gpu.lock"

log "--- 1. The wrapper hands its lease to Kit tools explicitly"
out=$($wrapper "check: hand-off" $kit try $lease inherit); st=$?
check "Kit helper inherits the wrapper's lease" "0 inherited" "$st ${out% *}"
out=$($wrapper "check: probe" $probe inspect $work/missing.aimodel 2>&1); st=$?
check "aether-probe runs at once under the wrapper (missing model: exit 1, no wait)" "1 1 0" \
  "$st $(print -r -- $out | grep -c 'Invalid model') $(print -r -- $out | grep -c waiting)"
check "lease released after the command" "none" "$(holder)"

log "--- 2. A wrapper interrupted by SIGTERM stops its command tree before releasing"
$wrapper "check: supervise" /bin/sh -c 'sleep 300 & echo $! > "$0/grandchild"; echo $$ > "$0/child"; wait' $work > /dev/null 2>&1 &
outer=$!
n=0; while [[ ! -s $work/grandchild || ! -s $work/child || "$(holder)" == none ]] && (( n < 500 )); do sleep 0.02; (( n += 1 )); done
inner=$(holder); child=$(cat $work/child); grandchild=$(cat $work/grandchild)
check "the command runs with the wrapper's pid in the lease" "$inner" "$(ps -o ppid= -p $child | tr -d ' ')"
kill -TERM $inner
wait $outer; st=$?
check "wrapper exit status" "130" "$st"
check "command stopped" "gone" "$(kill -0 $child 2>/dev/null && print alive || print gone)"
check "command's own child stopped" "gone" "$(kill -0 $grandchild 2>/dev/null && print alive || print gone)"
check "lease released after the command stopped" "none" "$(holder)"
check "interruption logged" "1" "$(grep -c 'interrupted pid=.*check: supervise' $AETHER_GPU_WRAPPER_DIR/gpu-lock.log)"

log "--- 3. A pid-less lease is logged explicitly and never removed by a waiting wrapper"
mkdir $lease
$wrapper "check: pidless" /usr/bin/true > /dev/null 2>&1 &
outer=$!
n=0; while (( $(grep -c 'no pid file' $AETHER_GPU_WRAPPER_DIR/gpu-lock.log) == 0 )) && (( n < 500 )); do sleep 0.02; (( n += 1 )); done
check "pid-less lease logged" "1" "$(grep -c 'busy (no pid file' $AETHER_GPU_WRAPPER_DIR/gpu-lock.log)"
inner=$(pgrep -P $outer)
kill -TERM $inner; wait $outer 2>/dev/null
check "the waiting wrapper left the pid-less lease alone" "dir" "$([[ -d $lease && ! -e $lease/pid ]] && print dir || print changed)"
rmdir $lease

left=$(ls -A $work | grep -v -e '^wrapdir$' -e '^child$' -e '^grandchild$' | paste -sd ' ' -)
log "final directory: ${left:-(no lease or leftovers)}"
log "failures=$failures"
exit $failures
