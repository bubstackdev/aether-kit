#!/bin/zsh
# Serialize native GPU work (Core AI inference, MPS exports/oracles, native test
# suites) across aether-kit worktrees and agents, and with Atelier. Usage:
#   ~/.cache/aether-kit/with-gpu-lock.sh <label> <command> [args...]
# Takes ~/.cache/aether-kit/gpu.lock (lockf; orders aether-kit jobs), then
# Atelier's shared lease /tmp/atelier-gpu.lock (mkdir directory + pid file, see
# AetherKit SharedGPULease and Atelier's SharedGPULock). Never reclaims another
# holder's lease; a stale or pid-less lease is logged and waited on. Logs to
# gpu-lock.log.
#
# Changes from the previous wrapper (AetherKit 0.0.2 GPU lease review):
# - Hands its lease to the command explicitly: exports AETHER_GPU_LEASE_HOLDER=<pid>,
#   so Kit tools (aether-probe, the native suites) inherit it instead of waiting for
#   their own parent. Kit tools inherit nothing without this hand-off.
# - Supervises the command: on INT/TERM/HUP it terminates the command's whole
#   process tree (TERM, then KILL after 10 s) before releasing the lease, so no GPU
#   work outlives the lease. (SIGKILL of the wrapper cannot be handled; Kit tools
#   then reclaim the stale lease once the holder is gone.)
# - Installs its trap before taking the lease, and releases by renaming the lease
#   aside before deleting it, so a signal mid-acquire or mid-release does not leave
#   a pid-less lease at the shared path.
# - Logs a pid-less lease explicitly instead of as "holder unknown".
# AETHER_GPU_LEASE_PATH and AETHER_GPU_WRAPPER_DIR override the lease and lock
# locations (testing only); unset, the paths are the shared ones above.
set -u
label=$1; shift
dir=${AETHER_GPU_WRAPPER_DIR:-$HOME/.cache/aether-kit}
lease=${AETHER_GPU_LEASE_PATH:-/tmp/atelier-gpu.lock}
log=$dir/gpu-lock.log
echo "$(date '+%F %T') waiting  pid=$$ $label" >> $log
exec /usr/bin/lockf -k $dir/gpu.lock /bin/zsh -c '
  log=$1; label=$2; lease=$3; shift 3
  created=0 child=0
  release() {
    # Only this holder lease: one naming this pid, or the one it just created and has not named yet.
    local holder=$(cat $lease/pid 2>/dev/null)
    [[ $holder == $$ || ( $created == 1 && -z $holder ) ]] || return 0
    local aside=$lease.released-$$-$RANDOM$RANDOM
    mv $lease $aside 2>/dev/null && rm -rf $aside
  }
  descendants() { local p; for p in $(pgrep -P $1 2>/dev/null); do print $p; descendants $p; done }
  stop_child() {
    (( child > 0 )) || return 0
    local pids=($child $(descendants $child)) survivors=() n=0 p
    kill -TERM $pids 2>/dev/null
    while (( n < 100 )); do
      survivors=()
      for p in $pids; do kill -0 $p 2>/dev/null && survivors+=($p); done
      (( ${#survivors} )) || break
      sleep 0.1; (( n += 1 ))
    done
    (( ${#survivors} )) && kill -KILL $survivors 2>/dev/null
    wait $child 2>/dev/null
  }
  on_signal() {
    stop_child
    release
    echo "$(date "+%F %T") interrupted pid=$$ $label (command stopped before release)" >> $log
    exit 130
  }
  trap on_signal INT TERM HUP
  waited=0
  until mkdir $lease 2>/dev/null; do
    if (( waited % 60 == 0 )); then
      holder=$(cat $lease/pid 2>/dev/null)
      if [[ -z $holder ]]; then
        note="no pid file: being created, or abandoned by a holder killed mid-create; never reclaimed automatically, remove $lease by hand if no GPU job is running"
      elif ! kill -0 $holder 2>/dev/null; then note="stale holder $holder"
      else note="holder $holder"; fi
      echo "$(date "+%F %T") atelier-lease busy ($note) pid=$$ $label" >> $log
    fi
    sleep 5; (( waited += 5 ))
  done
  created=1
  print -n $$ > $lease/pid.tmp.$$ && mv -f $lease/pid.tmp.$$ $lease/pid
  echo "$(date "+%F %T") acquired pid=$$ $label" >> $log
  export AETHER_GPU_LEASE_HOLDER=$$
  "$@" <&0 &
  child=$!
  wait $child; rc=$?
  child=0
  release
  echo "$(date "+%F %T") released pid=$$ $label rc=$rc" >> $log
  exit $rc' _ $log $label $lease "$@"
