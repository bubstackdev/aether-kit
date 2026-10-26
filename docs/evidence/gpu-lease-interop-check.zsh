#!/bin/zsh
# Kit SharedGPULease vs Atelier's SharedGPULock (verbatim source compiled into atelier-lease-sim) and the development
# wrapper's protocol, on a PRIVATE lease directory. Never touches /tmp/atelier-gpu.lock.
# Usage: interop-check.zsh ATELIER_SIM KIT_HELPER AETHER_PROBE WORK_DIR
set -u
sim=$1 kit=$2 probe=$3 work=$4
lease=$work/gpu.lock
[[ $lease == /tmp/atelier-gpu.lock* || $lease == /private/tmp/atelier-gpu.lock* ]] && { echo "refusing the shared path"; exit 64 }
# Every aether-probe below, including one started by a nested shell, uses the private lease.
export AETHER_GPU_LEASE_PATH=$lease
unset AETHER_GPU_LEASE_HOLDER
mkdir -p $work
failures=0
log() { print -r -- "$(date '+%H:%M:%S') $*" }
check() { # check LABEL EXPECTED ACTUAL
  if [[ "$2" == "$3" ]]; then log "PASS $1: $3"; else log "FAIL $1: expected [$2] got [$3]"; (( failures += 1 )); fi
}
holder() { cat $lease/pid 2>/dev/null || print -n none }
start_holder() { # start_holder CMD... ; sets hpid once "ready" is printed
  local fifo=$work/ready.$RANDOM
  mkfifo $fifo
  "$@" > $fifo 2>/dev/null &
  hpid=$!
  read -r line < $fifo
  rm -f $fifo
  [[ $line == ready ]] || log "holder did not report ready ($line)"
}
wait_gone() { local n=0; while [[ -e $lease ]] && (( n < 500 )); do sleep 0.02; (( n += 1 )); done }

log "atelier-lease-sim $(shasum -a 256 $sim | cut -c1-16) kit-helper $(shasum -a 256 $kit | cut -c1-16) aether-probe $(shasum -a 256 $probe | cut -c1-16)"
log "lease directory $lease"

log "--- 1. Atelier holds: Kit fail-fast, timed wait and aether-probe all see it held"
start_holder $sim hold $lease; a=$hpid
check "lease names the Atelier holder" "$a" "$(holder)"
out=$($kit try $lease); check "Kit fail-fast" "2 held $a" "$? $out"
out=$($kit wait $lease 0.5); check "Kit timed wait" "3 timedOut $a" "$? $out"
err=$(AETHER_GPU_LEASE_PATH=$lease AETHER_GPU_LEASE_TIMEOUT=0.5 $probe inspect $work/missing.aimodel 2>&1 >/dev/null); st=$?
check "aether-probe inspect times out before touching the model" "1 1" "$st $(print -r -- $err | grep -c "Timed out waiting for the shared GPU lease held by process $a")"

log "--- 2. Kit waits; Atelier's default SIGTERM handling releases; Kit acquires"
$kit wait $lease 10 > $work/wait.out & w=$!
sleep 0.3
check "Kit waiter still waiting" "running" "$(kill -0 $w 2>/dev/null && print running || print exited)"
kill -TERM $a; wait $a 2>/dev/null
wait $w; check "Kit waiter acquired after Atelier released" "0 acquired" "$? $(cat $work/wait.out)"
check "lease free" "none" "$(holder)"

log "--- 3. Kit holds: Atelier is reserved; Kit's SIGTERM release lets Atelier in"
start_holder $kit default $lease; k=$hpid
check "lease names the Kit holder" "$k" "$(holder)"
out=$($sim try $lease); check "Atelier while Kit holds" "2 reserved" "$? $out"
kill -TERM $k; wait $k 2>/dev/null; check "Kit holder terminated by SIGTERM (143)" "143" "$?"
check "Kit released on SIGTERM" "none" "$(holder)"
out=$($sim try $lease); check "Atelier after Kit released" "0 acquired" "$? $out"

log "--- 4. Stale leases: each side reclaims the other's SIGKILLed holder (shared .reclaim flock)"
start_holder $sim hold $lease; a=$hpid
kill -KILL $a; wait $a 2>/dev/null
check "stale Atelier lease left behind" "$a" "$(holder)"
out=$($kit try $lease); check "Kit reclaims the dead Atelier lease" "0 owned" "$? $out"
start_holder $kit default $lease; k=$hpid
kill -KILL $k; wait $k 2>/dev/null
check "stale Kit lease left behind" "$k" "$(holder)"
out=$($sim try $lease); check "Atelier reclaims the dead Kit lease" "0 acquired" "$? $out"
check "reclaim file is private" "600" "$(stat -f %Lp $work/gpu.lock.reclaim)"

log "--- 5. aether-probe waits for Atelier, then runs and releases on its error path"
start_holder $sim hold $lease; a=$hpid
AETHER_GPU_LEASE_PATH=$lease $probe inspect $work/missing.aimodel > /dev/null 2> $work/probe.err & p=$!
sleep 0.5
check "aether-probe reports waiting" "1" "$(grep -c "waiting for the GPU lease $lease held by process $a" $work/probe.err)"
kill -TERM $a; wait $a 2>/dev/null
wait $p; st=$?
check "aether-probe ran after the lease freed (model missing: exit 1)" "1 1" "$st $(grep -c "Invalid model" $work/probe.err)"
check "aether-probe released the lease" "none" "$(holder)"

log "--- 6. aether-probe under a wrapper-style parent that holds the lease: inherited only on explicit hand-off"
/bin/zsh -c '
  lease=$1 probe=$2 model=$3
  until mkdir $lease 2>/dev/null; do sleep 0.05; done
  print -n $$ > $lease/pid.tmp.$$ && mv -f $lease/pid.tmp.$$ $lease/pid
  release() { [[ "$(cat $lease/pid 2>/dev/null)" == "$$" ]] && rm -rf $lease }
  trap "release; exit 130" INT TERM HUP
  AETHER_GPU_LEASE_PATH=$lease AETHER_GPU_LEASE_TIMEOUT=0.5 $probe inspect $model 2> $lease.plain.err >/dev/null
  print "$(grep -c "Invalid model" $lease.plain.err) $(grep -c "held by ancestor process $$, which did not hand it" $lease.plain.err)"
  AETHER_GPU_LEASE_HOLDER=$$ AETHER_GPU_LEASE_PATH=$lease AETHER_GPU_LEASE_TIMEOUT=2 $probe inspect $model 2> $lease.child.err >/dev/null
  print "$(grep -c "Invalid model" $lease.child.err) $(grep -c waiting $lease.child.err)"
  print "holder-after-child $(cat $lease/pid)"
  print "wrapper $$"
  release
' _ $lease $probe $work/missing.aimodel > $work/wrapper.out
check "without a hand-off, aether-probe waits for its ancestor (and says why)" "0 1" "$(sed -n 1p $work/wrapper.out)"
check "with AETHER_GPU_LEASE_HOLDER, aether-probe inherits (ran, never waited)" "1 0" "$(sed -n 2p $work/wrapper.out)"
check "lease still the wrapper's after the child" "$(sed -n 4p $work/wrapper.out | cut -d' ' -f2)" "$(sed -n 3p $work/wrapper.out | cut -d' ' -f2)"
check "wrapper released" "none" "$(holder)"

log "--- 7. Kit waits on a wrapper-style holder"
/bin/zsh -c '
  lease=$1
  until mkdir $lease 2>/dev/null; do sleep 0.05; done
  print -n $$ > $lease/pid.tmp.$$ && mv -f $lease/pid.tmp.$$ $lease/pid
  release() { [[ "$(cat $lease/pid 2>/dev/null)" == "$$" ]] && rm -rf $lease }
  trap "release; exit 130" INT TERM HUP
  print ready
  while true; do sleep 0.05; done
' _ $lease > $work/wr.fifo.out &
z=$!
n=0; while [[ "$(holder)" != "$z" ]] && (( n < 200 )); do sleep 0.02; (( n += 1 )); done
out=$($kit try $lease); check "Kit fail-fast vs wrapper holder" "2 held $z" "$? $out"
$kit wait $lease 10 > $work/wait2.out & w=$!
sleep 0.3
kill -TERM $z; wait $z 2>/dev/null; check "wrapper trap exit" "130" "$?"
wait $w; check "Kit acquired after the wrapper released" "0 acquired" "$? $(cat $work/wait2.out)"

log "final directory: $(ls -A $work | grep -v -e '\.out$' -e '\.err$' | paste -sd ' ' -)"
log "failures=$failures"
exit $failures
