"""Workload supervision shared by the H3 qualification controllers (Scripts/qualify-h3-*.py).

A controller runs each workload in its own session and process group, so a terminal's Ctrl-C or a supervisor's
SIGTERM or SIGHUP reaches only the controller. `install()` turns the first of those signals into KeyboardInterrupt,
which the controller's cleanup handles with `stop()`: SIGTERM to the whole group, then SIGKILL at the grace deadline,
until no member is left.

Later signals are queued, never raised: once the controller is interrupted, and while it starts or stops a workload,
nothing can cut its cleanup short. They don't shorten the grace period either. A GPU workload runs under
~/.cache/aether-kit/with-gpu-lock.sh, whose shell stops its command (TERM, then KILL after 10 s) and only then releases
the shared lease; killing that shell earlier would leave the lease naming a dead holder, which the wrapper never
reclaims. Controllers that run under the wrapper give it a 15 s grace period.
"""
import os
import signal
import subprocess
import time

SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
#: After the SIGKILL, how long `stop()` waits for the group to empty before reporting it unjoined.
KILL_WAIT_SECONDS = 10

_state = dict(exiting=False, starting=False, stopping=False)
pending = []           # signals queued while they could not interrupt


def _on_signal(signum, frame):
    if _state["exiting"] or _state["starting"] or _state["stopping"]:
        pending.append(signum)
        return
    _state["exiting"] = True
    raise KeyboardInterrupt(f"signal {signum}")


def install():
    """Routes Ctrl-C, SIGTERM and SIGHUP through the controller's interruption (see the module notes)."""
    for sig in SIGNALS:
        signal.signal(sig, _on_signal)


def interrupted():
    """Whether the controller was interrupted, or a signal was queued while it couldn't interrupt."""
    return _state["exiting"] or bool(pending)


def _interrupt():
    _state["exiting"] = True
    raise KeyboardInterrupt(f"signal {pending[0]}" if pending else "interrupted")


def start(command, **kwargs):
    """Starts `command` as its own session and process group. An interrupted controller starts nothing, so a later
    stage never takes the GPU lease only to drop it. A signal that arrives while it starts is queued; the workload is
    then stopped here before the interruption is raised, so it is never left running unowned."""
    if interrupted():
        _interrupt()
    _state["starting"] = True
    try:
        proc = subprocess.Popen(command, start_new_session=True, **kwargs)
    finally:
        _state["starting"] = False
    if pending:
        stop(proc, grace=15)
        _interrupt()
    return proc


def _group_alive(proc):
    proc.poll()   # a reaped leader no longer holds the group open
    try:
        os.killpg(proc.pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal_group(proc, sig):
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def _wait_for_group(proc, seconds):
    deadline = time.monotonic() + seconds
    while _group_alive(proc) and time.monotonic() < deadline:
        time.sleep(0.05)


def stop(proc, grace):
    """Stops the whole process group `proc` leads, whether or not its leader has exited: SIGTERM, then SIGKILL once
    `grace` seconds pass with any member left, then waits for the group to empty. A leader that exits at once on
    SIGTERM (lockf, say) doesn't end the wait while its children run on. Signals meanwhile are queued (the controller
    is then `interrupted()`). Returns whether the group is gone; never raises for a signal, so it is safe in cleanup."""
    was_stopping, _state["stopping"] = _state["stopping"], True
    try:
        if _group_alive(proc):
            _signal_group(proc, signal.SIGTERM)
            _wait_for_group(proc, grace)
            if _group_alive(proc):
                _signal_group(proc, signal.SIGKILL)
                _wait_for_group(proc, KILL_WAIT_SECONDS)
        if proc.poll() is None:
            proc.wait()
        return not _group_alive(proc)
    finally:
        _state["stopping"] = was_stopping
        if pending:
            _state["exiting"] = True
