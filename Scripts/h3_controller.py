"""Workload supervision shared by the H3 qualification controllers (Scripts/qualify-h3-*.py).

A controller runs each workload in its own session and process group, so a terminal's Ctrl-C or a supervisor's
SIGTERM or SIGHUP reaches only the controller. `install()` turns the first of those signals into KeyboardInterrupt,
which the controller's cleanup handles by stopping and joining the workload. No later signal can cut that cleanup
short: once the controller is interrupted, or while it is starting or stopping a workload, further signals are queued,
and while a workload is being stopped they escalate it to SIGKILL. The workload is never abandoned.
"""
import os
import signal
import subprocess

SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

_workloads = {}        # pid -> Popen: workload groups this controller leads and has not joined
_state = dict(exiting=False, starting=False, stopping=False)
pending = []           # signals queued while they could not interrupt


def _kill_workloads(sig):
    for pid in list(_workloads):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            pass


def _on_signal(signum, frame):
    if _state["exiting"] or _state["stopping"]:
        # Already interrupted, or stopping a workload: escalate, never abandon it.
        pending.append(signum)
        _kill_workloads(signal.SIGKILL)
        return
    if _state["starting"]:
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


def start(command, **kwargs):
    """Starts `command` as its own session and process group and registers it. A signal that arrives while it starts
    is queued; the workload is then stopped and joined here before the interruption is raised, so it is never left
    running unowned."""
    _state["starting"] = True
    try:
        proc = subprocess.Popen(command, start_new_session=True, **kwargs)
        _workloads[proc.pid] = proc
    finally:
        _state["starting"] = False
    if pending:
        stop(proc, grace=5, force=True)
        _state["exiting"] = True
        raise KeyboardInterrupt(f"signal {pending[0]}")
    return proc


def stop(proc, grace, force=False):
    """Stops and joins the process group `proc` leads: SIGTERM, then SIGKILL after `grace` seconds. With `force`, the
    group is signalled even when its leader has already exited. Signals that arrive meanwhile escalate to SIGKILL and
    are queued: the controller is then `interrupted()`, and any later signal escalates too. Never raises for a signal,
    so it is safe in cleanup code."""
    was_stopping, _state["stopping"] = _state["stopping"], True
    try:
        if force or proc.poll() is None:
            for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, None)):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    pass
        proc.wait()
        _workloads.pop(proc.pid, None)
    finally:
        _state["stopping"] = was_stopping
        if pending:
            _state["exiting"] = True
