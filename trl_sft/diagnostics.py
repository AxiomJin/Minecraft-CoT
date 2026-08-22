"""Hang diagnostics: stall watchdog + heartbeat callback for SFTTrainer.

A multi-node hang surfaces only as NCCL's `Watchdog caught collective operation
timeout ... ran for 600000ms`, which says *that* some ranks stopped making
progress but never *where* in Python they are blocked -- so the root cause
(data loading? tokenization? a forward op? an actual collective mismatch?)
stays guesswork. The watchdog below closes that gap: it notices when steps stop
advancing and dumps every thread's traceback (and its dataloader workers')
straight to stderr, i.e. into the job log, well before NCCL aborts the process.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import signal
import sys
import threading
import time
from typing import Dict, List, Union

from transformers import TrainerCallback

logger = logging.getLogger(__name__)

_HEARTBEAT: Dict[str, Union[float, int, str]] = {"t": 0.0, "step": -1, "phase": "startup"}


def _rank_tag() -> str:
    return f"rank={os.environ.get('RANK', '?')} local_rank={os.environ.get('LOCAL_RANK', '?')}"


def _child_pids(pid: int) -> List[int]:
    """Direct child PIDs (Linux), used to reach forked dataloader workers."""
    try:
        with open(f"/proc/{pid}/task/{pid}/children", "r") as fh:
            return [int(p) for p in fh.read().split()]
    except OSError:
        return []


def _dump_all_stacks(reason: str) -> None:
    """Dump this process' thread stacks, then ask dataloader workers to dump theirs."""
    tag = _rank_tag()
    print(f"\n===== [stall-watchdog] {tag} {reason} =====", file=sys.stderr, flush=True)
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    for child in _child_pids(os.getpid()):
        print(f"===== [stall-watchdog] {tag} dumping child pid={child} =====", file=sys.stderr, flush=True)
        try:
            # Handler registered in _install_stall_watchdog() and inherited on fork.
            os.kill(child, signal.SIGUSR1)
        except OSError as exc:
            print(f"[stall-watchdog] cannot signal pid={child}: {exc}", file=sys.stderr, flush=True)
    time.sleep(1.0)  # let children flush their tracebacks before we return
    sys.stderr.flush()


def install_stall_watchdog(threshold_sec: float, max_dumps: int = 3) -> None:
    """Start a daemon thread that dumps Python stacks when training stalls.

    `threshold_sec` should sit comfortably below the NCCL collective timeout
    (default 600s) so the stacks are captured while the process is still alive.
    """
    if threshold_sec <= 0:
        return

    try:
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
    except (AttributeError, ValueError, OSError):  # platform without SIGUSR1 support
        pass

    _HEARTBEAT["t"] = time.time()

    def _loop() -> None:
        dumps, dumped_for_step = 0, None
        while True:
            time.sleep(min(30.0, max(5.0, threshold_sec / 4)))
            age = time.time() - float(_HEARTBEAT["t"])
            if age < threshold_sec:
                continue
            if _HEARTBEAT["step"] != dumped_for_step:  # a fresh stall -> dump again
                dumps, dumped_for_step = 0, _HEARTBEAT["step"]
            if dumps >= max_dumps:
                continue
            dumps += 1
            _dump_all_stacks(
                f"no progress for {age:.0f}s (step={_HEARTBEAT['step']} "
                f"phase={_HEARTBEAT['phase']}, dump {dumps}/{max_dumps})"
            )

    threading.Thread(target=_loop, name="stall-watchdog", daemon=True).start()
    logger.info(f"[stall-watchdog] armed: dumping Python stacks after {threshold_sec:.0f}s without step progress")


class HeartbeatCallback(TrainerCallback):
    """Feeds the stall watchdog, so a hang can be pinned to a specific step/phase."""

    def _beat(self, state, phase: str) -> None:
        _HEARTBEAT.update(t=time.time(), step=state.global_step, phase=phase)

    def on_train_begin(self, args, state, control, **kwargs):
        self._beat(state, "train_begin")
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self._beat(state, "step_begin")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self._beat(state, "step_end")
        return control

    def on_save(self, args, state, control, **kwargs):
        self._beat(state, "save")
        return control
