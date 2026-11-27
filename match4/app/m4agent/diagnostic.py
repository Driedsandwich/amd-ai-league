"""Opt-in grader diagnostics shared across question processes."""
import fcntl
import os
import re
import socket

from .common import result

COUNTER_PATH = "/tmp/x052_m4_diag.count"
NAME = re.compile(r"[A-Za-z0-9_.-]+")


def claim_invocation():
    fd = os.open(COUNTER_PATH, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+", encoding="ascii") as counter:
        fcntl.flock(counter, fcntl.LOCK_EX)
        count = counter.read(1)
        if count == "2":
            return False
        counter.seek(0)
        counter.write("2" if count == "1" else "1")
        counter.truncate()
        return True


def network_available():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            connection.settimeout(1)
            connection.connect(("1.1.1.1", 443))
        return 1
    except OSError:
        return 0


def diagnostic_result(args, job, net):
    # Only identifier-like names can be echoed, even for unrecognized keys or flags.
    flags = sorted(flag for flag, _ in args.options if flag.startswith("--") and NAME.fullmatch(flag[2:]))
    keys = sorted(key for key in job["input_keys"] if NAME.fullmatch(key))
    scheme = re.match(r"([A-Za-z][A-Za-z0-9+.-]*):", job["url"] or "")
    return result(f"DIAG flags={','.join(flags)} pos={args.positional_count} "
                  f"inkeys={','.join(keys)} url={scheme[1].lower() if scheme else 'none'} "
                  f"corpus={int(bool(job['corpus']))} net={net}", ["diag"])
