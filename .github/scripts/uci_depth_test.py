#!/usr/bin/env python3
"""UCI smoke test for AB-JChess.

Drives the engine with "go depth N" and asserts that:
  * every depth 1..N is reported through an "info depth N ... pv ..." line
  * a "bestmove" is finally emitted

Every "info depth" line the engine emits is echoed to this script's stdout, so
the CI logs show each layer's search output.

Keeps stdin open: a bare heredoc would close stdin and the engine reads EOF as
quit->stop, killing the search before it iterates.

Usage: uci_depth_test.py <engine> <nnue-file> [--depth N]
"""

import argparse
import queue
import subprocess
import sys
import threading
import time

# Jieqi (揭棋) has a very large branching factor: each extra depth costs
# roughly 2-3x the previous one, so a full "go depth 20" takes many hours and
# would blow the GitHub Actions 6-hour job limit. Depth 10 completes quickly
# while still exercising the whole iterative-deepening output path.
DEFAULT_DEPTH = 10
# Generous wall-clock budget (depth 10 normally finishes in a few minutes).
DEADLINE_SEC = 600


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine")
    parser.add_argument("nnue")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    args = parser.parse_args()
    target_depth = args.depth

    # text=True + bufsize=1 gives proper line buffering (bufsize=1 is invalid in
    # binary mode and only prints a RuntimeWarning in Python 3.12+).
    proc = subprocess.Popen(
        [args.engine],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    cmds = [
        "uci\n",
        f"setoption name EvalFile value {args.nnue}\n",
        "isready\n",
        "position startpos\n",
        f"go depth {target_depth}\n",
    ]
    for c in cmds:
        proc.stdin.write(c)
    proc.stdin.flush()

    # Read the engine's stdout through a dedicated reader thread feeding a queue.
    # Do NOT use select()+readline() on the same stream: Python's TextIOWrapper
    # may pull several lines into its internal buffer in one read() call, and
    # select() on the pipe then reports "not ready" while a line (e.g. the
    # final "bestmove", which the engine emits only ~0.5ms after the last
    # "info depth" line) is still sitting unread in that buffer. That race made
    # the driver wait out the whole deadline even though the engine had already
    # answered. A blocking reader thread never misses buffered data.
    lines = queue.Queue()

    def reader() -> None:
        for line in proc.stdout:
            lines.put(line)

    threading.Thread(target=reader, daemon=True).start()

    depths = set()
    bestmove = None
    deadline = time.time() + DEADLINE_SEC
    while time.time() < deadline:
        try:
            line = lines.get(timeout=1.0)
        except queue.Empty:
            continue
        line = line.rstrip("\n")
        if line.startswith("info depth ") and " pv " in line:
            print(line, flush=True)
            try:
                depths.add(int(line.split()[2]))
            except (IndexError, ValueError):
                pass
        if line.startswith("bestmove"):
            bestmove = line
            break

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.stdin.close()
    proc.wait()

    print("BESTMOVE:", bestmove)
    print("DEPTHS_SEEN:", sorted(depths))

    if bestmove is None:
        print(f"FAIL: go depth {target_depth} did not produce a bestmove")
        return 1
    missing = sorted(set(range(1, target_depth + 1)) - depths)
    if missing:
        print(f"FAIL: missing depths: {missing}")
        return 1
    print(f"PASS: go depth {target_depth} iterated depths 1..{target_depth} "
          "and produced a bestmove")
    return 0


if __name__ == "__main__":
    sys.exit(main())
