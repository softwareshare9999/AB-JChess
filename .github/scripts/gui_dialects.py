#!/usr/bin/env python3
"""Replay the FEN dialects and move histories that Jieqi GUIs actually send.

This is the regression test for the PengfeiChess report: given a position where Black is to
move, the engine must answer with a Black move. AB-JChess used to accept only five- or
six-field FENs, refuse PengfeiChess's seven-field form (`board side pools captured - -
no-capture number`) with `info string bad fen`, keep the previous position, and then answer
`go` for that stale position -- i.e. it moved for Red when Black was to move.

Usage: gui_dialects.py <engine> <net> [--depth N]

Exit status 0 when every dialect is accepted and produces a move for the side to move.
"""
from __future__ import annotations

import argparse
import queue
import re
import subprocess
import sys
import threading

DEPTH = 1
TIMEOUT_SEC = 60
START = "xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX"
POOLS = "A2B2N2R2C2P5a2b2n2r2c2p5"

# Black to move with 14 face-down pieces on their start points; the pool below is deliberately one identity short
# of them, which is what a user trimming the piece counts in the GUI produces.
SHORT_BOARD = "xxxxkxxxx/9/1x5x1/x1x1R1x1x/9/9/9/3N5/9/3K5"
SHORT_POOL = "r2a2b2n2c2p3"

# (name, position command, side to move) -- the engine must produce a move for that side.
CASES = [
    ("pikafish 5-field",      f"position fen {START} w {POOLS} 0 1", "w"),
    ("jieqibox 6-field",      f"position fen {START} w {POOLS} - 0 1", "w"),
    ("pengfei 7-field",       f"position fen {START} w {POOLS} - - 0 1", "w"),
    ("pengfei 8-field",       f"position fen {START} w {POOLS} R1 - - 0 1", "w"),
    ("pengfei 7-field black", f"position fen {START} b {POOLS} - - 0 1", "b"),
    ("ace uci_pf",            f"position fen {START} ${POOLS} w - - 0 1", "w"),
    ("three fields",          f"position fen {START} w {POOLS}", "w"),
    ("red written as r",      f"position fen {START} r {POOLS} 0 1", "w"),
    # PengfeiChess' analysis mode saves and replays positions as a bare `board side`.
    ("analysis 2-field",      "position fen r1baka2r/5n3/n1c1b2c1/p1p1p1p1p/9/2P3P2/P3P3P/NC4NC1/9/R1BAKAB1R w", "w"),
    # A pool short of the face-down pieces on the board. A GUI lets the user trim the remaining-piece counts, so
    # this arrives in normal play, and it used to be refused ("pool smaller than face-down count"), which made the
    # engine answer `bestmove (none)`; the GUI then resynced and discarded the edit. The engine now covers the
    # shortfall, reports it as an info string, and still moves.
    ("pool short by one",     f"position fen {SHORT_BOARD} b {SHORT_POOL} - - 0 1", "b"),
    ("pool empty with dark",  f"position fen {SHORT_BOARD} b - - - 0 1", "b"),
    ("history with reveal",   "position startpos moves a3a4P", "b"),
    ("history with x marker", "position startpos moves a3a4Px", "b"),
    ("startpos",              "position startpos", "w"),
]

RED = re.compile(r"^[a-i][0-9][a-i][0-9]$")


def ask(engine: str, net: str, position: str) -> tuple[list[str], str | None]:
    proc = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: [lines.put(x) for x in proc.stdout], daemon=True).start()
    for cmd in ("uci", f"setoption name EvalFile value {net}", "isready", position,
                f"go depth {DEPTH}", "quit"):
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()
    out, best = [], None
    try:
        while True:
            line = lines.get(timeout=TIMEOUT_SEC)
            out.append(line.rstrip("\n"))
            if line.startswith("bestmove"):
                best = line.split()[1]
                break
    except queue.Empty:
        pass
    proc.stdin.close()
    proc.wait(timeout=TIMEOUT_SEC)
    return out, best


def main() -> int:
    global DEPTH
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("net")
    ap.add_argument("--depth", type=int, default=DEPTH)
    a = ap.parse_args()
    DEPTH = a.depth

    failures = 0
    for name, position, side in CASES:
        out, best = ask(a.engine, a.net, position)
        problems = []
        bad = [x for x in out if "bad fen" in x or "bad move" in x or "refusing to search" in x]
        if bad:
            problems.append(bad[0].replace("info string ", ""))
        if best in (None, "(none)", "0000"):
            problems.append(f"no usable bestmove (got {best})")
        elif not RED.match(best):
            problems.append(f"unexpected bestmove {best}")
        # A move by the wrong side: for these positions the side to move differs, so compare the
        # reply against the other side's known reply from the same start position.
        if not problems and side == "b" and best == "a3a4":
            problems.append("answered a RED move while BLACK is to move")
        if not problems and side == "w" and best in ("a6a5", "i6i5"):
            problems.append("answered a BLACK move while RED is to move")
        status = "ok  " if not problems else "FAIL"
        print(f"{status} {name:22} side={side} bestmove={best}  {'; '.join(problems)}")
        failures += bool(problems)

    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
