"""Bot subprocess. Speaks length-prefixed pickles over the inherited stdin/stdout.

Requests:  ("init", spec, seed)  ->  ("ok", None, cpu) | ("err", message, cpu)
           ("act", obs)          ->  ("ok", actions, cpu) | ("err", message, cpu)
``cpu`` is the CPU time (seconds) the request took inside this process.
The bot's own prints go to stderr so they can't corrupt the protocol.
"""

from __future__ import annotations

import os
import pickle
import socket
import struct
import sys
import time
import traceback

from inkarena.arena.loader import load_policy_class

HEADER = struct.Struct("<I")


def read_msg(f):
    head = f.read(HEADER.size)
    if len(head) < HEADER.size:
        return None
    (n,) = HEADER.unpack(head)
    return pickle.loads(f.read(n))


def write_msg(f, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    f.write(HEADER.pack(len(data)) + data)
    f.flush()


class _NoNetwork(socket.socket):
    def __init__(self, *args, **kwargs):
        raise PermissionError("network access is disabled for bots")


def _block_network() -> None:
    socket.socket = _NoNetwork
    socket.create_connection = lambda *a, **k: _NoNetwork()
    socket.socketpair = lambda *a, **k: _NoNetwork()


def main() -> None:
    proto_in = os.fdopen(os.dup(0), "rb")
    proto_out = os.fdopen(os.dup(1), "wb")
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    _block_network()

    policy = None
    while True:
        msg = read_msg(proto_in)
        if msg is None:
            return
        start = time.process_time()
        try:
            if msg[0] == "init":
                _, spec, seed = msg
                policy = load_policy_class(spec)(seed)
                reply = ("ok", None)
            elif msg[0] == "act":
                reply = ("ok", policy.act(msg[1]))
            else:
                reply = ("err", f"unknown request {msg[0]!r}")
        except Exception as e:
            reply = ("err", "".join(traceback.format_exception_only(type(e), e)).strip())
        cpu = time.process_time() - start
        try:
            write_msg(proto_out, (*reply, cpu))
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            write_msg(proto_out, ("err", f"unpicklable output: {e}", cpu))


if __name__ == "__main__":
    main()
