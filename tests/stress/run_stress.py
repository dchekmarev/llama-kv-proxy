# tests/stress/run_stress.py

"""Stress harness for llama-kv-proxy (no real LLM).

Spawns the fake backend (tests/fake_backend.py) and the real proxy
(python -m app) as subprocesses, then hammers the proxy with N threads
of chat requests (stream + non-stream, small + big) and verifies the proxy
stays responsive and never leaks slots:

  steady        N workers x R requests, backend answers after --delay (20 s)
  backend_error backend forced to answer 500; proxy must fail fast, no hang
  backend_down  backend SIGKILLed mid-flight; clients must get errors, not
                hangs; proxy must recover after backend restart
  disconnect    clients drop the socket mid-response; slots must be freed
  half_request  clients send a partial HTTP body and vanish
  dangling      many half-open sockets left idle while normal traffic runs
  chaos         mixed normal / disconnect / error traffic

After every scenario a slot-drain check runs: exactly n_slots concurrent
requests must all finish within delay + margin. A leaked slot makes one of
them queue (or 503) and the check fails. The fake backend additionally
counts same-slot concurrency violations (a proxy bug by definition).

Usage:
  python tests/stress/run_stress.py                 # full suite, 20 s delay
  python tests/stress/run_stress.py --fast          # 2 s delay, quick matrix
  python tests/stress/run_stress.py --scenario steady --workers 8
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N_SLOTS = 2

# Proxy log line: "<ts> LEVEL [request_id] logger: <message>". We correlate a
# slot acquire (after_acquire) with its release (slot_release) by request_id,
# which the proxy stamps on every log record (and which background tasks
# inherit). This is the authoritative slot-exclusivity signal: the slot lock
# guarantees the proxy never holds one slot twice, so any same-slot overlap
# reconstructed here is a real bug (the backend's own counter can false-positive
# on client disconnects, where it learns of the abort after the proxy has
# already released the slot).
_SLOT_LOG_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \w+ \[([0-9a-fA-F]+)\] \S+: "
    r"(after_acquire|slot_release) g=(.*?) key=(\S+)"
)


def _parse_ts(s: str) -> float:
    # All timestamps share the host's local tz; only relative ordering matters.
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f").timestamp()  # noqa: DTZ007


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@dataclass
class Outcome:
    scenario: str
    label: str
    ok: int = 0
    failed: int = 0
    errors: list = field(default_factory=list)
    latencies: list = field(default_factory=list)

    def add(self, label: str, ok: bool, latency: float, err: str = ""):
        self.latencies.append(latency)
        if ok:
            self.ok += 1
        else:
            self.failed += 1
            if len(self.errors) < 10:
                self.errors.append(f"{label}: {err}")

    @property
    def total(self):
        return self.ok + self.failed

    def summary(self) -> str:
        if not self.latencies:
            return "no requests"
        ls = sorted(self.latencies)
        p50 = ls[len(ls) // 2]
        p99 = ls[min(len(ls) - 1, int(len(ls) * 0.99))]
        return (
            f"total={self.total} ok={self.ok} failed={self.failed} "
            f"p50={p50:.1f}s p99={p99:.1f}s max={ls[-1]:.1f}s"
        )


class Harness:
    def __init__(self, args):
        self.args = args
        self.delay = args.delay
        self.be_port = free_port()
        self.px_port = free_port()
        self.be_url = f"http://127.0.0.1:{self.be_port}"
        self.px_url = f"http://127.0.0.1:{self.px_port}"
        self.tmp = tempfile.mkdtemp(prefix="stress_")
        os.makedirs(os.path.join(self.tmp, "meta"), exist_ok=True)
        self.be_proc: subprocess.Popen | None = None
        self.px_proc: subprocess.Popen | None = None
        self.px_log = open(os.path.join(self.tmp, "proxy.log"), "w")  # noqa: SIM115
        self.be_log = open(os.path.join(self.tmp, "backend.log"), "w")  # noqa: SIM115
        self.violations_base = 0
        self.px_log_offset = 0

    # ---------- process management ----------

    def start_backend(self):
        self.be_proc = subprocess.Popen(
            [
                sys.executable,
                os.path.join(ROOT, "tests", "fake_backend.py"),
                "--port", str(self.be_port),
                "--n-slots", str(N_SLOTS),
            ],
            stdout=self.be_log,
            stderr=subprocess.STDOUT,
        )
        self._wait_up(self.be_url + "/v1/models")

    def start_proxy(self):
        env = dict(os.environ)
        env.update(
            {
                "BACKENDS": json.dumps(
                    [{"url": self.be_url, "n_slots": N_SLOTS}]
                ),
                "PORT": str(self.px_port),
                "META_DIR": os.path.join(self.tmp, "meta"),
                "BIN_CACHE_DIR": "",
                "ACQUIRE_TIMEOUT": "90",
                "REQUEST_TIMEOUT": "120",
                "SLOT_POLL_INTERVAL_S": "2",
                "MODEL_ID_TTL": "60",
                "EVICT_INTERVAL_S": "3600",
                "BIN_RECONCILE_INTERVAL_S": "0",
                "LOG_LEVEL": "INFO",
            }
        )
        self.px_proc = subprocess.Popen(
            [sys.executable, "-m", "app"],
            env=env,
            cwd=ROOT,
            stdout=self.px_log,
            stderr=subprocess.STDOUT,
        )
        self._wait_up(self.px_url + "/proxy/health", timeout=30)

    def stop_all(self):
        for p in (self.px_proc, self.be_proc):
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        self.px_log.close()
        self.be_log.close()

    @staticmethod
    def _wait_up(url: str, timeout: float = 20):
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            try:
                r = httpx.get(url, timeout=3)
                if r.status_code < 500:
                    return
                last = f"HTTP {r.status_code}"
            except Exception as e:  # noqa: BLE001
                last = str(e)
            time.sleep(0.3)
        raise RuntimeError(f"server did not come up at {url}: {last}")

    # ---------- fake backend control ----------

    def control(self, **kw):
        r = httpx.post(self.be_url + "/_control", json=kw, timeout=5)
        r.raise_for_status()

    def stats(self) -> dict:
        return httpx.get(self.be_url + "/_stats", timeout=5).json()

    def check_violations(self, scenario: str, out: Outcome):
        # Informational only: the backend's counter can false-positive on
        # client disconnects (it learns of the abort after the proxy released
        # the slot). The authoritative check is check_slot_exclusivity_proxy.
        st = self.stats()
        new = st["slot_violations"] - self.violations_base
        if new:
            print(f"   [info] backend saw {new} same-slot overlap(s) "
                  f"(abort-lag artifact; see proxy-side check)")
        self.violations_base = st["slot_violations"]

    def check_slot_exclusivity_proxy(self, scenario: str, out: Outcome):
        """Reconstruct per-slot hold intervals from the proxy log and assert no
        two requests held the same slot at the same time."""
        # Let in-flight background release logs (stream reader / bg save) land.
        time.sleep(0.5)
        path = os.path.join(self.tmp, "proxy.log")
        with open(path) as f:
            f.seek(self.px_log_offset)
            new_lines = f.readlines()
            self.px_log_offset = f.tell()

        acq: dict[str, tuple[float, str]] = {}
        rel: dict[str, tuple[float, str]] = {}
        for line in new_lines:
            m = _SLOT_LOG_RE.match(line)
            if not m:
                continue
            ts = _parse_ts(m.group(1))
            rid, kind, g = m.group(2), m.group(3), m.group(4)
            (acq if kind == "after_acquire" else rel)[rid] = (ts, g)

        by_g: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        for rid, (ats, g) in acq.items():
            if rid in rel:
                rts, _ = rel[rid]
                by_g[g].append((ats, rts, rid))

        overlaps = []
        for g, ivs in by_g.items():
            ivs.sort()
            for prev, cur in pairwise(ivs):
                if cur[0] < prev[1]:
                    overlaps.append(
                        f"slot={g} req={prev[2]}[{prev[0]:.3f},{prev[1]:.3f}) "
                        f"overlaps req={cur[2]}[{cur[0]:.3f},{cur[1]:.3f})"
                    )
        if overlaps:
            out.add("slot_exclusivity", False, 0.0,
                    f"{len(overlaps)} same-slot overlap(s): " + overlaps[0])

    def drain_check(self, scenario: str, out: Outcome, delay: float | None = None,
                    margin: float = 15.0):
        """n_slots concurrent requests must all finish within delay+margin."""
        d = self.delay if delay is None else delay
        budget = d + margin

        def one(_):
            t0 = time.time()
            r = httpx.post(
                self.px_url + "/v1/chat/completions",
                json=make_payload(big=False, stream=False, tag=f"drain{time.time()}"),
                timeout=httpx.Timeout(budget + 30, connect=5),
            )
            return time.time() - t0, r.status_code, r.text[:200]

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=N_SLOTS) as ex:
            results = list(ex.map(one, range(N_SLOTS)))
        wall = time.time() - t0
        for i, (lat, status, text) in enumerate(results):
            ok = status == 200 and lat <= budget
            out.add(f"drain[{i}]", ok, lat, f"status={status} lat={lat:.1f}s {text}")
        if wall > budget:
            out.add("drain_wall", False, wall, f"wall={wall:.1f}s > budget={budget:.1f}s")

    # ---------- request builders ----------

    def chat(self, big: bool, stream: bool, tag: str, timeout: float) -> tuple[int, str]:
        payload = make_payload(big=big, stream=stream, tag=tag)
        if stream:
            with httpx.Client(timeout=httpx.Timeout(timeout, connect=5)) as c, \
                    c.stream("POST", self.px_url + "/v1/chat/completions", json=payload) as r:
                body = b"".join(r.iter_bytes())
                return r.status_code, body.decode("utf-8", "ignore")
        r = httpx.post(
            self.px_url + "/v1/chat/completions",
            json=payload,
            timeout=httpx.Timeout(timeout, connect=5),
        )
        return r.status_code, r.text

    def chat_stream_partial(self, big: bool, tag: str, read_chunks: int,
                            timeout: float) -> tuple[int, str]:
        """Start a streaming request, read a few chunks, then drop the socket."""
        payload = make_payload(big=big, stream=True, tag=tag)
        c = httpx.Client(timeout=httpx.Timeout(timeout, connect=5))
        try:
            with c.stream("POST", self.px_url + "/v1/chat/completions", json=payload) as r:
                got = 0
                for chunk in r.iter_bytes():
                    got += 1
                    if got >= read_chunks:
                        break
            return r.status_code, f"read {got} chunks then dropped"
        finally:
            c.close()

    # ---------- scenarios ----------

    def scenario_steady(self, out: Outcome):
        a = self.args
        per = a.requests_per_worker
        self.control(mode="ok", delay=self.delay, chunk_interval=0.5, n_chunks=20)

        def worker(w: int):
            for j in range(per):
                big = (w + j) % 2 == 0
                stream = (w + j) % 3 != 0
                t0 = time.time()
                try:
                    status, body = self.chat(big, stream, f"s{w}-{j}", a.req_timeout)
                    # 503 = slot exhaustion backpressure (fast fail, not a hang)
                    ok = status in (200, 503)
                    if stream and status == 200 and "[DONE]" not in body:
                        ok = False
                        body = "missing [DONE]"
                    out.add(f"w{w}.{j}", ok, time.time() - t0,
                            f"status={status} body={body[:120]}")
                except Exception as e:  # noqa: BLE001
                    out.add(f"w{w}.{j}", False, time.time() - t0, repr(e))

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(worker, range(a.workers)))
        self.check_violations("steady", out)
        self.drain_check("steady", out)
        self.check_slot_exclusivity_proxy("steady", out)

    def scenario_backend_error(self, out: Outcome):
        a = self.args
        self.control(mode="error", status_code=500, delay=0.2)
        try:

            def one(i: int):
                stream = i % 2 == 0
                t0 = time.time()
                try:
                    status, body = self.chat(False, stream, f"e{i}", 30)
                    # non-stream -> proxy maps to 502; stream -> 500 passthrough
                    ok = status in (500, 502)
                    out.add(f"err{i}", ok, time.time() - t0,
                            f"status={status} body={body[:120]}")
                except Exception as e:  # noqa: BLE001
                    out.add(f"err{i}", False, time.time() - t0, repr(e))

            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                list(ex.map(one, range(a.workers * 2)))
        finally:
            self.control(mode="ok", delay=0.2)
        self.check_violations("backend_error", out)
        self.drain_check("backend_error", out, delay=0.2)
        self.check_slot_exclusivity_proxy("backend_error", out)
        self.control(delay=self.delay)

    def scenario_backend_down(self, out: Outcome):
        a = self.args
        self.control(mode="ok", delay=5.0)

        def one(i: int):
            t0 = time.time()
            try:
                status, body = self.chat(False, i % 2 == 0, f"d{i}", 60)
                # after the kill: 500 JSON (non-stream) or SSE error event (stream)
                ok = status == 500 or "stream interrupted" in body
                out.add(f"down{i}", ok, time.time() - t0,
                        f"status={status} body={body[:120]}")
            except Exception as e:  # noqa: BLE001
                out.add(f"down{i}", False, time.time() - t0, repr(e))

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(one, i) for i in range(a.workers)]
            time.sleep(2.0)
            if self.be_proc and self.be_proc.poll() is None:
                self.be_proc.kill()
                self.be_proc.wait(timeout=5)
            for f in futs:
                f.result()
        # restart the backend and verify the proxy recovers
        self.start_backend()
        self.control(mode="ok", delay=2.0)
        self.drain_check("backend_down", out, delay=2.0)
        self.check_slot_exclusivity_proxy("backend_down", out)
        self.control(delay=self.delay)

    def scenario_disconnect(self, out: Outcome):
        a = self.args
        # long stream: 3 s first byte, then 30 chunks x 0.5 s = ~18 s total
        self.control(mode="ok", delay=3.0, chunk_interval=0.5, n_chunks=30)
        try:
            def one(i: int):
                t0 = time.time()
                try:
                    status, info = self.chat_stream_partial(False, f"dc{i}", 2, 60)
                    out.add(f"dc{i}", status == 200, time.time() - t0, info)
                except Exception as e:  # noqa: BLE001
                    out.add(f"dc{i}", False, time.time() - t0, repr(e))

            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                list(ex.map(one, range(a.workers)))
            # give the proxy a moment to notice the disconnects, then verify
            # the slots are actually free with a tight budget.
            time.sleep(5.0)
            self.check_violations("disconnect", out)
            self.drain_check("disconnect", out, delay=3.0, margin=10.0)
            self.check_slot_exclusivity_proxy("disconnect", out)
        finally:
            self.control(delay=self.delay, n_chunks=20)

    def scenario_half_request(self, out: Outcome):
        a = self.args
        self.control(mode="ok", delay=2.0)

        def one(i: int):
            t0 = time.time()
            try:
                s = socket.create_connection(("127.0.0.1", self.px_port), timeout=5)
                s.sendall(
                    b"POST /v1/chat/completions HTTP/1.1\r\n"
                    b"Host: x\r\nContent-Type: application/json\r\n"
                    b"Content-Length: 5000\r\n\r\n"
                    b'{"messages": [{"role": "user", "content": "par'
                )
                time.sleep(0.3)
                s.close()
                out.add(f"half{i}", True, time.time() - t0)
            except Exception as e:  # noqa: BLE001
                out.add(f"half{i}", False, time.time() - t0, repr(e))

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(one, range(a.workers * 2)))
        time.sleep(2.0)
        self.check_violations("half_request", out)
        self.drain_check("half_request", out, delay=2.0)
        self.check_slot_exclusivity_proxy("half_request", out)
        self.control(delay=self.delay)

    def scenario_dangling(self, out: Outcome):
        a = self.args
        self.control(mode="ok", delay=2.0)
        socks = []
        try:
            # Half-open: incomplete request headers, socket left idle. The
            # proxy must hold these without blocking real traffic.
            for i in range(30):
                s = socket.create_connection(("127.0.0.1", self.px_port), timeout=5)
                s.sendall(b"GET / HTTP/1.1\r\nHost: x\r\nX-Partial: 1\r\n")
                socks.append(s)
            time.sleep(1.0)

            def one(i: int):
                t0 = time.time()
                try:
                    status, _ = self.chat(False, i % 2 == 0, f"dl{i}", 30)
                    ok = status == 200
                    out.add(f"dl{i}", ok, time.time() - t0, f"status={status}")
                except Exception as e:  # noqa: BLE001
                    out.add(f"dl{i}", False, time.time() - t0, repr(e))

            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                list(ex.map(one, range(a.workers)))
        finally:
            for s in socks:
                try:
                    s.close()
                except OSError:
                    pass
        self.check_violations("dangling", out)
        self.drain_check("dangling", out, delay=2.0)
        self.check_slot_exclusivity_proxy("dangling", out)
        self.control(delay=self.delay)

    def scenario_chaos(self, out: Outcome):
        a = self.args
        self.control(mode="ok", delay=3.0, chunk_interval=0.3, n_chunks=10)
        try:

            def worker(w: int):
                for j in range(a.requests_per_worker):
                    kind = (w * 7 + j * 3) % 10
                    t0 = time.time()
                    try:
                        if kind < 6:
                            status, body = self.chat(
                                (w + j) % 2 == 0, (w + j) % 2 == 1, f"c{w}-{j}", 60
                            )
                            ok = status == 200
                            out.add(f"c{w}.{j}", ok, time.time() - t0,
                                    f"status={status} body={body[:100]}")
                        else:
                            status, info = self.chat_stream_partial(
                                False, f"c{w}-{j}", 1, 60
                            )
                            out.add(f"c{w}.{j}", status == 200, time.time() - t0, info)
                    except Exception as e:  # noqa: BLE001
                        out.add(f"c{w}.{j}", False, time.time() - t0, repr(e))

            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                list(ex.map(worker, range(a.workers)))
            self.check_violations("chaos", out)
            self.drain_check("chaos", out, delay=3.0)
            self.check_slot_exclusivity_proxy("chaos", out)
        finally:
            self.control(mode="ok", delay=self.delay, n_chunks=20, chunk_interval=0.5)


def make_payload(big: bool, stream: bool, tag: str) -> dict:
    if big:
        text = f"tag {tag} " + "lorem ipsum dolor sit amet " * 60  # ~480+ words
    else:
        text = f"tag {tag} hello"
    return {
        "model": "fake-model",
        "messages": [
            {"role": "system", "content": "You are a helpful fake assistant."},
            {"role": "user", "content": text},
        ],
        "stream": stream,
        "max_tokens": 50,
    }


SCENARIOS = {
    "steady": "scenario_steady",
    "backend_error": "scenario_backend_error",
    "backend_down": "scenario_backend_down",
    "disconnect": "scenario_disconnect",
    "half_request": "scenario_half_request",
    "dangling": "scenario_dangling",
    "chaos": "scenario_chaos",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--requests-per-worker", type=int, default=8)
    ap.add_argument("--delay", type=float, default=20.0)
    ap.add_argument("--req-timeout", type=float, default=0,
                    help="per-request timeout; default covers worst-case slot queue")
    ap.add_argument("--scenario", default="all",
                    choices=list(SCENARIOS) + ["all"])
    ap.add_argument("--fast", action="store_true",
                    help="2 s backend delay, fewer requests (quick matrix)")
    args = ap.parse_args()

    if args.fast:
        args.delay = 2.0
        args.requests_per_worker = min(args.requests_per_worker, 4)
    if args.req_timeout <= 0:
        # Worst-case steady pileup: a request can queue behind every other
        # request. With N_SLOTS parallel slots and a max per-request hold of
        # (delay + n_chunks*chunk_interval), bound the latency accordingly.
        # The drain_check (not this timeout) is what detects a true freeze.
        max_hold = args.delay + 20 * 0.5  # steady uses n_chunks=20, ci=0.5
        total = args.workers * args.requests_per_worker
        args.req_timeout = (total / N_SLOTS) * max_hold + 60

    h = Harness(args)
    results: list[Outcome] = []
    try:
        h.start_backend()
        h.start_proxy()
        names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
        for name in names:
            out = Outcome(scenario=name, label=name)
            t0 = time.time()
            print(f"== scenario {name} ==", flush=True)
            getattr(h, SCENARIOS[name])(out)
            wall = time.time() - t0
            verdict = "PASS" if out.failed == 0 else "FAIL"
            print(f"   {out.summary()} wall={wall:.1f}s -> {verdict}", flush=True)
            for e in out.errors:
                print(f"     ERR {e}", flush=True)
            results.append(out)
    finally:
        h.stop_all()

    print("\n================ SUMMARY ================")
    failed = 0
    for out in results:
        verdict = "PASS" if out.failed == 0 else "FAIL"
        if out.failed:
            failed += 1
        print(f"{out.scenario:16s} {verdict}  {out.summary()}")
    print(f"tmp dir: {h.tmp}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
