#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass


@dataclass
class Result:
    index: int
    status: int
    latency_s: float
    body: str


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def request_once(base_url: str, model: str, index: int, delay_ms: int, timeout: float, stagger_ms: int) -> Result:
    if stagger_ms > 0:
        time.sleep(index * stagger_ms / 1000.0)
    url = f"{base_url.rstrip('/')}/upstream/{model}/slow-respond?echo=req-{index}&delay={delay_ms}ms"
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        status = exc.code
    except Exception as exc:
        body = repr(exc)
        status = 0
    return Result(index=index, status=status, latency_s=time.perf_counter() - start, body=body)


def poll_running(base_url: str, stop: threading.Event, interval_s: float, samples: list[dict]) -> None:
    url = f"{base_url.rstrip('/')}/running"
    while not stop.is_set():
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            replicas = [row.get("replica") or row.get("model") for row in payload.get("running", [])]
            samples.append({"t": time.time(), "replicas": sorted(replicas)})
        except Exception as exc:
            samples.append({"t": time.time(), "error": repr(exc)})
        stop.wait(interval_s)


def main() -> int:
    parser = argparse.ArgumentParser(description="HTTP autoscale benchmark for llama-swappo.")
    parser.add_argument("--base-url", default="http://127.0.0.1:19000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--delay-ms", type=int, default=1000)
    parser.add_argument("--stagger-ms", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.5)
    args = parser.parse_args()

    samples: list[dict] = []
    stop = threading.Event()
    poller = threading.Thread(target=poll_running, args=(args.base_url, stop, args.poll_interval_s, samples), daemon=True)
    poller.start()

    started = time.perf_counter()
    results: list[Result] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(request_once, args.base_url, args.model, i, args.delay_ms, args.timeout_s, args.stagger_ms)
            for i in range(args.requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    elapsed = time.perf_counter() - started
    stop.set()
    poller.join(timeout=2)

    latencies = [result.latency_s for result in results if result.status == 200]
    status_counts = Counter(str(result.status) for result in results)
    observed_replicas = sorted(
        {
            replica
            for sample in samples
            for replica in sample.get("replicas", [])
            if replica == args.model or replica.startswith(args.model + "#")
        }
    )
    max_running_replicas = max(
        (
            len(
                [
                    replica
                    for replica in sample.get("replicas", [])
                    if replica == args.model or replica.startswith(args.model + "#")
                ]
            )
            for sample in samples
        ),
        default=0,
    )

    summary = {
        "base_url": args.base_url,
        "model": args.model,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "delay_ms": args.delay_ms,
        "stagger_ms": args.stagger_ms,
        "elapsed_s": elapsed,
        "throughput_req_s": args.requests / elapsed if elapsed else 0.0,
        "status_counts": dict(status_counts),
        "latency_s": {
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
        },
        "observed_replicas": observed_replicas,
        "max_running_replicas": max_running_replicas,
        "running_samples": samples[-10:],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
