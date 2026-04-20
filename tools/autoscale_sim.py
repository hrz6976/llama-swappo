#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import statistics
from dataclasses import dataclass, field


@dataclass
class Replica:
    id: int
    capacity: int
    state: str = "ready"
    ready_at: float = 0.0
    completions: list[float] = field(default_factory=list)
    last_done: float = 0.0

    def refresh(self, now: float) -> None:
        while self.completions and self.completions[0] <= now:
            self.last_done = heapq.heappop(self.completions)
        if self.state == "starting" and self.ready_at <= now:
            self.state = "ready"

    @property
    def in_flight(self) -> int:
        return len(self.completions)

    @property
    def queue_ratio(self) -> float:
        if self.capacity <= 0:
            return 1.0
        return self.in_flight / self.capacity

    def can_accept(self) -> bool:
        return self.in_flight < self.capacity


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def least_loaded(replicas: list[Replica]) -> Replica:
    def score(replica: Replica) -> float:
        value = replica.queue_ratio
        if replica.state == "starting":
            value += 0.5
        if not replica.can_accept():
            value += 1.0
        return value

    return min(replicas, key=score)


def should_scale_up(
    replicas: list[Replica],
    now: float,
    max_replicas: int,
    cooldown_s: float,
    last_scale_up: float | None,
    queue_ratio: float,
) -> bool:
    if len(replicas) >= max_replicas:
        return False
    if last_scale_up is not None and now - last_scale_up < cooldown_s:
        return False

    active = 0
    for replica in replicas:
        if replica.state not in {"ready", "starting"}:
            continue
        active += 1
        if replica.queue_ratio < queue_ratio and replica.can_accept():
            return False
    return active > 0


def simulate(args: argparse.Namespace) -> dict:
    if args.arrival == "burst":
        arrivals = [0.0 for _ in range(args.requests)]
    else:
        arrivals = [i / args.rps for i in range(args.requests)]

    replicas = [Replica(id=0, capacity=args.capacity)]
    last_scale_up: float | None = None
    accepted = 0
    rejected = 0
    latencies: list[float] = []
    replica_counts: list[int] = []
    events = []

    for req_id, now in enumerate(arrivals):
        for replica in replicas:
            replica.refresh(now)

        if should_scale_up(replicas, now, args.max_replicas, args.cooldown_s, last_scale_up, args.scale_up_queue_ratio):
            replica = Replica(
                id=len(replicas),
                capacity=args.capacity,
                state="starting",
                ready_at=now + args.cold_start_s,
            )
            replicas.append(replica)
            last_scale_up = now
            events.append({"t": round(now, 3), "event": "scale_up", "replicas": len(replicas)})

        replica = least_loaded(replicas)
        if not replica.can_accept():
            rejected += 1
            events.append({"t": round(now, 3), "event": "reject", "request": req_id, "replicas": len(replicas)})
            replica_counts.append(len(replicas))
            continue

        accepted += 1
        begin_service = max(now, replica.ready_at)
        done = begin_service + args.service_s
        heapq.heappush(replica.completions, done)
        latencies.append(done - now)
        replica_counts.append(len(replicas))

    for replica in replicas:
        while replica.completions:
            replica.last_done = heapq.heappop(replica.completions)

    return {
        "requests": args.requests,
        "accepted": accepted,
        "rejected": rejected,
        "capacity_per_replica": args.capacity,
        "max_replicas": args.max_replicas,
        "cooldown_s": args.cooldown_s,
        "cold_start_s": args.cold_start_s,
        "scale_up_queue_ratio": args.scale_up_queue_ratio,
        "replicas_started": len(replicas),
        "max_observed_replicas": max(replica_counts) if replica_counts else 1,
        "latency_s": {
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
        },
        "events": events[: args.max_events],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate llama-swappo autoscale routing behavior.")
    parser.add_argument("--arrival", choices=["burst", "rps"], default="burst")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--rps", type=float, default=2.0)
    parser.add_argument("--service-s", type=float, default=4.0)
    parser.add_argument("--cold-start-s", type=float, default=20.0)
    parser.add_argument("--capacity", type=int, default=4, help="Proxy concurrencyLimit per replica.")
    parser.add_argument("--max-replicas", type=int, default=6)
    parser.add_argument("--cooldown-s", type=float, default=30.0)
    parser.add_argument("--scale-up-queue-ratio", type=float, default=0.75)
    parser.add_argument("--max-events", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(simulate(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
