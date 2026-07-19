/**
 * The generation ACK contract (plan §3.1). A wake defines a *generation*: at
 * capture time we record `target_total[sender] = total` for every sender that
 * currently has `unread > 0`. The generation is acknowledged when, for ALL
 * captured senders, `min(cursor, total) >= target_total` — cursor/total based,
 * so concurrent arrivals (which only raise `total`) never un-ack it.
 */
import type { SenderStatus } from "./cli";

/** sender -> target_total captured at wake time. */
export type Generation = Record<string, number>;

function consumed(status: SenderStatus): number {
  return Math.min(status.cursor, status.total);
}

/** Capture a target map for every currently-unread sender. */
export function captureGeneration(senders: Record<string, SenderStatus>): Generation {
  const gen: Generation = {};
  for (const [name, status] of Object.entries(senders)) {
    if (status.unread > 0) {
      gen[name] = status.total;
    }
  }
  return gen;
}

/**
 * Acknowledged when every captured sender has reached its target. A captured
 * sender that has vanished from the snapshot is treated as satisfied.
 */
export function isAcknowledged(gen: Generation, senders: Record<string, SenderStatus>): boolean {
  for (const [name, target] of Object.entries(gen)) {
    const status = senders[name];
    if (status === undefined) {
      continue; // vanished -> satisfied
    }
    if (consumed(status) < target) {
      return false;
    }
  }
  return true;
}

/**
 * Monotonic progress metric: sum over captured senders of `min(consumed, target)`
 * (a vanished sender counts as fully satisfied). Rises only as the lead drains,
 * so an increase between probes means genuine progress that refreshes the budget.
 */
export function consumedTotal(gen: Generation, senders: Record<string, SenderStatus>): number {
  let sum = 0;
  for (const [name, target] of Object.entries(gen)) {
    const status = senders[name];
    if (status === undefined) {
      sum += target;
    } else {
      sum += Math.min(consumed(status), target);
    }
  }
  return sum;
}

/**
 * A strictly-newer generation exists (ACK_STALLED rule 2): over the union of
 * captured and current sender keys — a missing captured target defaulting to 0 —
 * some sender has `current.total > captured_target` AND currently `unread > 0`
 * (the unread guard suppresses arrivals independently consumed between probes).
 */
export function hasNewerGeneration(
  gen: Generation,
  senders: Record<string, SenderStatus>,
): boolean {
  for (const [name, status] of Object.entries(senders)) {
    const capturedTarget = gen[name] ?? 0;
    if (status.total > capturedTarget && status.unread > 0) {
      return true;
    }
  }
  return false;
}
