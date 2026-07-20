import { describe, expect, it } from "vitest";
import {
  captureGeneration,
  consumedTotal,
  hasNewerGeneration,
  isAcknowledged,
} from "../src/generation";
import type { SenderStatus } from "../src/cli";

const s = (total: number, cursor: number): SenderStatus => ({
  total,
  cursor,
  unread: total - Math.min(cursor, total),
});

describe("captureGeneration", () => {
  it("captures target_total only for currently-unread senders", () => {
    const gen = captureGeneration({ alice: s(3, 1), bob: s(2, 2), carol: s(0, 0) });
    // bob fully read, carol empty -> excluded
    expect(gen).toEqual({ alice: 3 });
  });
});

describe("isAcknowledged", () => {
  it("requires every captured sender to reach its target", () => {
    const gen = { alice: 3, bob: 2 };
    expect(isAcknowledged(gen, { alice: s(3, 3), bob: s(2, 1) })).toBe(false);
    expect(isAcknowledged(gen, { alice: s(3, 3), bob: s(2, 2) })).toBe(true);
  });

  it("treats a vanished captured sender as satisfied", () => {
    expect(isAcknowledged({ alice: 3 }, {})).toBe(true);
  });

  it("clamps cursor to total (over-cursor still acks)", () => {
    expect(isAcknowledged({ alice: 3 }, { alice: s(3, 9) })).toBe(true);
  });

  it("is race-safe: later arrivals raise total but do not un-ack the captured generation", () => {
    // captured target 3; lead read all 3, then a 4th arrived (total 4, cursor 3)
    expect(isAcknowledged({ alice: 3 }, { alice: s(4, 3) })).toBe(true);
  });
});

describe("consumedTotal (progress metric)", () => {
  it("is monotonic as the cursor advances (below target)", () => {
    const gen = { alice: 60 }; // captured target for a >50 backlog
    const before = consumedTotal(gen, { alice: s(60, 10) });
    const after = consumedTotal(gen, { alice: s(60, 12) });
    expect(after).toBeGreaterThan(before);
  });

  it("caps contribution at the captured target", () => {
    expect(consumedTotal({ alice: 5 }, { alice: s(60, 50) })).toBe(5);
  });
});

describe("hasNewerGeneration", () => {
  it("recognizes a brand-new sender via union / default-0 with unread>0", () => {
    // stalled on A target 5; B's first-ever message arrives
    expect(hasNewerGeneration({ alice: 5 }, { alice: s(5, 0), bob: s(1, 0) })).toBe(true);
  });

  it("suppresses arrive-then-consume (higher total but unread==0)", () => {
    expect(hasNewerGeneration({ alice: 5 }, { alice: s(5, 0), bob: s(1, 1) })).toBe(false);
  });

  it("is false when only the captured, still-unread generation is present", () => {
    expect(hasNewerGeneration({ alice: 5 }, { alice: s(5, 0) })).toBe(false);
  });

  it("recognizes more messages from a captured sender (total grows, still unread)", () => {
    expect(hasNewerGeneration({ alice: 5 }, { alice: s(7, 0) })).toBe(true);
  });
});
