import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  pickSavePath: vi.fn(),
}));

vi.mock("./api", () => ({
  isTauri: true,
  pickSavePath: mocks.pickSavePath,
}));

import { saveDialog } from "./dialog";

describe("saveDialog", () => {
  beforeEach(() => mocks.pickSavePath.mockReset());

  it("asks the backend for a save grant bound to the purpose", async () => {
    mocks.pickSavePath.mockResolvedValue("/home/qa/Videos/cut");
    const save = await saveDialog("interchange");

    const chosen = await save?.({ title: "Export", defaultPath: "/home/qa/Videos/cut.xml" });

    expect(chosen).toBe("/home/qa/Videos/cut");
    expect(mocks.pickSavePath).toHaveBeenCalledWith("interchange", {
      title: "Export",
      defaultPath: "/home/qa/Videos/cut.xml",
    });
  });

  it("reports a cancelled dialog as null", async () => {
    mocks.pickSavePath.mockResolvedValue(null);
    const save = await saveDialog("project");

    expect(await save?.({})).toBeNull();
  });
});
