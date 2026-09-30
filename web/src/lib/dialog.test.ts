import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  pickSavePath: vi.fn(),
  pickOpenPaths: vi.fn(),
}));

vi.mock("./api", () => ({
  isTauri: true,
  pickSavePath: mocks.pickSavePath,
  pickOpenPaths: mocks.pickOpenPaths,
}));

import { openDialog, saveDialog } from "./dialog";

it("passes opaque native paths through the host open dialog", async () => {
  const path = "opentake-path-v1:unix:2f746d702f636c6970ff2e706e67";
  mocks.pickOpenPaths.mockResolvedValue([path]);
  const open = await openDialog();
  expect(await open?.({ multiple: true })).toEqual([path]);
  expect(mocks.pickOpenPaths).toHaveBeenCalledWith({ multiple: true });
});

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
