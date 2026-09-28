// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/api/core", () => ({ invoke: mocks.invoke }));

function frameEnvelope(): ArrayBuffer {
  const bytes = new Uint8Array(16);
  bytes.set([79, 84, 70, 49], 0); // OTF1
  new DataView(bytes.buffer).setUint32(4, 1280, true);
  new DataView(bytes.buffer).setUint32(8, 720, true);
  bytes.set([0xff, 0xd8, 0xff, 0xd9], 12);
  return bytes.buffer;
}

describe("binary composite preview IPC", () => {
  const createObjectURL = vi.fn(() => "blob:still-frame");
  const revokeObjectURL = vi.fn();

  beforeEach(() => {
    vi.resetModules();
    vi.stubGlobal("window", { __TAURI_INTERNALS__: {} });
    const NativeURL = URL;
    vi.stubGlobal("URL", class extends NativeURL {
      static createObjectURL = createObjectURL;
      static revokeObjectURL = revokeObjectURL;
    });
    mocks.invoke.mockReset().mockResolvedValue(frameEnvelope());
    createObjectURL.mockClear();
    revokeObjectURL.mockClear();
  });

  afterEach(() => vi.unstubAllGlobals());

  it("decodes dimensions and JPEG bytes and releases the temporary URL", async () => {
    const { compositeFrame } = await import("./api");
    const request = {
      frame: 10.9, projectEpoch: 1, timelineVersion: 2,
      sessionId: "preview", sessionGeneration: 3, seekGeneration: 4,
    };
    const image = await compositeFrame(request);
    expect(mocks.invoke).toHaveBeenCalledWith("composite_frame", {
      request: { ...request, frame: 10 }, maxSize: undefined,
    });
    expect(image).toMatchObject({ width: 1280, height: 720, dataUrl: "blob:still-frame" });
    const [blob] = createObjectURL.mock.calls[0] as [Blob];
    expect(blob.type).toBe("image/jpeg");
    expect(Array.from(new Uint8Array(await blob.arrayBuffer()))).toEqual([0xff, 0xd8, 0xff, 0xd9]);
    image?.release?.();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:still-frame");
  });

  it("rejects a malformed native response before making an object URL", async () => {
    const { decodeCompositeFrameResponse } = await import("./api");
    const bytes = frameEnvelope();
    new Uint8Array(bytes)[0] = 0;
    expect(() => decodeCompositeFrameResponse(bytes)).toThrow("Invalid binary preview frame");
    expect(createObjectURL).not.toHaveBeenCalled();
  });
});
