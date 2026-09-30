import { describe, expect, it } from "vitest";
import { displayNativePath, joinNativePath, nativePathName, nativePathParent, replaceNativePathSuffix } from "./nativePath";

describe("native paths", () => {
  it("keeps Unicode paths and both directory separators", () => {
    expect(displayNativePath("/media/片段.mp4")).toBe("/media/片段.mp4");
    expect(joinNativePath("C:\\media", "片段.mp4")).toBe("C:\\media\\片段.mp4");
  });

  it("labels raw bytes without turning the label into an IPC path", () => {
    const path = "opentake-path-v1:unix:2f746d702f636c69702dff2e6d7034";
    expect(displayNativePath(path)).toBe("/tmp/clip-\\xff.mp4");
    expect(nativePathName(path)).toBe("clip-\\xff.mp4");
    expect(joinNativePath("opentake-path-v1:unix:2f746d702fff", "片段.mp4")).toBe("opentake-path-v1:unix:2f746d702fff2fe78987e6aeb52e6d7034");
  });

  it("keeps Windows native units and paired Unicode characters", () => {
    expect(displayNativePath("opentake-path-v1:windows:0043003a005cd800002ed83ddc0e")).toBe("C:\\\\ud800.🐎");
    expect(joinNativePath("opentake-path-v1:windows:0043003a005cd800", "x.mp4")).toBe("opentake-path-v1:windows:0043003a005cd800005c0078002e006d00700034");
  });

  it("rejects invalid encodings", () => {
    for (const path of ["opentake-path-v1:unix:0", "opentake-path-v1:unix:00", "opentake-path-v1:other:ff"]) expect(() => displayNativePath(path)).toThrow();
  });

  it("finds parents before formatting escapes", () => {
    expect(nativePathParent("opentake-path-v1:unix:2f746d702f636c6970ff")).toBe("opentake-path-v1:unix:2f746d70");
    expect(nativePathParent("opentake-path-v1:windows:0043003a005cd800")).toBe("opentake-path-v1:windows:0043003a005c");
    expect(nativePathName("opentake-path-v1:windows:0043003a005cd800")).toBe("\\ud800");
  });

  it("preserves native filename units when proposing exports", () => {
    expect(replaceNativePathSuffix("/tmp/Film.OPENTAKE", ".opentake", ".mp4")).toBe("/tmp/Film.mp4");
    expect(replaceNativePathSuffix("/tmp/Film", ".opentake", ".srt")).toBe("/tmp/Film.srt");
    expect(replaceNativePathSuffix("opentake-path-v1:unix:2f746d702f636c6970ff2e6f70656e74616b65", ".opentake", ".mp4"))
      .toBe("opentake-path-v1:unix:2f746d702f636c6970ff2e6d7034");
    expect(replaceNativePathSuffix("opentake-path-v1:windows:0043003a005cd800002e004f00500045004e00540041004b0045", ".opentake", ".srt"))
      .toBe("opentake-path-v1:windows:0043003a005cd800002e007300720074");
  });
});
