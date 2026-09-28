import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ProjectOpenNotices } from "../../store/projectStore";

const bannerState = vi.hoisted(() => ({
  locale: "zh-CN" as "zh-CN" | "en",
  projectEpoch: 3,
  openNotices: null as ProjectOpenNotices | null,
  setOpenNotices: vi.fn(),
}));

// Server rendering reads a zustand store's initial state, so the banner reads
// this fixture instead (as the compatibility banner test does).
vi.mock("../../store/projectStore", () => ({
  useProjectStore: (selector: (state: object) => unknown) =>
    selector({
      projectEpoch: bannerState.projectEpoch,
      openNotices: bannerState.openNotices,
      setOpenNotices: bannerState.setOpenNotices,
    }),
}));

vi.mock("../../i18n", async () => {
  const { DICTS } = await import("../../i18n/dict");
  const translate = (key: string, vars?: Record<string, string | number>) => {
    const template = DICTS[bannerState.locale][key] ?? key;
    return template.replace(/\{(\w+)\}/g, (_match, name: string) =>
      vars && name in vars ? String(vars[name]) : `{${name}}`,
    );
  };
  return { useT: () => translate, t: translate };
});

import { ProjectOpenNoticesBanner } from "./ProjectOpenNoticesBanner";

describe("ProjectOpenNoticesBanner", () => {
  beforeEach(() => {
    bannerState.locale = "zh-CN";
    bannerState.projectEpoch = 3;
    bannerState.openNotices = null;
  });

  it("lists every notice on its own line in the active language with a dismiss button", () => {
    bannerState.openNotices = {
      projectEpoch: 3,
      warnings: [
        "media.json:offline-media:a",
        "media.json:offline-media:b",
        "generation-log.json:moved-aside:generation-log.oversized-1.json",
      ],
      mediaNames: { a: "Intro", b: "Outro" },
    };
    const zh = renderToStaticMarkup(<ProjectOpenNoticesBanner />);
    expect(zh).toContain("white-space:pre-line");
    expect(zh).toContain(
      "可重新链接：Intro、Outro\n生成记录过大，已移到项目内的 generation-log.oversized-1.json",
    );
    expect(zh).toContain('aria-label="知道了"');

    bannerState.locale = "en";
    const en = renderToStaticMarkup(<ProjectOpenNoticesBanner />);
    expect(en).toContain("relink them to use them: Intro, Outro");
    expect(en).toContain('aria-label="Dismiss"');
  });

  it("renders nothing without notices or for another project", () => {
    expect(renderToStaticMarkup(<ProjectOpenNoticesBanner />)).toBe("");
    bannerState.openNotices = {
      projectEpoch: 2,
      warnings: ["media.json:offline-media:a"],
      mediaNames: {},
    };
    expect(renderToStaticMarkup(<ProjectOpenNoticesBanner />)).toBe("");
  });
});
