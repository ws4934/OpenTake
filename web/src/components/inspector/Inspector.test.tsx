// @vitest-environment happy-dom

import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Clip, Timeline } from "../../lib/types";
import { useEditorUiStore } from "../../store/uiStore";
import { useMediaStore } from "../../store/mediaStore";
import { useProjectStore } from "../../store/projectStore";
import * as edit from "../../store/editActions";
import * as api from "../../lib/api";
import { t } from "../../i18n";
import { Inspector } from "./Inspector";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

function visualClip(overrides: Partial<Clip> = {}): Clip {
  return {
    id: "clip-1",
    mediaRef: "media-1",
    mediaType: "video",
    sourceClipType: "video",
    startFrame: 0,
    durationFrames: 90,
    trimStartFrame: 0,
    trimEndFrame: 90,
    speed: 1,
    volume: 1,
    fadeInFrames: 0,
    fadeOutFrames: 0,
    fadeInInterpolation: "linear",
    fadeOutInterpolation: "linear",
    opacity: 1,
    transform: {
      centerX: 0.5,
      centerY: 0.5,
      width: 1,
      height: 1,
      rotation: 0,
      flipHorizontal: false,
      flipVertical: false,
    },
    crop: { left: 0, top: 0, right: 0, bottom: 0 },
    ...overrides,
  };
}

function timelineWith(clip: Clip): Timeline {
  return {
    fps: 30,
    width: 1920,
    height: 1080,
    settingsConfigured: true,
    tracks: [
      {
        id: "video-1",
        name: "Video 1",
        type: "video",
        muted: false,
        hidden: false,
        syncLocked: false,
        clips: [clip],
      },
    ],
  };
}

afterEach(() => {
  document.body.replaceChildren();
  useEditorUiStore.setState({
    selectedClipIds: new Set(),
    inspectorTab: "video",
    keyframesPanelVisible: false,
  });
  useMediaStore.setState({ items: [], folders: [], importing: false, error: null });
  useProjectStore.getState().clearProjectSnapshot();
});

describe("Inspector completion surface", () => {
  it("commits speed through the retime command rather than generic properties", async () => {
    const clip = visualClip();
    const speed = vi.spyOn(edit, "setClipSpeed").mockResolvedValue(undefined);
    const properties = vi.spyOn(edit, "setClipProperties").mockResolvedValue(undefined);
    useProjectStore.setState({ timeline: timelineWith(clip) });
    useEditorUiStore.setState({ selectedClipIds: new Set([clip.id]), inspectorTab: "video" });
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    try {
      await act(async () => root.render(<Inspector />));
      const field = container.querySelector<HTMLElement>(`[role="spinbutton"][aria-label="${t("inspector.field.speed")}"]`);
      expect(field).not.toBeNull();
      await act(async () => field!.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true })));
      const input = container.querySelector<HTMLInputElement>(`input[aria-label="${t("inspector.field.speed")}"]`);
      expect(input).not.toBeNull();
      await act(async () => {
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, "2.00");
        input!.dispatchEvent(new InputEvent("input", { bubbles: true }));
      });
      await act(async () => input!.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true })));
      expect(speed).toHaveBeenCalledExactlyOnceWith([clip.id], 2);
      expect(properties).not.toHaveBeenCalled();
    } finally {
      await act(async () => root.unmount());
      speed.mockRestore();
      properties.mockRestore();
    }
  });

  it("keeps compact transform, crop, and HSL glyphs inside 24px hit frames", async () => {
    const clip = visualClip();
    useProjectStore.setState({ timeline: timelineWith(clip) });
    useEditorUiStore.setState({
      selectedClipIds: new Set([clip.id]),
      inspectorTab: "video",
    });
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<Inspector />));

    for (const title of [
      t("inspector.action.resetTransform"),
      t("inspector.action.cropEditStart"),
    ]) {
      const button = container.querySelector<HTMLButtonElement>(`button[title="${title}"]`);
      expect(button?.style.width, title).toBe("24px");
      expect(button?.style.height, title).toBe("24px");
    }

    await act(async () => useEditorUiStore.setState({ inspectorTab: "color" }));
    const hslReset = container.querySelector<HTMLButtonElement>(
      `button[title="${t("inspector.action.resetHslSecondary")}"]`,
    );
    expect(hslReset?.style.width).toBe("24px");
    expect(hslReset?.style.height).toBe("24px");
    await act(async () => root.unmount());
  });

  it("renders a generated media source without entering a selector update loop", async () => {
    useProjectStore.setState({
      timeline: {
        fps: 30,
        width: 1920,
        height: 1080,
        settingsConfigured: false,
        tracks: [],
      },
      projectPath: "/tmp/demo.opentake",
    });
    useMediaStore.setState({
      items: [
        {
          id: "stem-vocals",
          name: "Mix Vocals",
          type: "audio",
          duration: 5,
          hasAudio: true,
          generationInput: {
            prompt: "stem:vocals",
            model: "opentake-center-v1",
            duration: 5,
            aspectRatio: "audio",
            provider: "local",
            status: "ready",
            progress: 1,
          },
        },
      ],
      folders: [],
      importing: false,
      error: null,
    });
    useEditorUiStore.setState({
      selectedClipIds: new Set(),
      previewMediaId: "stem-vocals",
    });

    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<Inspector />));

    expect(container.textContent).toContain("Mix Vocals");
    expect(container.textContent).toContain("opentake-center-v1");
    await act(async () => root.unmount());
  });

  it("four_states_tabs_fields_and_lanes", async () => {
    const clip = visualClip();
    useProjectStore.setState({ timeline: timelineWith(clip), projectPath: "/tmp/demo.opentake" });
    useEditorUiStore.setState({ selectedClipIds: new Set([clip.id]), inspectorTab: "video" });

    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<Inspector />));

    const tabs = [...container.querySelectorAll<HTMLElement>('[role="tab"]')];
    expect(tabs.map((tab) => tab.textContent)).toEqual(["视频", "AI 编辑"]);
    expect(tabs[0]?.getAttribute("aria-selected")).toBe("true");

    await act(async () => tabs[1]?.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(container.querySelector('[data-testid="ai-edit-tab"]')).not.toBeNull();

    await act(async () => root.unmount());
  });

  it("keeps every compact keyframe navigation glyph inside a 24px target", async () => {
    const clip = visualClip();
    useProjectStore.setState({ timeline: timelineWith(clip), projectPath: "/tmp/demo.opentake" });
    useEditorUiStore.setState({
      selectedClipIds: new Set([clip.id]),
      inspectorTab: "video",
      activeFrame: 10,
    });
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<Inspector />));

    const controls = [...container.querySelectorAll<HTMLButtonElement>(
      'button[aria-label="跳到上一个关键帧"], button[aria-label="跳到下一个关键帧"], button[aria-label="在播放头处添加关键帧"]',
    )];
    expect(controls.length).toBeGreaterThan(0);
    expect(controls.every((button) => button.style.width === "24px")).toBe(true);
    expect(controls.every((button) => button.style.height === "24px")).toBe(true);

    await act(async () => root.unmount());
  });

  it("seeks both playhead values when navigating to the next keyframe", async () => {
    const clip = visualClip({
      opacityTrack: {
        keyframes: [
          { frame: 0, value: 1, interpolationOut: "linear" },
          { frame: 60, value: 0.5, interpolationOut: "linear" },
        ],
      },
    });
    useProjectStore.setState({ timeline: timelineWith(clip), projectPath: "/tmp/demo.opentake" });
    useEditorUiStore.setState({
      selectedClipIds: new Set([clip.id]),
      inspectorTab: "video",
      currentFrame: 10,
      activeFrame: 10,
    });
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<Inspector />));

    const next = [...container.querySelectorAll<HTMLButtonElement>(
      `button[aria-label="${t("inspector.keyframe.next")}"]`,
    )].find((button) => !button.disabled);
    expect(next).not.toBeUndefined();
    await act(async () => next!.dispatchEvent(new MouseEvent("click", { bubbles: true })));

    expect(useEditorUiStore.getState().currentFrame).toBe(60);
    expect(useEditorUiStore.getState().activeFrame).toBe(60);
    await act(async () => root.unmount());
  });

  it("labels numeric controls and disables animated writes outside the clip", async () => {
    const crop = { left: 0, top: 0, right: 0, bottom: 0 };
    const clip = visualClip({
      startFrame: 100,
      durationFrames: 20,
      positionTrack: { keyframes: [{ frame: 0, value: { a: 0, b: 0 }, interpolationOut: "linear" }] },
      scaleTrack: { keyframes: [{ frame: 0, value: { a: 1, b: 1 }, interpolationOut: "linear" }] },
      rotationTrack: { keyframes: [{ frame: 0, value: 0, interpolationOut: "linear" }] },
      opacityTrack: { keyframes: [{ frame: 0, value: 1, interpolationOut: "linear" }] },
      cropTrack: { keyframes: [{ frame: 0, value: crop, interpolationOut: "linear" }] },
      volumeTrack: { keyframes: [{ frame: 0, value: 1, interpolationOut: "linear" }] },
    });
    useProjectStore.setState({ timeline: timelineWith(clip) });
    useMediaStore.setState({
      items: [{ id: clip.mediaRef, name: "clip.mp4", type: "video", duration: 1, hasAudio: true }],
      folders: [],
      importing: false,
      error: null,
    });
    useEditorUiStore.setState({
      selectedClipIds: new Set([clip.id]),
      inspectorTab: "video",
      activeFrame: 99,
    });
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<Inspector />));

    for (const label of ["缩放", "旋转", "不透明度", "X 位置", "Y 位置", "左侧", "顶部", "右侧", "底部"]) {
      const control = container.querySelector<HTMLElement>(`[role="spinbutton"][aria-label="${label}"]`);
      expect(control, label).not.toBeNull();
      expect(control?.getAttribute("aria-disabled"), label).toBe("true");
      expect(control?.tabIndex, label).toBe(-1);
    }
    expect(container.querySelector('[aria-label="Value"]')).toBeNull();
    expect(container.querySelector<HTMLSelectElement>('[aria-label="选择裁剪比例"]')?.disabled).toBe(true);

    await act(async () => useEditorUiStore.setState({ inspectorTab: "audio" }));
    const volume = container.querySelector<HTMLElement>('[role="spinbutton"][aria-label="音量"]');
    expect(volume?.getAttribute("aria-disabled")).toBe("true");

    await act(async () => root.unmount());
  });

  it("normalizes fractional playback frames for animated inspector edits", async () => {
    const clip = visualClip({
      startFrame: 100,
      durationFrames: 20,
      opacityTrack: {
        keyframes: [{ frame: 10, value: 0.5, interpolationOut: "linear" }],
      },
    });
    useProjectStore.setState({ timeline: timelineWith(clip) });
    useEditorUiStore.setState({
      selectedClipIds: new Set([clip.id]),
      inspectorTab: "video",
      activeFrame: 110.8,
    });
    const upsert = vi.spyOn(edit, "upsertKeyframe").mockResolvedValue();
    const remove = vi.spyOn(edit, "removeKeyframe").mockResolvedValue();
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<Inspector />));

    const opacity = container.querySelector<HTMLElement>(
      '[role="spinbutton"][aria-label="不透明度"]',
    )!;
    // Arrow keys preview locally and commit once the key is released.
    await act(async () => {
      opacity.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }));
    });
    expect(upsert).not.toHaveBeenCalled();
    await act(async () => {
      opacity.dispatchEvent(new KeyboardEvent("keyup", { key: "ArrowDown", bubbles: true }));
    });
    expect(upsert).toHaveBeenCalledWith(
      clip.id,
      "opacity",
      110,
      expect.objectContaining({ kind: "scalar" }),
    );

    const removeAtPlayhead = container.querySelector<HTMLButtonElement>(
      'button[aria-label="移除播放头处的关键帧"]',
    )!;
    await act(async () => removeAtPlayhead.click());
    expect(remove).toHaveBeenCalledWith(clip.id, "opacity", 110);

    upsert.mockRestore();
    remove.mockRestore();
    await act(async () => root.unmount());
  });

  it("opens_video_and_ai_tabs_for_one_linked_av_selection", async () => {
    const video = visualClip({ id: "video", linkGroupId: "linked" });
    const audio = visualClip({
      id: "audio",
      mediaRef: "media-1",
      mediaType: "audio",
      linkGroupId: "linked",
    });
    const linkedTimeline = timelineWith(video);
    linkedTimeline.tracks.push({
      id: "audio-1",
      type: "audio",
      muted: false,
      hidden: false,
      syncLocked: true,
      clips: [audio],
    });
    useProjectStore.setState({ timeline: linkedTimeline });
    useEditorUiStore.setState({
      selectedClipIds: new Set([video.id, audio.id]),
      inspectorTab: "video",
    });

    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<Inspector />));

    expect(container.textContent).not.toContain("已选择 2 项");
    expect(
      [...container.querySelectorAll<HTMLElement>('[role="tab"]')].map((tab) => tab.textContent),
    ).toEqual(["视频", "AI 编辑"]);
    await act(async () => root.unmount());
  });

  it("creates edits and deletes a polygon mask through the undoable command route", async () => {
    const clip = visualClip();
    useProjectStore.setState({ timeline: timelineWith(clip), projectPath: "/tmp/demo.opentake" });
    useEditorUiStore.setState({ selectedClipIds: new Set([clip.id]), inspectorTab: "video" });
    const setMasks = vi.spyOn(edit, "setMasks").mockResolvedValue();
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<Inspector />));

    const maskSection = [...container.querySelectorAll("section")].find((section) =>
      section.textContent?.includes("蒙版"),
    );
    expect(maskSection).not.toBeUndefined();
    const enabled = maskSection?.querySelector<HTMLInputElement>('input[type="checkbox"]');
    await act(async () => enabled?.click());
    expect(setMasks).toHaveBeenLastCalledWith(
      [clip.id],
      [expect.objectContaining({ shape: expect.objectContaining({ kind: "circle" }) })],
    );

    const select = maskSection?.querySelector<HTMLSelectElement>("select");
    if (select) select.value = "poly";
    await act(async () => select?.dispatchEvent(new Event("change", { bubbles: true })));
    expect(setMasks).toHaveBeenLastCalledWith(
      [clip.id],
      [expect.objectContaining({ shape: expect.objectContaining({ kind: "poly" }) })],
    );
    expect(maskSection?.textContent).toContain("添加点");

    const deleteButton = [...(maskSection?.querySelectorAll("button") ?? [])].find(
      (button) => button.textContent === "删除蒙版",
    );
    await act(async () => deleteButton?.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(setMasks).toHaveBeenLastCalledWith([clip.id], []);

    setMasks.mockRestore();
    await act(async () => root.unmount());
  });

  it("analyzes, displays, and resets stabilization through production actions", async () => {
    const clip = visualClip();
    useProjectStore.setState({ timeline: timelineWith(clip), projectPath: "/tmp/demo.opentake" });
    useEditorUiStore.setState({ selectedClipIds: new Set([clip.id]), inspectorTab: "video" });
    const solution = {
      model: "opentake.motion-smoothing",
      modelVersion: 1,
      sourceIdentity: clip.mediaRef,
      strength: 1,
      cropMargin: 0,
      keyframes: [
        { frame: 0, translationX: 0, translationY: 0, rotationDegrees: 0 },
        { frame: 89, translationX: 0.02, translationY: -0.01, rotationDegrees: 0 },
      ],
    };
    const analyze = vi
      .spyOn(edit, "analyzeAndApplyStabilization")
      .mockResolvedValue(solution);
    const cancel = vi.spyOn(edit, "cancelStabilizationAnalysis").mockResolvedValue(true);
    const reset = vi.spyOn(edit, "resetStabilization").mockResolvedValue();
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<Inspector />));

    const analyzeButton = container.querySelector<HTMLButtonElement>(
      '[data-testid="stabilization-section"] button',
    );
    expect(analyzeButton).not.toBeNull();
    await act(async () => {
      analyzeButton?.click();
      await Promise.resolve();
    });
    expect(analyze).toHaveBeenCalledWith(clip.id);

    await act(async () => {
      useProjectStore.setState({
        timeline: timelineWith(visualClip({ stabilization: solution })),
      });
    });
    const stabilization = container.querySelector('[data-testid="stabilization-section"]');
    expect(stabilization?.textContent).toContain("opentake.motion-smoothing v1");
    expect(stabilization?.textContent).toContain("100%");
    let failReanalysis!: (reason: Error) => void;
    analyze.mockImplementationOnce(
      () =>
        new Promise((_resolve, reject) => {
          failReanalysis = reject;
        }),
    );
    const reanalyzeButton = [...(stabilization?.querySelectorAll("button") ?? [])].find(
      (button) => button.textContent === "重新分析",
    );
    await act(async () => {
      reanalyzeButton?.click();
      await Promise.resolve();
    });
    expect(reanalyzeButton?.disabled).toBe(true);
    const cancelButton = [...(stabilization?.querySelectorAll("button") ?? [])].find(
      (button) => button.textContent === "取消分析",
    );
    expect(cancelButton).not.toBeNull();
    await act(async () => cancelButton?.click());
    expect(cancel).toHaveBeenCalledOnce();
    await act(async () => failReanalysis(new Error("cancelled")));
    expect(stabilization?.querySelector('[role="alert"]')).toBeNull();
    const resetButton = [...(stabilization?.querySelectorAll("button") ?? [])].find(
      (button) => button.textContent === "重置防抖",
    );
    await act(async () => resetButton?.click());
    expect(reset).toHaveBeenCalledWith(clip.id);

    analyze.mockRestore();
    cancel.mockRestore();
    reset.mockRestore();
    await act(async () => root.unmount());
  });

  it("analyzes, reports progress, displays, and resets loudness normalization", async () => {
    const clip = visualClip();
    useProjectStore.setState({ timeline: timelineWith(clip), projectPath: "/tmp/demo.opentake" });
    useMediaStore.setState({
      items: [{ id: clip.mediaRef, name: "speech.wav", type: "video", duration: 3, hasAudio: true }],
      folders: [],
      importing: false,
      error: null,
    });
    useEditorUiStore.setState({ selectedClipIds: new Set([clip.id]), inspectorTab: "audio", activeFrame: 42 });
    const normalization = {
      targetLufs: -16,
      truePeakCeilingDbtp: -1,
      inputIntegratedLufs: -23,
      inputTruePeakDbtp: -8,
      gainDb: 7,
      outputIntegratedLufs: -16,
      outputTruePeakDbtp: -1,
    };
    const listen = vi.spyOn(api, "onLoudnessProgress").mockImplementation(async (_id, handler) => {
      handler({ clipId: clip.id, done: 40, total: 100 });
      return () => {};
    });
    const analyze = vi.spyOn(edit, "analyzeAndApplyLoudness").mockResolvedValue(normalization);
    const reset = vi.spyOn(edit, "setLoudnessNormalization").mockResolvedValue();
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<Inspector />));

    const section = container.querySelector('[data-testid="loudness-section"]');
    const analyzeButton = [...(section?.querySelectorAll("button") ?? [])].find(
      (button) => button.textContent === "分析并应用",
    );
    await act(async () => {
      analyzeButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(listen).toHaveBeenCalledWith(clip.id, expect.any(Function));
    expect(analyze).toHaveBeenCalledWith(clip.id, -16, -1);

    await act(async () => {
      useProjectStore.setState({
        timeline: timelineWith(visualClip({ loudnessNormalization: normalization })),
      });
    });
    const result = container.querySelector('[data-testid="loudness-result"]');
    expect(result?.textContent).toContain("-23.0 → -16.0 LUFS");
    expect(result?.textContent).toContain("+7.0 dB");
    const resetButton = [...(section?.querySelectorAll("button") ?? [])].find(
      (button) => button.textContent === "重置响度",
    );
    await act(async () => resetButton?.click());
    expect(reset).toHaveBeenCalledWith(clip.id, null);

    listen.mockRestore();
    analyze.mockRestore();
    reset.mockRestore();
    await act(async () => root.unmount());
  });

  it("prepares, previews, cancels, displays, and resets audio denoise", async () => {
    const clip = visualClip();
    useProjectStore.setState({ timeline: timelineWith(clip), projectPath: "/tmp/demo.opentake" });
    useMediaStore.setState({
      items: [{ id: clip.mediaRef, name: "speech-noise.wav", type: "video", duration: 5, hasAudio: true }],
      folders: [],
      importing: false,
      error: null,
    });
    useEditorUiStore.setState({ selectedClipIds: new Set([clip.id]), inspectorTab: "audio" });
    const denoise = { mode: "voice" as const, strength: 0.85, previewEnabled: true };
    const unlisten = vi.fn();
    const listen = vi.spyOn(api, "onDenoiseProgress").mockImplementation(async (_id, handler) => {
      handler({ clipId: clip.id, done: 60, total: 100 });
      return unlisten;
    });
    const prepare = vi
      .spyOn(edit, "prepareAndApplyAudioDenoise")
      .mockResolvedValue(denoise);
    const setDenoise = vi.spyOn(edit, "setAudioDenoise").mockResolvedValue();
    const cancel = vi.spyOn(edit, "cancelDenoiseAnalysis").mockResolvedValue(true);
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<Inspector />));

    const section = container.querySelector('[data-testid="denoise-section"]');
    const mode = section?.querySelector<HTMLSelectElement>('select[aria-label="模式"]');
    const strength = section?.querySelector<HTMLInputElement>('input[aria-label="强度"]');
    await act(async () => {
      if (mode) mode.value = "voice";
      mode?.dispatchEvent(new Event("change", { bubbles: true }));
      if (strength) {
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set?.call(
          strength,
          "0.85",
        );
      }
      strength?.dispatchEvent(new InputEvent("input", { bubbles: true }));
    });
    const applyButton = [...(section?.querySelectorAll("button") ?? [])].find(
      (button) => button.textContent === "应用降噪",
    );
    await act(async () => {
      applyButton?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(listen).toHaveBeenCalledWith(clip.id, expect.any(Function));
    expect(prepare).toHaveBeenCalledWith(clip.id, "voice", 0.85, true);
    expect(unlisten).toHaveBeenCalledOnce();

    await act(async () => {
      useProjectStore.setState({ timeline: timelineWith(visualClip({ audioDenoise: denoise })) });
    });
    expect(container.querySelector('[data-testid="denoise-result"]')?.textContent).toContain(
      "人声 · 85% · 预览开启",
    );
    const preview = section?.querySelector<HTMLInputElement>('input[aria-label="预览降噪"]');
    await act(async () => preview?.click());
    expect(setDenoise).toHaveBeenCalledWith(clip.id, { ...denoise, previewEnabled: false });

    let rejectReapply!: (reason: Error) => void;
    prepare.mockImplementationOnce(
      () =>
        new Promise((_resolve, reject) => {
          rejectReapply = reject;
        }),
    );
    const reapplyButton = [...(section?.querySelectorAll("button") ?? [])].find(
      (button) => button.textContent === "重新应用",
    );
    await act(async () => {
      reapplyButton?.click();
      await Promise.resolve();
    });
    const cancelButton = [...(section?.querySelectorAll("button") ?? [])].find(
      (button) => button.textContent === "取消处理",
    );
    await act(async () => cancelButton?.click());
    expect(cancel).toHaveBeenCalledOnce();
    await act(async () => rejectReapply(new Error("denoise_cancelled")));
    expect(section?.querySelector('[role="alert"]')).toBeNull();

    const resetButton = [...(section?.querySelectorAll("button") ?? [])].find(
      (button) => button.textContent === "重置降噪",
    );
    await act(async () => resetButton?.click());
    expect(setDenoise).toHaveBeenLastCalledWith(clip.id, null);

    listen.mockRestore();
    prepare.mockRestore();
    setDenoise.mockRestore();
    cancel.mockRestore();
    await act(async () => root.unmount());
  });

  it("separates stems locally with privacy copy, progress, cancellation, and success", async () => {
    const clip = visualClip();
    useProjectStore.setState({ timeline: timelineWith(clip), projectPath: "/tmp/demo.opentake" });
    useMediaStore.setState({
      items: [{ id: clip.mediaRef, name: "mix.wav", type: "video", duration: 5, hasAudio: true }],
      folders: [],
      importing: false,
      error: null,
    });
    useEditorUiStore.setState({ selectedClipIds: new Set([clip.id]), inspectorTab: "audio" });
    const unlisten = vi.fn();
    const listen = vi.spyOn(api, "onStemSeparationProgress").mockImplementation(async (_id, handler) => {
      handler({ sourceAssetId: clip.mediaRef, done: 700, total: 1000 });
      return unlisten;
    });
    const separate = vi.spyOn(api, "separateAudioStems").mockResolvedValue({
      vocalsAssetId: "vocals",
      accompanimentAssetId: "music",
      sourceSha256: "a".repeat(64),
      execution: "local:opentake-center-v1",
      modelSha256: "b".repeat(64),
      vocalSdrImprovementDb: 60,
    });
    const cancel = vi.spyOn(api, "cancelStemSeparation").mockResolvedValue(true);
    const importTracks = vi.spyOn(api, "importStemsToTracks").mockResolvedValue({
      clipIds: ["vocal-clip", "music-clip"],
      actionName: "Import Stems To Tracks",
    });
    const undo = vi.spyOn(edit, "undo").mockResolvedValue({} as never);
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<Inspector />));

    const section = container.querySelector('[data-testid="stem-separation-section"]');
    expect(section?.textContent).toContain("音频不会上传");
    const button = [...(section?.querySelectorAll("button") ?? [])].find(
      (candidate) => candidate.textContent === "分离人声与伴奏",
    );
    await act(async () => {
      button?.click();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(listen).toHaveBeenCalledWith(clip.mediaRef, expect.any(Function));
    expect(separate).toHaveBeenCalledWith(clip.mediaRef, "local", null, null, false);
    expect(unlisten).toHaveBeenCalledOnce();
    expect(container.querySelector('[data-testid="stem-separation-result"]')?.textContent).toContain(
      "已将人声和伴奏添加到素材库",
    );
    await act(async () => useMediaStore.setState((state) => ({
      items: [
        ...state.items,
        { id: "vocals", name: "Vocals", type: "audio", duration: 5, hasAudio: true, path: "/tmp/vocals.wav" },
        { id: "music", name: "Music", type: "audio", duration: 5, hasAudio: true, path: "/tmp/music.wav" },
      ],
    })));
    expect(section?.querySelectorAll("audio")).toHaveLength(2);
    const importButton = [...(section?.querySelectorAll("button") ?? [])].find(
      (candidate) => candidate.textContent === "导入为两条对齐音轨",
    );
    await act(async () => importButton?.click());
    expect(importTracks).toHaveBeenCalledWith("vocals", "music", 42);
    const undoButton = [...(section?.querySelectorAll("button") ?? [])].find(
      (candidate) => candidate.textContent === "撤销音轨导入",
    );
    await act(async () => undoButton?.click());
    expect(undo).toHaveBeenCalledOnce();

    let rejectSeparation!: (reason: Error) => void;
    separate.mockImplementationOnce(
      () =>
        new Promise((_resolve, reject) => {
          rejectSeparation = reject;
        }),
    );
    await act(async () => {
      button?.click();
      await Promise.resolve();
    });
    const cancelButton = [...(section?.querySelectorAll("button") ?? [])].find(
      (candidate) => candidate.textContent === "取消分离",
    );
    await act(async () => cancelButton?.click());
    expect(cancel).toHaveBeenCalledOnce();
    await act(async () => rejectSeparation(new Error("stem_separation_cancelled")));
    expect(section?.querySelector('[role="alert"]')).toBeNull();

    listen.mockRestore();
    separate.mockRestore();
    cancel.mockRestore();
    importTracks.mockRestore();
    undo.mockRestore();
    await act(async () => root.unmount());
  });

  it("adds reorders adjusts toggles and removes generic effects through undoable commands", async () => {
    const clip = visualClip({
      effects: [
        { name: "grayscale", params: {}, enabled: true },
        { name: "invert", params: { amount: 0.5 }, enabled: true },
      ],
    });
    useProjectStore.setState({ timeline: timelineWith(clip), projectPath: "/tmp/demo.opentake" });
    useEditorUiStore.setState({ selectedClipIds: new Set([clip.id]), inspectorTab: "video" });
    const setEffects = vi.spyOn(edit, "setEffects").mockResolvedValue();
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<Inspector />));

    const section = container.querySelector('[data-testid="generic-effects-section"]');
    expect(section).not.toBeNull();
    const add = [...(section?.querySelectorAll("button") ?? [])].find(
      (button) => button.textContent === "添加效果",
    );
    await act(async () => add?.click());
    expect(setEffects).toHaveBeenLastCalledWith(
      [clip.id],
      expect.arrayContaining([expect.objectContaining({ name: "grayscale" })]),
    );

    let items = [...(section?.querySelectorAll<HTMLElement>('[data-testid="generic-effect-item"]') ?? [])];
    const moveUp = items[2]?.querySelector<HTMLButtonElement>('button[aria-label="上移效果"]');
    await act(async () => moveUp?.click());
    expect(setEffects.mock.calls.at(-1)?.[1].map((effect) => effect.name)).toEqual([
      "grayscale",
      "grayscale",
      "invert",
    ]);

    items = [...(section?.querySelectorAll<HTMLElement>('[data-testid="generic-effect-item"]') ?? [])];
    const amount = items[0]?.querySelector<HTMLInputElement>('input[type="range"]');
    await act(async () => {
      if (amount) {
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set?.call(
          amount,
          "0.35",
        );
      }
      amount?.dispatchEvent(new InputEvent("input", { bubbles: true }));
    });
    expect(setEffects.mock.calls.at(-1)?.[1][0]?.params.amount).not.toBe(0.35);
    await act(async () => amount?.dispatchEvent(new PointerEvent("pointerup", { bubbles: true })));
    expect(setEffects.mock.calls.at(-1)?.[1][0]?.params.amount).toBe(0.35);

    const enabled = items[0]?.querySelector<HTMLInputElement>('input[type="checkbox"]');
    await act(async () => enabled?.click());
    expect(setEffects.mock.calls.at(-1)?.[1][0]?.enabled).toBe(false);

    const remove = items[0]?.querySelector<HTMLButtonElement>('button[aria-label="删除效果"]');
    await act(async () => remove?.click());
    expect(setEffects.mock.calls.at(-1)?.[1]).toHaveLength(2);

    setEffects.mockRestore();
    await act(async () => root.unmount());
  });
  describe("generic effect amount slider", () => {
    async function renderAmountSlider(setEffects: ReturnType<typeof vi.spyOn>) {
      const clip = visualClip({ effects: [{ name: "grayscale", params: { amount: 0 }, enabled: true }] });
      useProjectStore.setState({ timeline: timelineWith(clip), projectPath: "/tmp/demo.opentake" });
      useEditorUiStore.setState({ selectedClipIds: new Set([clip.id]), inspectorTab: "video", toast: null });
      const container = document.createElement("div");
      document.body.append(container);
      const root = createRoot(container);
      await act(async () => root.render(<Inspector />));
      const slider = container.querySelector<HTMLInputElement>(
        '[data-testid="generic-effect-item"] input[type="range"]',
      )!;
      const setValue = (value: string, type: "input" | "change" = "input") => {
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set?.call(slider, value);
        slider.dispatchEvent(new Event(type, { bubbles: true }));
      };
      return { clip, container, root, slider, setValue, setEffects };
    }

    it("previews a drag locally and commits the final amount once on release", async () => {
      const { clip, container, root, slider, setValue, setEffects } = await renderAmountSlider(
        vi.spyOn(edit, "setEffects").mockResolvedValue(),
      );
      await act(async () => {
        for (let step = 1; step <= 20; step++) setValue(String(step / 20));
      });
      expect(setEffects).not.toHaveBeenCalled();
      expect(container.querySelector('[data-testid="generic-effect-item"]')?.textContent).toContain("100%");

      await act(async () => {
        slider.dispatchEvent(new Event("change", { bubbles: true }));
        slider.dispatchEvent(new PointerEvent("pointerup", { bubbles: true }));
      });
      expect(setEffects).toHaveBeenCalledOnce();
      expect(setEffects).toHaveBeenCalledWith([clip.id], [
        expect.objectContaining({ name: "grayscale", params: { amount: 1 } }),
      ]);

      setEffects.mockRestore();
      await act(async () => root.unmount());
    });

    it("commits a held arrow key once on key release instead of on every change", async () => {
      const { root, slider, setValue, setEffects } = await renderAmountSlider(
        vi.spyOn(edit, "setEffects").mockResolvedValue(),
      );
      await act(async () => {
        for (let step = 1; step <= 5; step++) {
          slider.dispatchEvent(
            new KeyboardEvent("keydown", { key: "ArrowRight", repeat: step > 1, bubbles: true }),
          );
          setValue(String(step / 100));
          setValue(String(step / 100), "change");
        }
      });
      expect(setEffects).not.toHaveBeenCalled();
      await act(async () => {
        slider.dispatchEvent(new KeyboardEvent("keyup", { key: "ArrowRight", bubbles: true }));
      });
      expect(setEffects).toHaveBeenCalledOnce();
      expect(setEffects.mock.calls[0]?.[1][0]?.params.amount).toBe(0.05);

      setEffects.mockRestore();
      await act(async () => root.unmount());
    });

    it("reports a rejected commit and restores the committed amount", async () => {
      const unhandled = vi.fn();
      const onUnhandled = (event: PromiseRejectionEvent) => unhandled(event.reason);
      window.addEventListener("unhandledrejection", onUnhandled);
      const stale = Object.assign(new Error("project changed"), { code: "staleProject" });
      const { container, root, slider, setValue, setEffects } = await renderAmountSlider(
        vi.spyOn(edit, "setEffects").mockRejectedValue(stale),
      );
      try {
        await act(async () => setValue("0.8"));
        await act(async () => slider.dispatchEvent(new PointerEvent("pointerup", { bubbles: true })));
        await act(async () => new Promise((resolve) => setTimeout(resolve, 0)));

        expect(setEffects).toHaveBeenCalledOnce();
        expect(useEditorUiStore.getState().toast?.message).toContain("project changed");
        expect(container.querySelector('[data-testid="generic-effect-item"]')?.textContent).toContain("0%");
        expect(unhandled).not.toHaveBeenCalled();
      } finally {
        window.removeEventListener("unhandledrejection", onUnhandled);
        setEffects.mockRestore();
        await act(async () => root.unmount());
      }
    });
  });
  describe("async section tasks follow their own target", () => {
    const audioClip = (overrides: Partial<Clip> = {}) =>
      visualClip({ id: "clip-a", mediaRef: "mix-a", ...overrides });

    function setMedia(...ids: string[]) {
      useMediaStore.setState({
        items: ids.map((id) => ({ id, name: `${id}.wav`, type: "video" as const, duration: 5, hasAudio: true })),
        folders: [],
        importing: false,
        error: null,
      });
    }

    async function renderAudioTab(clips: Clip[], selected: string) {
      useProjectStore.setState({
        timeline: { ...timelineWith(clips[0]!), tracks: [{ ...timelineWith(clips[0]!).tracks[0]!, clips }] },
        projectPath: "/tmp/demo.opentake",
      });
      useEditorUiStore.setState({ selectedClipIds: new Set([selected]), inspectorTab: "audio" });
      const container = document.createElement("div");
      document.body.append(container);
      const root = createRoot(container);
      await act(async () => root.render(<Inspector />));
      return { container, root };
    }

    const buttonIn = (container: HTMLElement, section: string, label: string) =>
      [...(container.querySelector(`[data-testid="${section}"]`)?.querySelectorAll("button") ?? [])].find(
        (button) => button.textContent === label,
      );

    it("drops a stem result that finishes after the clip's source media changed", async () => {
      setMedia("mix-a", "mix-b");
      const listen = vi.spyOn(api, "onStemSeparationProgress").mockResolvedValue(() => {});
      let finish!: (result: api.StemSeparationResult) => void;
      const separate = vi.spyOn(api, "separateAudioStems").mockImplementation(
        () => new Promise((resolve) => { finish = resolve; }),
      );
      const { container, root } = await renderAudioTab([audioClip()], "clip-a");
      await act(async () => {
        buttonIn(container, "stem-separation-section", "分离人声与伴奏")?.click();
        await Promise.resolve();
        await Promise.resolve();
      });
      expect(separate).toHaveBeenCalledWith("mix-a", "local", null, null, false);

      // Swap Media keeps the clip (and this section instance) but changes its source.
      await act(async () =>
        useProjectStore.setState({ timeline: timelineWith(audioClip({ mediaRef: "mix-b" })) }),
      );
      await act(async () => {
        finish({
          vocalsAssetId: "vocals-a",
          accompanimentAssetId: "music-a",
          sourceSha256: "a".repeat(64),
          execution: "local:opentake-center-v1",
          modelSha256: "b".repeat(64),
          vocalSdrImprovementDb: 60,
        });
        await Promise.resolve();
      });

      expect(container.querySelector('[data-testid="stem-separation-result"]')).toBeNull();
      expect(buttonIn(container, "stem-separation-section", "导入为两条对齐音轨")).toBeUndefined();
      expect(buttonIn(container, "stem-separation-section", "分离人声与伴奏")?.disabled).toBe(false);

      listen.mockRestore();
      separate.mockRestore();
      await act(async () => root.unmount());
    });

    it("keeps a stem result and a loudness failure on the clip that started them", async () => {
      setMedia("mix-a", "mix-b");
      const other = audioClip({ id: "clip-b", mediaRef: "mix-b", startFrame: 90 });
      vi.spyOn(api, "onStemSeparationProgress").mockResolvedValue(() => {});
      vi.spyOn(api, "onLoudnessProgress").mockResolvedValue(() => {});
      let finishStems!: (result: api.StemSeparationResult) => void;
      vi.spyOn(api, "separateAudioStems").mockImplementation(
        () => new Promise((resolve) => { finishStems = resolve; }),
      );
      let failLoudness!: (reason: Error) => void;
      vi.spyOn(edit, "analyzeAndApplyLoudness").mockImplementation(
        () => new Promise((_resolve, reject) => { failLoudness = reject; }),
      );
      const { container, root } = await renderAudioTab([audioClip(), other], "clip-a");
      await act(async () => {
        buttonIn(container, "stem-separation-section", "分离人声与伴奏")?.click();
        buttonIn(container, "loudness-section", "分析并应用")?.click();
        await Promise.resolve();
        await Promise.resolve();
      });

      await act(async () => useEditorUiStore.setState({ selectedClipIds: new Set([other.id]) }));
      await act(async () => {
        finishStems({
          vocalsAssetId: "vocals-a",
          accompanimentAssetId: "music-a",
          sourceSha256: "a".repeat(64),
          execution: "local:opentake-center-v1",
          modelSha256: "b".repeat(64),
          vocalSdrImprovementDb: 60,
        });
        failLoudness(new Error("decoder failed"));
        await Promise.resolve();
        await Promise.resolve();
      });

      expect(container.querySelector('[data-testid="stem-separation-result"]')).toBeNull();
      expect(container.querySelector('[data-testid="loudness-section"] [role="alert"]')).toBeNull();
      // The first clip's failed normalization is still reported, as a toast.
      expect(useEditorUiStore.getState().toast?.message).toContain("decoder failed");

      vi.restoreAllMocks();
      await act(async () => root.unmount());
    });
  });
});
