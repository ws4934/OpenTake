import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MediaList, Timeline } from "../lib/types";

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const srv = vi.hoisted(() => {
  const timeline: Timeline = {
    fps: 30,
    width: 1920,
    height: 1080,
    settingsConfigured: true,
    tracks: [],
  };
  const media: MediaList = {
    items: [
      {
        id: "m1",
        name: "clip",
        type: "video",
        duration: 10,
        hasAudio: true,
        path: "/tmp/clip.mov",
      },
    ],
    folders: [],
  };
  const order: string[] = [];
  return {
    timeline,
    media,
    order,
    createdPath: null as string | null,
    stopBoundary: vi.fn(async () => order.push("stop")),
    projectOpen: vi.fn(async () => {
      order.push("open");
      return {
        timeline,
        projectEpoch: 4,
        version: 7,
        projectPath: "/tmp/core-resolved.opentake",
        compatibilityReadOnly: false,
        compatibilityBlockers: [],
      };
    }),
    projectNew: vi.fn(async (path: string | null = null) => {
      order.push("new");
      srv.createdPath = path;
      return {
        timeline,
        projectEpoch: 5,
        version: 0,
        projectPath: path,
        compatibilityReadOnly: false,
        compatibilityBlockers: [],
      };
    }),
    projectSave: vi.fn(async (path: string | null) => path ?? ""),
    sampleProjectMaterialize: vi.fn(async () => "/tmp/data/Projects/quick-tutorial-copy/Tutorial.opentake"),
    getMedia: vi.fn(async () => media),
    openDialog: vi.fn(async () => undefined),
    save: vi.fn(async (..._args: unknown[]): Promise<string | null> => "/tmp/fresh.opentake"),
    saveDialog: vi.fn(async () => srv.save),
    message: vi.fn(async (..._args: unknown[]): Promise<string> => "Cancel"),
    messageDialog: vi.fn(async () => srv.message),
    lifecycleClaimFailedClose: vi.fn(async () => true),
    lifecycleResolveFailedClose: vi.fn(async (_id: number, _choice: string) => {}),
    getDefaultProjectDir: vi.fn(async () => ""),
    checkPathExists: vi.fn(async (_path: string) => false),
    // A refresh snapshot to return instead of the default one.
    refreshSnapshot: null as null | Record<string, unknown>,
  };
});

vi.mock("../lib/api", () => ({
  projectOpen: srv.projectOpen,
  projectNew: srv.projectNew,
  projectSave: srv.projectSave,
  sampleProjectMaterialize: srv.sampleProjectMaterialize,
  getDefaultProjectDir: srv.getDefaultProjectDir,
  checkPathExists: srv.checkPathExists,
  lifecycleClaimFailedClose: srv.lifecycleClaimFailedClose,
  lifecycleResolveFailedClose: srv.lifecycleResolveFailedClose,
  getTimeline: async () => srv.refreshSnapshot ?? ({
    timeline: srv.timeline,
    projectEpoch: 5,
    version: 0,
    projectPath: srv.createdPath,
    compatibilityReadOnly: false,
    compatibilityBlockers: [],
  }),
  canUndo: async () => false,
  canRedo: async () => false,
  getMedia: srv.getMedia,
  motionDocumentList: vi.fn(async () => []),
  motionDocumentCreate: vi.fn(async () => { throw new Error("unused"); }),
  motionDocumentRead: vi.fn(async () => { throw new Error("unused"); }),
  motionDocumentHash: vi.fn(async () => "0".repeat(64)),
  motionDocumentPatch: vi.fn(async () => { throw new Error("unused"); }),
  motionPreview: vi.fn(async () => { throw new Error("unused"); }),
  motionPreviewCancel: vi.fn(async () => false),
}));

vi.mock("../components/preview/nativePlaybackSession", () => ({
  stopNativePlaybackForProjectBoundary: srv.stopBoundary,
}));

vi.mock("../lib/dialog", () => ({
  saveDialog: srv.saveDialog,
  openDialog: srv.openDialog,
  messageDialog: srv.messageDialog,
}));

import {
  newProjectAndEnter,
  openProjectPath,
  openProjectViaDialog,
  openSampleProject,
  resolveFailedClose,
  saveCurrentProject,
  saveCurrentProjectAs,
} from "./projectActions";
import { useEditorUiStore } from "./uiStore";
import { useMediaStore } from "./mediaStore";
import { forceRefresh } from "./sync";
import { projectOpenNotices } from "../lib/projectMessages";
import { useProjectStore } from "./projectStore";
import { useRecentStore } from "./recentStore";
import { useI18nStore } from "../i18n";
import { useMotionStudioStore } from "./motionStudioStore";

const defaultMotionFlushSave = useMotionStudioStore.getState().flushSave;

beforeEach(() => {
  srv.message.mockReset();
  srv.message.mockResolvedValue("Cancel");
  srv.messageDialog.mockReset();
  srv.messageDialog.mockResolvedValue(srv.message);
  srv.lifecycleClaimFailedClose.mockReset();
  srv.lifecycleClaimFailedClose.mockResolvedValue(true);
  srv.lifecycleResolveFailedClose.mockReset();
  srv.lifecycleResolveFailedClose.mockResolvedValue(undefined);
  srv.save.mockReset();
  srv.save.mockResolvedValue("/tmp/fresh.opentake");
  srv.saveDialog.mockReset();
  srv.saveDialog.mockResolvedValue(srv.save);
  srv.getDefaultProjectDir.mockReset();
  srv.getDefaultProjectDir.mockResolvedValue("");
  srv.checkPathExists.mockReset();
  srv.checkPathExists.mockResolvedValue(false);
  useMotionStudioStore.setState({
    dirtyFiles: { "index.html": false, "styles.css": false },
    conflict: null,
    savingFile: null,
    publishPhase: "idle",
    flushSave: defaultMotionFlushSave,
  });
});

describe("newProjectAndEnter default path", () => {
  beforeEach(() => {
    srv.projectNew.mockClear();
    useEditorUiStore.setState({ view: "home", toast: null });
    useI18nStore.setState({ locale: "zh-CN" });
  });

  afterEach(() => {
    useProjectStore.setState({ projectEpoch: 0, projectPath: null, timelineVersion: 0 });
  });

  it("passes the first unused localized sibling to the native save panel", async () => {
    srv.getDefaultProjectDir.mockResolvedValue("/Users/qa/Documents/OpenTake");
    srv.checkPathExists
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce(false);
    srv.save.mockResolvedValueOnce(null);

    await newProjectAndEnter();

    expect(srv.checkPathExists.mock.calls.map(([path]) => path)).toEqual([
      "/Users/qa/Documents/OpenTake/未命名.opentake",
      "/Users/qa/Documents/OpenTake/未命名 2.opentake",
      "/Users/qa/Documents/OpenTake/未命名 3.opentake",
    ]);
    expect(srv.save).toHaveBeenCalledWith({
      title: "新建项目",
      defaultPath: "/Users/qa/Documents/OpenTake/未命名 3.opentake",
      filters: [{ name: "OpenTake", extensions: ["opentake"] }],
    });
    expect(srv.projectNew).not.toHaveBeenCalled();
  });

  it("preserves Windows separators when constructing the default path", async () => {
    useI18nStore.setState({ locale: "en" });
    srv.getDefaultProjectDir.mockResolvedValue("C:\\Users\\qa\\Documents\\OpenTake");
    srv.save.mockResolvedValueOnce(null);

    await newProjectAndEnter();

    expect(srv.checkPathExists).toHaveBeenCalledWith(
      "C:\\Users\\qa\\Documents\\OpenTake\\Untitled.opentake",
    );
    expect(srv.save).toHaveBeenCalledWith(expect.objectContaining({
      defaultPath: "C:\\Users\\qa\\Documents\\OpenTake\\Untitled.opentake",
    }));
  });

  it("falls back to the containing directory when existence probing fails", async () => {
    srv.getDefaultProjectDir.mockResolvedValue("/Users/qa/Documents/OpenTake");
    srv.checkPathExists.mockRejectedValueOnce(new Error("filesystem probe unavailable"));
    srv.save.mockResolvedValueOnce(null);

    await newProjectAndEnter();

    expect(srv.save).toHaveBeenCalledWith(expect.objectContaining({
      defaultPath: "/Users/qa/Documents/OpenTake",
    }));
    expect(srv.projectNew).not.toHaveBeenCalled();
    expect(useEditorUiStore.getState().toast).toBeNull();
  });

  it("keeps the browser fallback independent of native path probing", async () => {
    srv.saveDialog.mockResolvedValueOnce(null);

    await newProjectAndEnter();

    expect(srv.getDefaultProjectDir).not.toHaveBeenCalled();
    expect(srv.checkPathExists).not.toHaveBeenCalled();
    expect(srv.projectNew).toHaveBeenCalledWith(null);
    expect(useEditorUiStore.getState().view).toBe("editor");
  });
});

describe("openProjectPath", () => {
  beforeEach(() => {
    srv.order.length = 0;
    srv.createdPath = null;
    srv.stopBoundary.mockClear();
    srv.projectOpen.mockClear();
    srv.projectNew.mockReset();
    srv.projectNew.mockImplementation(async (path: string | null = null) => {
      srv.order.push("new");
      srv.createdPath = path;
      return {
        timeline: srv.timeline,
        projectEpoch: 5,
        version: 0,
        projectPath: path,
        compatibilityReadOnly: false,
        compatibilityBlockers: [],
      };
    });
    srv.getMedia.mockReset();
    srv.getMedia.mockImplementation(async () => srv.media);
    srv.projectSave.mockReset();
    srv.projectSave.mockImplementation(async (path: string | null) => path ?? "");
    srv.openDialog.mockReset();
    srv.openDialog.mockResolvedValue(undefined);
    useMediaStore.setState({ items: [], folders: [], importing: false, error: null });
    useRecentStore.setState({ recents: [] });
    useProjectStore.setState({ projectPath: null, projectEpoch: 0, timelineVersion: 0 });
    useEditorUiStore.setState({ view: "home", toast: null });
    useI18nStore.setState({ locale: "zh-CN" });
  });

  it("refreshes the media mirror after opening a project", async () => {
    await openProjectPath("/tmp/demo.opentake");

    expect(useProjectStore.getState().projectPath).toBe("/tmp/core-resolved.opentake");
    expect(useRecentStore.getState().recents[0]?.path).toBe("/tmp/core-resolved.opentake");
    expect(useMediaStore.getState().items.map((item) => item.id)).toEqual(["m1"]);
    expect(useEditorUiStore.getState().view).toBe("editor");
  });

  it("keeps open notices naming offline media until dismissed, not on later refreshes", async () => {
    const warnings = ["media.json:offline-media:m1", "media.json:offline-media:gone"];
    srv.projectOpen.mockImplementationOnce(async () => ({
      timeline: srv.timeline,
      projectEpoch: 11,
      version: 0,
      projectPath: "/tmp/unsafe.opentake",
      compatibilityReadOnly: false,
      compatibilityBlockers: [],
      compatibilityWarnings: warnings,
    }));

    await openProjectPath("/tmp/unsafe.opentake");

    const notices = useProjectStore.getState().openNotices;
    expect(notices).toEqual({ projectEpoch: 11, warnings, mediaNames: { m1: "clip" } });
    expect(projectOpenNotices(notices!.warnings, notices!.mediaNames)).toEqual([
      "以下素材的路径不安全，已作为离线素材打开，可重新链接：clip、gone",
    ]);
    // A notice is not a transient toast that could vanish or be replaced.
    expect(useEditorUiStore.getState().toast).toBeNull();

    // Dismissed, it stays dismissed: a refresh of the same project carries
    // the same warnings but never records them again.
    useProjectStore.getState().setOpenNotices(null);
    srv.refreshSnapshot = {
      timeline: structuredClone(srv.timeline),
      projectEpoch: 11,
      version: 1,
      projectPath: "/tmp/unsafe.opentake",
      compatibilityReadOnly: false,
      compatibilityBlockers: [],
      compatibilityWarnings: warnings,
    };
    try {
      await forceRefresh();
    } finally {
      srv.refreshSnapshot = null;
    }
    expect(useProjectStore.getState().timelineVersion).toBe(1);
    expect(useProjectStore.getState().openNotices).toBeNull();
    expect(useEditorUiStore.getState().toast).toBeNull();
  });

  it("tells the user an oversized generation log was moved aside", async () => {
    useI18nStore.setState({ locale: "en" });
    srv.projectOpen.mockImplementationOnce(async () => ({
      timeline: srv.timeline,
      projectEpoch: 12,
      version: 0,
      projectPath: "/tmp/big-log.opentake",
      compatibilityReadOnly: false,
      compatibilityBlockers: [],
      compatibilityWarnings: [
        "generation-log.json:moved-aside:generation-log.oversized-1.json",
      ],
    }));

    await openProjectPath("/tmp/big-log.opentake");

    const notices = useProjectStore.getState().openNotices!;
    expect(projectOpenNotices(notices.warnings, notices.mediaNames)).toEqual([
      "The generation log was too large; it was moved to generation-log.oversized-1.json inside the project and a new log was started.",
    ]);
  });

  it("opens a project without warnings silently", async () => {
    await openProjectPath("/tmp/demo.opentake");
    expect(useProjectStore.getState().openNotices).toBeNull();
    expect(useEditorUiStore.getState().toast).toBeNull();
  });

  it("waits for a debounced Motion Studio save before replacing project authority", async () => {
    const motionSave = deferred<void>();
    const flushSave = vi.fn(async () => {
      await motionSave.promise;
      useMotionStudioStore.setState({
        dirtyFiles: { "index.html": false, "styles.css": false },
      });
    });
    useMotionStudioStore.setState({
      dirtyFiles: { "index.html": true, "styles.css": false },
      conflict: null,
      savingFile: null,
      flushSave,
    });

    const opening = openProjectPath("/tmp/demo.opentake");
    await Promise.resolve();
    expect(flushSave).toHaveBeenCalledOnce();
    expect(srv.projectOpen).not.toHaveBeenCalled();

    motionSave.resolve();
    await opening;
    expect(srv.projectOpen).toHaveBeenCalledWith("/tmp/demo.opentake");
  });

  it("blocks a project boundary while Motion Studio has an unresolved conflict", async () => {
    useMotionStudioStore.setState({
      dirtyFiles: { "index.html": true, "styles.css": false },
      conflict: { file: "index.html", localSource: "<main>mine</main>" },
      savingFile: null,
      flushSave: vi.fn(async () => undefined),
    });

    await expect(openProjectPath("/tmp/demo.opentake")).rejects.toThrow("Motion Studio");
    expect(srv.projectOpen).not.toHaveBeenCalled();

    useMotionStudioStore.setState({
      dirtyFiles: { "index.html": false, "styles.css": false },
      conflict: null,
    });
  });

  it("blocks a project boundary while a Motion Studio publish is committing", async () => {
    useMotionStudioStore.setState({
      dirtyFiles: { "index.html": false, "styles.css": false },
      conflict: null,
      savingFile: null,
      publishPhase: "committing",
      flushSave: vi.fn(async () => undefined),
    });

    await expect(openProjectPath("/tmp/demo.opentake")).rejects.toThrow("Motion Studio");
    expect(srv.projectOpen).not.toHaveBeenCalled();
  });

  it("clears a media error from the previously open project", async () => {
    useMediaStore.getState().setError("old project import failed");

    await openProjectPath("/tmp/demo.opentake");

    expect(useMediaStore.getState().error).toBeNull();
  });

  it("grants recursive access when selecting an opentake bundle directory", async () => {
    const open = vi.fn(async () => "/tmp/demo.opentake");
    srv.openDialog.mockResolvedValueOnce(open);

    await openProjectViaDialog();

    expect(open).toHaveBeenCalledWith({
      directory: true,
      multiple: false,
      recursive: true,
    });
    expect(srv.projectOpen).toHaveBeenCalledWith("/tmp/demo.opentake");
  });

  it("reports a native picker failure before project-open delegation", async () => {
    srv.openDialog.mockRejectedValueOnce(new Error("picker unavailable"));

    await expect(openProjectViaDialog()).rejects.toThrow("picker unavailable");

    expect(useEditorUiStore.getState().toast?.message).toBe("打开失败：picker unavailable");
    expect(srv.projectOpen).not.toHaveBeenCalled();
  });

  it("preserves media transient state when project open fails", async () => {
    const oldFolder = { id: "old-folder", name: "Old", parentFolderId: null };
    useMediaStore.setState({
      items: srv.media.items,
      folders: [oldFolder],
      importing: true,
      error: "old project error",
    });
    const failure = { code: "engine", message: "project open timed out after 15s" };
    srv.projectOpen.mockRejectedValueOnce(failure);

    await expect(openProjectPath("/tmp/broken.opentake")).rejects.toBe(failure);

    expect(useMediaStore.getState().importing).toBe(true);
    expect(useMediaStore.getState().error).toBe("old project error");
    expect(useMediaStore.getState().items).toEqual(srv.media.items);
    expect(useMediaStore.getState().folders).toEqual([oldFolder]);
    expect(useEditorUiStore.getState().toast?.message).toBe(
      "打开失败：project open timed out after 15s",
    );
  });

  it("clears the old catalog immediately, then installs the opened project catalog", async () => {
    useMediaStore.setState({
      items: [{ ...srv.media.items[0]!, id: "old-item" }],
      folders: [{ id: "old-folder", name: "Old", parentFolderId: null }],
    });
    const nextCatalog = deferred<MediaList>();
    srv.getMedia.mockImplementationOnce(() => nextCatalog.promise);

    const opening = openProjectPath("/tmp/demo.opentake");
    await vi.waitFor(() => {
      expect(useProjectStore.getState().projectPath).toBe("/tmp/core-resolved.opentake");
    });
    expect(useMediaStore.getState().items).toEqual([]);
    expect(useMediaStore.getState().folders).toEqual([]);

    nextCatalog.resolve(srv.media);
    await opening;
    expect(useMediaStore.getState().items.map((item) => item.id)).toEqual(["m1"]);
  });

  it("never restores the old catalog when the opened project media refresh fails", async () => {
    useMediaStore.setState({
      items: [{ ...srv.media.items[0]!, id: "old-item" }],
      folders: [{ id: "old-folder", name: "Old", parentFolderId: null }],
    });
    srv.getMedia.mockRejectedValueOnce(new Error("media refresh failed"));

    await expect(openProjectPath("/tmp/demo.opentake")).rejects.toThrow("media refresh failed");

    expect(useProjectStore.getState().projectPath).toBe("/tmp/core-resolved.opentake");
    expect(useMediaStore.getState().items).toEqual([]);
    expect(useMediaStore.getState().folders).toEqual([]);
  });

  it("resets project-scoped UI runtime only after a successful project open", async () => {
    useEditorUiStore.setState({
      isPlaying: true,
      currentFrame: 91,
      activeFrame: 91,
      selectedClipIds: new Set(["old-clip"]),
      previewMediaId: "old-media",
      layoutPreset: "vertical",
      agentPanelVisible: false,
    });

    await openProjectPath("/tmp/reset.opentake");

    const ui = useEditorUiStore.getState();
    expect(ui.isPlaying).toBe(false);
    expect(ui.currentFrame).toBe(0);
    expect(ui.activeFrame).toBe(0);
    expect(ui.selectedClipIds.size).toBe(0);
    expect(ui.previewMediaId).toBeNull();
    expect(ui.layoutPreset).toBe("vertical");
    expect(ui.agentPanelVisible).toBe(false);
  });

  it("stops native playback before opening a project whose version collides", async () => {
    useProjectStore.setState({ projectEpoch: 3, timelineVersion: 7 });

    await openProjectPath("/tmp/collision.opentake");

    expect(srv.order.slice(0, 2)).toEqual(["stop", "open"]);
    expect(useProjectStore.getState().projectEpoch).toBe(4);
  });

  it("stops native playback before creating a fresh project", async () => {
    await newProjectAndEnter();

    expect(srv.order.slice(0, 2)).toEqual(["stop", "new"]);
    expect(srv.projectNew).toHaveBeenCalledWith("/tmp/fresh.opentake");
    expect(srv.projectSave).not.toHaveBeenCalled();
    expect(useProjectStore.getState().projectEpoch).toBe(5);
    expect(useProjectStore.getState().projectPath).toBe("/tmp/fresh.opentake");
  });

  it("passes an extensionless dialog path through and adopts the backend bundle path", async () => {
    srv.save.mockResolvedValueOnce("/home/qa/Videos/Vlog");
    srv.projectNew.mockImplementationOnce(async (path: string | null = null) => {
      srv.order.push("new");
      expect(path).toBe("/home/qa/Videos/Vlog");
      srv.createdPath = "/home/qa/Videos/Vlog.opentake";
      return {
        timeline: srv.timeline,
        projectEpoch: 5,
        version: 0,
        projectPath: "/home/qa/Videos/Vlog.opentake",
        compatibilityReadOnly: false,
        compatibilityBlockers: [],
      };
    });

    await newProjectAndEnter();

    expect(srv.saveDialog).toHaveBeenCalledWith("project");
    expect(srv.projectNew).toHaveBeenCalledWith("/home/qa/Videos/Vlog");
    expect(useProjectStore.getState().projectPath).toBe("/home/qa/Videos/Vlog.opentake");
    expect(useRecentStore.getState().recents[0]?.path).toBe("/home/qa/Videos/Vlog.opentake");
  });

  it("preserves the current project when initial creation fails", async () => {
    const oldTimeline: Timeline = {
      ...srv.timeline,
      tracks: [
        {
          id: "old-track",
          type: "video",
          muted: false,
          hidden: false,
          syncLocked: true,
          clips: [],
        },
      ],
    };
    useProjectStore.setState({
      projectEpoch: 3,
      timelineVersion: 12,
      timeline: oldTimeline,
      projectPath: "/tmp/current.opentake",
      lastSavedVersion: 12,
    });
    useMediaStore.setState({ items: srv.media.items, folders: [], error: "old media state" });
    const failure = { code: "engine", message: "project create timed out after 15s" };
    srv.projectNew.mockRejectedValueOnce(failure);

    await expect(newProjectAndEnter()).rejects.toBe(failure);

    const project = useProjectStore.getState();
    expect(project.projectEpoch).toBe(3);
    expect(project.timelineVersion).toBe(12);
    expect(project.timeline).toBe(oldTimeline);
    expect(project.projectPath).toBe("/tmp/current.opentake");
    expect(useMediaStore.getState().items).toEqual(srv.media.items);
    expect(useMediaStore.getState().error).toBe("old media state");
    expect(useEditorUiStore.getState().view).toBe("home");
    expect(useEditorUiStore.getState().toast?.message).toBe(
      "创建失败：project create timed out after 15s",
    );
    expect(srv.projectSave).not.toHaveBeenCalled();
  });
});

describe("openSampleProject", () => {
  beforeEach(() => {
    srv.order.length = 0;
    srv.sampleProjectMaterialize.mockClear();
    srv.projectOpen.mockClear();
    srv.projectOpen.mockImplementation(async () => ({
      timeline: srv.timeline,
      projectEpoch: 8,
      version: 2,
      projectPath: "/tmp/data/Projects/quick-tutorial-copy/Tutorial.opentake",
      compatibilityReadOnly: false,
      compatibilityBlockers: [],
    }));
    srv.getMedia.mockResolvedValue(srv.media);
    useRecentStore.setState({
      recents: [{ path: "/tmp/User.opentake", name: "User", openedAt: 1 }],
    });
    useEditorUiStore.setState({ view: "home", toast: null });
    useI18nStore.setState({ locale: "en" });
  });

  it("opens a durable tutorial copy and registers it in recent projects", async () => {
    await openSampleProject("quick-tutorial", true);

    expect(srv.sampleProjectMaterialize).toHaveBeenCalledWith("quick-tutorial");
    expect(srv.projectOpen).toHaveBeenCalledWith(
      "/tmp/data/Projects/quick-tutorial-copy/Tutorial.opentake",
    );
    expect(useRecentStore.getState().recents.map(({ name }) => name)).toEqual(["Tutorial", "User"]);
    expect(useProjectStore.getState().projectPath).not.toContain("/cache/");
    expect(useRecentStore.getState().recents[0].path).toBe(useProjectStore.getState().projectPath);
    expect(useEditorUiStore.getState().view).toBe("editor");
    expect(useEditorUiStore.getState().toast?.message).toContain("Tutorial project opened");
  });

  it("does not open or mutate recents when materialization fails", async () => {
    srv.sampleProjectMaterialize.mockRejectedValueOnce(new Error("download failed"));

    await expect(openSampleProject("quick-tutorial", true)).rejects.toThrow("download failed");

    expect(srv.projectOpen).not.toHaveBeenCalled();
    expect(useRecentStore.getState().recents.map(({ name }) => name)).toEqual(["User"]);
    expect(useEditorUiStore.getState().view).toBe("home");
    expect(useEditorUiStore.getState().toast?.message).toContain("download failed");
  });
});

describe("saveCurrentProject", () => {
  beforeEach(() => {
    srv.projectSave.mockReset();
    srv.projectSave.mockImplementation(async (path: string | null) => path ?? "");
    useProjectStore.setState({
      snapshotMutationRevision: 0,
      projectEpoch: 1,
      projectPath: "/tmp/unknown.opentake",
      timelineVersion: 9,
      lastSavedVersion: 8,
    });
    useEditorUiStore.setState({ toast: null });
    useI18nStore.setState({ locale: "zh-CN" });
  });

  it("surfaces a production-shaped string rejection and keeps the document dirty", async () => {
    srv.projectSave.mockRejectedValueOnce(
      "project is compatibility read-only because this build does not understand future fields",
    );

    await saveCurrentProject();

    expect(useEditorUiStore.getState().toast?.message).toBe(
      "保存失败：project is compatibility read-only because this build does not understand future fields",
    );
    expect(useProjectStore.getState().lastSavedVersion).toBe(8);
  });

  const saveFailures: Array<{
    code: string;
    params?: Record<string, string>;
    zh: string;
    en: string;
  }> = [
    {
      code: "projectStorageFull",
      zh: "磁盘空间不足，无法保存项目。",
      en: "There is not enough disk space to save the project.",
    },
    {
      code: "projectPermissionDenied",
      zh: "OpenTake 没有写入项目文件夹的权限。",
      en: "OpenTake is not allowed to write to the project folder.",
    },
    {
      code: "projectIo",
      params: { kind: "broken pipe" },
      zh: "无法读取或写入项目文件（broken pipe）。",
      en: "The project files could not be read or written (broken pipe).",
    },
    {
      code: "projectInvalidManifest",
      zh: "素材列表中有一个无法再次打开的文件路径，因此项目未保存。",
      en: "The media list contains a file path that could not be opened again, so the project was not saved.",
    },
    {
      code: "projectComponentTooLarge",
      params: { file: "generation-log.json", sizeMib: "17.0", limitMib: "16" },
      zh: "generation-log.json 大小为 17.0 MiB，超过了项目文件允许的 16 MiB。",
      en: "generation-log.json is 17.0 MiB, more than the 16 MiB a project file may hold.",
    },
    {
      code: "projectPartialCommit",
      zh: "时间线已保存，但素材列表更新失败。请再次保存以完成。",
      en: "The timeline was saved, but updating the media list failed. Save again to finish.",
    },
    {
      code: "projectDurabilityUnconfirmed",
      zh: "项目已保存，但磁盘未确认写入。请再次保存，确保断电后修改不会丢失。",
      en: "The project was saved, but the disk did not confirm the write. Save again to make sure the changes survive a power failure.",
    },
    {
      code: "projectRecoveryRequired",
      zh: "项目无法安全保存。旧版本已保留在项目旁一个以“.opentake-backup”结尾的隐藏文件夹中；在项目再次成功保存之前请保留它。",
      en: 'The project could not be saved safely. The previous version was kept next to the project in a hidden folder ending in ".opentake-backup"; keep it until the project saves successfully again.',
    },
  ];

  for (const failure of saveFailures) {
    it(`shows ${failure.code} in the active language`, async () => {
      for (const [locale, prefix, expected] of [
        ["zh-CN", "保存失败：", failure.zh],
        ["en", "Save failed: ", failure.en],
      ] as const) {
        useI18nStore.setState({ locale });
        useEditorUiStore.setState({ toast: null });
        srv.projectSave.mockRejectedValueOnce({
          code: failure.code,
          message: "English server message",
          ...(failure.params ? { params: failure.params } : {}),
        });

        await saveCurrentProject();

        expect(useEditorUiStore.getState().toast?.message).toBe(`${prefix}${expected}`);
      }
    });
  }

  it("keeps the server message for an unknown code or missing params", async () => {
    for (const rejection of [
      { code: "projectSomethingNew", message: "Something new failed." },
      { code: "projectComponentTooLarge", message: "project.json is 70.0 MiB, too large." },
      { code: "internal", message: "Project operation failed" },
    ]) {
      useEditorUiStore.setState({ toast: null });
      srv.projectSave.mockRejectedValueOnce(rejection);
      await saveCurrentProject();
      expect(useEditorUiStore.getState().toast?.message).toBe(`保存失败：${rejection.message}`);
    }
  });

  it("queues one follow-up save when the document advances during an in-flight save", async () => {
    const first = deferred<string>();
    const second = deferred<string>();
    srv.projectSave
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise);

    const saving = saveCurrentProject();
    useProjectStore.getState().replaceProjectSnapshot({
      timeline: srv.timeline,
      version: 10,
      projectEpoch: 1,
      projectPath: "/tmp/unknown.opentake",
      compatibilityReadOnly: false,
      compatibilityBlockers: [],
    });
    first.resolve("/tmp/unknown.opentake");

    await vi.waitFor(() => expect(srv.projectSave).toHaveBeenCalledTimes(2));
    expect(useProjectStore.getState().lastSavedVersion).toBe(8);

    second.resolve("/tmp/unknown.opentake");
    await saving;
    expect(useProjectStore.getState().lastSavedVersion).toBe(10);
  });

  it("suppresses a stale failure after switching projects", async () => {
    const first = deferred<string>();
    srv.projectSave.mockImplementationOnce(() => first.promise);

    const saving = saveCurrentProject();
    useProjectStore.getState().replaceProjectSnapshot({
      timeline: srv.timeline,
      projectEpoch: 2,
      version: 3,
      projectPath: "/tmp/new.opentake",
      compatibilityReadOnly: false,
      compatibilityBlockers: [],
    });
    first.reject("old project save failed");
    await saving;

    expect(useEditorUiStore.getState().toast).toBeNull();
    expect(useProjectStore.getState().lastSavedVersion).toBe(3);
  });

  it("coalesces overlapping autosave and keyboard requests", async () => {
    const first = deferred<string>();
    srv.projectSave.mockImplementationOnce(() => first.promise);

    const autosave = saveCurrentProject();
    const keyboardSave = saveCurrentProject();
    expect(srv.projectSave).toHaveBeenCalledTimes(1);

    first.resolve("/tmp/unknown.opentake");
    await Promise.all([autosave, keyboardSave]);
    expect(srv.projectSave).toHaveBeenCalledTimes(1);
    expect(useProjectStore.getState().lastSavedVersion).toBe(9);
  });

  it("queues an explicit save for a clean project opened during another project save", async () => {
    const first = deferred<string>();
    srv.projectSave
      .mockImplementationOnce(() => first.promise)
      .mockRejectedValueOnce("project B is compatibility read-only");

    const projectASave = saveCurrentProject();
    useProjectStore.getState().replaceProjectSnapshot({
      timeline: srv.timeline,
      projectEpoch: 2,
      version: 3,
      projectPath: "/tmp/project-b.opentake",
      compatibilityReadOnly: true,
      compatibilityBlockers: ["project.json:futureTimeline"],
    });
    const projectBSave = saveCurrentProject();
    first.resolve("/tmp/unknown.opentake");
    await Promise.all([projectASave, projectBSave]);

    expect(srv.projectSave).toHaveBeenCalledTimes(2);
    expect(useEditorUiStore.getState().toast?.message).toBe(
      "保存失败：project B is compatibility read-only",
    );
    expect(useProjectStore.getState().lastSavedVersion).toBe(3);
  });

  it("suppresses a failure after the initiating snapshot revision changes", async () => {
    const first = deferred<string>();
    srv.projectSave.mockImplementationOnce(() => first.promise);
    useProjectStore.setState({ lastSavedVersion: 9 });

    const saving = saveCurrentProject();
    useProjectStore.getState().replaceProjectSnapshot({
      timeline: srv.timeline,
      version: 9,
      projectEpoch: 1,
      projectPath: "/tmp/unknown.opentake",
      compatibilityReadOnly: false,
      compatibilityBlockers: [],
    });
    first.reject("stale save failure");
    await saving;

    expect(useEditorUiStore.getState().toast).toBeNull();
    expect(srv.projectSave).toHaveBeenCalledTimes(1);
  });

  it("does not redirect a stale queued request to a different dirty project", async () => {
    const first = deferred<string>();
    srv.projectSave.mockImplementationOnce(() => first.promise);

    const projectASave = saveCurrentProject();
    useProjectStore.getState().replaceProjectSnapshot({
      timeline: srv.timeline,
      projectEpoch: 2,
      version: 3,
      projectPath: "/tmp/project-b.opentake",
      compatibilityReadOnly: true,
      compatibilityBlockers: ["project.json:futureTimeline"],
    });
    const staleProjectBSave = saveCurrentProject();
    useProjectStore.getState().replaceProjectSnapshot({
      timeline: srv.timeline,
      projectEpoch: 3,
      version: 4,
      projectPath: "/tmp/project-c.opentake",
      compatibilityReadOnly: false,
      compatibilityBlockers: [],
    });
    useProjectStore.getState().replaceProjectSnapshot({
      timeline: srv.timeline,
      version: 5,
      projectEpoch: 3,
      projectPath: "/tmp/unknown.opentake",
      compatibilityReadOnly: false,
      compatibilityBlockers: [],
    });
    first.resolve("/tmp/unknown.opentake");
    await Promise.all([projectASave, staleProjectBSave]);

    expect(srv.projectSave).toHaveBeenCalledTimes(1);
    expect(useProjectStore.getState().lastSavedVersion).toBe(4);
    expect(useProjectStore.getState().timelineVersion).toBe(5);
  });
});

describe("saveCurrentProjectAs", () => {
  beforeEach(() => {
    srv.projectSave.mockReset();
    srv.projectSave.mockImplementation(async (path: string | null) => path ?? "");
    useRecentStore.setState({ recents: [] });
    useProjectStore.setState({
      snapshotMutationRevision: 0,
      projectEpoch: 1,
      projectPath: "/tmp/current.opentake",
      compatibilityReadOnly: false,
      timelineVersion: 9,
      lastSavedVersion: 8,
    });
    useEditorUiStore.setState({ toast: null });
    useI18nStore.setState({ locale: "zh-CN" });
  });

  it("adopts the core-returned Save As path only after publication succeeds", async () => {
    srv.projectSave.mockResolvedValueOnce("/tmp/canonical-fresh.opentake");

    await saveCurrentProjectAs();

    expect(srv.projectSave).toHaveBeenCalledWith(
      "/tmp/fresh.opentake",
      1,
      "/tmp/current.opentake",
    );
    expect(useProjectStore.getState().projectPath).toBe("/tmp/canonical-fresh.opentake");
    expect(useProjectStore.getState().lastSavedVersion).toBe(9);
    expect(useRecentStore.getState().recents[0]?.path).toBe(
      "/tmp/canonical-fresh.opentake",
    );
  });

  it("passes an extensionless Save As dialog path through unchanged", async () => {
    srv.save.mockResolvedValueOnce("/home/qa/Videos/Copy");
    srv.projectSave.mockResolvedValueOnce("/home/qa/Videos/Copy.opentake");

    await saveCurrentProjectAs();

    expect(srv.saveDialog).toHaveBeenCalledWith("project");
    expect(srv.projectSave).toHaveBeenCalledWith(
      "/home/qa/Videos/Copy",
      1,
      "/tmp/current.opentake",
    );
    expect(useProjectStore.getState().projectPath).toBe("/home/qa/Videos/Copy.opentake");
    expect(useRecentStore.getState().recents[0]?.path).toBe("/home/qa/Videos/Copy.opentake");
  });

  it("coalesces overlapping Save As gestures into one native publication", async () => {
    const publication = deferred<string>();
    srv.projectSave.mockImplementation(() => publication.promise);

    const first = saveCurrentProjectAs();
    await vi.waitFor(() => expect(srv.projectSave).toHaveBeenCalledOnce());
    const overlapping = saveCurrentProjectAs();
    publication.resolve("/tmp/canonical-fresh.opentake");
    await Promise.all([first, overlapping]);

    expect(overlapping).toBe(first);
    expect(srv.saveDialog).toHaveBeenCalledTimes(1);
    expect(srv.save).toHaveBeenCalledTimes(1);
    expect(srv.projectSave).toHaveBeenCalledTimes(1);
    expect(useProjectStore.getState().projectPath).toBe("/tmp/canonical-fresh.opentake");
  });

  it("adopts a completed Save As for the same project without marking newer edits saved", async () => {
    const publication = deferred<string>();
    srv.projectSave.mockImplementationOnce(() => publication.promise);

    const saving = saveCurrentProjectAs();
    await vi.waitFor(() => expect(srv.projectSave).toHaveBeenCalledOnce());
    useProjectStore.getState().replaceProjectSnapshot({
      timeline: srv.timeline,
      projectEpoch: 1,
      version: 10,
      projectPath: "/tmp/current.opentake",
      compatibilityReadOnly: false,
      compatibilityBlockers: [],
    });

    publication.resolve("/tmp/canonical-fresh.opentake");
    await saving;

    expect(useProjectStore.getState().projectPath).toBe("/tmp/canonical-fresh.opentake");
    expect(useProjectStore.getState().timelineVersion).toBe(10);
    expect(useProjectStore.getState().lastSavedVersion).toBe(8);
    expect(useRecentStore.getState().recents[0]?.path).toBe(
      "/tmp/canonical-fresh.opentake",
    );
  });

  it("accepts a same-epoch Save As completion when a refresh already mirrors its new path", async () => {
    const publication = deferred<string>();
    srv.projectSave.mockImplementationOnce(() => publication.promise);

    const saving = saveCurrentProjectAs();
    await vi.waitFor(() => expect(srv.projectSave).toHaveBeenCalledOnce());
    useProjectStore.getState().replaceProjectSnapshot({
      timeline: srv.timeline,
      projectEpoch: 1,
      version: 10,
      projectPath: "/tmp/canonical-fresh.opentake",
      compatibilityReadOnly: false,
      compatibilityBlockers: [],
    });

    publication.resolve("/tmp/canonical-fresh.opentake");
    await saving;

    expect(useProjectStore.getState().projectPath).toBe("/tmp/canonical-fresh.opentake");
    expect(useProjectStore.getState().lastSavedVersion).toBe(8);
    expect(useRecentStore.getState().recents[0]?.path).toBe(
      "/tmp/canonical-fresh.opentake",
    );
  });

  it("reports a Save As failure that still belongs to the same edited project", async () => {
    const publication = deferred<string>();
    srv.projectSave.mockImplementationOnce(() => publication.promise);

    const saving = saveCurrentProjectAs();
    await vi.waitFor(() => expect(srv.projectSave).toHaveBeenCalledOnce());
    useProjectStore.getState().replaceProjectSnapshot({
      timeline: srv.timeline,
      projectEpoch: 1,
      version: 10,
      projectPath: "/tmp/current.opentake",
      compatibilityReadOnly: false,
      compatibilityBlockers: [],
    });

    publication.reject(new Error("same-project publication failed"));
    await expect(saving).rejects.toThrow("same-project publication failed");

    expect(useProjectStore.getState().projectPath).toBe("/tmp/current.opentake");
    expect(useProjectStore.getState().lastSavedVersion).toBe(8);
    expect(useEditorUiStore.getState().toast?.message).toBe(
      "保存失败：same-project publication failed",
    );
  });

  it("does not adopt a stale Save As completion after the active project changes", async () => {
    const publication = deferred<string>();
    srv.projectSave.mockImplementationOnce(() => publication.promise);

    const saving = saveCurrentProjectAs();
    await vi.waitFor(() => expect(srv.projectSave).toHaveBeenCalledOnce());
    useProjectStore.getState().replaceProjectSnapshot({
      timeline: srv.timeline,
      projectEpoch: 2,
      version: 3,
      projectPath: "/tmp/project-b.opentake",
      compatibilityReadOnly: false,
      compatibilityBlockers: [],
    });

    publication.resolve("/tmp/stale-project-a-copy.opentake");
    await saving;

    expect(useProjectStore.getState().projectPath).toBe("/tmp/project-b.opentake");
    expect(useProjectStore.getState().lastSavedVersion).toBe(3);
    expect(useRecentStore.getState().recents).toEqual([]);
  });

  it("does not show a stale Save As failure on the replacement project", async () => {
    const publication = deferred<string>();
    srv.projectSave.mockImplementationOnce(() => publication.promise);

    const saving = saveCurrentProjectAs();
    await vi.waitFor(() => expect(srv.projectSave).toHaveBeenCalledOnce());
    useProjectStore.getState().replaceProjectSnapshot({
      timeline: srv.timeline,
      projectEpoch: 2,
      version: 3,
      projectPath: "/tmp/project-b.opentake",
      compatibilityReadOnly: false,
      compatibilityBlockers: [],
    });

    publication.reject(new Error("project A publication failed"));
    await expect(saving).rejects.toThrow("project A publication failed");

    expect(useProjectStore.getState().projectPath).toBe("/tmp/project-b.opentake");
    expect(useEditorUiStore.getState().toast).toBeNull();
  });

  it("preserves the active path and dirty state when Save As fails", async () => {
    srv.projectSave.mockRejectedValueOnce(new Error("destination denied"));

    await expect(saveCurrentProjectAs()).rejects.toThrow("destination denied");

    expect(useProjectStore.getState().projectPath).toBe("/tmp/current.opentake");
    expect(useProjectStore.getState().lastSavedVersion).toBe(8);
    expect(useRecentStore.getState().recents).toEqual([]);
    expect(useEditorUiStore.getState().toast?.message).toBe("保存失败：destination denied");
  });

  it("does not open a Save As flow for compatibility read-only projects", async () => {
    useProjectStore.setState({ compatibilityReadOnly: true });

    await saveCurrentProjectAs();

    expect(srv.projectSave).not.toHaveBeenCalled();
    expect(useProjectStore.getState().projectPath).toBe("/tmp/current.opentake");
  });
});

describe("project boundaries save the current project first", () => {
  const openers = {
    openProjectPath: () => openProjectPath("/tmp/other.opentake"),
    newProjectAndEnter: () => newProjectAndEnter(),
    openSampleProject: () => openSampleProject("quick-tutorial", false),
  } as const;
  const replacement = {
    openProjectPath: srv.projectOpen,
    newProjectAndEnter: srv.projectNew,
    openSampleProject: srv.projectOpen,
  } as const;

  beforeEach(() => {
    srv.projectOpen.mockClear();
    srv.projectNew.mockClear();
    srv.sampleProjectMaterialize.mockClear();
    srv.projectSave.mockReset();
    srv.projectSave.mockImplementation(async (path: string | null) => path ?? "/tmp/current.opentake");
    useProjectStore.setState({
      snapshotMutationRevision: 0,
      projectEpoch: 1,
      projectPath: "/tmp/current.opentake",
      timelineVersion: 7,
      lastSavedVersion: 5,
    });
    useEditorUiStore.setState({ view: "editor", toast: null });
    useI18nStore.setState({ locale: "zh-CN" });
  });

  afterEach(() => {
    useProjectStore.setState({ projectEpoch: 0, projectPath: null, timelineVersion: 0, lastSavedVersion: 0 });
  });

  it.each(Object.keys(openers) as Array<keyof typeof openers>)(
    "%s saves unsaved edits before replacing the session",
    async (name) => {
      await openers[name]();

      expect(srv.projectSave).toHaveBeenCalledWith(null, 1, "/tmp/current.opentake");
      expect(srv.projectSave.mock.invocationCallOrder[0]).toBeLessThan(
        replacement[name].mock.invocationCallOrder[0],
      );
    },
  );

  it("saves before materializing a sample project", async () => {
    await openSampleProject("quick-tutorial", false);

    expect(srv.projectSave.mock.invocationCallOrder[0]).toBeLessThan(
      srv.sampleProjectMaterialize.mock.invocationCallOrder[0],
    );
  });

  it("saves edits made during sample download before opening the new copy", async () => {
    srv.sampleProjectMaterialize.mockImplementationOnce(async () => {
      useProjectStore.setState({ timelineVersion: 8 });
      return "/tmp/data/Projects/quick-tutorial-copy/Tutorial.opentake";
    });

    await openSampleProject("quick-tutorial", false);

    expect(srv.projectSave).toHaveBeenCalledTimes(2);
    expect(srv.projectSave.mock.invocationCallOrder[1]).toBeGreaterThan(
      srv.sampleProjectMaterialize.mock.invocationCallOrder[0],
    );
    expect(srv.projectSave.mock.invocationCallOrder[1]).toBeLessThan(
      srv.projectOpen.mock.invocationCallOrder[0],
    );
  });

  it("keeps the current project if edits made during sample download cannot be saved", async () => {
    srv.sampleProjectMaterialize.mockImplementationOnce(async () => {
      useProjectStore.setState({ timelineVersion: 8 });
      return "/tmp/data/Projects/quick-tutorial-copy/Tutorial.opentake";
    });
    srv.projectSave.mockResolvedValueOnce("/tmp/current.opentake");
    srv.projectSave.mockRejectedValueOnce(new Error("disk full"));

    await expect(openSampleProject("quick-tutorial", false)).rejects.toThrow("当前工程的修改未能保存");

    expect(srv.projectOpen).not.toHaveBeenCalled();
    expect(useProjectStore.getState().projectPath).toBe("/tmp/current.opentake");
    expect(useProjectStore.getState().timelineVersion).toBe(8);
    expect(useProjectStore.getState().lastSavedVersion).toBe(7);
  });

  it.each(Object.keys(openers) as Array<keyof typeof openers>)(
    "%s keeps the current project when its save fails",
    async (name) => {
      srv.projectSave.mockRejectedValueOnce(new Error("disk full"));

      await expect(openers[name]()).rejects.toThrow("当前工程的修改未能保存");

      expect(replacement[name]).not.toHaveBeenCalled();
      expect(srv.sampleProjectMaterialize).not.toHaveBeenCalled();
      expect(useProjectStore.getState().projectPath).toBe("/tmp/current.opentake");
      expect(useProjectStore.getState().lastSavedVersion).toBe(5);
      expect(useEditorUiStore.getState().toast?.message).toContain("当前工程的修改未能保存");
    },
  );

  it.each(Object.keys(openers) as Array<keyof typeof openers>)(
    "%s offers Save As, Don't Save and Cancel after a failed save",
    async (name) => {
      srv.projectSave.mockRejectedValueOnce(new Error("volume unplugged"));

      await expect(openers[name]()).rejects.toThrow("当前工程的修改未能保存");

      expect(srv.message).toHaveBeenCalledOnce();
      const [, options] = srv.message.mock.calls[0] as [string, { buttons: unknown }];
      expect(options.buttons).toEqual({ yes: "另存为…", no: "不保存", cancel: "取消" });
      expect(replacement[name]).not.toHaveBeenCalled();
    },
  );

  it.each(Object.keys(openers) as Array<keyof typeof openers>)(
    "%s continues without saving or deleting after Don't Save",
    async (name) => {
      srv.projectSave.mockRejectedValue(new Error("bundle was deleted"));
      srv.message.mockResolvedValueOnce("不保存");

      await openers[name]();

      // Asked once even though a sample open checks the boundary twice.
      expect(srv.message).toHaveBeenCalledOnce();
      expect(replacement[name]).toHaveBeenCalled();
      // Only the failed in-place saves ran: no Save As, nothing else written.
      for (const [path] of srv.projectSave.mock.calls) expect(path).toBeNull();
    },
  );

  it("saves elsewhere with Save As and then switches", async () => {
    srv.projectSave.mockRejectedValueOnce(new Error("disk full"));
    srv.projectSave.mockResolvedValueOnce("/tmp/rescued.opentake");
    srv.message.mockResolvedValueOnce("Yes");
    srv.save.mockResolvedValueOnce("/tmp/rescued");

    await openProjectPath("/tmp/other.opentake");

    expect(srv.projectSave).toHaveBeenLastCalledWith(
      "/tmp/rescued",
      1,
      "/tmp/current.opentake",
    );
    expect(useRecentStore.getState().recents.map(({ path }) => path)).toContain(
      "/tmp/rescued.opentake",
    );
    expect(srv.projectOpen).toHaveBeenCalledWith("/tmp/other.opentake");
  });

  it("asks again when Save As is cancelled, and Cancel keeps the project", async () => {
    srv.projectSave.mockRejectedValueOnce(new Error("disk full"));
    srv.message.mockResolvedValueOnce("另存为…").mockResolvedValueOnce("取消");
    srv.save.mockResolvedValueOnce(null);

    await expect(openProjectPath("/tmp/other.opentake")).rejects.toThrow(
      "当前工程的修改未能保存",
    );

    expect(srv.message).toHaveBeenCalledTimes(2);
    expect(srv.projectOpen).not.toHaveBeenCalled();
    expect(useProjectStore.getState().projectPath).toBe("/tmp/current.opentake");
  });

  it.each(Object.keys(openers) as Array<keyof typeof openers>)(
    "%s does not save a clean project",
    async (name) => {
      useProjectStore.setState({ lastSavedVersion: 7 });

      await openers[name]();

      expect(srv.projectSave).not.toHaveBeenCalled();
      expect(replacement[name]).toHaveBeenCalled();
    },
  );
});

describe("resolveFailedClose", () => {
  beforeEach(() => {
    srv.projectSave.mockReset();
    srv.projectSave.mockImplementation(async (path: string | null) => path ?? "/tmp/current.opentake");
    useProjectStore.setState({
      snapshotMutationRevision: 0,
      projectEpoch: 1,
      projectPath: "/tmp/current.opentake",
      timelineVersion: 7,
      lastSavedVersion: 7,
      compatibilityReadOnly: false,
    });
    useI18nStore.setState({ locale: "en" });
  });

  afterEach(() => {
    useProjectStore.setState({ projectEpoch: 0, projectPath: null, timelineVersion: 0, lastSavedVersion: 0 });
  });

  it.each([
    ["Don't Save", "discard"],
    ["No", "discard"],
    ["Cancel", "cancel"],
  ])("answers %s with %s", async (button, choice) => {
    srv.message.mockResolvedValueOnce(button);

    await resolveFailedClose({ id: 7, intent: "exit", message: "No space left on device" });

    expect(srv.lifecycleClaimFailedClose).toHaveBeenCalledWith(7);
    expect(srv.message.mock.calls[0]?.[0]).toContain("No space left on device");
    expect(srv.lifecycleResolveFailedClose).toHaveBeenCalledWith(7, choice);
    expect(srv.projectSave).not.toHaveBeenCalled();
  });

  it("retries the close after a successful Save As", async () => {
    srv.message.mockResolvedValueOnce("Save As…");
    srv.save.mockResolvedValueOnce("/tmp/elsewhere.opentake");

    await resolveFailedClose({ id: 3, intent: "hide", message: "bundle was deleted" });

    expect(srv.projectSave).toHaveBeenCalledWith("/tmp/elsewhere.opentake", 1, "/tmp/current.opentake");
    expect(srv.lifecycleResolveFailedClose).toHaveBeenCalledWith(3, "retry");
  });

  it("hands the choice to the native prompt when the dialog cannot be shown", async () => {
    srv.message.mockRejectedValueOnce(new Error("dialog.message not allowed"));

    await resolveFailedClose({ id: 9, intent: "exit", message: "disk full" });

    expect(srv.lifecycleResolveFailedClose).toHaveBeenCalledWith(9, "native");
  });

  it("does not prompt when the native fallback already owns the choice", async () => {
    srv.lifecycleClaimFailedClose.mockResolvedValueOnce(false);

    await resolveFailedClose({ id: 2, intent: "exit", message: "disk full" });

    expect(srv.message).not.toHaveBeenCalled();
    expect(srv.lifecycleResolveFailedClose).not.toHaveBeenCalled();
  });
});
