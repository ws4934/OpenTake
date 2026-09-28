/**
 * Project lifecycle gestures driven from the Home view. "New" starts a fresh
 * session and enters the editor; "Open" picks an `.opentake` bundle (a directory
 * on disk) via the native dialog, opens it in the core, records it in recents,
 * and enters the editor. All paths degrade gracefully outside Tauri so the
 * browser shell can still navigate into the editor.
 */

import * as api from "../lib/api";
import type { RuntimeTimelineSnapshot } from "../lib/types";
import { forceRefresh } from "./sync";
import { useEditorUiStore } from "./uiStore";
import { useProjectStore } from "./projectStore";
import { useRecentStore } from "./recentStore";
import { refreshMedia, resetProjectMediaState, useMediaStore } from "./mediaStore";
import { openDialog, saveDialog } from "../lib/dialog";
import { t } from "../i18n";
import { projectErrorMessage } from "../lib/projectMessages";
import { stopNativePlaybackForProjectBoundary } from "../components/preview/nativePlaybackSession";
import { useMotionStudioStore } from "./motionStudioStore";

const PROJECT_EXT = "opentake";

async function flushMotionStudioBeforeProjectBoundary(): Promise<void> {
  const motion = useMotionStudioStore.getState();
  await motion.flushSave();
  const latest = useMotionStudioStore.getState();
  if (
    latest.savingFile ||
    latest.conflict ||
    ["validating", "rendering", "encoding", "committing"].includes(latest.publishPhase) ||
    latest.dirtyFiles["index.html"] ||
    latest.dirtyFiles["styles.css"]
  ) {
    throw new Error(
      "Motion Studio 更改尚未保存，请先解决保存错误或版本冲突 / Motion Studio changes are not saved; resolve the save error or revision conflict first.",
    );
  }
}

function pathSeparator(path: string): "/" | "\\" {
  return path.lastIndexOf("\\") > path.lastIndexOf("/") ? "\\" : "/";
}

async function unusedDefaultProjectPath(defaultDir: string): Promise<string | undefined> {
  if (!defaultDir) return undefined;
  const joiner = defaultDir.endsWith("/") || defaultDir.endsWith("\\")
    ? ""
    : pathSeparator(defaultDir);
  const untitled = t("home.untitled");

  try {
    for (let ordinal = 1; ; ordinal += 1) {
      const suffix = ordinal === 1 ? "" : ` ${ordinal}`;
      const candidate = `${defaultDir}${joiner}${untitled}${suffix}.${PROJECT_EXT}`;
      if (!(await api.checkPathExists(candidate))) return candidate;
    }
  } catch {
    // An unknown candidate must not be passed to NSSavePanel because an
    // existing `.opentake` directory would be entered as a folder. Point the
    // panel at the containing directory instead and let it choose a safe name.
    return defaultDir;
  }
}

/**
 * New project. Mirrors upstream `AppState.createNewProject` (`NSSavePanel`):
 * prompt for a save location + name (default `~/Documents/OpenTake`), then
 * create the session and **immediately write the `.opentake` bundle to disk** so
 * the project has a real location (the user's complaint was "new project can't
 * choose where it saves"). Records it in recents and enters the editor.
 *
 * Outside Tauri (browser shell) there is no save panel — fall back to a fresh
 * in-memory session so the UI is still explorable.
 */
export async function newProjectAndEnter(): Promise<void> {
  try {
    const save = await saveDialog();
    if (!save) {
      await flushMotionStudioBeforeProjectBoundary();
      await saveCurrentProjectBeforeBoundary();
      await stopNativePlaybackForProjectBoundary();
      const snapshot = await api.projectNew(null);
      useProjectStore.getState().replaceProjectSnapshot(snapshot);
      resetProjectMediaState();
      await forceRefresh();
      useEditorUiStore.getState().resetProjectRuntimeState();
      useEditorUiStore.getState().setView("editor");
      return;
    }

    const defaultDir = await api.getDefaultProjectDir().catch(() => "");
    const defaultPath = await unusedDefaultProjectPath(defaultDir);

    const chosen = await save({
      title: t("home.newProject"),
      defaultPath,
      filters: [{ name: "OpenTake", extensions: [PROJECT_EXT] }],
    });
    if (typeof chosen !== "string") return; // cancelled

    // Pass the dialog result unchanged: the native dialog authorizes exactly
    // this path, and the backend appends `.opentake` beside it when the dialog
    // did not (GTK never does).
    const requestedPath = chosen;
    await flushMotionStudioBeforeProjectBoundary();
    await saveCurrentProjectBeforeBoundary();
    await stopNativePlaybackForProjectBoundary();
    // The desktop command persists a separate fresh session first and only
    // replaces the live project after that bundle can be reopened. A failed
    // initial save therefore leaves the current project and UI untouched.
    const snapshot = await api.projectNew(requestedPath);
    useProjectStore.getState().replaceProjectSnapshot(snapshot);
    resetProjectMediaState();
    await forceRefresh();
    const committedPath = snapshot.projectPath ?? requestedPath;
    useProjectStore.getState().markSaved();
    useRecentStore.getState().add(committedPath);
    useEditorUiStore.getState().resetProjectRuntimeState();
    useEditorUiStore.getState().setView("editor");
  } catch (error) {
    useEditorUiStore
      .getState()
      .pushToast(t("project.createFailed", { error: projectLifecycleErrorMessage(error) }));
    throw error;
  }
}

interface SaveSnapshot {
  snapshotMutationRevision: number;
  projectEpoch: number;
  projectPath: string;
  timelineVersion: number;
}

let saveInFlight: Promise<void> | null = null;
let activeSaveSnapshot: SaveSnapshot | null = null;
let queuedExplicitSave: SaveSnapshot | null = null;

function captureSaveSnapshot(): SaveSnapshot | null {
  const current = useProjectStore.getState();
  if (!current.projectPath) return null;
  return {
    snapshotMutationRevision: current.snapshotMutationRevision,
    projectEpoch: current.projectEpoch,
    projectPath: current.projectPath,
    timelineVersion: current.timelineVersion,
  };
}

function sameSnapshot(left: SaveSnapshot, right: SaveSnapshot): boolean {
  return (
    left.snapshotMutationRevision === right.snapshotMutationRevision &&
    left.projectEpoch === right.projectEpoch &&
    left.projectPath === right.projectPath &&
    left.timelineVersion === right.timelineVersion
  );
}

function sameProject(snapshot: SaveSnapshot): boolean {
  const current = useProjectStore.getState();
  return (
    current.projectEpoch === snapshot.projectEpoch && current.projectPath === snapshot.projectPath
  );
}

function currentProjectNeedsSave(): boolean {
  const current = useProjectStore.getState();
  return Boolean(current.projectPath) && current.timelineVersion !== current.lastSavedVersion;
}

/** Persist the open project before another session replaces it. The core only
 *  swaps sessions, and autosave's debounce (or an earlier failed save it never
 *  retries) would otherwise drop the latest edits silently. The coordinator
 *  reports failures without rejecting, so the dirty state is re-checked: if the
 *  save did not land, the boundary is refused and the current project stays. */
async function saveCurrentProjectBeforeBoundary(): Promise<void> {
  if (!currentProjectNeedsSave()) return;
  await saveCurrentProject();
  if (currentProjectNeedsSave()) throw new Error(t("project.unsavedBlocksSwitch"));
}

async function runSaveCoordinator(): Promise<void> {
  while (true) {
    const explicitRequest = queuedExplicitSave;
    queuedExplicitSave = null;
    const snapshot = captureSaveSnapshot();
    if (!snapshot) return;
    if (explicitRequest && !sameProject(explicitRequest)) return;
    activeSaveSnapshot = snapshot;

    try {
      const savedPath = await api.projectSave(
        null,
        snapshot.projectEpoch,
        snapshot.projectPath,
      );
      if (sameProject(snapshot)) {
        useRecentStore.getState().markSaved(savedPath || snapshot.projectPath);
      }
    } catch (error) {
      activeSaveSnapshot = null;
      const afterFailure = captureSaveSnapshot();
      const failureIsCurrent = Boolean(afterFailure && sameSnapshot(snapshot, afterFailure));
      if (failureIsCurrent) {
        useEditorUiStore
          .getState()
          .pushToast(t("project.saveFailed", { error: projectErrorMessage(error) }));
      }
      if (queuedExplicitSave) continue;
      if (failureIsCurrent) return;
      if (sameProject(snapshot) && currentProjectNeedsSave()) continue;
      return;
    }

    activeSaveSnapshot = null;
    const after = useProjectStore.getState();
    if (
      sameProject(snapshot) &&
      after.snapshotMutationRevision === snapshot.snapshotMutationRevision &&
      after.timelineVersion === snapshot.timelineVersion
    ) {
      after.markSaved(snapshot.timelineVersion);
    }
    if (queuedExplicitSave) continue;
    if (sameProject(snapshot) && currentProjectNeedsSave()) continue;
    return;
  }
}

/**
 * Save the open project back to its bundle (`project_save(None)`). Used by the
 * Cmd/Ctrl+S shortcut and the debounced autosave. Concurrent triggers share one
 * coordinator; if the document advances while a save is in flight, one fresh
 * save follows before the new version can be marked persisted. Completions are
 * bound to the initiating project identity so an old project cannot mark or
 * toast a newly opened project.
 */
export function saveCurrentProject(): Promise<void> {
  const request = captureSaveSnapshot();
  if (!request) return Promise.resolve();
  if (saveInFlight) {
    if (!activeSaveSnapshot || !sameSnapshot(request, activeSaveSnapshot)) {
      queuedExplicitSave = request;
    }
    return saveInFlight;
  }
  queuedExplicitSave = request;
  const run = runSaveCoordinator();
  const tracked = run.finally(() => {
    if (saveInFlight === tracked) saveInFlight = null;
  });
  saveInFlight = tracked;
  return tracked;
}

/** Save the current project to a newly chosen `.opentake` bundle and adopt that
 *  path as the live session. The core performs an atomic Save As (including
 *  project-local media) and only changes its retained root after publication
 *  succeeds; the front-end mirrors the returned canonical path afterwards.
 *  Overlapping gestures share one operation so two native publications cannot
 *  race the front-end's path ownership. */
let saveAsInFlight: Promise<void> | null = null;

export function saveCurrentProjectAs(): Promise<void> {
  if (saveAsInFlight) return saveAsInFlight;
  const run = runSaveCurrentProjectAs();
  const tracked = run.finally(() => {
    if (saveAsInFlight === tracked) saveAsInFlight = null;
  });
  saveAsInFlight = tracked;
  return tracked;
}

async function runSaveCurrentProjectAs(): Promise<void> {
  const project = useProjectStore.getState();
  const request = captureSaveSnapshot();
  if (!request || project.compatibilityReadOnly) return;
  const requestIsExactCurrent = () => {
    const current = captureSaveSnapshot();
    return Boolean(current && sameSnapshot(request, current));
  };
  const requestProjectIsCurrent = (committedPath?: string) => {
    const current = useProjectStore.getState();
    return (
      current.projectEpoch === request.projectEpoch &&
      (current.projectPath === request.projectPath || current.projectPath === committedPath)
    );
  };
  try {
    const save = await saveDialog();
    if (!save || !requestIsExactCurrent()) return;
    const selected = await save({
      title: t("menu.saveAs"),
      defaultPath: request.projectPath,
      filters: [{ name: "OpenTake", extensions: [PROJECT_EXT] }],
    });
    if (typeof selected !== "string" || !requestIsExactCurrent()) return;
    await flushMotionStudioBeforeProjectBoundary();
    if (!requestIsExactCurrent()) return;
    const committedPath = await api.projectSave(
      selected,
      request.projectEpoch,
      request.projectPath,
    );
    if (!requestProjectIsCurrent(committedPath)) return;
    const savedSnapshotIsCurrent = requestIsExactCurrent();
    const current = useProjectStore.getState();
    if (current.projectPath !== committedPath) current.setProjectPath(committedPath);
    if (savedSnapshotIsCurrent) {
      useProjectStore.getState().markSaved(request.timelineVersion);
    }
    useRecentStore.getState().add(committedPath);
  } catch (error) {
    if (requestProjectIsCurrent()) {
      useEditorUiStore.getState().pushToast(
        t("project.saveFailed", { error: projectLifecycleErrorMessage(error) }),
      );
    }
    throw error;
  }
}

/** Open `path` (a `.opentake` bundle), refresh the mirror, record it, and enter
 *  the editor. Used by both the dialog flow and the recents list. */
function projectLifecycleErrorMessage(error: unknown): string {
  return projectErrorMessage(error);
}

/** Keep the recoverable problems the core handled while opening a project
 *  for the notices banner, naming media as the opened manifest does. Later
 *  refreshes of the same project carry the same warnings but never come
 *  through here, so the notices are not repeated. */
function recordProjectOpenNotices(snapshot: RuntimeTimelineSnapshot): void {
  const warnings = snapshot.compatibilityWarnings ?? [];
  const mediaNames = Object.fromEntries(
    useMediaStore.getState().items.map((item) => [item.id, item.name]),
  );
  useProjectStore.getState().setOpenNotices(
    warnings.length === 0 ? null : { projectEpoch: snapshot.projectEpoch, warnings, mediaNames },
  );
}

export async function openProjectPath(path: string): Promise<void> {
  let snap: Awaited<ReturnType<typeof api.projectOpen>>;
  try {
    await flushMotionStudioBeforeProjectBoundary();
    await saveCurrentProjectBeforeBoundary();
    await stopNativePlaybackForProjectBoundary();
    snap = await api.projectOpen(path);
  } catch (error) {
    const message = projectLifecycleErrorMessage(error);
    useEditorUiStore.getState().pushToast(t("project.openFailed", { error: message }));
    throw error;
  }
  useProjectStore.getState().replaceProjectSnapshot(snap);
  resetProjectMediaState();
  useProjectStore.getState().markSaved();
  if (snap.projectPath) useRecentStore.getState().add(snap.projectPath);
  await refreshMedia();
  useEditorUiStore.getState().resetProjectRuntimeState();
  useEditorUiStore.getState().setView("editor");
  recordProjectOpenNotices(snap);
}

/** Pick a project bundle with the native dialog, then open it. `.opentake`
 *  bundles are directories, so the picker is a directory chooser (mirrors
 *  upstream's package-as-folder open panel). */
export async function openProjectViaDialog(): Promise<void> {
  let delegatedToProjectOpen = false;
  try {
    const open = await openDialog();
    if (!open) {
      // Browser shell: no file system. Just enter the editor on the demo mirror.
      useEditorUiStore.getState().setView("editor");
      return;
    }
    const selected = await open({ directory: true, multiple: false, recursive: true });
    if (typeof selected !== "string") return; // cancelled
    delegatedToProjectOpen = true;
    await openProjectPath(selected);
  } catch (error) {
    // openProjectPath reports its own downstream failures. Dialog acquisition
    // and picker failures happen before delegation, so report them here.
    if (!delegatedToProjectOpen) {
      useEditorUiStore.getState().pushToast(
        t("project.openFailed", { error: projectLifecycleErrorMessage(error) }),
      );
    }
    throw error;
  }
}

/** Materialize and open an independent, durable sample project and record it
 * in the recent-project registry. Tutorial routing is explicit so the Home card can
 * request the guided variant only after the project has opened successfully. */
export async function openSampleProject(slug: string, startTutorial: boolean): Promise<void> {
  try {
    await saveCurrentProjectBeforeBoundary();
    const path = await api.sampleProjectMaterialize(slug);
    await flushMotionStudioBeforeProjectBoundary();
    await saveCurrentProjectBeforeBoundary();
    await stopNativePlaybackForProjectBoundary();
    const snapshot = await api.projectOpen(path);
    useProjectStore.getState().replaceProjectSnapshot(snapshot);
    resetProjectMediaState();
    useProjectStore.getState().markSaved();
    useRecentStore.getState().add(path);
    await refreshMedia();
    useEditorUiStore.getState().resetProjectRuntimeState();
    useEditorUiStore.getState().setView("editor");
    if (startTutorial) {
      useEditorUiStore.getState().pushToast(t("home.tutorialStarted"));
    }
  } catch (error) {
    useEditorUiStore.getState().pushToast(
      t("home.sampleFailed", { error: projectLifecycleErrorMessage(error) }),
    );
    throw error;
  }
}
