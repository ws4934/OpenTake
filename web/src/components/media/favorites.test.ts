import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FavoriteSyncResult } from "../../lib/types";

const { syncProjectFavorites, getMedia } = vi.hoisted(() => ({
  syncProjectFavorites: vi.fn(),
  getMedia: vi.fn(),
}));
vi.mock("../../lib/api", () => ({ syncProjectFavorites, getMedia }));

function makeLocalStorage(): Storage {
  const map = new Map<string, string>();
  return {
    getItem: (key) => map.get(key) ?? null,
    setItem: (key, value) => void map.set(key, String(value)),
    removeItem: (key) => void map.delete(key),
    clear: () => map.clear(),
    key: (index) => [...map.keys()][index] ?? null,
    get length() {
      return map.size;
    },
  } as Storage;
}

const KEY = "opentake.favorites";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

async function setCurrentProject(projectEpoch: number, projectPath: string) {
  const { useProjectStore } = await import("../../store/projectStore");
  const project = { projectEpoch, projectPath };
  useProjectStore.setState(project);
  return project;
}

describe("migrateLocalFavorites", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  beforeEach(() => {
    vi.resetModules();
    vi.stubGlobal("localStorage", makeLocalStorage());
    syncProjectFavorites.mockReset().mockResolvedValue({
      media: { items: [], folders: [] },
      migratedLegacyAssetIds: [],
      failures: [],
    });
    getMedia.mockReset().mockResolvedValue({ items: [], folders: [] });
  });

  it("keeps a recovered result when a pre-completion favorite catalog arrives late", async () => {
    const { migrateLocalFavorites, applyFavoriteMigrationOutcome } = await import("./favorites");
    const { useMediaStore } = await import("../../store/mediaStore");
    const project = await setCurrentProject(31, "/recovered.opentake");
    const ready = { id: "held", name: "Held", type: "image" as const, duration: 0, hasAudio: false, generationStatus: "none" as const };
    const downloading = { ...ready, generationStatus: "downloading" as const, generationProgress: 0.8 };
    const pending = deferred<FavoriteSyncResult>();
    syncProjectFavorites.mockReturnValueOnce(pending.promise);
    const migration = migrateLocalFavorites([downloading], project);

    // Recovery publishes Ready while the favorite command's older DTO is in flight.
    useMediaStore.setState({ items: [ready] });
    getMedia.mockResolvedValueOnce({ items: [{ ...ready, favorite: true }], folders: [] });
    pending.resolve({ media: { items: [downloading], folders: [] }, migratedLegacyAssetIds: [], failures: [] });
    applyFavoriteMigrationOutcome(project, await migration);

    expect(useMediaStore.getState().items[0]).toMatchObject({ generationStatus: "none", favorite: true });
  });

  it("sends only current-project legacy ids and removes only confirmed migrations", async () => {
    localStorage.setItem(KEY, JSON.stringify(["a", "b", "other-project"]));
    syncProjectFavorites.mockResolvedValueOnce({
      media: { items: [], folders: [] },
      migratedLegacyAssetIds: ["a"],
      failures: [{ assetId: "b", message: "offline" }],
    });
    const project = await setCurrentProject(4, "/project-4.opentake");
    const { migrateLocalFavorites } = await import("./favorites");

    const outcome = await migrateLocalFavorites([{ id: "a" }, { id: "b" }], project);

    expect(syncProjectFavorites).toHaveBeenCalledWith(["a", "b"], project);
    expect(outcome.failures).toEqual([{ assetId: "b", message: "offline" }]);
    expect(JSON.parse(localStorage.getItem(KEY) as string)).toEqual(["b", "other-project"]);
  });

  it("reconciles manifest favorites once per project identity even without local data", async () => {
    const { migrateLocalFavorites } = await import("./favorites");
    const project10 = await setCurrentProject(10, "/project-10.opentake");

    expect((await migrateLocalFavorites([{ id: "a" }], project10)).synced).toBe(true);
    expect((await migrateLocalFavorites([{ id: "a" }], project10)).synced).toBe(false);
    const project11 = await setCurrentProject(11, "/project-11.opentake");
    expect((await migrateLocalFavorites([{ id: "a" }], project11)).synced).toBe(true);

    expect(syncProjectFavorites).toHaveBeenNthCalledWith(1, [], project10);
    expect(syncProjectFavorites).toHaveBeenNthCalledWith(2, [], project11);
  });

  it("retains local ids and allows retry when synchronization rejects", async () => {
    localStorage.setItem(KEY, JSON.stringify(["a"]));
    syncProjectFavorites.mockRejectedValueOnce(new Error("library unavailable"));
    const { migrateLocalFavorites } = await import("./favorites");
    const project = await setCurrentProject(7, "/project-7.opentake");

    await expect(migrateLocalFavorites([{ id: "a" }], project)).rejects.toThrow(
      "library unavailable",
    );
    await migrateLocalFavorites([{ id: "a" }], project);

    expect(syncProjectFavorites).toHaveBeenCalledTimes(2);
    expect(JSON.parse(localStorage.getItem(KEY) as string)).toEqual(["a"]);
  });

  it("retries a partial failure in the same project after the loop backoff", async () => {
    let now = 1000;
    vi.spyOn(Date, "now").mockImplementation(() => now);
    localStorage.setItem(KEY, JSON.stringify(["a"]));
    syncProjectFavorites
      .mockResolvedValueOnce({
        media: { items: [], folders: [] },
        migratedLegacyAssetIds: [],
        failures: [{ assetId: "a", message: "offline" }],
      })
      .mockResolvedValueOnce({
        media: { items: [], folders: [] },
        migratedLegacyAssetIds: ["a"],
        failures: [],
      });
    const project = await setCurrentProject(8, "/project-8.opentake");
    const { migrateLocalFavorites } = await import("./favorites");

    expect((await migrateLocalFavorites([{ id: "a" }], project)).failures).toHaveLength(1);
    expect((await migrateLocalFavorites([{ id: "a" }], project)).synced).toBe(false);
    now += 251;
    expect((await migrateLocalFavorites([{ id: "a" }], project)).failures).toHaveLength(0);

    expect(syncProjectFavorites).toHaveBeenCalledTimes(2);
    expect(localStorage.getItem(KEY)).toBeNull();
  });

  it("waits for the project media mirror before completing an identity with legacy ids", async () => {
    localStorage.setItem(KEY, JSON.stringify(["a"]));
    const { migrateLocalFavorites } = await import("./favorites");
    const project = await setCurrentProject(12, "/project-12.opentake");

    expect((await migrateLocalFavorites([], project)).synced).toBe(false);
    expect(syncProjectFavorites).not.toHaveBeenCalled();

    await migrateLocalFavorites([{ id: "a" }], project);
    expect(syncProjectFavorites).toHaveBeenCalledWith(["a"], project);
  });

  it("treats same-epoch bundle paths as distinct migration identities", async () => {
    const { migrateLocalFavorites } = await import("./favorites");
    const projectA = await setCurrentProject(20, "/A.opentake");
    await migrateLocalFavorites([{ id: "a" }], projectA);
    const projectB = await setCurrentProject(20, "/B.opentake");
    await migrateLocalFavorites([{ id: "a" }], projectB);

    expect(syncProjectFavorites).toHaveBeenCalledTimes(2);
  });

  it("does not complete or consume legacy ids for a superseded same-epoch path", async () => {
    const pending = deferred<{
      media: { items: []; folders: [] };
      migratedLegacyAssetIds: string[];
      failures: [];
    }>();
    syncProjectFavorites.mockReturnValueOnce(pending.promise);
    localStorage.setItem(KEY, JSON.stringify(["a"]));
    const projectA = await setCurrentProject(25, "/A.opentake");
    const { migrateLocalFavorites } = await import("./favorites");
    const staleMigration = migrateLocalFavorites([{ id: "a" }], projectA);
    const projectB = await setCurrentProject(25, "/B.opentake");

    pending.resolve({
      media: { items: [], folders: [] },
      migratedLegacyAssetIds: ["a"],
      failures: [],
    });

    expect((await staleMigration).synced).toBe(false);
    expect(JSON.parse(localStorage.getItem(KEY) as string)).toEqual(["a"]);
    await migrateLocalFavorites([{ id: "a" }], projectB);
    expect(syncProjectFavorites).toHaveBeenCalledTimes(2);
  });

  it("rejects a migration result from the previous path at the same epoch", async () => {
    const { applyFavoriteMigrationOutcome } = await import("./favorites");
    const { useMediaStore } = await import("../../store/mediaStore");
    const projectA = await setCurrentProject(30, "/A.opentake");
    await setCurrentProject(30, "/B.opentake");
    useMediaStore.setState({
      items: [{ id: "b", name: "B", type: "video", duration: 1, hasAudio: false }],
      folders: [{ id: "folder-b", name: "B" }],
      error: "B error",
    });

    expect(
      applyFavoriteMigrationOutcome(projectA, {
        synced: true,
        failures: [{ assetId: "late-a", message: "A failure" }],
      }),
    ).toBe(false);
    expect(useMediaStore.getState().items.map((item) => item.id)).toEqual(["b"]);
    expect(useMediaStore.getState().folders.map((folder) => folder.id)).toEqual(["folder-b"]);
    expect(useMediaStore.getState().error).toBe("B error");
  });
});
