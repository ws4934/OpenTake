import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ invoke: vi.fn() }));

vi.mock("@tauri-apps/api/core", () => ({ invoke: mocks.invoke }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn() }));

describe("project_save error envelope", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.stubGlobal("window", { __TAURI_INTERNALS__: {} });
    mocks.invoke.mockReset();
  });

  afterEach(() => vi.unstubAllGlobals());

  it("keeps the code and string params of a rejected save for translation", async () => {
    mocks.invoke.mockRejectedValueOnce({
      code: "projectComponentTooLarge",
      message: "generation-log.json is 17.0 MiB, more than the 16 MiB a project file may hold.",
      params: { file: "generation-log.json", sizeMib: "17.0", limitMib: "16", bogus: 3 },
    });
    const { projectSave, TauriCommandError } = await import("./api");
    const { projectErrorMessage } = await import("./projectMessages");
    const { useI18nStore } = await import("../i18n");
    useI18nStore.setState({ locale: "zh-CN" });

    const error = await projectSave(null, 1, "/tmp/a.opentake").catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(TauriCommandError);
    const commandError = error as InstanceType<typeof TauriCommandError>;
    expect(commandError.code).toBe("projectComponentTooLarge");
    // Non-string values are dropped rather than passed to the translation.
    expect(commandError.params).toEqual({
      file: "generation-log.json",
      sizeMib: "17.0",
      limitMib: "16",
    });
    expect(projectErrorMessage(error)).toBe(
      "generation-log.json 大小为 17.0 MiB，超过了项目文件允许的 16 MiB。",
    );
  });

  it("gives a rejection without params an empty map", async () => {
    mocks.invoke.mockRejectedValueOnce({ code: "projectIo", message: "The project files could not be read or written (other error)." });
    const { projectSave, TauriCommandError } = await import("./api");
    const error = await projectSave(null, 1, "/tmp/a.opentake").catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(TauriCommandError);
    expect((error as InstanceType<typeof TauriCommandError>).params).toEqual({});
  });
});
