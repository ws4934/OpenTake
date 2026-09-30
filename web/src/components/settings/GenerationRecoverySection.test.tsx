// @vitest-environment happy-dom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { GenerationRecoveryRecord } from "../../lib/types";
import { useEditorUiStore } from "../../store/uiStore";
import { t } from "../../i18n";
const api = vi.hoisted(() => ({ list: vi.fn(), discard: vi.fn(), open: vi.fn() }));
vi.mock("../../lib/api", () => ({ listGenerationRecoveries: api.list, discardGenerationRecovery: api.discard }));
vi.mock("../../store/projectActions", () => ({ openProjectViaDialog: api.open }));
import { GenerationRecoverySection } from "./GenerationRecoverySection";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const record: GenerationRecoveryRecord = { jobId: "saved", projectPath: "/old/Project.opentake", recordedAt: 1, resultCount: 1, byteSize: 1024, active: false, outcomeUnknown: false, discardIncomplete: false };
let host: HTMLDivElement;
let root: Root;
beforeEach(() => {
  vi.clearAllMocks();
  api.list.mockResolvedValue([record]);
  api.discard.mockResolvedValue([]);
  api.open.mockResolvedValue(true);
  useEditorUiStore.setState({ settingsOpen: true });
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => { await act(async () => root.unmount()); host.remove(); });
async function render() { await act(async () => root.render(<GenerationRecoverySection />)); }
function button(label: string) { return [...host.querySelectorAll<HTMLButtonElement>("button")].find(b => b.textContent === label)!; }

it("offers the original project in the native picker and uses the normal save boundary", async () => {
  await render();
  expect(host.textContent).toContain(record.projectPath);
  await act(async () => button(t("generation.recovery.open")).click());
  expect(api.open).toHaveBeenCalledWith(record.projectPath);
  expect(useEditorUiStore.getState().settingsOpen).toBe(false);
});
it("keeps recovery visible when the native project picker is cancelled", async () => {
  api.open.mockResolvedValue(false);
  await render();
  await act(async () => button(t("generation.recovery.open")).click());
  expect(useEditorUiStore.getState().settingsOpen).toBe(true);
  expect(host.textContent).toContain(record.projectPath);
});
it("requires a second explicit action before deleting saved results", async () => {
  await render();
  await act(async () => button(t("generation.recovery.discard")).click());
  expect(api.discard).not.toHaveBeenCalled();
  await act(async () => button(t("common.cancel")).click());
  expect(api.discard).not.toHaveBeenCalled();
  await act(async () => button(t("generation.recovery.discard")).click());
  await act(async () => button(t("generation.recovery.confirmDiscard")).click());
  expect(api.discard).toHaveBeenCalledWith(record.jobId, true);
  expect(host.textContent).toContain(t("generation.recovery.empty"));
});
it("keeps failed cleanup available for a retry", async () => {
  api.discard.mockRejectedValue(new Error("disk refused cleanup"));
  await render();
  await act(async () => button(t("generation.recovery.discard")).click());
  await act(async () => button(t("generation.recovery.confirmDiscard")).click());
  expect(host.querySelector('[role="alert"]')?.textContent).toContain("disk refused cleanup");
  expect(button(t("generation.recovery.confirmDiscard")).disabled).toBe(false);
});
it("does not offer removal while a paid submission is still active", async () => {
  api.list.mockResolvedValue([{ ...record, active: true }]);
  await render();
  expect(button(t("generation.recovery.discard")).disabled).toBe(true);
  expect(host.textContent).toContain(t("generation.recovery.active"));
});
it("surfaces a damaged store instead of an empty successful report", async () => {
  api.list.mockRejectedValue(new Error("invalid recovery file"));
  await render();
  expect(host.querySelector('[role="alert"]')?.textContent).toContain("invalid recovery file");
  expect(host.textContent).not.toContain(t("generation.recovery.empty"));
});
it("refreshes a task that has finished while settings stayed open", async () => {
  api.list.mockResolvedValueOnce([{ ...record, active: true }]).mockResolvedValueOnce([record]);
  await render();
  expect(button(t("generation.recovery.discard")).disabled).toBe(true);
  await act(async () => button(t("generation.recovery.refresh")).click());
  expect(api.list).toHaveBeenCalledTimes(2);
  expect(button(t("generation.recovery.discard")).disabled).toBe(false);
});
