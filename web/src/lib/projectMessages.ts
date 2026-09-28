/**
 * User-facing text for project persistence results. The core reports save
 * failures with stable `project*` codes and open-time notices as
 * file-qualified strings; this module turns both into text in the active
 * language, keeping the core's English message for anything it does not know.
 */

import { t, type TFunction } from "../i18n";

function isStringRecord(value: unknown): value is Record<string, string> {
  return (
    typeof value === "object" &&
    value !== null &&
    Object.values(value).every((entry) => typeof entry === "string")
  );
}

function rawMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "object" && error !== null && "message" in error) {
    const message = (error as { message?: unknown }).message;
    if (typeof message === "string") return message;
  }
  return String(error);
}

/** The message for a failed project command: the translation of its
 *  `project*` code when there is one, otherwise the core's own message. */
export function projectErrorMessage(error: unknown): string {
  if (typeof error === "object" && error !== null && "code" in error) {
    const { code, params } = error as { code?: unknown; params?: unknown };
    if (typeof code === "string" && code.startsWith("project")) {
      const key = `projectError.${code}`;
      const text = t(key, isStringRecord(params) ? params : undefined);
      // An unknown code, or one whose values did not arrive, keeps the
      // core's message rather than showing a key or a placeholder.
      if (text !== key && !/\{\w+\}/.test(text)) return text;
    }
  }
  return rawMessage(error);
}

/** One translated notice per kind of recoverable problem handled when a
 *  project was opened. `mediaNames` names media by id; an unknown id is
 *  shown as is. */
export function projectOpenNotices(
  warnings: readonly string[],
  mediaNames: Readonly<Record<string, string>>,
  translate: TFunction = t,
): string[] {
  const offline: string[] = [];
  const ignoredProxy: string[] = [];
  const notices: string[] = [];
  const nameOf = (id: string) => mediaNames[id] ?? id;
  for (const warning of warnings) {
    const offlineId = warning.match(/^media\.json:offline-media:(.+)$/)?.[1];
    const proxyId = warning.match(/^media\.json:ignored-proxy:(.+)$/)?.[1];
    const aside = warning.match(/^generation-log\.json:moved-aside:(.+)$/)?.[1];
    if (offlineId !== undefined) offline.push(nameOf(offlineId));
    else if (proxyId !== undefined) ignoredProxy.push(nameOf(proxyId));
    else if (aside !== undefined) {
      notices.push(translate("projectOpen.generationLogMovedAside", { file: aside }));
    } else notices.push(translate("projectOpen.otherNotice", { notice: warning }));
  }
  const separator = translate("projectOpen.nameSeparator");
  if (ignoredProxy.length > 0) {
    notices.unshift(translate("projectOpen.ignoredProxy", { names: ignoredProxy.join(separator) }));
  }
  if (offline.length > 0) {
    notices.unshift(translate("projectOpen.offlineMedia", { names: offline.join(separator) }));
  }
  return notices;
}
