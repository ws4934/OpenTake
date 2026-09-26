/** Controls that never own text/value editing keys. All other input types,
 * including number and date/time segments, must keep their native shortcuts. */
const NON_TEXT_INPUT_TYPES: ReadonlySet<string> = new Set([
  "button", "checkbox", "color", "file", "hidden", "image", "radio",
  "range", "reset", "submit",
]);

/** Shared boundary for DOM shortcuts and native application-menu commands. */
export function isTextEntry(target: EventTarget | null): boolean {
  if (typeof Element === "undefined" || !(target instanceof Element)) return false;
  const editable = target.closest<HTMLElement>("input, textarea, [contenteditable]");
  if (!editable) return false;
  if (editable.isContentEditable || editable.matches("textarea")) return true;
  return editable.matches("input") &&
    !NON_TEXT_INPUT_TYPES.has((editable as HTMLInputElement).type);
}

export function hasTextEntryFocus(): boolean {
  return typeof document !== "undefined" && isTextEntry(document.activeElement);
}
