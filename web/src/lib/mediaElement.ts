/**
 * Release a discarded `<video>`/`<audio>` element's decoder and buffered media
 * now instead of whenever it is garbage-collected: WebKit and Chromium keep
 * both until then, so elements that mount and unmount as the playhead or a
 * scrolled grid moves pile up. Runs in a microtask and skips an element that
 * is still in the document, so it only ever affects an element React removed
 * (not one a StrictMode effect re-run keeps mounted).
 */
export function releaseMediaElement(element: HTMLMediaElement): void {
  queueMicrotask(() => {
    if (element.isConnected) return;
    element.pause();
    element.removeAttribute("src");
    element.load();
  });
}
