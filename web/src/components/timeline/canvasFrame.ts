/**
 * Size a canvas to a `cssWidth` x `cssHeight` box at `dpr` and paint one frame
 * into its 2D context. Assigning `width` or `height` reallocates (and clears)
 * the backing store even when the value is unchanged, so they are assigned
 * only when the size changes. Every frame instead starts from the canvas a
 * resize leaves: the whole store is cleared and `paint` runs between `save()`
 * and `restore()`, so no context state carries over to the next frame.
 */
export function paintCanvasFrame(
  canvas: HTMLCanvasElement,
  cssWidth: number,
  cssHeight: number,
  dpr: number,
  paint: (ctx: CanvasRenderingContext2D) => void,
): void {
  const width = Math.ceil(cssWidth * dpr);
  const height = Math.ceil(cssHeight * dpr);
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.save();
  try {
    ctx.clearRect(0, 0, width, height);
    paint(ctx);
  } finally {
    ctx.restore();
  }
}
