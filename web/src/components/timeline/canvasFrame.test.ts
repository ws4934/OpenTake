import { describe, expect, it } from "vitest";
import { paintCanvasFrame } from "./canvasFrame";

function fakeCanvas(options: { context?: boolean } = {}) {
  const sizeWrites: string[] = [];
  const ops: string[] = [];
  let width = 300;
  let height = 150;
  const ctx = {
    save: () => ops.push("save"),
    restore: () => ops.push("restore"),
    clearRect: (...args: number[]) => ops.push(`clearRect ${args.join(",")}`),
  };
  const canvas = {
    get width() {
      return width;
    },
    set width(value: number) {
      sizeWrites.push(`width=${value}`);
      width = value;
    },
    get height() {
      return height;
    },
    set height(value: number) {
      sizeWrites.push(`height=${value}`);
      height = value;
    },
    style: { width: "", height: "" },
    getContext: () => (options.context === false ? null : ctx),
  };
  return { canvas: canvas as unknown as HTMLCanvasElement, ctx, sizeWrites, ops };
}

describe("paintCanvasFrame", () => {
  it("sizes the backing store only when the device size changes", () => {
    const { canvas, sizeWrites } = fakeCanvas();

    paintCanvasFrame(canvas, 640.5, 200, 1.5, () => {});
    expect(sizeWrites).toEqual(["width=961", "height=300"]);
    expect(canvas.style.width).toBe("640.5px");
    expect(canvas.style.height).toBe("200px");

    for (let frame = 0; frame < 3; frame += 1) paintCanvasFrame(canvas, 640.5, 200, 1.5, () => {});
    expect(sizeWrites).toHaveLength(2);

    paintCanvasFrame(canvas, 640.5, 200, 2, () => {});
    expect(sizeWrites).toEqual(["width=961", "height=300", "width=1281", "height=400"]);
  });

  it("clears the whole store and scopes context state to each frame", () => {
    const { canvas, ctx, ops } = fakeCanvas();

    paintCanvasFrame(canvas, 100, 50, 1.25, (painted) => {
      expect(painted).toBe(ctx);
      ops.push("paint");
    });

    expect(ops).toEqual(["save", "clearRect 0,0,125,63", "paint", "restore"]);
  });

  it("restores the context when painting throws", () => {
    const { canvas, ops } = fakeCanvas();

    expect(() =>
      paintCanvasFrame(canvas, 100, 50, 1, () => {
        throw new Error("paint failed");
      }),
    ).toThrow("paint failed");
    expect(ops[ops.length - 1]).toBe("restore");
  });

  it("sizes a canvas without a 2D context and skips painting", () => {
    const { canvas, sizeWrites } = fakeCanvas({ context: false });
    let painted = false;

    paintCanvasFrame(canvas, 100, 50, 1, () => {
      painted = true;
    });

    expect(sizeWrites).toEqual(["width=100", "height=50"]);
    expect(painted).toBe(false);
  });
});
