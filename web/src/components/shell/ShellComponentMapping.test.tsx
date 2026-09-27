import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

interface ExactOwner {
  owner: string;
  source: URL;
  liveBoundary: string;
  visibleEvidence: URL;
}

const exactOwners: ExactOwner[] = [
  {
    owner: "AiEditTab",
    source: new URL("../inspector/AiEditTab.tsx", import.meta.url),
    liveBoundary: "edit.setClipProperties",
    visibleEvidence: new URL("../inspector/AiEditTab.test.tsx", import.meta.url),
  },
  {
    owner: "MusicTab",
    source: new URL("../media/MusicTab.tsx", import.meta.url),
    liveBoundary: "addMediaToTimeline",
    visibleEvidence: new URL("../media/MusicTab.test.tsx", import.meta.url),
  },
  {
    owner: "TransitionTab",
    source: new URL("../media/TransitionTab.tsx", import.meta.url),
    liveBoundary: "setTransition",
    visibleEvidence: new URL("../media/TransitionTab.test.tsx", import.meta.url),
  },
  {
    owner: "TransformOverlay",
    source: new URL("../preview/TransformOverlay.tsx", import.meta.url),
    liveBoundary: "edit.setTransformAtFrame",
    visibleEvidence: new URL("../preview/TransformOverlay.interaction.test.tsx", import.meta.url),
  },
  {
    owner: "CropOverlay",
    source: new URL("../preview/CropOverlay.tsx", import.meta.url),
    liveBoundary: "edit.setClipProperties",
    visibleEvidence: new URL("../../lib/cropOverlay.test.ts", import.meta.url),
  },
];

describe("ShellComponentMapping", () => {
  it("shell_components_keep_their_live_editing_boundaries", () => {
    for (const entry of exactOwners) {
      const source = readFileSync(entry.source, "utf8");
      const evidence = readFileSync(entry.visibleEvidence, "utf8");
      expect(source).toContain(`export function ${entry.owner}`);
      expect(source).toContain(entry.liveBoundary);
      expect(evidence).toMatch(/describe\(|it\(/);
    }

    const transitionCanvasEvidence = readFileSync(
      new URL("../timeline/timelineOverlays.test.ts", import.meta.url),
      "utf8",
    );
    expect(transitionCanvasEvidence).toContain(
      "paints a cut marker for a valid adjacent cross dissolve",
    );
  });
});
