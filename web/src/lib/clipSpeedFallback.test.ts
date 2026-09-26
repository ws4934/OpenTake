import { describe, expect, it } from "vitest";
import { createFallbackStore } from "./fallback";
import type { Clip } from "./types";

function fixture(starts = [0, 30], linkedAudio = false) {
  const fallback = createFallbackStore();
  const template = fallback.getTimeline().timeline.tracks[0].clips[0];
  fallback.reset();
  const videoTrack = fallback.editApply({ type: "insertTrack", kind: "video" }).affectedClipIds[0];
  const animated = (id: string, start: number, audio = false): Clip => ({
    ...structuredClone(template), id, startFrame: start, durationFrames: 30,
    trimStartFrame: 0, trimEndFrame: 0, speed: 1,
    mediaType: audio ? "audio" : "video", sourceClipType: audio ? "audio" : "video",
    linkGroupId: linkedAudio && start === 0 ? "av" : undefined,
    transitionOut: undefined, fadeInFrames: 10, fadeOutFrames: 20,
    opacityTrack: { keyframes: [
      { frame: 0, value: 0, interpolationOut: "linear" },
      { frame: 30, value: 1, interpolationOut: "linear" },
    ] },
  });
  const entries = starts.map((start, index) => ({
    clip: animated(`video-${index}`, start), targetTrackId: videoTrack, startFrame: start,
  }));
  if (linkedAudio) {
    const audioTrack = fallback.editApply({ type: "insertTrack", kind: "audio" }).affectedClipIds[0];
    entries.push(...starts.map((start, index) => ({
      clip: animated(`audio-${index}`, start, true), targetTrackId: audioTrack, startFrame: start,
    })));
  }
  const pasted = fallback.editApply({ type: "pasteClips", entries });
  expect(pasted.changed).toBe(true);
  return { fallback, ids: pasted.affectedClipIds };
}

describe("browser demo retime boundary", () => {
  it.each([[2, 15], [0.5, 60]])("retimes %sx animation and linked following chains", (speed, duration) => {
    const { fallback, ids } = fixture([0, 30], true);
    const result = fallback.editApply({ type: "setClipSpeed", clipIds: [ids[0]], speed, ripple: true });
    expect(result.changed).toBe(true);
    for (const track of fallback.getTimeline().timeline.tracks) {
      expect(track.clips[0].durationFrames).toBe(duration);
      expect(track.clips[0].opacityTrack?.keyframes.map(({ frame, value }) => [frame, value]))
        .toEqual([[0, 0], [duration, 1]]);
      expect(track.clips[1].startFrame).toBe(duration);
      expect(track.clips[0].fadeOutFrames).toBe(speed === 2 ? 5 : 20);
    }
  });

  it("rejects a later target collision without partially publishing the candidate", () => {
    const { fallback, ids } = fixture([0, 30, 100]);
    const before = fallback.getTimeline();
    const result = fallback.editApply({ type: "setClipSpeed", clipIds: ids.slice(0, 2), speed: 0.5, ripple: true });
    expect(result.changed).toBe(false);
    expect(fallback.getTimeline()).toEqual(before);
  });

  it("leaves gapped clips in place and preserves versions on no-op or invalid edits", () => {
    const { fallback, ids } = fixture([0, 45]);
    const before = fallback.getTimeline();
    for (const speed of [1, 0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(fallback.editApply({ type: "setClipSpeed", clipIds: [ids[0]], speed, ripple: true }).changed).toBe(false);
      expect(fallback.getTimeline()).toEqual(before);
    }
    fallback.editApply({ type: "setClipSpeed", clipIds: [ids[0]], speed: 2, ripple: true });
    expect(fallback.getTimeline().timeline.tracks[0].clips[1].startFrame).toBe(45);
  });

  it("processes reversed and duplicate selections only once at their current positions", () => {
    const { fallback, ids } = fixture([0, 30, 60]);
    fallback.editApply({ type: "setClipSpeed", clipIds: [ids[1], ids[0], ids[1]], speed: 2, ripple: true });
    expect(fallback.getTimeline().timeline.tracks[0].clips.map((clip) => [clip.startFrame, clip.durationFrames]))
      .toEqual([[0, 15], [15, 15], [30, 30]]);
  });

  it.each([{ speed: 2 }, { durationFrames: 15 }])("generic timing %j rescales animation without ripple", (properties) => {
    const { fallback, ids } = fixture();
    fallback.editApply({ type: "setClipProperties", clipIds: [ids[0]], properties });
    const clips = fallback.getTimeline().timeline.tracks[0].clips;
    expect(clips[0].opacityTrack?.keyframes.map(({ frame, value }) => [frame, value])).toEqual([[0, 0], [15, 1]]);
    expect(clips[1].startFrame).toBe(30);
  });
});
