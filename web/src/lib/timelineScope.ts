import type { Timeline } from "./types";

/** The timeline the editor shows and plays: the open nested sequence (compound
 *  clip) named by `activeNestedSequenceId`, else the root timeline. Pure, so
 *  the stores, the playback engine and the menus share it without importing
 *  one another. */
export function currentTimelineOf(
  root: Timeline,
  activeNestedSequenceId: string | null,
): Timeline {
  if (activeNestedSequenceId === null) return root;
  return (
    root.nestedSequences?.find((sequence) => sequence.id === activeNestedSequenceId)
      ?.timeline ?? root
  );
}
