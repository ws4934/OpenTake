/**
 * ScrubbableNumberField (SPEC §6.6). Warm-colored, right-aligned, tabular value.
 * Horizontal drag changes the value (Shift x10, Cmd x0.1); a 3px threshold
 * distinguishes drag from click; click switches to a text input (Enter/blur
 * commit, Esc cancel). `mixed` shows an em dash.
 *
 * Like upstream's field, a gesture previews its value locally (`onChange`) and
 * commits once when it ends: a drag on release, held Up/Down keys on key
 * release (or blur/unmount). A key gesture commits through the `onCommit` of
 * the render it started in, so its target (e.g. the keyframe frame) cannot
 * move while the key is held.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { LAYOUT } from "../../lib/theme";
import { useImeComposition } from "../../hooks/useImeComposition";
import { holdGestureCommit, runTimelineEdit } from "../../store/editActions";

interface Props {
  ariaLabel?: string;
  disabled?: boolean;
  value: number;
  mixed?: boolean;
  min: number;
  max: number;
  /** Display units changed per pixel of horizontal drag. */
  sensitivity: number;
  /** Format the numeric value into display text (without suffix handled here). */
  format: (v: number) => string;
  /** Inverse of `format` for typed text (suffix already stripped). Defaults to `Number`. */
  parse?: (text: string) => number | null;
  suffix?: string;
  width?: number;
  onChange?: (v: number) => void; // during a drag or key hold (optional live)
  /** Commit the gesture's final value. A returned promise that rejects is
   *  reported as an edit-failure toast. */
  onCommit: (v: number) => void | PromiseLike<unknown>;
  /** Override the rendered text (e.g. "-∞ dB" for the volume floor). */
  displayTextOverride?: (v: number) => string | null;
}

interface KeyGesture {
  startValue: number;
  value: number;
  commit: Props["onCommit"];
  /** Unregisters the flush that lets undo/redo land this gesture first. */
  release: () => void;
}

/** The last committed value while the mirror has not caught up with it (its
 *  edit is still in flight, or `value` has not refreshed yet). It is shown,
 *  and the next gesture starts from it instead of the stale mirrored value,
 *  so quick successive taps accumulate. */
interface CommitSeed {
  value: number;
  /** `value` prop when the seed was last committed. */
  basis: number;
  pending: number;
}

function isPromiseLike(value: unknown): value is PromiseLike<unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as PromiseLike<unknown>).then === "function"
  );
}

function isArrowStepKey(key: string): key is "ArrowUp" | "ArrowDown" {
  return key === "ArrowUp" || key === "ArrowDown";
}

export function ScrubbableNumberField(p: Props) {
  // Handlers read the latest props: a drag or edit that outlives a re-render
  // must report through the current callbacks, never the ones captured at mount.
  const latest = useRef(p);
  latest.current = p;
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  // Value previewed by an active drag or key gesture (null when idle).
  const [live, setLive] = useState<number | null>(null);
  // Re-render after a rejected commit drops the seed below.
  const [, setSeedRevision] = useState(0);
  // Text being edited (null when not editing). Committed on unmount, so an edit
  // interrupted by a selection switch lands on this instance's target.
  const draftRef = useRef<string | null>(null);
  const dragRef = useRef<{
    startX: number;
    startValue: number;
    moved: boolean;
    pointerId: number;
    captureTarget: HTMLElement;
  } | null>(null);
  const provisionalRef = useRef<number | null>(null);
  const keyGestureRef = useRef<KeyGesture | null>(null);
  const seedRef = useRef<CommitSeed | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const displayRef = useRef<HTMLSpanElement>(null);
  const restoreDisplayFocusRef = useRef(false);
  const ime = useImeComposition();

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus();
      inputRef.current?.select();
      return;
    }
    if (restoreDisplayFocusRef.current) {
      restoreDisplayFocusRef.current = false;
      displayRef.current?.focus();
    }
  }, [editing]);

  const clamp = (v: number) =>
    Math.max(latest.current.min, Math.min(latest.current.max, v));

  const seededValue = (value: number): number | null => {
    const seed = seedRef.current;
    return seed && (seed.pending > 0 || seed.basis === value) ? seed.value : null;
  };
  const seeded = seededValue(p.value);
  const shown = live ?? seeded ?? p.value;
  const text = (() => {
    if (p.mixed && live === null && seeded === null) return "—";
    const override = p.displayTextOverride?.(shown);
    if (override) return override;
    return p.format(shown) + (p.suffix ?? "");
  })();

  /** Commit `value` through `commit` (by default the current `onCommit`). An
   *  asynchronous commit seeds the display and the next gesture until the
   *  mirror catches up; its rejection is reported as an edit-failure toast. */
  const commitValue = useCallback(
    (value: number, commit: Props["onCommit"] = latest.current.onCommit) => {
      const basis = latest.current.value;
      const result = commit(value);
      // A synchronous commit already updated the parent's value.
      if (!isPromiseLike(result)) return;
      const seed = seedRef.current ?? { value, basis, pending: 0 };
      seed.value = value;
      seed.basis = basis;
      seed.pending += 1;
      seedRef.current = seed;
      runTimelineEdit(
        Promise.resolve(result).then(
          () => {
            seed.pending -= 1;
            if (seed.pending === 0 && seed.basis !== latest.current.value && seedRef.current === seed) {
              seedRef.current = null;
            }
          },
          (error: unknown) => {
            seed.pending -= 1;
            if (seedRef.current === seed && seed.value === value) {
              seedRef.current = null;
              setSeedRevision((revision) => revision + 1);
            }
            throw error;
          },
        ),
      );
    },
    [],
  );

  // A seed stops applying once nothing is in flight and the mirror moved on.
  useEffect(() => {
    const seed = seedRef.current;
    if (seed && seed.pending === 0 && seed.basis !== p.value) seedRef.current = null;
  }, [p.value]);

  /** End the arrow-key gesture and return its value, or null when none was
   *  active. A committed gesture lands its value (when it changed); a
   *  cancelled one restores the start value in the parent's live preview. */
  const finishKeyGesture = useCallback(
    (commit: boolean): number | null => {
      const gesture = keyGestureRef.current;
      keyGestureRef.current = null;
      if (!gesture) return null;
      gesture.release();
      setLive(null);
      if (!commit) {
        if (gesture.value !== gesture.startValue) latest.current.onChange?.(gesture.startValue);
        return gesture.startValue;
      }
      if (gesture.value !== gesture.startValue) commitValue(gesture.value, gesture.commit);
      return gesture.value;
    },
    [commitValue],
  );

  const stepKeyGesture = (direction: 1 | -1, modifier: number) => {
    const q = latest.current;
    let gesture = keyGestureRef.current;
    if (!gesture) {
      const start = seededValue(q.value) ?? q.value;
      gesture = {
        startValue: start,
        value: start,
        commit: q.onCommit,
        release: holdGestureCommit(() => void finishKeyGesture(true)),
      };
      keyGestureRef.current = gesture;
    }
    gesture.value = clamp(gesture.value + direction * q.sensitivity * modifier);
    setLive(gesture.value);
    q.onChange?.(gesture.value);
  };

  // Unmounting mid-gesture (e.g. the Inspector remounts for another clip)
  // commits to this field's own target, like upstream's onDisappear.
  useEffect(() => () => void finishKeyGesture(true), [finishKeyGesture]);

  const startEditing = (initial: string) => {
    draftRef.current = initial;
    setDraft(initial);
    setEditing(true);
  };

  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      if (p.disabled) return;
      e.preventDefault();
      // A pending key gesture lands first; the drag continues from its value.
      const keyed = finishKeyGesture(true);
      const captureTarget = e.currentTarget as HTMLElement;
      captureTarget.focus();
      dragRef.current = {
        startX: e.clientX,
        startValue: keyed ?? seededValue(p.value) ?? p.value,
        moved: false,
        pointerId: e.pointerId,
        captureTarget,
      };
      provisionalRef.current = null;
      captureTarget.setPointerCapture(e.pointerId);
    },
    [finishKeyGesture, p.disabled, p.value],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      const d = dragRef.current;
      if (!d) return;
      const dx = e.clientX - d.startX;
      if (!d.moved && Math.abs(dx) < LAYOUT.dragThreshold) return;
      d.moved = true;
      let mult = latest.current.sensitivity;
      if (e.shiftKey) mult *= 10;
      if (e.metaKey) mult *= 0.1;
      const next = clamp(d.startValue + dx * mult);
      provisionalRef.current = next;
      setLive(next);
      latest.current.onChange?.(next);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const onPointerUp = useCallback(
    (_e: React.PointerEvent) => {
      const d = dragRef.current;
      dragRef.current = null;
      if (!d) return;
      try {
        d.captureTarget.releasePointerCapture(d.pointerId);
      } catch {
        // Capture may already be gone when the browser ends the gesture.
      }
      setLive(null);
      const q = latest.current;
      if (q.disabled) {
        provisionalRef.current = null;
        return;
      }
      if (d.moved && provisionalRef.current !== null) {
        commitValue(provisionalRef.current);
        provisionalRef.current = null;
      } else {
        startEditing(q.format(d.startValue));
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const cancelPointer = useCallback(() => {
    const d = dragRef.current;
    const previewed = provisionalRef.current;
    dragRef.current = null;
    provisionalRef.current = null;
    if (!d) return;
    setLive(null);
    // Take back the parent's live preview of the abandoned drag.
    if (previewed !== null && previewed !== d.startValue) latest.current.onChange?.(d.startValue);
    try {
      d.captureTarget.releasePointerCapture(d.pointerId);
    } catch {
      // `lostpointercapture` means there is no capture left to release.
    }
    d.captureTarget.focus();
  }, []);

  useEffect(() => {
    if (!p.disabled) return;
    const d = dragRef.current;
    const previewed = provisionalRef.current;
    dragRef.current = null;
    provisionalRef.current = null;
    finishKeyGesture(false);
    setLive(null);
    if (d && previewed !== null && previewed !== d.startValue) {
      latest.current.onChange?.(d.startValue);
    }
    restoreDisplayFocusRef.current = false;
    if (d) {
      try {
        d.captureTarget.releasePointerCapture(d.pointerId);
      } catch {
        // The browser may already have released capture while disabling.
      }
    }
    draftRef.current = null;
    setEditing(false);
  }, [finishKeyGesture, p.disabled]);

  const commitDraft = useCallback(() => {
    const text = draftRef.current;
    draftRef.current = null;
    if (text === null) return;
    const q = latest.current;
    const cleaned = text.replace(q.suffix ?? "", "").replace(",", ".").trim();
    const parsed = q.parse ? q.parse(cleaned) : Number(cleaned);
    if (!q.disabled && parsed !== null && Number.isFinite(parsed)) commitValue(clamp(parsed));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [commitValue]);

  // Unmounting mid-edit (e.g. the Inspector remounts for another clip before
  // the input blurs) commits to this field's own target, like a blur would.
  useEffect(() => commitDraft, [commitDraft]);

  const finishEditing = useCallback((restoreFocus: boolean) => {
    draftRef.current = null;
    restoreDisplayFocusRef.current = restoreFocus;
    setEditing(false);
  }, []);

  const commitEdit = useCallback((restoreFocus: boolean) => {
    commitDraft();
    finishEditing(restoreFocus);
  }, [commitDraft, finishEditing]);

  const beginEditing = () => {
    if (p.disabled) return;
    const keyed = finishKeyGesture(true);
    startEditing(p.format(keyed ?? seededValue(p.value) ?? p.value));
  };

  if (editing) {
    return (
      <input
        ref={inputRef}
        aria-label={p.ariaLabel ?? "Value"}
        disabled={p.disabled}
        value={draft}
        onChange={(e) => {
          draftRef.current = e.target.value;
          setDraft(e.target.value);
        }}
        onBlur={() => commitEdit(false)}
        onKeyDown={(e) => {
          // Enter/Escape that confirm or cancel an IME candidate stay with the IME.
          if (ime.isComposingKeyDown(e)) return;
          if (e.key === "Enter") commitEdit(true);
          else if (e.key === "Escape") finishEditing(true);
        }}
        {...ime.compositionHandlers}
        className="tabular"
        style={{
          width: p.width ?? 56,
          textAlign: "right",
          background: "var(--bg-raised)",
          border: "var(--bw-thin) solid var(--border-primary)",
          borderRadius: "var(--radius-xs)",
          color: "var(--accent-primary)",
          fontSize: "var(--fs-sm)",
          padding: "1px 4px",
        }}
      />
    );
  }

  return (
    <span
      ref={displayRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={cancelPointer}
      onLostPointerCapture={cancelPointer}
      role="spinbutton"
      aria-label={p.ariaLabel ?? "Value"}
      aria-valuemin={p.min}
      aria-valuemax={p.max}
      aria-valuenow={p.mixed && live === null ? undefined : shown}
      aria-valuetext={text}
      aria-disabled={p.disabled || undefined}
      tabIndex={p.disabled ? -1 : 0}
      data-interaction-state={p.disabled ? "disabled" : "enabled"}
      onBlur={() => void finishKeyGesture(true)}
      onKeyDown={(e) => {
        if (p.disabled) return;
        if (e.key === "Escape" && (dragRef.current || keyGestureRef.current)) {
          e.preventDefault();
          finishKeyGesture(false);
          cancelPointer();
          return;
        }
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          beginEditing();
          return;
        }
        if (!isArrowStepKey(e.key)) return;
        e.preventDefault();
        const modifier = (e.shiftKey ? 10 : 1) * (e.metaKey || e.ctrlKey ? 0.1 : 1);
        stepKeyGesture(e.key === "ArrowUp" ? 1 : -1, modifier);
      }}
      onKeyUp={(e) => {
        if (isArrowStepKey(e.key)) finishKeyGesture(true);
      }}
      className="tabular"
      style={{
        width: p.width ?? 56,
        display: "inline-block",
        textAlign: "right",
        color: p.mixed && live === null ? "var(--text-tertiary)" : "var(--accent-primary)",
        fontSize: "var(--fs-sm)",
        cursor: p.disabled ? "not-allowed" : "ew-resize",
        userSelect: "none",
      }}
    >
      {text}
    </span>
  );
}
