/**
 * TextTab (SPEC §6.3). Inspector tab for text clips. Edits `textContent` (text
 * box) plus the full `textStyle` (font / size / color / alignment / background /
 * border / shadow). Text commits on blur; discrete style controls commit
 * immediately via SetClipProperties (the backend writes `clip.text_style`, the
 * render layer re-rasterizes the text box on the next `timeline_changed`).
 * Color pickers report continuously while dragging, so a pick previews locally
 * and commits once, like upstream's `debouncedCommitTextStyles`.
 */

import { useEffect, useRef, useState } from "react";
import { AlignCenter, AlignLeft, AlignRight, type LucideIcon } from "lucide-react";
import * as edit from "../../store/editActions";
import { Icon } from "../ui/Icon";
import { ScrubbableNumberField } from "./ScrubbableNumberField";
import { RADIUS, SPACE } from "../../lib/theme";
import type { TFunction } from "../../i18n";
import type { Clip, Rgba, TextAlignment, TextStyle } from "../../lib/types";

const COLOR_SWATCH_SIZE = SPACE.lgXl;

/** Idle time after the last color a picker reports before it commits
 *  (upstream `debouncedCommitClipProperties` uses the same 400 ms). WebKit
 *  fires `change` together with every `input` while its color panel is
 *  dragged, so no native event marks the end of the gesture there. */
const COLOR_COMMIT_DELAY_MS = 400;

/** Style colors a picker can preview before they are committed. */
type PendingColors = Partial<Record<"color" | "background" | "border" | "shadow", Rgba>>;

function withColors(style: TextStyle, colors: PendingColors): TextStyle {
  return {
    ...style,
    ...(colors.color && { color: colors.color }),
    ...(colors.background && { background: { ...style.background, color: colors.background } }),
    ...(colors.border && { border: { ...style.border, color: colors.border } }),
    ...(colors.shadow && { shadow: { ...style.shadow, color: colors.shadow } }),
  };
}

/** Same default as `DEFAULT_TEXT_STYLE` in editActions / domain `TextStyle`. */
function completeTextStyle(style: TextStyle | undefined): TextStyle {
  return {
    fontName: style?.fontName ?? "Helvetica-Bold",
    fontSize: style?.fontSize ?? 96,
    fontScale: style?.fontScale ?? 1,
    color: { r: 1, g: 1, b: 1, a: 1, ...style?.color },
    alignment: style?.alignment ?? "center",
    shadow: {
      enabled: style?.shadow?.enabled ?? true,
      color: { r: 0, g: 0, b: 0, a: 0.6, ...style?.shadow?.color },
      offsetX: style?.shadow?.offsetX ?? 0,
      offsetY: style?.shadow?.offsetY ?? -2,
      blur: style?.shadow?.blur ?? 6,
    },
    background: {
      enabled: style?.background?.enabled ?? false,
      color: { r: 0, g: 0, b: 0, a: 0.6, ...style?.background?.color },
    },
    border: {
      enabled: style?.border?.enabled ?? false,
      color: { r: 0, g: 0, b: 0, a: 1, ...style?.border?.color },
    },
  };
}

/** A short, opinionated list of common font families. Free-text is also allowed
 *  so any installed system font name works (the rasterizer resolves it). */
const FONT_OPTIONS = [
  "Helvetica-Bold",
  "Helvetica",
  "Arial-BoldMT",
  "ArialMT",
  "TimesNewRomanPS-BoldMT",
  "Georgia",
  "Courier-Bold",
  "Verdana",
];

const ALIGN_ICON: Record<TextAlignment, LucideIcon> = {
  left: AlignLeft,
  center: AlignCenter,
  right: AlignRight,
};

export function TextTab({ clip, t }: { clip: Clip; t: TFunction }) {
  const [value, setValue] = useState(clip.textContent ?? "");
  const [style, setStyle] = useState<TextStyle>(() => completeTextStyle(clip.textStyle));
  const styleRef = useRef(style);
  styleRef.current = style;
  // The mirrored style a rejected commit falls back to.
  const committedStyle = useRef(clip.textStyle);
  committedStyle.current = clip.textStyle;
  // Color picks previewed locally and not yet committed.
  const pendingColors = useRef<{
    colors: PendingColors;
    timer: ReturnType<typeof setTimeout>;
    release: () => void;
  } | null>(null);

  // Typed text not yet committed (null when clean).
  const pendingText = useRef<string | null>(null);

  // Reset local state when the selected clip (or its persisted style) changes.
  useEffect(() => {
    pendingText.current = null;
    setValue(clip.textContent ?? "");
  }, [clip.id, clip.textContent]);
  useEffect(() => {
    setStyle(completeTextStyle(clip.textStyle));
  }, [clip.id, clip.textStyle]);

  const commitText = () => {
    const next = pendingText.current;
    pendingText.current = null;
    if (next === null || next === (clip.textContent ?? "")) return;
    edit.runTimelineEdit(
      edit.setClipProperties([clip.id], { textContent: next }).catch((error: unknown) => {
        // Keep the typed text so the next blur retries it.
        if (pendingText.current === null) pendingText.current = next;
        throw error;
      }),
    );
  };
  // The Inspector remounts per clip, so switching clips mid-edit unmounts this
  // tab before the textarea blurs: commit to the clip the text was typed for.
  const commitTextRef = useRef(commitText);
  commitTextRef.current = commitText;
  useEffect(() => () => commitTextRef.current(), []);

  // Commit a whole new style. Pending color picks are folded into it, so a
  // discrete edit made while a pick is pending commits both at once.
  const commitStyle = (next: TextStyle) => {
    const pending = pendingColors.current;
    pendingColors.current = null;
    if (pending) {
      clearTimeout(pending.timer);
      pending.release();
    }
    const style = pending ? withColors(next, pending.colors) : next;
    setStyle(style);
    edit.runTimelineEdit(
      edit.setClipProperties([clip.id], { textStyle: style }).catch((error: unknown) => {
        setStyle(completeTextStyle(committedStyle.current));
        throw error;
      }),
    );
  };

  const flushStyle = () => {
    if (pendingColors.current) commitStyle(styleRef.current);
  };
  const flushStyleRef = useRef(flushStyle);
  flushStyleRef.current = flushStyle;
  // Switching clips unmounts this tab: land a pending color on its own clip.
  useEffect(() => () => flushStyleRef.current(), []);

  // Preview a color the picker reports and commit it once the picker goes
  // idle, loses focus, or a history command needs it landed first.
  const previewColor = (target: keyof PendingColors, color: Rgba) => {
    setStyle((current) => withColors(current, { [target]: color }));
    const pending = pendingColors.current;
    if (pending) clearTimeout(pending.timer);
    pendingColors.current = {
      colors: { ...pending?.colors, [target]: color },
      timer: setTimeout(() => flushStyleRef.current(), COLOR_COMMIT_DELAY_MS),
      release: pending?.release ?? edit.holdGestureCommit(() => flushStyleRef.current()),
    };
  };

  return (
    <section style={{ display: "flex", flexDirection: "column", gap: "var(--space-lg)" }}>
      <div>
        <SectionLabel label={t("inspector.section.text")} />
        <textarea
          aria-label={t("inspector.section.text")}
          value={value}
          placeholder={t("inspector.textPlaceholder")}
          onChange={(e) => {
            pendingText.current = e.target.value;
            setValue(e.target.value);
          }}
          onBlur={commitText}
          rows={4}
          style={{
            width: "100%",
            resize: "vertical",
            minHeight: 80,
            padding: "var(--space-sm)",
            fontSize: "var(--fs-sm)",
            color: "var(--text-primary)",
            background: "var(--bg-elevated)",
            border: "var(--bw-thin) solid var(--border-primary)",
            borderRadius: 4,
            fontFamily: "var(--font-sans)",
            outline: "none",
          }}
        />
      </div>

      <div>
        <SectionLabel label={t("inspector.section.textStyle")} />

        <Row label={t("inspector.field.fontFamily")}>
          <select
            aria-label={t("inspector.field.fontFamily")}
            value={style.fontName}
            onChange={(e) => commitStyle({ ...style, fontName: e.target.value })}
            style={{
              maxWidth: 120,
              fontSize: "var(--fs-sm)",
              color: "var(--accent-primary)",
              background: "var(--bg-raised)",
              border: "var(--bw-thin) solid var(--border-primary)",
              borderRadius: "var(--radius-xs)",
              padding: "1px 4px",
            }}
          >
            {(FONT_OPTIONS.includes(style.fontName)
              ? FONT_OPTIONS
              : [style.fontName, ...FONT_OPTIONS]
            ).map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </Row>

        <Row label={t("inspector.field.fontSize")}>
          <ScrubbableNumberField
            ariaLabel={t("inspector.field.fontSize")}
            value={style.fontSize}
            min={4}
            max={512}
            sensitivity={0.5}
            format={(v) => v.toFixed(0)}
            width={56}
            onCommit={(v) => commitStyle({ ...style, fontSize: v })}
          />
        </Row>

        <Row label={t("inspector.field.textColor")}>
          <ColorSwatch
            label={t("inspector.field.textColor")}
            color={style.color}
            onPreview={(color) => previewColor("color", color)}
            onDone={flushStyle}
          />
        </Row>

        <Row label={t("inspector.field.alignment")}>
          <div style={{ display: "inline-flex", gap: 2 }}>
            {(["left", "center", "right"] as TextAlignment[]).map((a) => (
              <AlignButton
                key={a}
                align={a}
                active={style.alignment === a}
                title={t(`inspector.align.${a}`)}
                onClick={() => commitStyle({ ...style, alignment: a })}
              />
            ))}
          </div>
        </Row>

        <ToggleColorRow
          label={t("inspector.field.background")}
          enabled={style.background.enabled}
          color={style.background.color}
          onToggle={(enabled) =>
            commitStyle({ ...style, background: { ...style.background, enabled } })
          }
          onColor={(color) => previewColor("background", color)}
          onColorDone={flushStyle}
        />

        <ToggleColorRow
          label={t("inspector.field.border")}
          enabled={style.border.enabled}
          color={style.border.color}
          onToggle={(enabled) =>
            commitStyle({ ...style, border: { ...style.border, enabled } })
          }
          onColor={(color) => previewColor("border", color)}
          onColorDone={flushStyle}
        />

        <ToggleColorRow
          label={t("inspector.field.shadow")}
          enabled={style.shadow.enabled}
          color={style.shadow.color}
          onToggle={(enabled) =>
            commitStyle({ ...style, shadow: { ...style.shadow, enabled } })
          }
          onColor={(color) => previewColor("shadow", color)}
          onColorDone={flushStyle}
        />
      </div>
    </section>
  );
}

function SectionLabel({ label }: { label: string }) {
  return (
    <div
      style={{
        marginBottom: "var(--space-sm)",
        fontSize: "var(--fs-xxs)",
        fontWeight: "var(--fw-semibold)",
        letterSpacing: "var(--tracking-wide)",
        color: "var(--text-muted)",
        textTransform: "uppercase",
      }}
    >
      {label}
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div
      style={{
        minHeight: 24,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: "var(--space-sm)",
      }}
    >
      <span
        title={label}
        style={{
          minWidth: 0,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          fontSize: "var(--fs-xs)",
          color: "var(--text-tertiary)",
        }}
      >
        {label}
      </span>
      <span
        style={{ flexShrink: 0, display: "inline-flex", alignItems: "center", gap: "var(--space-xs)" }}
      >
        {children}
      </span>
    </div>
  );
}

/** A toggle checkbox + (when enabled) a color swatch, on one row. Used for the
 *  text background, border, and shadow fills. */
function ToggleColorRow({
  label,
  enabled,
  color,
  onToggle,
  onColor,
  onColorDone,
}: {
  label: string;
  enabled: boolean;
  color: Rgba;
  onToggle: (enabled: boolean) => void;
  onColor: (color: Rgba) => void;
  onColorDone: () => void;
}) {
  return (
    <Row label={label}>
      <input
        type="checkbox"
        aria-label={label}
        checked={enabled}
        style={{ accentColor: "var(--accent-primary)", cursor: "pointer" }}
        onChange={(e) => onToggle(e.target.checked)}
      />
      {enabled && (
        <ColorSwatch label={label} color={color} onPreview={onColor} onDone={onColorDone} />
      )}
    </Row>
  );
}

function AlignButton({
  align,
  active,
  title,
  onClick,
}: {
  align: TextAlignment;
  active: boolean;
  title: string;
  onClick: () => void;
}) {
  return (
    <button
      title={title}
      aria-label={title}
      aria-pressed={active}
      onClick={onClick}
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: 24,
        height: 24,
        color: active ? "var(--text-primary)" : "var(--text-tertiary)",
        background: active ? "var(--bg-raised)" : "transparent",
        border: `var(--bw-thin) solid ${active ? "var(--accent-primary)" : "var(--border-primary)"}`,
        borderRadius: "var(--radius-xs)",
        cursor: "pointer",
      }}
    >
      <Icon icon={ALIGN_ICON[align]} size={13} />
    </button>
  );
}

/** A native color picker bound to an `Rgba`. The picker edits RGB; alpha is
 *  preserved verbatim (text colors are usually opaque, fills keep their alpha).
 *  Every reported color is a preview; `onDone` (blur) ends the gesture. */
function ColorSwatch({
  label,
  color,
  onPreview,
  onDone,
}: {
  label: string;
  color: Rgba;
  onPreview: (color: Rgba) => void;
  onDone: () => void;
}) {
  return (
    <input
      aria-label={label}
      type="color"
      value={rgbaToHex(color)}
      onChange={(e) => onPreview({ ...hexToRgb(e.target.value), a: color.a })}
      onBlur={onDone}
      style={{
        width: COLOR_SWATCH_SIZE,
        height: COLOR_SWATCH_SIZE,
        padding: 0,
        border: "var(--bw-thin) solid var(--border-primary)",
        borderRadius: RADIUS.xs,
        background: "transparent",
        cursor: "pointer",
      }}
    />
  );
}

function channelHex(value: number): string {
  const clamped = Math.max(0, Math.min(255, Math.round(value * 255)));
  return clamped.toString(16).padStart(2, "0");
}

function rgbaToHex(color: Rgba): string {
  return `#${channelHex(color.r)}${channelHex(color.g)}${channelHex(color.b)}`;
}

function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const raw = hex.replace("#", "");
  const expanded =
    raw.length === 3
      ? raw
          .split("")
          .map((ch) => ch + ch)
          .join("")
      : raw;
  const parsed = Number.parseInt(expanded, 16);
  if (!Number.isFinite(parsed)) return { r: 1, g: 1, b: 1 };
  return {
    r: ((parsed >> 16) & 0xff) / 255,
    g: ((parsed >> 8) & 0xff) / 255,
    b: (parsed & 0xff) / 255,
  };
}
