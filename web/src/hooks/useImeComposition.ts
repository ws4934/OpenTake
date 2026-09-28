/**
 * IME composition guard for text fields that submit on Enter.
 *
 * Chinese, Japanese and Korean input methods use Enter to confirm the current
 * candidate. Browsers still dispatch a keydown for that keystroke:
 * - Chromium and Firefox mark it `isComposing` (keyCode 229) and fire it
 *   before `compositionend`;
 * - WebKit (WKWebView, the macOS desktop shell) fires `compositionend` first
 *   and then a keydown with keyCode 229 but `isComposing === false`.
 * A field must leave every such keydown to the IME instead of submitting.
 * A plain Enter keydown after `compositionend` (keyCode 13) is a real Enter:
 * Chromium's Korean IME sends one after committing the syllable, and users
 * expect it to submit.
 */

import { useCallback, useMemo, useRef, type KeyboardEvent as ReactKeyboardEvent } from "react";

/** keyCode browsers report for a keydown the IME consumed ("Process" key). */
const IME_PROCESS_KEY_CODE = 229;

type KeyDownEvent = Pick<KeyboardEvent, "isComposing" | "keyCode">;

/** True when the keydown itself reports an active IME composition. */
export function isImeCompositionKeyDown(event: KeyDownEvent | ReactKeyboardEvent): boolean {
  const native = "nativeEvent" in event ? event.nativeEvent : event;
  return native.isComposing === true || native.keyCode === IME_PROCESS_KEY_CODE;
}

export interface ImeCompositionGuard {
  /** Spread onto the text field so the guard sees composition boundaries. */
  compositionHandlers: {
    onCompositionStart: () => void;
    onCompositionEnd: () => void;
  };
  /** Call first in the field's keydown handler: true means the keystroke
   *  belongs to the IME and must neither submit nor be prevented. */
  isComposingKeyDown: (event: ReactKeyboardEvent) => boolean;
}

export function useImeComposition(): ImeCompositionGuard {
  // Between compositionstart and compositionend, even in an engine that does
  // not flag the keydown itself.
  const composing = useRef(false);

  const onCompositionStart = useCallback(() => {
    composing.current = true;
  }, []);

  const onCompositionEnd = useCallback(() => {
    composing.current = false;
  }, []);

  const isComposingKeyDown = useCallback(
    (event: ReactKeyboardEvent) => composing.current || isImeCompositionKeyDown(event),
    [],
  );

  return useMemo(
    () => ({ compositionHandlers: { onCompositionStart, onCompositionEnd }, isComposingKeyDown }),
    [isComposingKeyDown, onCompositionEnd, onCompositionStart],
  );
}
