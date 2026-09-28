/**
 * Rounding that matches Rust's `f64::round` (and Swift's `.rounded()`):
 * halves round away from zero. `Math.round` rounds halves toward +Infinity, so
 * `Math.round(-1.5) === -1` while Rust gives `-2`. Every frame computation
 * that mirrors a Rust `.round()` must use this.
 */
export function roundHalfAwayFromZero(value: number): number {
  const magnitude = Math.round(Math.abs(value));
  return value < 0 && magnitude !== 0 ? -magnitude : magnitude;
}
