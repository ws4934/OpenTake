/**
 * Paired display formatters / parsers for Inspector number fields whose display
 * units differ from the stored value (upstream `displayMultiplier`). `parse`
 * receives the edited text without the field suffix and returns the raw value,
 * or `null` when it is not a number.
 */

function parseNumber(text: string): number | null {
  const value = Number(text);
  return text !== "" && Number.isFinite(value) ? value : null;
}

/** Stored 0..n ratio shown as a whole percent (0.5 ↔ "50"). */
export const percentUnits = {
  format: (v: number) => Math.round(v * 100).toString(),
  parse: (text: string): number | null => {
    const value = parseNumber(text);
    return value === null ? null : value / 100;
  },
};

/** Stored linear gain shown in decibels (1 ↔ "0.0"); "-∞" / "-inf" is silence. */
export const decibelUnits = {
  format: (v: number) => (20 * Math.log10(Math.max(1e-6, v))).toFixed(1),
  parse: (text: string): number | null => {
    if (/^-\s*(∞|inf(inity)?)$/i.test(text)) return 0;
    const value = parseNumber(text);
    return value === null ? null : 10 ** (value / 20);
  },
};
