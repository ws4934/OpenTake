/** Filesystem path strings are opaque IPC values. Filename operations preserve
 * the versioned representation; display labels never become filesystem paths. */
const PREFIX = "opentake-path-v1:";

function nativeUnits(path: string): { platform: "unix" | "windows"; units: number[] } | null {
  if (!path.startsWith(PREFIX)) return null;
  const match = /^opentake-path-v1:(unix|windows):([0-9a-f]+)$/i.exec(path);
  if (!match) throw new Error("Invalid native path encoding");
  const platform = match[1].toLowerCase() as "unix" | "windows";
  const width = platform === "unix" ? 2 : 4;
  if (match[2].length % width !== 0) throw new Error("Invalid native path encoding");
  const units = [];
  for (let index = 0; index < match[2].length; index += width) {
    const unit = Number.parseInt(match[2].slice(index, index + width), 16);
    if (unit === 0) throw new Error("Native paths cannot contain NUL");
    units.push(unit);
  }
  return { platform, units };
}

/** Show invalid Unix bytes and unpaired Windows surrogates as escapes. The
 * returned label is never passed back to filesystem commands. */
export function displayNativePath(path: string): string {
  const native = nativeUnits(path);
  if (!native) return path;
  const { platform, units } = native;
  if (platform === "windows") {
    let label = "";
    for (let index = 0; index < units.length; index += 1) {
      const unit = units[index];
      if (unit >= 0xd800 && unit <= 0xdbff && units[index + 1] >= 0xdc00 && units[index + 1] <= 0xdfff) {
        label += String.fromCharCode(unit, units[++index]);
      } else if (unit >= 0xd800 && unit <= 0xdfff) {
        label += `\\u${unit.toString(16).padStart(4, "0")}`;
      } else label += String.fromCharCode(unit);
    }
    return label;
  }
  const bytes = new Uint8Array(units);
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let label = "";
  for (let index = 0; index < bytes.length;) {
    const byte = bytes[index];
    const width = byte < 0x80 ? 1 : byte >= 0xc2 && byte <= 0xdf ? 2 : byte >= 0xe0 && byte <= 0xef ? 3 : byte >= 0xf0 && byte <= 0xf4 ? 4 : 0;
    if (width && index + width <= bytes.length) {
      try {
        label += decoder.decode(bytes.subarray(index, index + width));
        index += width;
        continue;
      } catch { /* This byte has no Unicode representation; escape it below. */ }
    }
    label += `\\x${byte.toString(16).padStart(2, "0")}`;
    index += 1;
  }
  return label;
}

export function nativePathName(path: string): string {
  const native = nativeUnits(path);
  if (!native) return path.split(/[\\/]/).filter(Boolean).pop() ?? "";
  let lastSeparator = -1;
  for (let index = 0; index < native.units.length; index += 1) {
    if (native.units[index] === 47 || (native.platform === "windows" && native.units[index] === 92)) lastSeparator = index;
  }
  const name = native.units.slice(lastSeparator + 1);
  return name.length ? displayNativePath(nativeWire(native.platform, name)) : "";
}

function nativeWire(platform: "unix" | "windows", units: number[]): string {
  const width = platform === "unix" ? 2 : 4;
  return `${PREFIX}${platform}:${units.map((unit) => unit.toString(16).padStart(width, "0")).join("")}`;
}

/** Replace a known filename suffix, or append the replacement when absent. */
export function replaceNativePathSuffix(path: string, suffix: string, replacement: string): string {
  const native = nativeUnits(path);
  if (!native) return (path.toLowerCase().endsWith(suffix.toLowerCase()) ? path.slice(0, -suffix.length) : path) + replacement;
  const encode = (text: string): number[] => native.platform === "unix"
    ? Array.from(new TextEncoder().encode(text))
    : Array.from({ length: text.length }, (_, index) => text.charCodeAt(index));
  const suffixUnits = encode(suffix.toLowerCase());
  const start = native.units.length - suffixUnits.length;
  const matches = start >= 0 && suffixUnits.every((unit, index) => {
    const original = native.units[start + index];
    return (original >= 65 && original <= 90 ? original + 32 : original) === unit;
  });
  const units = matches ? native.units.slice(0, start) : native.units;
  return nativeWire(native.platform, units.concat(encode(replacement)));
}

export function nativePathParent(path: string): string {
  const native = nativeUnits(path);
  if (!native) return path.replace(/[\\/][^\\/]*$/, "");
  let separator = -1;
  for (let index = 0; index < native.units.length; index += 1) {
    if (native.units[index] === 47 || (native.platform === "windows" && native.units[index] === 92)) separator = index;
  }
  if (separator < 0) return "";
  const rootLength = native.platform === "windows" && native.units[1] === 58 ? 3 : 1;
  return nativeWire(native.platform, native.units.slice(0, Math.max(separator, rootLength)));
}

/** Preserve the directory's native units when proposing a new filename. */
export function joinNativePath(directory: string, name: string): string {
  const native = nativeUnits(directory);
  if (!native) {
    const separator = directory.lastIndexOf("\\") > directory.lastIndexOf("/") ? "\\" : "/";
    return `${directory}${directory.endsWith("/") || directory.endsWith("\\") ? "" : separator}${name}`;
  }
  const { platform, units } = native;
  if (units[units.length - 1] !== 47 && (platform !== "windows" || units[units.length - 1] !== 92)) units.push(platform === "unix" ? 47 : 92);
  units.push(...(platform === "unix" ? new TextEncoder().encode(name) : Array.from({ length: name.length }, (_, index) => name.charCodeAt(index))));
  return nativeWire(platform, units);
}
