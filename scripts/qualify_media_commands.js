// Run this as a snippet in the packaged desktop application's developer console.
// Open a disposable project with 50 video assets and show the Media panel.
// Obtain ids before timing, choose an uncached poster time within every source:
// const media = await window.__TAURI_INTERNALS__.invoke("get_media");
// await qualifyMediaCommands(media.items.map(item => item.id), 0.83);
// Do not clear another project's caches to prepare the fixture.
//
// WebKit JavaScript runs in a separate process. A page timer alone cannot detect
// shell main-thread stalls: can_undo is a tiny synchronous native command, so its
// response latency includes the host's command dispatch/queue delay.
window.qualifyMediaCommands = async function (mediaRefs, timeSecs) {
  if (!Array.isArray(mediaRefs) || mediaRefs.length !== 50
      || mediaRefs.some(id => typeof id !== "string") || new Set(mediaRefs).size !== 50) {
    throw new Error("Provide exactly 50 distinct video media ids");
  }
  if (!Number.isFinite(timeSecs) || timeSecs < 0) {
    throw new Error("Choose a finite, uncached poster time within every source");
  }
  const grid = document.querySelector('[role="grid"][aria-label="Media"]');
  if (!grid) throw new Error("Show the Media grid before qualification");
  const invoke = window.__TAURI_INTERNALS__.invoke;
  const start = performance.now();
  const latencies = [];
  let busy = false;
  let heartbeat = Promise.resolve();
  let heartbeatError;
  const timer = setInterval(() => {
    const maximum = grid.scrollHeight - grid.clientHeight;
    if (maximum > 0) grid.scrollTop = (grid.scrollTop + 25) % maximum;
    if (busy) return;
    busy = true;
    const sent = performance.now();
    heartbeat = invoke("can_undo")
      .then(() => latencies.push(performance.now() - sent))
      .catch(error => { heartbeatError = error; })
      .finally(() => { busy = false; });
  }, 10);
  let results;
  try {
    results = await Promise.allSettled(mediaRefs.map(mediaRef =>
      invoke("generate_thumbnail", { mediaRef, timeSecs, includeSprite: false })
    ));
  } finally {
    clearInterval(timer);
  }
  await heartbeat;
  if (heartbeatError !== undefined) throw heartbeatError;
  if (!latencies.length) throw new Error("No native heartbeat samples recorded");
  latencies.sort((a, b) => a - b);
  return {
    elapsedMs: performance.now() - start,
    success: results.filter(result => result.status === "fulfilled").length,
    errors: results.filter(result => result.status === "rejected").map(result => result.reason),
    samples: latencies.length,
    maxNativeMs: Math.max(...latencies),
    p95NativeMs: latencies[Math.floor(latencies.length * 0.95)],
  };
};
