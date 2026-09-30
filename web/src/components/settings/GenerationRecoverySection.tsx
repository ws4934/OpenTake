import { useEffect, useRef, useState } from "react";
import { discardGenerationRecovery, listGenerationRecoveries } from "../../lib/api";
import type { GenerationRecoveryRecord } from "../../lib/types";
import { formatBytes } from "../../lib/storageFormat";
import { openProjectPath } from "../../store/projectActions";
import { useEditorUiStore } from "../../store/uiStore";
import { useT } from "../../i18n";

const buttonStyle = {
  minWidth: 24,
  minHeight: 24,
  padding: "4px 8px",
  borderRadius: "var(--radius-xs)",
  background: "var(--home-hover)",
};

export function GenerationRecoverySection() {
  const t = useT();
  const [records, setRecords] = useState<GenerationRecoveryRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const alive = useRef(false);
  useEffect(() => {
    alive.current = true;
    let current = true;
    setLoading(true);
    setError(null);
    listGenerationRecoveries().then(
      next => {
        if (current) { setRecords(next); setLoading(false); }
      },
      reason => {
        if (current) { setError(String(reason)); setLoading(false); }
      },
    );
    return () => { current = false; alive.current = false; };
  }, [revision]);

  const open = async (record: GenerationRecoveryRecord) => {
    setBusy(record.jobId);
    setError(null);
    try {
      await openProjectPath(record.projectPath);
      useEditorUiStore.getState().setSettingsOpen(false);
    } catch (reason) {
      if (alive.current) setError(String(reason));
    } finally {
      if (alive.current) setBusy(null);
    }
  };
  const discard = async (jobId: string) => {
    setBusy(jobId);
    setError(null);
    try {
      const next = await discardGenerationRecovery(jobId, true);
      if (alive.current) { setRecords(next); setConfirming(null); }
    } catch (reason) {
      if (alive.current) setError(String(reason));
    } finally {
      if (alive.current) setBusy(null);
    }
  };

  return (
    <section aria-label={t("generation.recovery.title")} style={{ display: "grid", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <h3 style={{ fontSize: "var(--fs-sm)", margin: 0 }}>{t("generation.recovery.title")}</h3>
        <button
          style={buttonStyle}
          disabled={busy !== null || loading}
          onClick={() => { setConfirming(null); setRevision(value => value + 1); }}
        >
          {t("generation.recovery.refresh")}
        </button>
      </div>
      <p style={{ fontSize: "var(--fs-xs)", color: "var(--text-tertiary)", margin: 0 }}>
        {t("generation.recovery.description")}
      </p>
      {error && <div role="alert">{error}</div>}
      {loading && <div role="status">{t("storage.loading")}</div>}
      {!error && records?.length === 0 && <p>{t("generation.recovery.empty")}</p>}
      {records?.map(record => (
        <div key={record.jobId} style={{ display: "grid", gap: 4, borderTop: "1px solid var(--border-primary)", paddingTop: 8 }}>
          <strong style={{ fontSize: "var(--fs-xs)", overflowWrap: "anywhere" }}>{record.projectPath}</strong>
          <span style={{ fontSize: "var(--fs-xs)", color: "var(--text-tertiary)" }}>
            {new Date(record.recordedAt * 1000).toLocaleString()} · {t("generation.recovery.results", { count: record.resultCount })} · {formatBytes(record.byteSize)}
          </span>
          {record.active && <span>{t("generation.recovery.active")}</span>}
          {record.outcomeUnknown && <span>{t("generation.outcomeUnknown")}</span>}
          {record.discardIncomplete && <span>{t("generation.recovery.cleanupIncomplete")}</span>}
          {confirming === record.jobId ? (
            <div>
              <p>{t("generation.recovery.discardConfirm")}</p>
              <button style={buttonStyle} disabled={busy !== null || loading} onClick={() => void discard(record.jobId)}>
                {t("generation.recovery.confirmDiscard")}
              </button>
              <button style={buttonStyle} disabled={busy !== null || loading} onClick={() => setConfirming(null)}>
                {t("common.cancel")}
              </button>
            </div>
          ) : (
            <div style={{ display: "flex", gap: 8 }}>
              <button style={buttonStyle} disabled={busy !== null || loading || record.discardIncomplete} onClick={() => void open(record)}>
                {t("generation.recovery.open")}
              </button>
              <button style={buttonStyle} disabled={busy !== null || loading || record.active} onClick={() => setConfirming(record.jobId)}>
                {t("generation.recovery.discard")}
              </button>
            </div>
          )}
        </div>
      ))}
    </section>
  );
}
