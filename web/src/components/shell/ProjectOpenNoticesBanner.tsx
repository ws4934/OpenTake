import { X } from "lucide-react";
import { useT } from "../../i18n";
import { HoverButton } from "../ui/HoverButton";
import { Icon } from "../ui/Icon";
import { projectOpenNotices } from "../../lib/projectMessages";
import { useProjectStore } from "../../store/projectStore";

/** Recoverable problems the core handled when the current project was opened
 *  (media opened offline, an ignored proxy, a generation log moved aside).
 *  Stays until dismissed, since the user may need to relink media. */
export function ProjectOpenNoticesBanner() {
  const openNotices = useProjectStore((state) => state.openNotices);
  const projectEpoch = useProjectStore((state) => state.projectEpoch);
  const setOpenNotices = useProjectStore((state) => state.setOpenNotices);
  const t = useT();

  if (!openNotices || openNotices.projectEpoch !== projectEpoch) return null;
  const notices = projectOpenNotices(openNotices.warnings, openNotices.mediaNames, t);
  if (notices.length === 0) return null;

  return (
    <div
      role="status"
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: "var(--space-sm)",
        padding: "var(--space-sm) var(--space-md)",
        background: "var(--bg-raised)",
        borderBottom: "var(--bw-thin) solid var(--status-warning)",
        color: "var(--text-primary)",
        fontSize: "var(--fs-sm)",
      }}
    >
      <strong style={{ whiteSpace: "nowrap" }}>{t("projectOpen.title")}</strong>
      <div style={{ flex: 1, whiteSpace: "pre-line" }}>{notices.join("\n")}</div>
      <HoverButton title={t("projectOpen.dismiss")} onClick={() => setOpenNotices(null)}>
        <Icon icon={X} size={13} />
      </HoverButton>
    </div>
  );
}
