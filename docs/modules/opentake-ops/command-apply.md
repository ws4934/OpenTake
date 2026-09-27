# 子系统：命令与事务（command.rs + editor_state.rs）

> 上级：[INDEX.md](INDEX.md) · 总览：[OVERVIEW.md](OVERVIEW.md)

## 职责

这是本模块的中枢：定义**唯一编辑入口** `EditCommand` 枚举与执行函数 `apply()`，把所有编辑收敛成一条 `withTimelineSwap` 式事务；并由 `EditorState` 持有可编辑文档与时间线快照/素材增量历史。撤销 / 校验 / 版本号因此只写一次。

源文件：
- `../../../crates/opentake-ops/src/command.rs`（`EditCommand` + `apply` + 各命令实现）
- `../../../crates/opentake-ops/src/editor_state.rs`（`EditorState` + `DocSnapshot` + 撤销栈）
- `../../../crates/opentake-ops/src/editor_state/manifest_delta.rs`（素材/文件夹字段级增量，保留带外写入）

## 关键类型 / 函数 / 算法

### EditCommand —— 统一编辑命令枚举

`EditCommand`（`command.rs`）是覆盖全部编辑表面的命令枚举。**注意：它是纯枚举，没有 serde derive**（见下「序列化陷阱」）。变体含：

- 放置 / 插入：`AddClips` / `AddClipsAutoTrack`（在按媒体类型新建的共享轨上放置）/ `InsertClips`（波纹插入）。
- 片段结构：`MoveClips` / `DuplicateClips`（Alt 拖拽深拷贝）/ `RemoveClips` / `SplitClip` / `TrimClips`。
- 调速：`SetClipSpeed { clip_ids, speed, ripple }`。Inspector 通过 `setClipSpeed` IPC 默认启用 ripple；时长按 `round(old_duration × old_speed / new_speed)`（至少一帧）计算，先按新旧时长比缩放所有动画关键帧，再裁边和限制淡入淡出。选中片段按调用顺序去重处理，未直接选中的联动非文本伙伴按时间线顺序追加，各自在**当前位置**移动本轨的紧邻后续链；有间隙的片段不移动。增长或链尾位移碰撞、整数溢出、非法速度及不支持的嵌套调速均原子拒绝，不覆盖其他片段、不消耗历史。一次 Undo 还原整次调整。普通 `SetClipProperties` 的 speed/durationFrames 同样缩放动画，但为 Agent 保留**非 ripple** 语义，需调用者处理片段间距；这与上游 Agent 的直接裁剪关键帧行为有意不同。稳定化采样轨的重定基仍由独立问题跟踪，不包含在本命令的动画修复中。
- 属性：`SetClipProperties` / `SetColorGrade` / `SetChromaKey` / `SetMasks` / `SetEffects` / `SwapMedia`（原位换 `media_ref` 保留全部编辑属性）。
- 关键帧：`SetKeyframes` / `StampKeyframe` / `RemoveKeyframe` / `MoveKeyframe` / `SetKeyframeInterpolation`（公开 API 用**绝对时间线帧**）。
- 波纹删除：`RippleDeleteRanges`（按帧区间）/ `RippleDeleteClips`（按选中片段）。
- 轨道 / 文本 / 链接：`AddTexts` / `Link` / `Unlink` / `RemoveTracks` / `InsertTrack` / `SetTrackProps`（mute/hide/sync-lock 切换）。
- 文件夹 / 媒体库：`CreateFolder` / `MoveToFolder` / `RenameMedia` / `RenameFolder` / `DeleteMedia` / `DeleteFolder`。
- 历史：`Undo` / `Redo`。

辅助载荷类型：`ClipEntry`（→ `PlaceSpec`）、`RenameEntry`、`TextEntry`、`ClipProperties`（`None` 字段不变；设标量值会清掉对应关键帧轨）、`KeyframeProperty` / `KeyframePayload`。

### apply() —— 事务执行壳

`apply(state: &mut EditorState, command: EditCommand, ids: &dyn IdGen) -> Result<EditResult, EditError>`：把命令分派到各实现函数。普通编辑实现走 `transact()`；`Undo`/`Redo` 与带 refusal 结果的 ripple 路径执行等价的 snapshot/restore/commit 流程：

```
fn transact(state, action_name, summarize, work):
    before  = state.snapshot()        // 整文档 Clone
    affected = match work(state):
        Ok(value) → value
        Err(error) → state.restore(before); return Err(error)
    after   = state.snapshot()
    timeline_changed = before.timeline != after.timeline
    manifest_changed = before.manifest != after.manifest
    changed = timeline_changed || manifest_changed
    if changed: state.commit(before)   // 推 before 入撤销栈、清 redo、version++
    return EditResult{ changed, timeline_changed, manifest_changed,
                       action_name, affected_clip_ids, timeline_version, summary }
```

即上游 `withTimelineSwap` 的泛化：从「整 timeline 交换」扩到「整文档（timeline + manifest）交换」。

### EditResult / EditError

- `EditResult { changed, timeline_changed, manifest_changed, action_name, affected_clip_ids, timeline_version, summary }`：`changed` 驱动 commit/version/通用同步；两个 domain flag 让 core 在 manifest 变化时额外发送 `MediaChanged`。Tauri `EditResultDto` 仍只投影面向前端的原 5 个业务字段。未变更命令报告**先前**的 version。
- `EditError::Invalid(String)`：输入校验失败（坏索引 / 缺片段 / 空载荷）。
- `EditError::Refused(String)`：波纹拒绝（sync-lock 跟随轨无法吸收位移）。

### EditorState —— 文档 + 撤销栈

`EditorState`（`editor_state.rs`）：
- 字段：`timeline` / `manifest`（文件夹命令改的是 manifest 不是 timeline）/ `undo_stack` / `redo_stack` / `version`。
- `DocSnapshot { timeline, manifest }`：事务变更检测与失败回滚使用的完整快照，整棵 `Clone` + `PartialEq`。
- `HistoryEntry`：时间线快照、定向 `ManifestDelta`、操作标签和事务版本。素材增量只保留被命令增删/修改的条目与文件夹，按 id 定位、按字段恢复。
- 查询：`version()` / `can_undo()` / `can_redo()` / `undo_depth()` / `find_clip(id) -> Option<ClipLocation>`（1:1 上游 `findClip`）/ `track_index(track_id)`。
- 事务内部 API（`pub(crate)`）：`snapshot` / `restore` / `commit` / `undo` / `redo`。

## 不变量与上游对齐

- **原子性**：`work` 返回 `Err` 时 `transact` 显式恢复 `before` 后返回，timeline、manifest、history、version 均不留下部分变化；波纹拒绝（`Err(Refused)`）执行同样的「整次不改」。
- **commit-if-changed**：只有 `before != after` 才入栈 + `version++`。无实质变化的命令（如 `SwapMedia` 换到相同 `media_ref`）返回 `changed = false`、不污染撤销栈。
- **撤销只作用于编辑事务**：`commit(before)` 记录时间线快照和逆向素材增量，并**清空 `redo_stack`**。Undo/Redo 保留当前素材清单中的无关导入；字段只有在本命令确实修改、且当前仍等于命令写入值时才恢复，后续带外写入优先。收藏、格式版本和提供商音色不属于普通历史。恢复后根据实际变化生成反向增量，保持显式注册/删除、文件夹命令和多轮撤销/重做语义。真实恢复才递增 `version`；非法时间线候选仍原子拒绝。
- **pin-by-id**：放置类命令在 `clear_region`（可能 prune / 移位索引）后用 `track_index(track_id)` 重新定位轨道，避免索引失效。
- **关键帧绝对帧**：命令公开 API 用绝对时间线帧，内部转 clip 相对偏移（拆分逻辑在 domain）。
- 单次 rename（媒体 / 文件夹）= 一元素 vec，与批量同走一个撤销组（对齐上游 `withUndoGroup`）。

## ⚠️ 序列化陷阱（高频 bug 来源）

- `EditCommand` 是**纯枚举，无 serde derive**。
- IPC 层另有 serde DTO `EditRequest`（在 `../../../src-tauri/src/commands.rs`），用 `#[serde(tag = "type", rename_all = "camelCase")]`，由 Tauri 命令 `edit_apply` 映射成 `EditCommand`。
- 因此**多词字段在前端线上必须是 camelCase**（如 `atFrame` / `trackIndex`）。历史上「删除 / 分割 / Inspector 全静默失效」就是 DTO 的 camelCase 没对齐导致反序列化失败。改 IPC 字段时，Rust DTO、前端 `web/src/lib/types.ts` 的 `EditRequest`、调用处三边必须同步；IPC 内若静默吞错，先加 `try/catch` 把错误暴露出来。
- 编辑命令 / IPC 的当前规格见 [core 命令路由](../../specs/core/2-command-routing.md)。

## 与其他子系统关系

- **调用 `ops/*`**：大多数命令在 `transact` 的 `work` 闭包里组合 `ops/` 算法；带 refusal 报告的 ripple 路径在等价的手写事务中调用（见 [ops-algorithms.md](ops-algorithms.md)）。
- **re-export `engines`**：`RippleDeleteRanges` 直接收 `FrameRange`（来自 [engines.md](engines.md)）。
- **依赖 `IdGen`**：新实体 id 由注入的生成器铸造（见 [intent-id.md](intent-id.md)）。
- **被 `opentake-core` 装配**：core 把 `EditorState` 包进 `Arc<Mutex<…>>` 权威容器，对 UI/Agent/MCP 暴露唯一 `apply` 入口，并经版本号 + 事件广播推前端。

---

> 上级：[INDEX.md](INDEX.md) · 总览：[OVERVIEW.md](OVERVIEW.md)
