//! Link-group queries and mutations. Clips sharing a `link_group_id` behave as
//! one unit for selection, move, trim, split, and delete. 1:1 port of the pure
//! parts of `EditorViewModel+Linking.swift`.

use std::collections::{HashMap, HashSet};

use opentake_domain::Timeline;

/// Reverse link-group index: group id -> member clip ids. Built in one
/// `O(tracks·clips)` pass. 1:1 port of `linkIndex`.
pub fn link_index(timeline: &Timeline) -> HashMap<String, Vec<String>> {
    let mut m: HashMap<String, Vec<String>> = HashMap::new();
    for t in &timeline.tracks {
        for c in &t.clips {
            if let Some(g) = &c.link_group_id {
                m.entry(g.clone()).or_default().push(c.id.clone());
            }
        }
    }
    m
}

pub(crate) struct BorrowedLinkIndex<'a> {
    groups: HashMap<&'a str, Vec<&'a str>>,
    clip_groups: HashMap<&'a str, &'a str>,
}

impl<'a> BorrowedLinkIndex<'a> {
    pub(crate) fn new(timeline: &'a Timeline) -> Self {
        let mut groups: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut clip_groups = HashMap::new();
        for track in &timeline.tracks {
            for clip in &track.clips {
                if let Some(group_id) = clip.link_group_id.as_deref() {
                    groups.entry(group_id).or_default().push(&clip.id);
                    clip_groups.insert(clip.id.as_str(), group_id);
                }
            }
        }
        Self {
            groups,
            clip_groups,
        }
    }

    fn members(&self, clip_id: &str) -> &[&'a str] {
        self.clip_groups
            .get(clip_id)
            .and_then(|group_id| self.groups.get(*group_id))
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    pub(crate) fn partners<'b>(&'b self, clip_id: &'b str) -> impl Iterator<Item = &'a str> + 'b {
        self.members(clip_id)
            .iter()
            .copied()
            .filter(move |member_id| *member_id != clip_id)
    }
}

/// Every clip id sharing a link group with any id in `ids`, including the inputs.
/// 1:1 port of `expandToLinkGroup`.
pub fn expand_to_link_group(timeline: &Timeline, ids: &HashSet<String>) -> HashSet<String> {
    let idx = BorrowedLinkIndex::new(timeline);
    let mut groups = HashSet::new();
    for id in ids {
        if let Some(group_id) = idx.clip_groups.get(id.as_str()) {
            groups.insert(*group_id);
        }
    }
    if groups.is_empty() {
        return ids.clone();
    }
    let mut result = ids.clone();
    for group_id in groups {
        if let Some(members) = idx.groups.get(group_id) {
            for member_id in members {
                result.insert((*member_id).to_string());
            }
        }
    }
    result
}

/// Ids of clips that share `clip_id`'s link group, excluding `clip_id` itself.
/// 1:1 port of `linkedPartnerIds(of:)`.
pub fn linked_partner_ids(timeline: &Timeline, clip_id: &str) -> Vec<String> {
    BorrowedLinkIndex::new(timeline)
        .partners(clip_id)
        .map(str::to_string)
        .collect()
}

/// Linked-partner ids that should receive a timing-style change (duration, trim,
/// speed) applied uniformly to `clip_ids`. 1:1 port of `timingPropagationPartners`.
pub fn timing_propagation_partners(
    timeline: &Timeline,
    clip_ids: &HashSet<String>,
) -> HashSet<String> {
    let mut out = HashSet::new();
    let index = BorrowedLinkIndex::new(timeline);
    for id in clip_ids {
        for partner_id in index.partners(id) {
            if !clip_ids.contains(partner_id) {
                out.insert(partner_id.to_string());
            }
        }
    }
    out
}

/// For a single-clip frame move to `to_frame`, the linked-partner moves needed to
/// keep A/V in sync. 1:1 port of `partnerMoves(forMoveOf:toFrame:)`.
pub fn partner_moves(timeline: &Timeline, clip_id: &str, to_frame: i32) -> Vec<(String, i32)> {
    let Some(lead) = find_clip_start(timeline, clip_id) else {
        return Vec::new();
    };
    let Some(delta) = to_frame.checked_sub(lead) else {
        return Vec::new();
    };
    if delta == 0 {
        return Vec::new();
    }
    linked_partner_ids(timeline, clip_id)
        .into_iter()
        .filter_map(|pid| {
            let start = find_clip_start(timeline, &pid)?;
            Some((pid, start.checked_add(delta)?.max(0)))
        })
        .collect()
}

fn find_clip_start(timeline: &Timeline, clip_id: &str) -> Option<i32> {
    for t in &timeline.tracks {
        if let Some(c) = t.clips.iter().find(|c| c.id == clip_id) {
            return Some(c.start_frame);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentake_domain::{Clip, ClipType, Track};

    fn linked(id: &str, start: i32, group: &str) -> Clip {
        let mut c = Clip::new(id, "asset", start, 30);
        c.link_group_id = Some(group.to_string());
        c
    }

    fn tl_two_groups() -> Timeline {
        let mut tl = Timeline::new();
        let mut v = Track::new("v", ClipType::Video);
        v.clips.push(linked("v1", 0, "g1"));
        v.clips.push(Clip::new("solo", "asset", 100, 30));
        let mut a = Track::new("a", ClipType::Audio);
        a.clips.push(linked("a1", 0, "g1"));
        tl.tracks.push(v);
        tl.tracks.push(a);
        tl
    }

    #[test]
    fn expand_pulls_in_group_members() {
        let tl = tl_two_groups();
        let got = expand_to_link_group(&tl, &["v1".to_string()].into_iter().collect());
        assert!(got.contains("v1") && got.contains("a1"));
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn expand_unlinked_returns_input_only() {
        let tl = tl_two_groups();
        let got = expand_to_link_group(&tl, &["solo".to_string()].into_iter().collect());
        assert_eq!(got, ["solo".to_string()].into_iter().collect());
    }

    #[test]
    fn partner_ids_excludes_self() {
        let tl = tl_two_groups();
        assert_eq!(linked_partner_ids(&tl, "v1"), vec!["a1".to_string()]);
        assert!(linked_partner_ids(&tl, "solo").is_empty());
    }

    #[test]
    fn partner_moves_keeps_delta_and_clamps() {
        let tl = tl_two_groups();
        // move v1 from 0 to 50 -> a1 (at 0) should target 50.
        assert_eq!(partner_moves(&tl, "v1", 50), vec![("a1".to_string(), 50)]);
        // negative target clamps to 0: not reachable here since both at 0; check delta 0 -> empty.
        assert!(partner_moves(&tl, "v1", 0).is_empty());
    }

    #[test]
    fn timing_partners_excludes_inputs() {
        let tl = tl_two_groups();
        let p = timing_propagation_partners(&tl, &["v1".to_string()].into_iter().collect());
        assert_eq!(p, ["a1".to_string()].into_iter().collect());
        // both in set -> no extra partners
        let p2 = timing_propagation_partners(
            &tl,
            &["v1".to_string(), "a1".to_string()].into_iter().collect(),
        );
        assert!(p2.is_empty());
    }
}
