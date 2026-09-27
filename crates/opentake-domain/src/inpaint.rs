//! Reusable fill plan for a static object-removal mask across video frames.

use std::collections::VecDeque;

use crate::Mask;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InpaintPlanError {
    Cancelled,
    EmptyMask,
    FullMask,
    TooLarge,
    Unfillable,
    DimensionsChanged,
}

struct FillStep {
    index: usize,
    known_neighbours: [usize; 8],
    count: usize,
}

/// The mask coverage and boundary fill order depend on dimensions, but never
/// on frame contents. Reuse one plan for all frames of a source clip.
pub struct InpaintPlan {
    width: u32,
    height: u32,
    blend: Vec<(usize, f64)>,
    fill: Vec<FillStep>,
}

impl InpaintPlan {
    pub fn prepare(
        mask: &Mask,
        width: u32,
        height: u32,
        cancelled: impl FnMut() -> bool,
    ) -> Result<Self, InpaintPlanError> {
        Self::prepare_with_coverage(width, height, cancelled, |x, y| mask.coverage(x, y))
    }

    fn prepare_with_coverage(
        width: u32,
        height: u32,
        mut cancelled: impl FnMut() -> bool,
        mut coverage_at: impl FnMut(f64, f64) -> f64,
    ) -> Result<Self, InpaintPlanError> {
        let w = width as usize;
        let h = height as usize;
        let pixels = w.checked_mul(h).ok_or(InpaintPlanError::TooLarge)?;
        pixels.checked_mul(4).ok_or(InpaintPlanError::TooLarge)?;
        let mut coverage = Vec::with_capacity(pixels);
        let mut known = Vec::with_capacity(pixels);
        let mut selected = 0_usize;
        for y in 0..h {
            if y % 32 == 0 && cancelled() {
                return Err(InpaintPlanError::Cancelled);
            }
            for x in 0..w {
                let value = coverage_at((x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64);
                coverage.push(value);
                known.push(value <= 0.001);
                selected += usize::from(value > 0.001);
            }
        }
        if selected == 0 {
            return Err(InpaintPlanError::EmptyMask);
        }
        if selected == pixels {
            return Err(InpaintPlanError::FullMask);
        }

        let mut queued = vec![false; pixels];
        let mut queue = VecDeque::new();
        for index in 0..pixels {
            if !known[index] && has_known_neighbour(index, w, h, &known) {
                queue.push_back(index);
                queued[index] = true;
            }
        }
        let mut fill = Vec::with_capacity(selected);
        while let Some(index) = queue.pop_front() {
            if fill.len().is_multiple_of(4096) && cancelled() {
                return Err(InpaintPlanError::Cancelled);
            }
            let mut known_neighbours = [0; 8];
            let mut count = 0;
            for neighbour in neighbours(index, w, h).into_iter().flatten() {
                if known[neighbour] {
                    known_neighbours[count] = neighbour;
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }
            fill.push(FillStep {
                index,
                known_neighbours,
                count,
            });
            known[index] = true;
            for neighbour in neighbours(index, w, h).into_iter().flatten() {
                if !known[neighbour] && !queued[neighbour] {
                    queue.push_back(neighbour);
                    queued[neighbour] = true;
                }
            }
        }
        if known.iter().any(|value| !value) {
            return Err(InpaintPlanError::Unfillable);
        }
        Ok(Self {
            width,
            height,
            blend: coverage
                .into_iter()
                .enumerate()
                .filter(|(_, alpha)| *alpha > 0.0)
                .collect(),
            fill,
        })
    }

    /// Fill RGB in place while preserving the original alpha channel. The
    /// original RGBA data is retained until every blend is complete.
    pub fn apply(
        &self,
        rgba: &mut [u8],
        mut cancelled: impl FnMut() -> bool,
    ) -> Result<(), InpaintPlanError> {
        if rgba.len() != self.width as usize * self.height as usize * 4 {
            return Err(InpaintPlanError::DimensionsChanged);
        }
        let mut filled = rgba.to_vec();
        for (step_index, step) in self.fill.iter().enumerate() {
            if step_index.is_multiple_of(4096) && cancelled() {
                return Err(InpaintPlanError::Cancelled);
            }
            let mut sums = [0_u32; 3];
            for &neighbour in &step.known_neighbours[..step.count] {
                let offset = neighbour * 4;
                sums[0] += u32::from(filled[offset]);
                sums[1] += u32::from(filled[offset + 1]);
                sums[2] += u32::from(filled[offset + 2]);
            }
            let offset = step.index * 4;
            for channel in 0..3 {
                filled[offset + channel] = (sums[channel] / step.count as u32) as u8;
            }
        }
        for &(index, alpha) in &self.blend {
            let offset = index * 4;
            for channel in 0..3 {
                rgba[offset + channel] = (f64::from(rgba[offset + channel]) * (1.0 - alpha)
                    + f64::from(filled[offset + channel]) * alpha)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
        Ok(())
    }
}

fn has_known_neighbour(index: usize, width: usize, height: usize, known: &[bool]) -> bool {
    neighbours(index, width, height)
        .into_iter()
        .flatten()
        .any(|neighbour| known[neighbour])
}

fn neighbours(index: usize, width: usize, height: usize) -> [Option<usize>; 8] {
    let x = index % width;
    let y = index / width;
    let mut result = [None; 8];
    let mut cursor = 0;
    for dy in -1_i32..=1 {
        for dx in -1_i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0 && ny >= 0 && nx < width as i32 && ny < height as i32 {
                result[cursor] = Some(ny as usize * width + nx as usize);
                cursor += 1;
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MaskShape, Point2};

    // Preserve the exact per-frame flood order and blending from the previous
    // implementation; comparison includes feathered and inverted polygons.
    fn original_boundary_fill(rgba: &mut [u8], width: usize, height: usize, mask: &Mask) {
        let mut coverage = Vec::new();
        let mut known = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let value = mask.coverage(
                    (x as f64 + 0.5) / width as f64,
                    (y as f64 + 0.5) / height as f64,
                );
                coverage.push(value);
                known.push(value <= 0.001);
            }
        }
        let original = rgba.to_vec();
        let mut filled = original.clone();
        let mut queued = vec![false; width * height];
        let mut queue = VecDeque::new();
        for index in 0..known.len() {
            if !known[index] && has_known_neighbour(index, width, height, &known) {
                queue.push_back(index);
                queued[index] = true;
            }
        }
        while let Some(index) = queue.pop_front() {
            let mut sums = [0_u32; 3];
            let mut count = 0;
            for neighbour in neighbours(index, width, height).into_iter().flatten() {
                if known[neighbour] {
                    let offset = neighbour * 4;
                    for channel in 0..3 {
                        sums[channel] += u32::from(filled[offset + channel]);
                    }
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }
            for channel in 0..3 {
                filled[index * 4 + channel] = (sums[channel] / count) as u8;
            }
            known[index] = true;
            for neighbour in neighbours(index, width, height).into_iter().flatten() {
                if !known[neighbour] && !queued[neighbour] {
                    queue.push_back(neighbour);
                    queued[neighbour] = true;
                }
            }
        }
        assert!(known.into_iter().all(|value| value));
        for (index, alpha) in coverage.into_iter().enumerate() {
            if alpha > 0.0 {
                for channel in 0..3 {
                    let offset = index * 4 + channel;
                    rgba[offset] = (f64::from(original[offset]) * (1.0 - alpha)
                        + f64::from(filled[offset]) * alpha)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    #[test]
    fn reuses_one_mask_scan_and_matches_original_polygon_frames() {
        for inverted in [false, true] {
            let mask = Mask {
                shape: MaskShape::Poly {
                    points: vec![
                        Point2::new(0.22, 0.24),
                        Point2::new(0.81, 0.31),
                        Point2::new(0.69, 0.81),
                        Point2::new(0.30, 0.76),
                    ],
                },
                feather: 0.08,
                invert: inverted,
                ..Mask::default()
            };
            let mut coverage_calls = 0;
            let plan = InpaintPlan::prepare_with_coverage(
                64,
                36,
                || false,
                |x, y| {
                    coverage_calls += 1;
                    mask.coverage(x, y)
                },
            )
            .unwrap();
            assert_eq!(coverage_calls, 64 * 36);
            for frame_index in 0..10_u32 {
                let pixels: Vec<_> = (0..64 * 36)
                    .flat_map(|index| {
                        let x = index % 64;
                        let y = index / 64;
                        [
                            ((x * 13 + y * 3 + frame_index * 11) % 256) as u8,
                            ((x * 5 + y * 7 + frame_index * 17) % 256) as u8,
                            ((x * 3 + y * 19 + frame_index * 23) % 256) as u8,
                            255,
                        ]
                    })
                    .collect();
                let mut actual = pixels.clone();
                let mut expected = pixels;
                original_boundary_fill(&mut expected, 64, 36, &mask);
                plan.apply(&mut actual, || false).unwrap();
                assert_eq!(actual, expected, "inverted={inverted}, frame={frame_index}");
            }
            assert_eq!(coverage_calls, 64 * 36);
        }
    }

    #[test]
    fn invalid_masks_keep_their_errors() {
        let empty = Mask {
            shape: MaskShape::Circle {
                center: Point2::new(2.0, 2.0),
                radius: Point2::new(0.1, 0.1),
            },
            ..Mask::default()
        };
        assert!(matches!(
            InpaintPlan::prepare(&empty, 64, 36, || false),
            Err(InpaintPlanError::EmptyMask)
        ));
        assert!(matches!(
            InpaintPlan::prepare(&Mask::default(), 64, 36, || false),
            Err(InpaintPlanError::FullMask)
        ));
    }

    #[test]
    fn cancellation_applies_during_preparation_and_per_frame() {
        let mask = Mask {
            shape: MaskShape::Circle {
                center: Point2::new(0.5, 0.5),
                radius: Point2::new(0.2, 0.2),
            },
            ..Mask::default()
        };
        assert!(matches!(
            InpaintPlan::prepare(&mask, 16, 12, || true),
            Err(InpaintPlanError::Cancelled)
        ));
        let plan = InpaintPlan::prepare(&mask, 16, 12, || false).unwrap();
        let mut rgba = vec![255; 16 * 12 * 4];
        assert_eq!(
            plan.apply(&mut rgba, || true),
            Err(InpaintPlanError::Cancelled)
        );
    }
}
