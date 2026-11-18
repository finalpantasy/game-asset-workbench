//! Skin-weight transfer: a new mesh takes the weights of the nearest vertices of a rigged mesh.
use crate::skin::Skin;
use crate::{v, Vec3};
use glam::Vec3 as V;
use rayon::prelude::*;
use std::collections::HashMap;

/// A rigged mesh indexed for nearest-vertex lookups (a hash grid of `cell`-sized boxes).
pub struct Source {
    verts: Vec<V>,
    skin: Skin,
    bones: usize,
    cell: f32,
    cells: HashMap<[i32; 3], Vec<u32>>,
}

impl Source {
    /// `cell` should be a few times the mesh's typical edge length.
    pub fn new(positions: &[Vec3], skin: Skin, bones: usize, cell: f32) -> Source {
        assert_eq!(positions.len(), skin.len(), "one skin entry per vertex");
        let verts: Vec<V> = positions.iter().map(v).collect();
        let mut cells: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
        for (i, p) in verts.iter().enumerate() { cells.entry(key(*p, cell)).or_default().push(i as u32); }
        Source { verts, skin, bones, cell, cells }
    }

    pub fn positions(&self) -> impl Iterator<Item = Vec3> + '_ { self.verts.iter().map(|p| p.to_array()) }
    pub fn skin(&self) -> &Skin { &self.skin }
    pub fn bones(&self) -> usize { self.bones }

    /// Weight of every bone at a point: the four nearest source vertices, the nearest dominating
    /// and the others softening seams between faces. Not normalized.
    pub fn dense(&self, p: Vec3) -> Vec<f32> {
        let p = v(&p);
        let mut near: Vec<(f32, u32)> = Vec::new();
        let c = key(p, self.cell);
        for r in 1..12 {
            near.clear();
            for z in c[2] - r..=c[2] + r {
                for y in c[1] - r..=c[1] + r {
                    for x in c[0] - r..=c[0] + r {
                        if let Some(ids) = self.cells.get(&[x, y, z]) {
                            for &i in ids { near.push((self.verts[i as usize].distance_squared(p), i)); }
                        }
                    }
                }
            }
            if near.len() >= 4 { break; }
        }
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut d = vec![0.0f32; self.bones];
        let d0 = near.first().map_or(0.0, |n| n.0.sqrt());
        for &(d2, i) in near.iter().take(4) {
            let k = 1.0 / (d2.sqrt() - d0 + self.cell * (2.0 / 15.0));
            for j in 0..4 { d[self.skin.joints[i as usize][j] as usize] += self.skin.weights[i as usize][j] * k; }
        }
        d
    }

    /// Four strongest bones per point, weights summing to one.
    pub fn transfer(&self, points: &[Vec3]) -> Skin {
        let top: Vec<([u16; 4], [f32; 4])> = points.par_iter().map(|p| {
            top4(&self.dense(*p))
        }).collect();
        Skin { joints: top.iter().map(|t| t.0).collect(), weights: top.iter().map(|t| t.1).collect() }
    }

    /// Every bone's weight at source vertex `i`.
    fn row(&self, i: usize) -> Vec<f32> {
        let mut d = vec![0.0f32; self.bones];
        for j in 0..4 { d[self.skin.joints[i][j] as usize] += self.skin.weights[i][j]; }
        d
    }

    /// Stores the four strongest bones of `d` at vertex `i`, summing to one.
    fn set_row(&mut self, i: usize, d: &[f32]) {
        let (j, w) = top4(d);
        self.skin.joints[i] = j;
        self.skin.weights[i] = w;
    }

    /// Hands `bone`'s weight to `to` past a joint: on the side of the plane through `point` that
    /// `normal` points to, the weight fades out over `band` meters. Use it where a rigger let a lower
    /// limb bone pull on skin above its joint. Returns how many vertices changed.
    pub fn fade_past(&mut self, bone: usize, point: Vec3, normal: Vec3, band: f32, to: usize) -> usize {
        let (o, n) = (v(&point), v(&normal).normalize_or_zero());
        let mut changed = 0;
        for i in 0..self.verts.len() {
            let s = (self.verts[i] - o).dot(n);
            if s <= 0.0 { continue; }
            let mut d = self.row(i);
            if d[bone] <= 0.0 { continue; }
            let t = (s / band.max(1e-6)).min(1.0);
            let lose = d[bone] * t * t * (3.0 - 2.0 * t);
            d[bone] -= lose;
            d[to] += lose;
            self.set_row(i, &d);
            changed += 1;
        }
        changed
    }

    /// Where `owner` is the strongest bone, any weight on a bone outside `allowed` goes to `owner`.
    /// Clears stray weights, such as a head bone pulling on a hoof. Returns how many vertices changed.
    pub fn confine(&mut self, owner: usize, allowed: &[usize]) -> usize {
        let mut changed = 0;
        for i in 0..self.verts.len() {
            let mut d = self.row(i);
            let top = (0..d.len()).max_by(|a, b| d[*a].total_cmp(&d[*b])).unwrap_or(0);
            if top != owner { continue; }
            let mut moved = 0.0;
            for (b, w) in d.iter_mut().enumerate() {
                if *w > 0.0 && b != owner && !allowed.contains(&b) { moved += *w; *w = 0.0; }
            }
            if moved > 0.0 {
                d[owner] += moved;
                self.set_row(i, &d);
                changed += 1;
            }
        }
        changed
    }

    /// Evens out the weights of `bones` over the surface: each round, every vertex that carries
    /// any of them takes the Gaussian-weighted average of its neighbors within `radius`. A vertex's
    /// total on `bones` stays the same, so other bones (hooves, head) keep their exact weights.
    /// Works on point positions alone, so it needs no mesh topology.
    pub fn smooth(&mut self, bones: &[usize], radius: f32, rounds: usize) {
        self.smooth_linked(bones, radius, rounds, |_, _| true)
    }

    /// [`Source::smooth`], averaging only over neighbor pairs that `link` accepts. Use it to keep
    /// skin that is close in space but not on one surface apart, such as the two inner thighs.
    pub fn smooth_linked(&mut self, bones: &[usize], radius: f32, rounds: usize, link: impl Fn(Vec3, Vec3) -> bool + Sync) {
        let mut on = vec![false; self.bones];
        for &b in bones { on[b] = true; }
        let r2 = radius * radius;
        let sig2 = (radius * 0.5).powi(2);
        let reach = (radius / self.cell).ceil() as i32;
        // neighbors (and their Gaussian weights) once; positions do not move
        let near: Vec<Vec<(u32, f32)>> = (0..self.verts.len()).into_par_iter().map(|i| {
            let p = self.verts[i];
            let c = key(p, self.cell);
            let mut out = Vec::new();
            for z in c[2] - reach..=c[2] + reach {
                for y in c[1] - reach..=c[1] + reach {
                    for x in c[0] - reach..=c[0] + reach {
                        if let Some(ids) = self.cells.get(&[x, y, z]) {
                            for &k in ids {
                                let d2 = self.verts[k as usize].distance_squared(p);
                                if d2 <= r2 && link(p.to_array(), self.verts[k as usize].to_array()) {
                                    out.push((k, (-d2 / (2.0 * sig2)).exp()));
                                }
                            }
                        }
                    }
                }
            }
            out
        }).collect();
        for _ in 0..rounds {
            let rows: Vec<Vec<f32>> = (0..self.verts.len()).into_par_iter().map(|i| self.row(i)).collect();
            let new: Vec<Option<Vec<f32>>> = (0..self.verts.len()).into_par_iter().map(|i| {
                let share: f32 = rows[i].iter().zip(&on).filter(|(_, o)| **o).map(|(w, _)| w).sum();
                if share <= 1e-4 { return None; }
                let mut avg = vec![0.0f32; self.bones];
                for &(k, g) in &near[i] {
                    let rk = &rows[k as usize];
                    let sk: f32 = rk.iter().zip(&on).filter(|(_, o)| **o).map(|(w, _)| w).sum();
                    if sk <= 1e-4 { continue; }
                    for b in bones { avg[*b] += g * rk[*b] / sk; }
                }
                let tot: f32 = avg.iter().sum();
                if tot <= 1e-9 { return None; }
                let mut d = rows[i].clone();
                for b in bones { d[*b] = share * avg[*b] / tot; }
                Some(d)
            }).collect();
            for (i, d) in new.into_iter().enumerate() {
                if let Some(d) = d { self.set_row(i, &d); }
            }
        }
    }
}

/// The four strongest entries of a dense weight row, normalized.
fn top4(d: &[f32]) -> ([u16; 4], [f32; 4]) {
    let mut r: Vec<(f32, usize)> = d.iter().copied().zip(0..).collect();
    r.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (mut j, mut w) = ([0u16; 4], [0.0f32; 4]);
    for k in 0..4.min(r.len()) { j[k] = r[k].1 as u16; w[k] = r[k].0.max(0.0); }
    let s: f32 = w.iter().sum::<f32>().max(1e-9);
    (j, w.map(|x| x / s))
}

fn key(p: V, cell: f32) -> [i32; 3] { (p / cell).floor().as_ivec3().to_array() }
