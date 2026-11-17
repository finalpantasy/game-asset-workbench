//! Closed-loop weight repair: pose the model in its clips, find triangles whose edges stretch or
//! squash past the deform-check limits, soften the weights of the vertices involved (a share of the
//! neighbors' average), and repeat while the stretch and squash shares keep falling.
//!
//! A transfer from a reference of another body leaves hard cuts where the new mesh has edges
//! crossing them. Blurring everything (smoothing rings) hurts the places that were fine; this only
//! touches the places the pose check flags, and keeps the best state it saw.
use crate::anim::{self, Mat4};
use crate::deform::{self, cone_volume, crossing_pairs, frames_for, xform, Limits, PoseSource, Prep};
use crate::geom::V3;
use crate::weights::top4;
use crate::Model;
use std::collections::HashSet;

pub struct RepairOptions {
    pub iters: usize,
    /// Share of the neighbors' mean mixed into a flagged vertex each iteration.
    pub lambda: f32,
    /// Fractions per clip (as deform-check's --steps).
    pub steps: usize,
    /// Rings of neighbors softened along with each flagged vertex.
    pub reach: usize,
    /// Share moved toward the dominant bone for vertices flagged only for squash (a fold that
    /// collapses on a half-and-half blend); 0 treats squash like stretch and softens.
    pub sharpen: f32,
    /// Share pulled toward the weights of the surface a flagged vertex crosses.
    pub pull: f32,
    /// Try candidate weights for each flagged vertex (blends with its neighbors, with its own main
    /// bones) and keep the one whose edges stretch and squash least over the clips; false softens
    /// toward the neighbors' mean instead.
    pub search: bool,
    /// Cost of moving a vertex's weights away from where this run started (per unit of change).
    pub anchor: f32,
}

impl Default for RepairOptions {
    fn default() -> Self {
        RepairOptions { iters: 60, lambda: 0.5, steps: 12, reach: 0, sharpen: 0.0, pull: 0.25, search: true, anchor: 0.5 }
    }
}

pub struct RepairResult {
    pub iterations: usize,
    pub best_iteration: usize,
    /// Score before and after: squares of each body region's worst-frame stretch and squash share and of the new-crossing share, each over its limit.
    pub score_before: f32,
    pub score_after: f32,
    pub groups_touched: usize,
}

fn sub3(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn len3(a: V3) -> f32 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

pub fn repair(model: &mut Model, borrowed: Option<&Model>, o: &RepairOptions) -> Result<RepairResult, String> {
    let prep = Prep::new(model, false)?;
    let limits = Limits::default();
    let nj = prep.skel.names.len();
    let (frames, jms): (Vec<_>, Vec<Vec<Mat4>>) = {
        let src = PoseSource { model, borrowed };
        let names = src.clip_names();
        if names.is_empty() {
            return Err("no clips to pose the model in (neither the file nor the reference has any)".into());
        }
        let frames = frames_for(&names, &[], o.steps)?;
        frames.iter().filter_map(|f| src.joint_matrices(f).map(|m| (f.clone(), m))).unzip()
    };
    if jms.is_empty() {
        return Err("could not pose the model in any frame".into());
    }
    // Limits per region: 1.5x what the reference does in the same frames (floors as deform-check), or
    // the floors alone without a reference.
    let nreg = prep.regions.len();
    let (mut lim_st, mut lim_sq) = (vec![limits.floor_region_pct; nreg], vec![limits.floor_region_pct; nreg]);
    let mut lim_int = limits.floor_intersect_area_pct * 6.0;
    let mut lim_vol = limits.volume_loss_pct;
    if let Some(rm) = borrowed {
        if let Ok(rp) = Prep::new(rm, false) {
            let rsrc = PoseSource { model: rm, borrowed: None };
            let meas = deform::measure(rm, &rp, &rsrc, &frames, &limits);
            for (ri, name) in prep.regions.iter().enumerate() {
                if let Some(rj) = rp.regions.iter().position(|n| n == name) {
                    lim_st[ri] = (meas.max_region(rj, true) * limits.max_x).max(limits.floor_region_pct);
                    lim_sq[ri] = (meas.max_region(rj, false) * limits.max_x).max(limits.floor_region_pct);
                }
            }
            lim_vol = (meas.worst_volume_loss(&rp).0 * limits.max_x).max(limits.floor_volume_loss_pct);
            lim_int = (meas.max_stat(|s| s.intersect_area_pct) * limits.max_x).max(limits.floor_intersect_area_pct);
        }
    }
    // Vertices of every welded group, adjacency between groups.
    let ng = prep.corner_group.iter().flatten().copied().max().map_or(0, |m| m as usize + 1);
    let mut verts: Vec<Vec<(u32, u32)>> = vec![Vec::new(); ng];
    let mut adj: Vec<HashSet<u32>> = vec![HashSet::new(); ng];
    for (i, &(mi, ti)) in prep.tris.iter().enumerate() {
        let t = model.meshes[mi as usize].tris[ti as usize];
        let g = prep.corner_group[i];
        for k in 0..3 {
            let e = (mi, t[k]);
            if !verts[g[k] as usize].contains(&e) {
                verts[g[k] as usize].push(e);
            }
            adj[g[k] as usize].insert(g[(k + 1) % 3]);
            adj[g[k] as usize].insert(g[(k + 2) % 3]);
        }
    }
    let mut dense: Vec<Vec<f32>> = vec![Vec::new(); ng];
    for g in 0..ng {
        if let Some(&(mi, v)) = verts[g].first() {
            let m = &model.meshes[mi as usize];
            let mut d = vec![0f32; nj];
            for c in 0..4 {
                let (j, w) = (m.joints[v as usize][c] as usize, m.weights[v as usize][c]);
                if w > 0.0 && j < nj {
                    d[j] += w;
                }
            }
            dense[g] = d;
        }
    }
    let min_edge = prep.height * 1e-4;
    let min_change = prep.height * limits.min_change_frac;

    // The score and, per group, how many frames flagged it.
    let evaluate = |model: &Model| -> (f32, Vec<u8>, Vec<Vec<u32>>, f32) {
        let mut hits = vec![0u8; ng];
        let mut partners: Vec<Vec<u32>> = vec![Vec::new(); ng];
        let (mut worst_st, mut worst_sq) = (vec![0f32; nreg], vec![0f32; nreg]);
        let (mut worst_int, mut worst_vol) = (0f32, 0f32);
        for jm in &jms {
            let pos: Vec<Vec<V3>> = model.meshes.iter().map(|m| anim::skin(m, jm)).collect();
            let (mut r_st, mut r_sq) = (vec![0f32; nreg], vec![0f32; nreg]);
            for (i, &(mi, ti)) in prep.tris.iter().enumerate() {
                let t = model.meshes[mi as usize].tris[ti as usize];
                let p = t.map(|v| pos[mi as usize][v as usize]);
                let e = [len3(sub3(p[1], p[0])), len3(sub3(p[2], p[1])), len3(sub3(p[0], p[2]))];
                let (mut mx, mut mn) = (1f32, 1f32);
                for k in 0..3 {
                    let r0 = prep.rest_edges[i][k];
                    if r0 < min_edge || (e[k] - r0).abs() < min_change {
                        continue;
                    }
                    let r = e[k] / r0;
                    mx = mx.max(r);
                    mn = mn.min(r);
                }
                let st = mx > limits.stretch_edge;
                let sq = mn < limits.squash_edge;
                if st {
                    r_st[prep.tri_region[i] as usize] += prep.rest_area[i];
                }
                if sq {
                    r_sq[prep.tri_region[i] as usize] += prep.rest_area[i];
                }
                if st || sq {
                    for g in prep.corner_group[i] {
                        hits[g as usize] |= if st { 1 } else { 0 } | if sq { 2 } else { 0 };
                    }
                }
            }
            for r in 0..nreg {
                let a = prep.region_area[r].max(1e-12);
                worst_st[r] = worst_st[r].max(100.0 * r_st[r] / a);
                worst_sq[r] = worst_sq[r].max(100.0 * r_sq[r] / a);
            }
            for z in &prep.joint_zones {
                if z.rest_volume.abs() > 0.0 {
                    let v = cone_volume(model, &prep, &z.tris, &pos, xform(&jm[z.joint], z.head));
                    worst_vol = worst_vol.max(100.0 * (1.0 - v / z.rest_volume).max(0.0));
                }
            }
            // New crossings since rest, as a share of the whole surface.
            let pairs: Vec<(u32, u32)> = crossing_pairs(model, &prep, &pos).into_iter().filter(|p| !prep.rest_pairs.contains(p)).collect();
            for &(a, b) in &pairs {
                let (ga, gb) = (prep.corner_group[a as usize], prep.corner_group[b as usize]);
                for x in ga {
                    for y in gb {
                        if x != y {
                            hits[x as usize] |= 4;
                            hits[y as usize] |= 4;
                            if partners[x as usize].len() < 24 && !partners[x as usize].contains(&y) {
                                partners[x as usize].push(y);
                            }
                            if partners[y as usize].len() < 24 && !partners[y as usize].contains(&x) {
                                partners[y as usize].push(x);
                            }
                        }
                    }
                }
            }
            let mut involved: Vec<u32> = pairs.iter().flat_map(|p| [p.0, p.1]).collect();
            involved.sort_unstable();
            involved.dedup();
            let area: f32 = involved.iter().map(|&i| prep.rest_area[i as usize]).sum();
            worst_int = worst_int.max(100.0 * area / prep.total_area.max(1e-12));
        }
        // Squares of each share over its limit, so the worst one weighs most, and a steep extra
        // term for a share past 85% of its limit (the places the check would fail).
        let mut score = 0f32;
        let mut add = |v: f32, lim: f32| {
            let r = v / lim.max(1e-6);
            score += r * r + 12.0 * (r - 0.85).max(0.0).powi(2) + 60.0 * (r - 1.0).max(0.0).powi(2);
        };
        for r in (0..nreg).filter(|&r| prep.region_area[r] > 0.0) {
            add(worst_st[r], lim_st[r]);
            add(worst_sq[r], lim_sq[r]);
        }
        add(worst_int, lim_int);
        add(worst_vol, lim_vol * 0.85);
        (score, hits, partners, worst_int / lim_int.max(1e-6))
    };
    let write_back = |model: &mut Model, dense: &[Vec<f32>], groups: &[usize]| {
        for &g in groups {
            let (j4, w4) = top4(&dense[g]);
            for &(mi, v) in &verts[g] {
                model.meshes[mi as usize].joints[v as usize] = j4;
                model.meshes[mi as usize].weights[v as usize] = w4;
            }
        }
    };

    // Local search state: rest positions, edges with their rest lengths, the skinned position of every
    // group in every frame, and the weights this run started with.
    let rest: Vec<V3> = (0..ng).map(|g| verts[g].first().map(|&(mi, v)| model.meshes[mi as usize].pos[v as usize]).unwrap_or([0.0; 3])).collect();
    let edges: Vec<Vec<(u32, f32)>> = (0..ng)
        .map(|g| {
            if dense[g].is_empty() {
                return Vec::new();
            }
            adj[g].iter().filter(|&&n| !dense[n as usize].is_empty()).map(|&n| (n, len3(sub3(rest[g], rest[n as usize])))).filter(|e| e.1 >= min_edge).collect()
        })
        .collect();
    let orig = dense.clone();
    let nf = jms.len();
    let skin_one = |d4: &([u16; 4], [f32; 4]), f: usize, p: V3| -> V3 {
        let mut out = [0f32; 3];
        for c in 0..4 {
            let w = d4.1[c];
            if w <= 0.0 {
                continue;
            }
            let q = xform(&jms[f][d4.0[c] as usize], p);
            for k in 0..3 {
                out[k] += w * q[k];
            }
        }
        out
    };
    let mut ppos: Vec<Vec<V3>> = if o.search { (0..nf).map(|f| (0..ng).map(|g| if dense[g].is_empty() { rest[g] } else { skin_one(&top4(&dense[g]), f, rest[g]) }).collect()).collect() } else { Vec::new() };
    let (st_t, sq_t) = (1.0 + (limits.stretch_edge - 1.0) * 0.7, 1.0 - (1.0 - limits.squash_edge) * 0.7);
    let pen = |e: f32, r0: f32| -> f32 {
        if (e - r0).abs() < min_change {
            return 0.0;
        }
        let r = e / r0;
        if r > st_t {
            (r - st_t).powi(2)
        } else if r < sq_t {
            (sq_t - r).powi(2)
        } else {
            0.0
        }
    };

    let snapshot = |model: &Model| -> Vec<(Vec<[u16; 4]>, Vec<[f32; 4]>)> { model.meshes.iter().map(|m| (m.joints.clone(), m.weights.clone())).collect() };
    let (score0, mut hits, mut partners, mut int_ratio) = evaluate(model);
    let mut cur = score0;
    let mut kept = snapshot(model);
    let mut best_it = 0usize;
    let mut touched: HashSet<usize> = HashSet::new();
    let mut stale = 0;
    let mut done = 0;
    for it in 1..=o.iters {
        done = it;
        // Crossings only count once they come close to their limit; softening everything that touches
        // a harmless crossing blurs good weights.
        let take = if int_ratio > 0.8 { 7 } else { 3 };
        let mut mark: Vec<bool> = (0..ng).map(|g| hits[g] & take != 0 && !dense[g].is_empty()).collect();
        if !mark.iter().any(|&m| m) {
            break;
        }
        for _ in 0..o.reach {
            let grown: Vec<usize> = (0..ng).filter(|&g| !mark[g] && !dense[g].is_empty() && adj[g].iter().any(|&n| mark[n as usize])).collect();
            for g in grown {
                mark[g] = true;
            }
        }
        let flagged: Vec<usize> = (0..ng).filter(|&g| mark[g]).collect();
        // Jacobi step over the flagged groups: soften, sharpen where only squash was flagged, and pull
        // crossing surfaces together.
        let mut next: Vec<(usize, Vec<f32>)> = Vec::with_capacity(flagged.len());
        if o.search {
            for &g in &flagged {
                if edges[g].is_empty() {
                    continue;
                }
                let cur = dense[g].clone();
                let mut cands: Vec<Vec<f32>> = Vec::new();
                let mix = |a: &[f32], b: &[f32], t: f32| -> Vec<f32> { a.iter().zip(b).map(|(x, y)| (1.0 - t) * x + t * y).collect() };
                let ns: Vec<usize> = adj[g].iter().map(|&n| n as usize).filter(|&n| !dense[n].is_empty()).collect();
                if !ns.is_empty() {
                    let inv = 1.0 / ns.len() as f32;
                    let mean: Vec<f32> = (0..nj).map(|j| ns.iter().map(|&n| dense[n][j]).sum::<f32>() * inv).collect();
                    for t in [0.3f32, 0.6, 1.0] {
                        cands.push(mix(&cur, &mean, t));
                    }
                    for &n in ns.iter().take(14) {
                        for t in [0.5f32, 1.0] {
                            cands.push(mix(&cur, &dense[n], t));
                        }
                    }
                }
                let mut order: Vec<usize> = (0..nj).filter(|&j| cur[j] > 1e-3).collect();
                order.sort_by(|&a, &b| cur[b].total_cmp(&cur[a]));
                for &j in order.iter().take(3) {
                    let mut one = vec![0f32; nj];
                    one[j] = 1.0;
                    for t in [0.3f32, 0.6] {
                        cands.push(mix(&cur, &one, t));
                    }
                }
                let cost = |c: &[f32]| -> (f32, ([u16; 4], [f32; 4])) {
                    let c4 = top4(c);
                    let mut sum = 0f32;
                    for f in 0..nf {
                        let p = skin_one(&c4, f, rest[g]);
                        for &(n, r0) in &edges[g] {
                            sum += pen(len3(sub3(p, ppos[f][n as usize])), r0);
                        }
                    }
                    let mut dd = vec![0f32; nj];
                    for k in 0..4 {
                        dd[c4.0[k] as usize] += c4.1[k];
                    }
                    let l1: f32 = (0..nj).map(|j| (dd[j] - orig[g][j]).abs()).sum();
                    (sum + o.anchor * l1, c4)
                };
                let (mut best_cost, _) = cost(&cur);
                let mut best: Option<([u16; 4], [f32; 4])> = None;
                for c in &cands {
                    let (k, c4) = cost(c);
                    if k < best_cost - 1e-6 {
                        best_cost = k;
                        best = Some(c4);
                    }
                }
                if let Some(c4) = best {
                    let mut dd = vec![0f32; nj];
                    for k in 0..4 {
                        dd[c4.0[k] as usize] += c4.1[k];
                    }
                    for f in 0..nf {
                        ppos[f][g] = skin_one(&c4, f, rest[g]);
                    }
                    dense[g] = dd.clone();
                    next.push((g, dd));
                }
            }
        } else {
        for &g in &flagged {
            let mut d = dense[g].clone();
            if o.sharpen > 0.0 && hits[g] & 3 == 2 && hits[g] & 4 == 0 {
                let dom = (0..nj).max_by(|&a, &b| dense[g][a].total_cmp(&dense[g][b])).unwrap_or(0);
                let k = o.sharpen;
                for j in 0..nj {
                    d[j] = (1.0 - k) * d[j] + if j == dom { k } else { 0.0 };
                }
                next.push((g, d));
                continue;
            }
            let ns: Vec<&Vec<f32>> = adj[g].iter().filter(|&&n| !dense[n as usize].is_empty()).map(|&n| &dense[n as usize]).collect();
            if !ns.is_empty() {
                let (inv, k) = (1.0 / ns.len() as f32, o.lambda);
                for j in 0..nj {
                    let mean: f32 = ns.iter().map(|v| v[j]).sum::<f32>() * inv;
                    d[j] = (1.0 - k) * d[j] + k * mean;
                }
            }
            if hits[g] & 4 != 0 {
                // Surfaces that cross each other should move together.
                let ps: Vec<&Vec<f32>> = partners[g].iter().filter(|&&q| !dense[q as usize].is_empty()).map(|&q| &dense[q as usize]).collect();
                if !ps.is_empty() {
                    let (inv, k) = (1.0 / ps.len() as f32, o.pull);
                    for j in 0..nj {
                        let mean: f32 = ps.iter().map(|v| v[j]).sum::<f32>() * inv;
                        d[j] = (1.0 - k) * d[j] + k * mean;
                    }
                }
            }
            next.push((g, d));
        }
        }
        let groups: Vec<usize> = next.iter().map(|x| x.0).collect();
        for (g, d) in next {
            dense[g] = d;
        }
        write_back(model, &dense, &groups);
        touched.extend(&groups);
        let (score, h, pt, ir) = evaluate(model);
        int_ratio = ir;
        if std::env::var_os("GAW_WEIGHTS_DEBUG").is_some() {
            eprintln!("repair pass {it}: {} places, score {score:.2} (best {cur:.2})", groups.len());
        }
        hits = h;
        partners = pt;
        if score < cur - 1e-4 {
            cur = score;
            kept = snapshot(model);
            best_it = it;
            stale = 0;
        } else {
            // A step can make things worse before the next ones make them better: keep going a few
            // passes, then fall back to the best state seen.
            stale += 1;
            if stale >= 12 {
                break;
            }
        }
    }
    for (m, (j, w)) in model.meshes.iter_mut().zip(kept) {
        m.joints = j;
        m.weights = w;
    }
    Ok(RepairResult { iterations: done, best_iteration: best_it, score_before: score0, score_after: cur, groups_touched: touched.len() })
}
