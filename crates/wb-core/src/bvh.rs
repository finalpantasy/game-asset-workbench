//! A small bounding-volume hierarchy over triangle (or any) boxes: nearest-item queries for the
//! weight transfer and all-pairs overlap traversal for the deformation check's intersection test.
use crate::geom::V3;

#[derive(Clone, Copy)]
struct Node {
    lo: V3,
    hi: V3,
    /// Leaf: first index into `items`. Inner node: index of the left child.
    first: u32,
    /// Leaf: number of items (> 0). Inner node: 0.
    count: u32,
    /// Inner node: index of the right child.
    right: u32,
}

pub struct Bvh {
    nodes: Vec<Node>,
    items: Vec<u32>,
    boxes: Vec<(V3, V3)>,
}

const LEAF: usize = 4;

fn box_d2(p: V3, lo: V3, hi: V3) -> f32 {
    let mut s = 0.0;
    for k in 0..3 {
        let d = (lo[k] - p[k]).max(0.0).max(p[k] - hi[k]);
        s += d * d;
    }
    s
}

fn overlap(a: &Node, b: &Node) -> bool {
    (0..3).all(|k| a.lo[k] <= b.hi[k] && b.lo[k] <= a.hi[k])
}

impl Bvh {
    /// Builds over item boxes (`boxes[i]` = low and high corner of item i).
    pub fn build(boxes: &[(V3, V3)]) -> Bvh {
        let mut b = Bvh { nodes: Vec::new(), items: (0..boxes.len() as u32).collect(), boxes: boxes.to_vec() };
        if !boxes.is_empty() {
            let n = boxes.len();
            b.rec(boxes, 0, n);
        }
        b
    }

    fn rec(&mut self, boxes: &[(V3, V3)], start: usize, end: usize) -> u32 {
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        let (mut clo, mut chi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for &i in &self.items[start..end] {
            let (l, h) = boxes[i as usize];
            for k in 0..3 {
                lo[k] = lo[k].min(l[k]);
                hi[k] = hi[k].max(h[k]);
                let c = 0.5 * (l[k] + h[k]);
                clo[k] = clo[k].min(c);
                chi[k] = chi[k].max(c);
            }
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(Node { lo, hi, first: start as u32, count: (end - start) as u32, right: 0 });
        if end - start <= LEAF {
            return id;
        }
        let axis = (0..3).max_by(|&a, &b| (chi[a] - clo[a]).total_cmp(&(chi[b] - clo[b]))).unwrap();
        if chi[axis] - clo[axis] <= 0.0 {
            return id; // all centers equal: keep as a (large) leaf
        }
        let mid = (start + end) / 2;
        let cen = |i: &u32| {
            let (l, h) = boxes[*i as usize];
            l[axis] + h[axis]
        };
        self.items[start..end].select_nth_unstable_by(mid - start, |a, b| cen(a).total_cmp(&cen(b)));
        let left = self.rec(boxes, start, mid);
        let right = self.rec(boxes, mid, end);
        let n = &mut self.nodes[id as usize];
        n.first = left;
        n.count = 0;
        n.right = right;
        id
    }

    /// The item with the smallest `item_d2` (squared distance to the query point `p`), pruning by box
    /// distance. Returns (item, squared distance).
    pub fn nearest(&self, p: V3, mut item_d2: impl FnMut(u32) -> f32) -> Option<(u32, f32)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<(u32, f32)> = None;
        let mut stack = vec![0u32];
        while let Some(ni) = stack.pop() {
            let n = self.nodes[ni as usize];
            if best.is_some_and(|b| box_d2(p, n.lo, n.hi) >= b.1) {
                continue;
            }
            if n.count > 0 {
                for &it in &self.items[n.first as usize..(n.first + n.count) as usize] {
                    let d = item_d2(it);
                    if best.map_or(true, |b| d < b.1) {
                        best = Some((it, d));
                    }
                }
            } else {
                let (l, r) = (n.first, n.right);
                let (dl, dr) = (box_d2(p, self.nodes[l as usize].lo, self.nodes[l as usize].hi), box_d2(p, self.nodes[r as usize].lo, self.nodes[r as usize].hi));
                // Visit the nearer child first (pushed last).
                if dl < dr {
                    stack.push(r);
                    stack.push(l);
                } else {
                    stack.push(l);
                    stack.push(r);
                }
            }
        }
        best
    }

    fn boxes_overlap(&self, i: u32, j: u32) -> bool {
        let (a, b) = (self.boxes[i as usize], self.boxes[j as usize]);
        (0..3).all(|k| a.0[k] <= b.1[k] && b.0[k] <= a.1[k])
    }

    /// Calls `f(a, b)` once for every pair of distinct items whose boxes overlap (a < b is not
    /// guaranteed; each unordered pair is visited once).
    pub fn self_pairs(&self, mut f: impl FnMut(u32, u32)) {
        if self.nodes.is_empty() {
            return;
        }
        let mut stack: Vec<(u32, u32)> = vec![(0, 0)];
        while let Some((a, b)) = stack.pop() {
            let (na, nb) = (self.nodes[a as usize], self.nodes[b as usize]);
            if a == b {
                if na.count > 0 {
                    let it = &self.items[na.first as usize..(na.first + na.count) as usize];
                    for i in 0..it.len() {
                        for j in i + 1..it.len() {
                            if self.boxes_overlap(it[i], it[j]) {
                                f(it[i], it[j]);
                            }
                        }
                    }
                } else {
                    stack.push((na.first, na.first));
                    stack.push((na.right, na.right));
                    stack.push((na.first, na.right));
                }
                continue;
            }
            if !overlap(&na, &nb) {
                continue;
            }
            match (na.count > 0, nb.count > 0) {
                (true, true) => {
                    for &i in &self.items[na.first as usize..(na.first + na.count) as usize] {
                        for &j in &self.items[nb.first as usize..(nb.first + nb.count) as usize] {
                            if self.boxes_overlap(i, j) {
                                f(i, j);
                            }
                        }
                    }
                }
                (false, _) if nb.count > 0 || vol(&na) >= vol(&nb) => {
                    stack.push((na.first, b));
                    stack.push((na.right, b));
                }
                _ => {
                    stack.push((a, nb.first));
                    stack.push((a, nb.right));
                }
            }
        }
    }
}

fn vol(n: &Node) -> f32 {
    (0..3).map(|k| (n.hi[k] - n.lo[k]).max(0.0)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_and_pairs_match_brute_force() {
        let mut s = 12345u32;
        let mut rnd = move || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / 16777216.0
        };
        let boxes: Vec<(V3, V3)> = (0..300)
            .map(|_| {
                let c = [rnd() * 10.0, rnd() * 10.0, rnd() * 10.0];
                let r = 0.1 + rnd() * 0.8;
                ([c[0] - r, c[1] - r, c[2] - r], [c[0] + r, c[1] + r, c[2] + r])
            })
            .collect();
        let bvh = Bvh::build(&boxes);
        for _ in 0..50 {
            let p = [rnd() * 10.0, rnd() * 10.0, rnd() * 10.0];
            let d2 = |i: u32| box_d2(p, boxes[i as usize].0, boxes[i as usize].1);
            let brute = (0..boxes.len() as u32).map(|i| (i, d2(i))).min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
            let got = bvh.nearest(p, d2).unwrap();
            assert!((got.1 - brute.1).abs() < 1e-6, "{got:?} vs {brute:?}");
        }
        let mut got: Vec<(u32, u32)> = Vec::new();
        bvh.self_pairs(|a, b| got.push((a.min(b), a.max(b))));
        got.sort_unstable();
        let mut want = Vec::new();
        for i in 0..boxes.len() {
            for j in i + 1..boxes.len() {
                let (a, b) = (boxes[i], boxes[j]);
                if (0..3).all(|k| a.0[k] <= b.1[k] && b.0[k] <= a.1[k]) {
                    want.push((i as u32, j as u32));
                }
            }
        }
        assert_eq!(got, want);
    }
}
