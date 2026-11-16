# judge: does it still look like the original?

`gaw judge` scores a candidate against a reference with FLIP (perceptual difference, 0 same to 1 very different), SSIM, and a score per body part or bone. Flags: `gaw help`.

## Modes

- Two PNGs: `judge ref.png test.png`.
- Two GLBs drawn from the same cameras: `judge ref.glb test.glb`.
- Goldens (the regression mode): make once per model, then check every candidate.
  - `judge --make-goldens MODEL.glb --name NAME --out-dir DIR` (never overwrites without `--force`).
  - `judge --golden-dir DIR/NAME CANDIDATE.glb --out-dir OUT` draws the candidate from the stored cameras and scores it against the stored pictures.
- `judge --strands MODEL.glb [--ref STOCK.glb]`: hair only, no reference needed (see `references/hair-strap.md`).

## Reading the verdict

Score = the worst of: worst view mean, worst region mean, worst part in its own close-up. On the shipped limits PASS is under 0.035, WARN to 0.055, FAIL above. Accepted pattern layouts of stock FF9 characters scored 0.028 to 0.034; layouts with real flaws scored 0.068 to 0.089. For a new game, make goldens from stock, judge a known-good and a known-bad candidate, and set `--warn` and `--fail` from those.

- `--only a,b` names the regions the change is meant for (kit part or bone group). The verdict then rests on the worst region outside them: collateral damage. Use it whenever the edit is local.
- Open `sheet-*.png` (reference | test | error) and the heat maps in `--out-dir`. A WARN you can explain by eye is fine. A PASS with a visible defect means the cameras missed it: add a saved view.
- Shading warnings mean mean brightness moved more than `--shading-tol`, or the candidate carries vertex colors the stock lacks. Strip stray vertex colors (`references/ff9.md`).

## Limits

- Judge does not separate small hair-quality differences, color seams, or two close belt versions. It is a regression gate, not a taste test.
- Overlay meshes that exist only in the candidate (FF9 daggers) are scored as their own regions. A candidate with those meshes at stock UVs on a new atlas scores very badly (smeared texels). That is the check working.
- The standard views are idle, run 0.75 s and walk 0.25 s at game size, plus close-ups. A problem only visible inside a hair card or in an undrawn pose needs a saved view.
