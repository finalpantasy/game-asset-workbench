# ship: sheets to game textures

`gaw ship SHEET.png ... --out-dir NEW` writes each sheet as a Memoria raw texture at the same `.png` name: a 60-byte header (width, height, data length, Unity texture format, mip count) and a full mip chain, rows flipped to Unity's bottom-left origin. The game loads it with no PNG decode and no run-time mip build. Flags: `gaw help`. The runtime is Memoria-specific; other games need their own writer.

What to know:

- `--format auto` picks DXT1 only when every texel is opaque, else DXT5. Zidane's two sheets both come out DXT5 (about 21.3 MiB each at 4096; RGBA32 with mips is 85.3 MiB).
- Mips are a 2x2 box on stored values, weighted by alpha. Alpha is rescaled per level so the share of texels at or above `--cutoff` (default 128) matches level 0. Tiny levels can only get close.
- It decodes the result back and reports PSNR, cutout mismatches and coverage in `ship-report.json`. Read it. `--no-verify` skips this; do not.
- A sheet the game rebuilds at run time (FF9's blink sheet, the head render texture) must stay PNG: `--skip-blink` with `--blink-sheet`, or `--png-only`. One private-runtime test shipped the blink sheet as DXT5 and it blinked correctly, so treat that as allowed for the current plugin, not as a rule.
- The HD plugin skips textures with more than one mip, so shipped sheets are not rebuilt at load.
- Battle check (11-16, private runtime, both sheets DXT5 incl. the blink sheet): in battle the HD plugin rebuilt the PNG sheets (log "Rebuilt texture ... mips 13") and left the shipped ones alone, as in the field. A view-matched battle close-up pair (same camera and pose) differs by 46.7 dB masked PSNR and 12 silhouette pixels of 1.78M at 3840x2688: no row flip, color and mips match. The blink SOURCE box showed in battle in both builds (1 frame each, 13 and 14 px, face seen from behind; the battle camera lost Zidane after about 44 frames).
- The battle camera is random per launch, so two launches almost never frame the same view. To compare, take 40 `battle-solo` frames per build (`run_battle_dxt.py ART LABEL`), then pick the pair with the best silhouette overlap (IoU, then masked PSNR). The earlier ~30 dB figure came from mismatched pose/view, not from DXT: unmatched pairs score 18-32 dB, the matched pair 46.7 dB.
- Shipping is the last step. Run `judge` on the PNG sheets first; DXT only adds compression noise (about 46 dB at a matched close-up, invisible by eye). Do not run `judge` across two launches of a battle; the camera differs.
- Never write the output into a game folder. Put it in a new folder and let the user or a test harness copy it into a private game copy.
