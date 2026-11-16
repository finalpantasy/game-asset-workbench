# ship: sheets to game textures

`gaw ship SHEET.png ... --out-dir NEW` writes each sheet as a Memoria raw texture at the same `.png` name: a 60-byte header (width, height, data length, Unity texture format, mip count) and a full mip chain, rows flipped to Unity's bottom-left origin. The game loads it with no PNG decode and no run-time mip build. Flags: `gaw help`. The runtime is Memoria-specific; other games need their own writer.

What to know:

- `--format auto` picks DXT1 only when every texel is opaque, else DXT5. Zidane's two sheets both come out DXT5 (about 21.3 MiB each at 4096; RGBA32 with mips is 85.3 MiB).
- Mips are a 2x2 box on stored values, weighted by alpha. Alpha is rescaled per level so the share of texels at or above `--cutoff` (default 128) matches level 0. Tiny levels can only get close.
- It decodes the result back and reports PSNR, cutout mismatches and coverage in `ship-report.json`. Read it. `--no-verify` skips this; do not.
- A sheet the game rebuilds at run time (FF9's blink sheet, the head render texture) must stay PNG: `--skip-blink` with `--blink-sheet`, or `--png-only`. One private-runtime test shipped the blink sheet as DXT5 and it blinked correctly, so treat that as allowed for the current plugin, not as a rule.
- The HD plugin skips textures with more than one mip, so shipped sheets are not rebuilt at load.
- Shipping is the last step. Run `judge` on the PNG sheets first; the DXT result only adds compression noise (about 30 dB against a PNG build, not visible by eye in the one game run).
- Never write the output into a game folder. Put it in a new folder and let the user or a test harness copy it into a private game copy.
