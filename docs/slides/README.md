# Presentation slides (offline)

Open `index.html` in Chrome or Firefox (double-click it). No network is needed: everything is in this folder.
If a browser refuses to autoplay from a file, run `python3 -m http.server 8000` in this folder and open
`http://localhost:8000/`; a video that still cannot autoplay shows its play controls.

| Key | Action |
|---|---|
| Right, Space, PageDown, Enter, click right edge | next (slide 7 reveals four steps first) |
| Left, PageUp, Backspace, click left edge | previous |
| Home / End | first / last slide |
| F | fullscreen |
| N | speaker notes panel |

Videos loop for as long as their slide is shown and rewind when you leave it. The URL keeps the slide number
(`index.html#7`), so a reload returns to the same slide. `index.html?notes=1` opens with notes on.

Slides: 1 title, 2 problem, 3 benchmark, 4 baseline reproduced, 5 four baseline failures (videos),
6 the 4.5 s cache, 7 method diagram (memory v2), 8 store the frame and recompute, 9 side-by-side GC011,
10 side-by-sides GC034 and GC026, 11 revisit-horizon chart and table (measured with memory v1), 12 ablation,
13 PlayWorld task GC045 (selected example, with the 36-task result), 14 limitations, 15 autoresearch loop,
16 future work.

Plain-language definitions of every term are in [../EXPLAINER.md](../EXPLAINER.md).

Rebuild the media with `scripts/build_slides_media.sh` (needs the run output folders; see the script).
Render slides to PNG for a visual check with `scripts/render_slides.sh OUT_DIR`.
Self-test of navigation and video start/stop: open `index.html?selftest=1#5` and read `<pre id="selftest">`.
