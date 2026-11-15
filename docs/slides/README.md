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

The main story runs from the single-camera memory problem through the method,
results and research loop. Slides 15–16 show the first, text-conditioned multi-agent
prototype: shared plan and weights with separate visual KV, then three scripted
scenarios (handoff, greeting, joint attention). Slides 17–21 show the actual cross-agent extension:
archive/splice/runtime reuse, the shared visual-KV architecture, and three
pose-matched revisit videos with FOV radar. The radar's green readouts come from
actual 40-layer injection traces. These are common-origin static-scene, serial
visits; they are functional diagnostics rather than benchmark scores.
Slide 22 is future work; the remaining slides are the appendix.


Plain-language definitions of every term are in [../EXPLAINER.md](../EXPLAINER.md).

Rebuild the media with `scripts/build_slides_media.sh` (needs the run output folders; see the script).
Render slides to PNG for a visual check with `scripts/render_slides.sh OUT_DIR`.
Self-test of navigation and video start/stop: open `index.html?selftest=1#5` and read `<pre id="selftest">`.
