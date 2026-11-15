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

The main story runs from the single-camera memory problem and benchmark through the method,
results, scope, and autoresearch loop. Slides 15–17 extend it to the multi-agent prototype:
what is reused, how shared text context branches into two histories, then the labeled views. Slide 18 closes with
shared visual evidence as the next step. The remaining slides are the results appendix.
The multi-agent videos use scripted controllers and independent visual histories;
they are not multi-agent benchmark results.

Plain-language definitions of every term are in [../EXPLAINER.md](../EXPLAINER.md).

Rebuild the media with `scripts/build_slides_media.sh` (needs the run output folders; see the script).
Render slides to PNG for a visual check with `scripts/render_slides.sh OUT_DIR`.
Self-test of navigation and video start/stop: open `index.html?selftest=1#5` and read `<pre id="selftest">`.
