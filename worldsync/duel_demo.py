"""Side-by-side demo video of two agents in one shared world, with a caption strip per agent showing its memory."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


class _Texts:
    """Caption text goes to drawtext via small text files in a temp dir (avoids filtergraph escaping)."""

    def __init__(self, root):
        self.root, self.count = Path(root), 0

    def __call__(self, text):
        self.count += 1
        path = self.root / f"t{self.count}.txt"
        path.write_text(text)
        return f"textfile='{path}':expansion=none"


def _wrap(text, width):
    """Word-wrap each explicit line of ``text`` to ``width`` characters."""
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            if line and len(line) + 1 + len(word) > width:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        lines.append(line)
    return lines


def _panel(texts, label, captions, width, strip, size=15, chars=None):
    """drawtext filters for one agent: a fixed title plus time-windowed caption lines in the strip below."""
    chars = chars or max(20, int(width / (size * 0.56)))
    filters = [f"drawtext=fontfile={BOLD}:{texts(label)}:fontcolor=white:fontsize=20:x=12:y=10:"
               "box=1:boxcolor=black@0.55:boxborderw=6"]
    for start, end, text in captions:
        for row, line in enumerate(_wrap(text, chars)[:4]):
            filters.append(f"drawtext=fontfile={FONT}:{texts(line)}:fontcolor=white:fontsize={size}:x=12:"
                           f"y=h-{strip}+10+{row * (size + 6)}:enable='between(t,{start:.3f},{end:.3f})'")
    return ",".join(filters)


def compose(panels, out, strip=110, width=640, height=352, title=None, fps=16):
    """Write ``out`` (mp4): the agents' rollouts side by side, each with a caption strip; nothing else touches disk.

    ``panels`` is a list of ``(video_path, label, captions)`` where captions are ``(start_s, end_s, text)`` windows.
    """
    with tempfile.TemporaryDirectory() as tmp:
        return _compose(_Texts(tmp), panels, out, strip, width, height, title, fps)


def _compose(texts, panels, out, strip, width, height, title, fps):
    graph, names = [], []
    for slot, (_, label, captions) in enumerate(panels):
        name = f"p{slot}"
        graph.append(f"[{slot}:v]setpts=N/{fps}/TB,scale={width}:{height},pad={width}:{height + strip}:0:0:color=0x1b1410,"
                     f"{_panel(texts, label, captions, width, strip)}[{name}]")
        names.append(f"[{name}]")
    last = "s"
    graph.append(f"{''.join(names)}hstack=inputs={len(panels)}[s]" if len(panels) > 1 else f"[p0]copy[s]")
    if title:
        graph.append(f"[s]drawtext=fontfile={BOLD}:{texts(title)}:fontcolor=0xffd27a:fontsize=18:"
                     f"x=(w-text_w)/2:y=h-28[t]")
        last = "t"
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for video, _, _ in panels:
        cmd += ["-i", str(video)]
    cmd += ["-filter_complex", ";".join(graph), "-map", f"[{last}]", "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-shortest", str(out)]
    subprocess.run(cmd, check=True)
    return out
