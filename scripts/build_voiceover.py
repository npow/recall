"""Add a local text-to-speech narration track to the presentation video.

Usage: PIPER_PY=<python with piper-tts> PIPER_MODEL=<voice.onnx> python scripts/build_voiceover.py [--video PATH] [--out PATH]
Reads reproduction/presentation-voiceover.md (## scene headings), synthesises each paragraph with Piper on this machine
(nothing is sent anywhere), time-stretches slightly (atempo 0.92-1.08) to fit the scene, places it at the scene start,
normalises loudness to about -16 LUFS and muxes AAC audio (128 kbps) into the video without re-encoding the picture.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAD = 0.3


def run(*args, **kw):
    return subprocess.run(list(map(str, args)), check=True, **kw)


def duration(path):
    out = run("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path, capture_output=True, text=True)
    return float(out.stdout.strip())


def script_paragraphs(path):
    parts = re.split(r"^## (\w+)\s*$", Path(path).read_text(), flags=re.M)
    return {parts[i]: " ".join(parts[i + 1].split()) for i in range(1, len(parts), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default=str(ROOT / "docs/demo/presentation.mp4"))
    ap.add_argument("--out")
    ap.add_argument("--storyboard", default=str(ROOT / "reproduction/presentation-storyboard.json"))
    ap.add_argument("--script", default=str(ROOT / "reproduction/presentation-voiceover.md"))
    ap.add_argument("--report", default=str(ROOT / "reproduction/presentation-voiceover-fit.json"))
    args = ap.parse_args()
    piper_py, model = os.environ["PIPER_PY"], os.environ["PIPER_MODEL"]
    durations = json.load(open(args.storyboard))["durations"]
    text = script_paragraphs(args.script)
    total = sum(durations.values())
    tmp = Path(tempfile.mkdtemp(prefix="voiceover-"))
    inputs, filters, rows, start = [], [], [], 0.0
    for n, (scene, seconds) in enumerate(durations.items()):
        raw = tmp / f"{scene}.wav"
        run(piper_py, "-m", "piper", "-m", model, "-f", raw, input=text[scene], text=True, capture_output=True)
        raw_len = duration(raw)
        room = seconds - PAD
        tempo = min(max(raw_len / room, 0.92), 1.08) if raw_len > room else 1.0
        fitted = raw_len / tempo
        rows.append({"scene": scene, "start": start, "scene_seconds": seconds, "audio_seconds": round(fitted, 2), "tempo": round(tempo, 3),
                     "fits": fitted <= room + 1e-6, "words": len(text[scene].split())})
        inputs += ["-i", raw]
        delay = int((start + 0.15) * 1000)
        filters.append(f"[{n + 1}:a]atempo={tempo:.4f},adelay={delay}|{delay}[a{n}]")
        start += seconds
    mix = "".join(f"[a{n}]" for n in range(len(durations)))
    graph = ";".join(filters) + f";{mix}amix=inputs={len(durations)}:normalize=0:dropout_transition=0,apad,atrim=0:{total},loudnorm=I=-16:TP=-1.5:LRA=7[out]"
    out = Path(args.out or args.video)
    staged = tmp / "muxed.mp4"
    run("ffmpeg", "-loglevel", "error", "-y", "-i", args.video, *inputs, "-filter_complex", graph, "-map", "0:v", "-map", "[out]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-shortest", staged)
    out.parent.mkdir(parents=True, exist_ok=True)
    staged.replace(out)
    Path(args.report).write_text(json.dumps({"voice": Path(model).stem, "scenes": rows}, indent=1) + "\n")
    print(json.dumps({"output": str(out), "video_seconds": duration(out), "over_budget": [r["scene"] for r in rows if not r["fits"]]}))


if __name__ == "__main__":
    main()
