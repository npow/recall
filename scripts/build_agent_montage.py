"""Three selected paired POV excerpts; preserve timing within each pair."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT/"docs/slides/media"


def main():
    selected = [("Basket handoff", "multi-agent-pov.mp4", 2.0),
                ("Greeting", "multi-agent-wave.mp4", .5),
                ("Joint attention", "multi-agent-point.mp4", 1.3)]
    command = ["ffmpeg", "-v", "error", "-y"]
    filters = []
    for i, (title, name, start) in enumerate(selected):
        command += ["-ss", str(start), "-t", "4", "-i", str(MEDIA/name)]
        filters.append(f"[{i}:v]setpts=PTS-STARTPTS,fps=30,setsar=1,pad=iw:ih+48:0:0:color=0x101724,"
                       "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
                       f"text='{title} - selected example':x=22:y=h-35:fontsize=23:fontcolor=white[v{i}]")
    graph = ";".join(filters)+";[v0][v1][v2]concat=n=3:v=1:a=0[v]"
    command += ["-filter_complex", graph, "-map", "[v]", "-an", "-t", "12", "-c:v", "libx264",
                "-crf", "24", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(MEDIA/"multi-agent-montage.mp4")]
    subprocess.run(command, check=True)
    rows = [
        ("handoff", "outputs/two-agent-pov/tiled.mp4", True, "Readable basket interaction; release and ownership unresolved."),
        ("wave-basket", "outputs/extra-agent-demos/wave/tiled.mp4", True, "Both visible companions wave; turn order differs across histories."),
        ("point-basket", "outputs/extra-agent-demos/look_at_river/tiled.mp4", False, "Gesture weak in Orange's view; later empty-hand attempt clearer."),
        ("wave-empty", "outputs/more-agent-demos/wave/tiled.mp4", False, "Orange's view duplicates Purple; reject from selected presentation."),
        ("high-five-empty", "outputs/more-agent-demos/high_five/tiled.mp4", False, "Palms clap or miss; reciprocal physical contact not shown."),
        ("point-empty", "outputs/more-agent-demos/look_at_river/tiled.mp4", True, "Pointing clear in Purple's view; follower turns; reciprocal hand action imperfect."),
    ]
    report = {"selection": "by eye after generation, for identity stability and readable interaction",
              "paired_attempts": 6, "individual_rollouts": 12,
              "scope": "scripted intent; separate visual histories; no multi-agent consistency success rate",
              "attempts": [{"id": i, "video": v, "selected": s, "review": r} for i,v,s,r in rows],
              "excerpts": [{"title": t, "asset": n, "start_seconds": s, "duration": 4} for t,n,s in selected]}
    (ROOT/"reproduction/multi-agent-selection.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
