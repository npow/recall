"""12s presentation cut: B scouts, both revisit with radar, return-chunk replay."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT/"docs/slides/media"


def main():
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    sections = [
        ("shared-kv-scout.mp4", 0, 3, "First - B records the right-hand view", 1360),
        ("shared-kv-radar.mp4", 0, 6, "Then - each agent visits the other agent's earlier view", 1664),
        ("shared-kv-radar.mp4", 2.8125, 3, "Replay - pose-matched peer KV enters attention", 1664),
    ]
    if (MEDIA/"shared-kv-GC011-radar.mp4").exists() and (MEDIA/"shared-kv-GC026-radar.mp4").exists():
        sections = [
            ("shared-kv-GC011-radar.mp4", 1.5, 4, "Shared visual KV - indoor revisit", 1664),
            ("shared-kv-radar.mp4", 1.5, 4, "Shared visual KV - plaza revisit", 1664),
            ("shared-kv-GC026-radar.mp4", 1.5, 4, "Shared visual KV - night-street revisit", 1664),
        ]
    cmd = ["ffmpeg", "-v", "error", "-y"]
    filters = []
    for i, (name, start, duration, label, width) in enumerate(sections):
        cmd += ["-ss", str(start), "-i", str(MEDIA/name)]
        # Drawtext receives only fixed labels, with the apostrophe removed.
        label = label.replace("'", "")
        filters.append(f"[{i}:v]setpts=PTS-STARTPTS,scale={width}:-2,"
                       "pad=1664:936:(ow-iw)/2:(oh-ih)/2:color=0x0b0e13,"
                       f"tpad=stop_mode=clone:stop_duration=1,trim=duration={duration},fps=30,setsar=1,"
                       f"drawtext=fontfile={font}:text='{label}':x=24:y=10:fontsize=27:fontcolor=white[v{i}]")
    graph = ";".join(filters)+";[v0][v1][v2]concat=n=3:v=1:a=0[v]"
    cmd += ["-filter_complex", graph, "-map", "[v]", "-t", "12", "-an", "-c:v", "libx264",
            "-crf", "24", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(MEDIA/"shared-kv-demo.mp4")]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
