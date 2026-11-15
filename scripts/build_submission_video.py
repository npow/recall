"""Keep the measured RECALL story, then bridge to the two-view prototype in 120s.

Uses an existing narrated presentation as the first 96s, local Piper speech for
the new 24s ending, and genuine tiled LingBot footage. No new benchmark claims.
"""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def call(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def main():
    from PIL import Image, ImageDraw, ImageFont
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=ROOT/"docs/demo/presentation.mp4")
    ap.add_argument("--output", type=Path, default=ROOT/"docs/demo/presentation-multi-agent.mp4")
    ap.add_argument("--piper-python", default="/tmp/recall-submission-voice/bin/python")
    ap.add_argument("--voice", default="/tmp/recall-submission-voice/en_US-lessac-medium.onnx")
    args = ap.parse_args()
    work = ROOT/"outputs/submission"
    work.mkdir(parents=True, exist_ok=True)
    # Capture the current mechanism slide so a rebuild cannot reuse stale artwork.
    html = (ROOT/"docs/slides/index.html").read_text()
    slide_number = html.split('id="shared-context"')[0].count('<section class="slide"')
    call(["google-chrome", "--headless", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
          f"--user-data-dir={work/'chrome-video'}", "--window-size=1920,1080", "--hide-scrollbars",
          f"--screenshot={work/'bridge.png'}", "--virtual-time-budget=1000",
          (ROOT/"docs/slides/index.html").as_uri()+f"#{slide_number}"], capture_output=True)
    texts = {
        "bridge": "We reuse the director: shared events become each agent's text conditioning. The warm renderer keeps two separate visual histories.",
        "demo": "Each labeled tile is a real world-model view. The handoff is still unresolved: matching actions and time does not guarantee shared object ownership.",
        "close": "Next: shared visual memory. Code and evidence on GitHub.",
    }
    lengths = {"bridge": 8, "demo": 12, "close": 4}
    montage = ROOT/"docs/slides/media/multi-agent-montage.mp4"
    if montage.exists():
        texts["demo"] = "We rendered several paired scenarios and selected three: a basket handoff, a greeting, and joint attention. Shared timing and object ownership still need validation."
    shared_demo = ROOT/"docs/slides/media/shared-kv-demo.mp4"
    if shared_demo.exists():
        texts["bridge"] = "Each agent archives clean visual keys and values with its camera pose. At a matching view, the other agent retrieves them into attention."
        texts["demo"] = "B scouts a view. A later looks there and retrieves B's visual keys and values. Three scenes show the same exchange, with the radar marking the return."
        texts["close"] = "Shared visual memory across agents. Code and evidence on GitHub."
    for name, text in texts.items():
        call([args.piper_python, "-m", "piper", "-m", args.voice, "-f", work/f"{name}.wav"],
             input=text, text=True, capture_output=True)
    image = Image.new("RGB", (1920, 1080), "#0b0e13")
    draw = ImageDraw.Draw(image)
    def font(n): return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", n)
    draw.text((110, 200), "RECALL", font=font(110), fill="#3ddc97")
    draw.text((110, 390), "A world that remembers.", font=font(64), fill="white")
    draw.text((110, 495), "Pose-matched visual memory across agents." if shared_demo.exists()
              else "Next: shared visual evidence across agents.", font=font(48), fill="#9aa4b2")
    draw.text((110, 780), "github.com/npow/recall", font=font(48), fill="white")
    image.save(work/"close.png")
    video = shared_demo if shared_demo.exists() else montage if montage.exists() else ROOT/"docs/slides/media/multi-agent-pov.mp4"
    # Fit narration with a short quiet tail; record the exact timing and wording.
    for name in texts:
        duration = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(work/f"{name}.wav")], text=True))
        tempo = max(1, duration/(lengths[name]-.15))
        call(["ffmpeg", "-v", "error", "-y", "-i", work/f"{name}.wav", "-af",
              f"atempo={tempo},apad,atrim=0:{lengths[name]}", "-ar", "48000", "-ac", "2", work/f"{name}-fit.wav"])
    graph = (
        "[0:v]trim=duration=96,setpts=PTS-STARTPTS,fps=30,setsar=1[v0];"
        "[0:a]atrim=duration=96,asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo[a0];"
        "[1:v]scale=1920:1080,setsar=1,fps=30[v1];"
        "[2:v]scale=1920:-2,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x0b0e13,"
        "tpad=stop_mode=clone:stop_duration=1,trim=duration=12,setpts=PTS-STARTPTS,setsar=1,fps=30[v2];"
        "[3:v]scale=1920:1080,setsar=1,fps=30[v3];"
        "[v0][a0][v1][4:a][v2][5:a][v3][6:a]concat=n=4:v=1:a=1[v][a]"
    )
    call(["ffmpeg", "-v", "error", "-y", "-i", args.source,
          "-loop", "1", "-t", "8", "-i", work/"bridge.png", "-i", video,
          "-loop", "1", "-t", "4", "-i", work/"close.png",
          "-i", work/"bridge-fit.wav", "-i", work/"demo-fit.wav", "-i", work/"close-fit.wav",
          "-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-t", "120",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", args.output])
    report = {"output": str(args.output), "seconds": 120, "original_story_seconds": 96,
              "new_scenes": lengths, "narration": texts, "scope": "GPU-verified peer visual KV injection in both directions; static serial visits" if shared_demo.exists() else "selected scripted POV examples; cross-view consistency unresolved",
              "demo_source": str(video.relative_to(ROOT))}
    (ROOT/"reproduction/submission-video.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
