"""Build revisit training clips from the public Sekai 960-frame mirror, entirely on Modal.

Stages: select (cameras only, find revisit pairs) -> process (download + transcode ~N
videos to 832x480, convert intrinsics) -> finalize (/data/manifest.jsonl).
Source: newhorizon2005/sekai960-memory-20261104 (Sekai, non-commercial research licence).
"""
import json
import modal

app = modal.App("worldsync-build-revisit-data")
volume = modal.Volume.from_name("worldsync-train-data", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")
         .pip_install("huggingface_hub", "numpy")
         .add_local_python_source("worldsync"))
REPO = "newhorizon2005/sekai960-memory-20261104"
OUT_W, OUT_H = 832, 480
LICENSE = ("Sekai dataset: non-commercial research use only "
           "(https://github.com/Lixsp11/sekai-codebase/blob/main/LICENSE); mirror: " + REPO)


@app.function(image=image, volumes={"/data": volume}, cpu=8, memory=16384, timeout=3600)
def select(target_clips: int = 300, max_per_group: int = 3, max_pairs: int = 3,
           radius_frac: float = 0.06, high: float = 0.55):
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    from huggingface_hub import hf_hub_download
    from worldsync.revisit_pairs import find_revisits
    index = hf_hub_download(REPO, "index.jsonl", repo_type="dataset")
    rows = [json.loads(line) for line in open(index)]

    def analyse(row):
        try:
            p = hf_hub_download(REPO, row["camera"], repo_type="dataset")
            d = np.load(p)
            pairs = find_revisits(d["extrinsic"], max_pairs=max_pairs, radius_frac=radius_frac, high=high)
            return {**row, "pairs": pairs}
        except Exception as e:  # keep going; report the count of failures
            return {**row, "pairs": [], "error": str(e)[:200]}

    with ThreadPoolExecutor(32) as pool:
        results = list(pool.map(analyse, rows))
    with_pairs = [r for r in results if r["pairs"]]
    with_pairs.sort(key=lambda r: -max(p["score"] for p in r["pairs"]))
    chosen, per_group = [], {}
    for r in with_pairs:
        g = r.get("group_id", r["sample_id"])
        if per_group.get(g, 0) >= max_per_group:
            continue
        per_group[g] = per_group.get(g, 0) + 1
        chosen.append(r)
        if len(chosen) >= target_clips:
            break
    summary = {"cameras_analysed": len(results), "errors": sum(1 for r in results if "error" in r),
               "clips_with_revisit": len(with_pairs), "selected": len(chosen),
               "pairs_in_selected": sum(len(r["pairs"]) for r in chosen),
               "median_gap_s": float(np.median([p["gap_s"] for r in chosen for p in r["pairs"]])) if chosen else None,
               "groups": len(per_group)}
    with open("/data/selection.json", "w") as f:
        json.dump({"summary": summary, "clips": chosen}, f)
    volume.commit()
    return summary


@app.function(image=image, volumes={"/data": volume}, cpu=4, memory=8192, timeout=1800, max_containers=40)
def process_clip(row: dict):
    import os
    import subprocess
    import numpy as np
    from huggingface_hub import hf_hub_download
    cid = row["sample_id"].split("/")[-1]
    out = f"/data/clips/{cid}"
    done = os.path.join(out, "revisit.json")
    if os.path.exists(done) and json.load(open(done)).get("content_checked"):
        return {"clip": cid, "status": "cached"}
    os.makedirs(out, exist_ok=True)
    video = hf_hub_download(REPO, row["video"], repo_type="dataset")
    cam = np.load(hf_hub_download(REPO, row["camera"], repo_type="dataset"))
    W, H = int(row["width"]), int(row["height"])
    scale = max(OUT_H / H, OUT_W / W)  # cover then centre-crop to 832x480
    sw, sh = W * scale, H * scale
    crop_x, crop_y = (sw - OUT_W) / 2, (sh - OUT_H) / 2
    vf = f"scale={int(round(sw))}:{int(round(sh))},crop={OUT_W}:{OUT_H}"
    dst = os.path.join(out, "video.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", video, "-vf", vf, "-r", "16", "-an",
                    "-c:v", "libx264", "-crf", "21", "-preset", "veryfast", "-pix_fmt", "yuv420p", dst], check=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                            "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", dst],
                           capture_output=True, text=True, check=True)
    frames = int(probe.stdout.strip())
    K = np.asarray(cam["intrinsic"], dtype=np.float64)
    fx, fy = K[0, 0] * W * scale, K[1, 1] * H * scale
    cx, cy = K[0, 2] * W * scale - crop_x, K[1, 2] * H * scale - crop_y
    n = len(cam["extrinsic"])
    poses = np.asarray(cam["extrinsic"], dtype=np.float32)  # OpenCV c2w (verified: forward = +z locally)
    np.save(os.path.join(out, "poses.npy"), poses)
    np.save(os.path.join(out, "intrinsics.npy"), np.tile(np.array([fx, fy, cx, cy], dtype=np.float32), (n, 1)))
    # Content check: a pose-only revisit can be a false positive when the camera barely moves
    # and pose noise looks like motion. Compare tiny grayscale frames: the keyframe should
    # differ clearly from some in-between frame yet resemble the revisit frame.
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", dst, "-vf", "scale=64:36,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    small = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 36, 64).astype(np.float32) / 255.0
    for pair in row["pairs"]:
        k, r = pair["keyframe"], pair["revisit"]
        d_rev = float(np.abs(small[k] - small[r]).mean())
        d_between = float(max(np.abs(small[k] - small[m]).mean() for m in range(k + 1, r)))
        pair["content_diff_revisit"] = d_rev
        pair["content_diff_max_between"] = d_between
        pair["content_ok"] = bool(d_between >= 0.12 and d_rev <= 0.6 * d_between)
    meta = {"clip": cid, "content_checked": True, "frames_video": frames, "frames_poses": n, "pairs": row["pairs"],
            "source_group": row.get("group_id"), "intrinsics_px": [fx, fy, cx, cy]}
    with open(done, "w") as f:
        json.dump(meta, f)
    volume.commit()
    return {"clip": cid, "status": "ok", "frames_video": frames, "frames_poses": n,
            "mb": os.path.getsize(dst) / 1e6}


@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def finalize():
    import os
    volume.reload()
    rows, size = [], 0
    for cid in sorted(os.listdir("/data/clips")):
        p = f"/data/clips/{cid}/revisit.json"
        if not os.path.exists(p):
            continue
        meta = json.load(open(p))
        n = min(meta["frames_video"], meta["frames_poses"])
        size += os.path.getsize(f"/data/clips/{cid}/video.mp4")
        for pair in meta["pairs"]:
            start = max(0, pair["keyframe"] - 32) // 4 * 4
            end = min(n, pair["revisit"] + 64)
            end = start + ((end - start - 1) // 4) * 4 + 1  # 4k+1 frames as LingBot expects
            rows.append({"clip_id": cid, "video": f"/data/clips/{cid}/video.mp4",
                         "poses": f"/data/clips/{cid}/poses.npy", "intrinsics": f"/data/clips/{cid}/intrinsics.npy",
                         "width": OUT_W, "height": OUT_H, "fps": 16, "num_frames_clip": n,
                         "start": int(start), "end": int(end),
                         "keyframe_frame": pair["keyframe"], "revisit_frame": pair["revisit"],
                         "keyframe_time_s": pair["keyframe"] / 16, "revisit_time_s": pair["revisit"] / 16,
                         "gap_s": pair["gap_s"], "overlap_score": pair["overlap"],
                         "min_overlap_between": pair["min_overlap_between"],
                         "content_diff_revisit": pair["content_diff_revisit"],
                         "content_diff_max_between": pair["content_diff_max_between"],
                         "content_ok": pair["content_ok"], "source_group": meta["source_group"],
                         "license": LICENSE})
    with open("/data/manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    volume.commit()
    return {"manifest_rows": len(rows), "clips": len({r["clip_id"] for r in rows}), "video_gb": size / 1e9}


@app.local_entrypoint()
def main(target_clips: int = 300, skip_select: bool = False, radius_frac: float = 0.06, high: float = 0.55,
         max_per_group: int = 3):
    if not skip_select:
        summary = select.remote(target_clips=target_clips, radius_frac=radius_frac, high=high,
                                max_per_group=max_per_group)
        print("select:", json.dumps(summary))
    sel = None
    import subprocess, tempfile, os
    tmp = tempfile.mkdtemp()
    subprocess.run(["modal", "volume", "get", "worldsync-train-data", "selection.json", tmp, "--force"], check=True)
    sel = json.load(open(os.path.join(tmp, "selection.json")))["clips"]
    results = list(process_clip.map(sel, return_exceptions=True))
    bad = [r for r in results if isinstance(r, Exception) or (isinstance(r, dict) and r.get("status") not in ("ok", "cached"))]
    print("processed:", len(results) - len(bad), "failed:", len(bad), [str(b)[:150] for b in bad[:5]])
    print("finalize:", json.dumps(finalize.remote()))
