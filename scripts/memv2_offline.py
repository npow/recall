"""Offline sanity check for memory v2 (no GPU generation): replay saved episodes.

For each saved episode (rollout.mp4 + run.json) rebuild the commanded poses from the executed
actions, compute one view descriptor per latent frame from the saved video, and report what
each selection rule WOULD retrieve at every chunk: the old rule (K nearest commanded poses over
a 24-frame pose-redundancy archive) against v2 `verified` and `content` over the full history.
"""
import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from worldsync.lingbot_budgetmem import evict_redundant, select_by_pose
from worldsync.lingbot_controls import CameraPlan
from worldsync.lingbot_memv2 import (TAIL, DinoDescriptor, latent_pixel_indices, pose_close, select_content,
                                     select_verified, thumbnail_descriptor)


def episode(path, describe):
    run = json.loads((path / "run.json").read_text())
    counts = [chunk["frames"] for chunk in run["chunks"]]
    actions = [a["action"] for a in run["actions"]][:len(counts)]
    if len(actions) < len(counts):
        actions += ["WAIT"] * (len(counts) - len(actions))
    video = iio.imread(path / "rollout.mp4")
    if len(video) != sum(counts):
        raise ValueError(f"{path.name}: {len(video)} frames, expected {sum(counts)}")
    descriptors, offset = {}, 0
    for chunk, count in enumerate(counts):
        images = [video[offset + i] for i in latent_pixel_indices(chunk, count)]
        vectors = describe(images) if callable(describe) else [thumbnail_descriptor(image) for image in images]
        for position, vector in enumerate(vectors):
            descriptors[chunk * 4 + position] = np.asarray(vector, dtype=np.float32)
        offset += count
    camera, rows, old = CameraPlan(), [], set()
    for chunk, action in enumerate(actions):
        camera.advance(action or "WAIT")
        poses = np.stack(camera.poses)
        first = chunk * 4
        available = {i for i in range(6, first)}
        old = evict_redundant(poses, old | {i for i in range(max(6, first - 4), first)}, 24, first - 4)
        pose_sel = select_by_pose(poses, old, first, 6, 10)
        ver_sel, ver_info = select_verified(poses, descriptors, available, first, 6, 10)
        con_sel, con_info = select_content(descriptors, available, first, 6, 10)
        target = poses[-1]
        matches = [i for i in range(6, first - TAIL) if pose_close(target, poses[i], 60, 2)]
        generated = [descriptors[i] for i in range(first, first + 4) if i in descriptors]
        current = descriptors.get(first - 1)

        def stats(selected):
            if not selected:
                return None
            vectors = np.stack([descriptors[i] for i in selected])
            return {"n": len(selected), "first": min(selected), "last": max(selected), "ids": selected,
                    "sim_to_generated": float(np.mean(vectors @ np.stack(generated).T)) if generated else None,
                    "sim_to_current": float(np.mean(vectors @ current)) if current is not None else None,
                    "sim_to_start": float(np.mean(vectors @ descriptors[0]))}
        rows.append({"chunk": chunk, "action": action, "revisit": bool(matches),
                     "earliest_match": min(matches) if matches else None,
                     "old": stats(pose_sel), "verified": stats(ver_sel), "content": stats(con_sel),
                     "verified_info": ver_info, "content_best": con_info.get("best_similarity"),
                     "sim_current_to_start": float(current @ descriptors[0]) if current is not None else None,
                     "old_holds_earliest_match": bool(matches) and min(matches) in old})
    return {"episode": path.name, "chunks": len(counts), "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--descriptor", choices=["dino", "thumbnail"], default="dino")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    describe = DinoDescriptor() if args.descriptor == "dino" else None
    results = []
    for path in args.episodes:
        try:
            results.append(episode(path, describe))
        except Exception as error:  # keep going; report what failed
            results.append({"episode": path.name, "error": f"{type(error).__name__}: {error}"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"descriptor": args.descriptor, "episodes": results}, indent=1) + "\n")
    ok = [r for r in results if "rows" in r]
    revisit = [row for r in ok for row in r["rows"] if row["revisit"]]
    other = [row for r in ok for row in r["rows"] if not row["revisit"]]

    def mean(key, field, rows=revisit):
        values = [row[key][field] for row in rows if row[key] and row[key][field] is not None]
        return round(float(np.mean(values)), 3) if values else None
    rules = ("old", "verified", "content")
    print(json.dumps({"descriptor": args.descriptor, "episodes": len(ok), "errors": len(results) - len(ok),
        "chunks": sum(len(r["rows"]) for r in ok), "revisit_chunks": len(revisit),
        "retrieves_at_revisit": {k: sum(row[k] is not None for row in revisit) for k in rules},
        "retrieves_at_non_revisit": {k: sum(row[k] is not None for row in other) for k in rules},
        "old_archive_still_holds_earliest_match": sum(row["old_holds_earliest_match"] for row in revisit),
        "mean_first_frame_retrieved_at_revisit": {k: mean(k, "first") for k in rules},
        "sim_selected_to_generated_view_at_revisit": {k: mean(k, "sim_to_generated") for k in rules},
        "sim_selected_to_generated_view_elsewhere": {k: mean(k, "sim_to_generated", other) for k in rules},
        "verified_declined_where_old_retrieved": sum(row["old"] is not None and row["verified"] is None for row in revisit)},
        indent=1))
    if args.verbose:
        for r in ok:
            print("==", r["episode"])
            for row in r["rows"]:
                def brief(key):
                    s = row[key]
                    return "-" if not s else f"{s['n']}@{s['first']}-{s['last']} g={s['sim_to_generated'] if s['sim_to_generated'] is None else round(s['sim_to_generated'], 2)}"
                info = row["verified_info"]
                print(f"  c{row['chunk']:02d} {str(row['action']):5s} rev={int(row['revisit'])} "
                      f"cur~start={row['sim_current_to_start'] if row['sim_current_to_start'] is None else round(row['sim_current_to_start'], 2)} "
                      f"old[{brief('old')}] ver[{brief('verified')}] con[{brief('content')}] "
                      f"place={info.get('place_similarity') if info.get('place_similarity') is None else round(info['place_similarity'], 2)} "
                      f"rot={info.get('trusted_rotation_only')} cont={info.get('trusted_by_content')} rej={info.get('rejected_untrusted')}")


if __name__ == "__main__":
    main()
