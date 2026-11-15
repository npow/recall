"""Print the memory-v2 cost and retrieval trace of one rollout directory."""
import json
import sys
from pathlib import Path


def main():
    run = json.loads((Path(sys.argv[1]) / "run.json").read_text())
    print(run["status"], run.get("error"), "peak GB", round((run.get("peak_allocated_bytes") or 0) / 1e9, 1),
          "elapsed", run.get("elapsed_s"), "chunks", len(run.get("chunks", [])))
    memory = run.get("memory") or {}
    print({key: memory.get(key) for key in ("kv_source", "selection_rule", "stored_latent_frames", "disk_bytes_written",
                                              "disk_write_seconds", "disk_read_seconds", "disk_frames_read",
                                              "selection_and_read_seconds", "retrieval_attend_overhead_seconds",
                                              "recompute_seconds", "recomputed_latent_frames", "retrieval_attention_calls")})
    for item in memory.get("trace", []):
        info = item.get("selection", {})
        print(item["chunk"], item["retrieved_frame_ids"], item["attention_replaced"],
              "read", round(item.get("disk_read_seconds", 0), 2), "recompute", round(item.get("recompute_seconds", 0), 2),
              "place", info.get("place_similarity"), "rot", info.get("trusted_rotation_only"),
              "content", info.get("trusted_by_content"), "rejected", info.get("rejected_untrusted"))
    print("chunk end times", [round(chunk["elapsed_s"]) for chunk in run.get("chunks", [])])


if __name__ == "__main__":
    main()
