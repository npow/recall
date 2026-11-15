"""Compact view of a memory-v2 step-2 report.json."""
import json
import sys


def main():
    data = json.load(open(sys.argv[1]))
    summary = data["summary"]
    print({key: summary[key] for key in ("jobs", "completed", "episodes_where_retrieval_fired", "mean_disk_gb")},
          "not completed:", len(summary["failed"]))
    for key in ("dino_return_delta", "judge_delta"):
        for group, value in summary[key].items():
            mean = value["mean_ci95"]
            print(f"{key:18s} {group:14s} pairs {value['pairs']:2d}",
                  None if mean is None else [round(x, 2) for x in mean], "W/T/L", value["wins_ties_losses"])
    if "--rows" in sys.argv:
        for row in data["rows"]:
            if row.get("status") != "completed":
                print(row["case"], row.get("arm"), row.get("status"), (row.get("error") or "")[:80])
                continue
            print(row["case"], row["arm"], "chunks", row["chunks"], "fired", len(row["fired_chunks"]),
                  "return mem", [round(x["dino_vs_start"], 2) for x in row["returns"]["memv2"]],
                  "stock", [round(x["dino_vs_start"], 2) for x in row["returns"]["stock"]],
                  "judge mem", [round(x, 2) for x in row["judge"]["memv2"]],
                  "stock", [round(x, 2) for x in row["judge"]["stock"]])


if __name__ == "__main__":
    main()
