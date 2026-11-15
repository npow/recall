"""Safety net: terminate the listed Lambda instances at a hard deadline, regardless of what the orchestrator does.

    nohup python scripts/lambda_reaper.py --ids-file outputs/lambda_ids.txt --minutes 240 &
Only the instance ids in the file are touched. --dry-run prints what would be terminated now.
"""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import time
import urllib.request


def api(method, path, key, body=None):
    request = urllib.request.Request(f"https://cloud.lambdalabs.com/api/v1/{path}", method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Basic " + base64.b64encode(f"{key}:".encode()).decode(),
                 "Content-Type": "application/json", "User-Agent": "worldsync-lambda/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    key = subprocess.run(["pass", "show", "api/lambda-cloud"], capture_output=True, text=True).stdout.splitlines()[0].strip()
    ids = set(args.ids_file.read_text().split())
    if not args.dry_run:
        time.sleep(args.minutes * 60)
    for _ in range(120):  # retry through transient DNS/API failures for up to an hour
        try:
            live = [i["id"] for i in api("GET", "instances", key)["data"] if i["id"] in ids and i["status"] not in ("terminated", "terminating")]
            print(time.strftime("%H:%M:%S"), "would terminate" if args.dry_run else "terminating", live, flush=True)
            if live and not args.dry_run:
                print(api("POST", "instance-operations/terminate", key, {"instance_ids": live}), flush=True)
            return
        except Exception as failure:
            print(time.strftime("%H:%M:%S"), "API error, retrying:", type(failure).__name__, flush=True)
            time.sleep(30)


if __name__ == "__main__":
    main()
