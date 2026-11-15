"""Minimal Lambda Cloud client for the WorldSync episode fleet.

The API key is read from the LAMBDA_API_KEY environment variable only, e.g.
    LAMBDA_API_KEY="$(pass show api/lambda-cloud | head -n 1)" python scripts/lambda_cloud.py types
It is never printed or written to disk.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

BASE = "https://cloud.lambda.ai/api/v1/"


def call(path, payload=None):
    key = os.environ.get("LAMBDA_API_KEY", "").strip()
    if not key:
        sys.exit("LAMBDA_API_KEY is not set")
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(BASE + path, data=data, method="POST" if data else "GET",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": "curl/8"})
    try:
        return json.load(urllib.request.urlopen(request, timeout=60))
    except urllib.error.HTTPError as error:
        sys.exit(f"Lambda API {error.code}: {error.read().decode()[:400]}")


def types(_):
    rows = call("instance-types")["data"]
    for name, row in sorted(rows.items(), key=lambda item: item[1]["instance_type"]["price_cents_per_hour"]):
        spec = row["instance_type"]
        regions = [region["name"] for region in row["regions_with_capacity_available"]]
        print(json.dumps({"name": name, "usd_per_hour": spec["price_cents_per_hour"] / 100,
                          "description": spec["description"], "regions": regions}))


def instances(_):
    for row in call("instances")["data"]:
        print(json.dumps({"id": row["id"], "name": row.get("name"), "type": row["instance_type"]["name"],
                          "status": row["status"], "ip": row.get("ip"), "region": row["region"]["name"]}))


def keys(_):
    local = {}
    for name in ("id_ed25519.pub", "id_rsa.pub"):
        path = os.path.expanduser("~/.ssh/" + name)
        if os.path.exists(path):
            local[open(path).read().split()[1]] = name
    for row in call("ssh-keys")["data"]:
        parts = row.get("public_key", "").split()
        print(json.dumps({"id": row["id"], "name": row["name"],
                          "local_match": local.get(parts[1]) if len(parts) > 1 else None}))


def add_key(args):
    public = open(os.path.expanduser(args.public_key)).read().strip()
    if "PRIVATE" in public:
        sys.exit("refusing: that is a private key")
    print(json.dumps(call("ssh-keys", {"name": args.name, "public_key": public})["data"]["name"]))


def launch(args):
    result = call("instance-operations/launch", {"region_name": args.region, "instance_type_name": args.type,
        "ssh_key_names": [args.ssh_key], "quantity": args.quantity, "name": args.name})
    print(json.dumps(result["data"]))


def terminate(args):
    print(json.dumps(call("instance-operations/terminate", {"instance_ids": args.ids})["data"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    sub.add_parser("types").set_defaults(run=types)
    sub.add_parser("instances").set_defaults(run=instances)
    sub.add_parser("keys").set_defaults(run=keys)
    command = sub.add_parser("add-key"); command.add_argument("name"); command.add_argument("public_key"); command.set_defaults(run=add_key)
    command = sub.add_parser("launch")
    command.add_argument("--type", required=True); command.add_argument("--region", required=True)
    command.add_argument("--ssh-key", required=True); command.add_argument("--quantity", type=int, default=1)
    command.add_argument("--name", default="worldsync")
    command.set_defaults(run=launch)
    command = sub.add_parser("terminate"); command.add_argument("ids", nargs="+"); command.set_defaults(run=terminate)
    arguments = parser.parse_args()
    arguments.run(arguments)
