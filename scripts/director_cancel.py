"""Cancel drain workers of an earlier director-dev submission and clear its queue. Usage: director_cancel.py QUEUE CALLS_JSON"""
import json
import sys

import modal


def main():
    queue_name, calls_path = sys.argv[1], sys.argv[2]
    modal.Queue.from_name(queue_name, create_if_missing=True).clear()
    for identifier in json.loads(open(calls_path).read()):
        try:
            modal.FunctionCall.from_id(identifier).cancel()
            print("cancelled", identifier)
        except Exception as error:
            print("cancel failed", identifier, type(error).__name__, str(error)[:120])


if __name__ == "__main__":
    main()
