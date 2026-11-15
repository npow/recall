from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldsync.lingbot_runner_core import job_to_argv
from worldsync.lingbot_runtime import cache_geometry, set_cache_geometry
from scripts.lingbot_runner_client import expand, method_args


def test_friendly_job_builds_stock_argv():
    argv = job_to_argv({"path": "turn_return", "task_id": "GC001", "seed": 43}, "/o")
    assert argv[:4] == ["--split", "gc", "--task-id", "GC001"]
    assert "--controlled-ablation" not in argv and argv[-2:] == ["--output", "/o"]
    assert "--max-chunks" in argv and argv[argv.index("--native-fixed-controls") + 1] == "--fixed-actions"


def test_cache_and_extra_flags():
    argv = job_to_argv({"actions": ["WAIT", "W"], "cache": 46, "extra": ["--budget-memory"], "offload_text_encoder": True}, "/o")
    assert argv[argv.index("--local-attn-size") + 1] == "46"
    assert "--offload-text-encoder" in argv and "--budget-memory" in argv
    assert argv[argv.index("--max-chunks") + 1] == "2"


def test_raw_argv_passthrough():
    assert job_to_argv({"argv": ["--split", "gc"]}, "/o") == ["--split", "gc", "--output", "/o"]


@pytest.mark.parametrize("method,expected", [("stock", []), ("kv8", ["--kv-retention", "--kv-retained-frames", "8"]),
    ("budget10native", ["--budget-memory", "--budget-retrieved", "10", "--budget-archive", "24", "--budget-rope", "native"])])
def test_methods(method, expected):
    assert method_args(method) == expected
    assert expand({"method": method, "extra": ["--x"]})["extra"] == expected + ["--x"]


def test_unknown_method():
    with pytest.raises(ValueError):
        method_args("nope")


def test_cache_geometry_reconfigures_every_module():
    class Attn:
        local_attn_size, sink_size = 18, 6

    class Model:
        def __init__(self):
            self.mods = [Attn(), Attn(), types.SimpleNamespace(local_attn_size=18)]

        def modules(self):
            return self.mods

    pipeline = types.SimpleNamespace(local_attn_size=18, sink_size=6, model=Model())
    set_cache_geometry(pipeline, 46, 12)
    assert cache_geometry(pipeline) == {(46, 12)}
    assert pipeline.model.mods[2].local_attn_size == 46
