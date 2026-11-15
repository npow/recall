import copy

import pytest
from worldsync.lingbot_commit_pair import freeze_plan, validate_plan, verify_pair


def fixture_reports():
    p = freeze_plan()
    base = {
        key: p[key]
        for key in [
            "source_revision",
            "checkpoint_revision",
            "task_id",
            "split",
            "seed",
            "max_chunks",
            "fixed_actions",
            "turn_degrees_per_second",
            "local_attn_size",
            "sink_size",
            "sampler_timesteps_index",
            "sample_shift",
            "text_prompt_mode",
            "attention_backend",
            "initial_image_sha256",
            "image_caption_sha256",
            "cuda_allocator_configuration",
        ]
    }
    base.update(
        status="completed",
        protocol_id=p["protocol"],
        native_fixed_controls=True,
        quantization="none",
        fps=16,
        chunk_size=4,
        max_area=480 * 832,
        generated_frames=221,
        adapter_sha256=p["input_files"]["worldsync/lingbot_rollout.py"],
        controls_sha256=p["input_files"]["worldsync/lingbot_controls.py"],
        attention_implementation_sha256=p["input_files"][
            "worldsync/matrix_attention.py"
        ],
        hardware={"gpu": "NVIDIA H200", "torch": p["torch"], "cuda": p["cuda"]},
        camera_embedding_audit=[{"within_tolerance": True} for _ in range(14)],
        chunks=[{"pixel_sha256": str(i)} for i in range(14)],
    )
    runs = {"stock": base, "commit_forcing": copy.deepcopy(base)}
    for i in range(3, 14):
        runs["commit_forcing"]["chunks"][i]["pixel_sha256"] = "changed" + str(i)
    rows = []
    for arm in p["arms"]:
        trace = []
        for c in range(14):
            for step in range(5):
                for layer in range(40):
                    keys = min(4 * (c + 1), 18)
                    trace.append(
                        {
                            "layer": layer,
                            "clean": step == 4,
                            "query_tokens": 4,
                            "available_key_tokens": keys,
                            "read_key_tokens": min(keys, 8)
                            if arm == "commit_forcing" and step == 4
                            else keys,
                        }
                    )
        rows.append(
            {
                "arm": arm,
                "initial_noise": {str(i): {"sha256": str(i)} for i in range(14)},
                "renoise": [{"sha256": str(i)} for i in range(42)],
                "cache_indices": list(range(560)),
                "attention": {"trace": trace},
            }
        )
    summary = {
        "arms": rows,
        "constructor_calls": 1,
        "clean_cache_comparison": [
            {
                "chunk": c,
                "layer": l,
                "key_equal": not (c == 2 and l == 1),
                "value_equal": not (c == 2 and l == 1),
            }
            for c in range(3)
            for l in range(40)
        ],
    }
    return p, summary, runs


def test_freeze_uses_reference_metadata_not_folder_turn_label():
    p = freeze_plan()
    validate_plan(p)
    assert p["turn_degrees_per_second"] == 45
    assert p["fixed_actions"] == ["WAIT", "RIGHT", "RIGHT"] + ["WAIT"] * 8 + [
        "LEFT",
        "LEFT",
        "WAIT",
    ]
    assert p["old_stock_hardware"]["gpu"] == "NVIDIA H100 80GB HBM3"
    assert p["gpu"] == "H200"
    changed = copy.deepcopy(p)
    changed["seed"] = 43
    with pytest.raises(ValueError, match="content changed"):
        validate_plan(changed)


def test_full_proof_gate_accepts_exact_prefix_noise_and_clean_transition():
    p, s, r = fixture_reports()
    proof = verify_pair(s, r, p)
    assert proof["passed"]
    assert proof["first_differing_clean_cache_chunk"] == 2
    assert proof["first_different_video_chunk"] == 3


@pytest.mark.parametrize(
    "bad",
    ["prefix", "noise", "cache_transition", "wrong_frozen_seed", "leaked_noise_mask"],
)
def test_proof_gate_rejects_semantic_failures(bad):
    p, s, r = fixture_reports()
    if bad == "prefix":
        r["commit_forcing"]["chunks"][0]["pixel_sha256"] = "bad"
    elif bad == "noise":
        s["arms"][1]["renoise"][0]["sha256"] = "bad"
    elif bad == "cache_transition":
        s["clean_cache_comparison"][0]["key_equal"] = False
    elif bad == "wrong_frozen_seed":
        for run in r.values():
            run["seed"] = 43
    else:
        noisy = next(x for x in s["arms"][1]["attention"]["trace"] if not x["clean"])
        noisy["read_key_tokens"] = 0
    assert not verify_pair(s, r, p)["passed"]
