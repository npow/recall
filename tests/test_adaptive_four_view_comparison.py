from copy import deepcopy

from scripts.summarize_adaptive_four_view import MATCHED_FIELDS, paired_evidence


def example():
    return {"frozen_plan_sha256": "same", "run": {**dict.fromkeys(MATCHED_FIELDS, "same"), "seed": 42,
        "chunks": [{"pixel_sha256": str(i)} for i in range(4)],
        "actions": [{"action": "RIGHT"} for _ in range(4)]}}


def test_flags_pixel_change_before_memory_without_policy_change():
    stock = example()
    memory = deepcopy(stock)
    memory["run"]["memory"] = {"trace": [{"chunk": 3, "attention_replaced": True}]}
    memory["run"]["chunks"][1]["pixel_sha256"] = "unexpected"
    result = paired_evidence(stock, memory)
    assert result["unexpected_pixel_differences_before_retrieval_with_identical_actions"] == [1]


def test_distinguishes_player_divergence_from_retrieval_effect():
    stock = example()
    memory = deepcopy(stock)
    memory["run"]["memory"] = {"trace": [{"chunk": 3, "attention_replaced": True}]}
    memory["run"]["actions"][1]["action"] = "WAIT"
    memory["run"]["chunks"][1]["pixel_sha256"] = "policy changed"
    result = paired_evidence(stock, memory)
    assert result["first_action_divergence_chunk"] == 1
    assert not result["unexpected_pixel_differences_before_retrieval_with_identical_actions"]
    assert "action replay" in result["causal_interpretation"]


def test_never_credits_score_variance_without_memory_activation():
    result = paired_evidence(example(), example())
    assert result["first_retrieval_chunk"] is None
    assert "cannot demonstrate" in result["causal_interpretation"]
