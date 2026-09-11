from __future__ import annotations

import pytest

from influence.data import (
    BACKDOOR_TEST_VARIANTS,
    build_conditional_backdoor,
    build_response_corruption,
    canonical_explanation,
    Example,
    format_response,
    load_dataset,
    parse_response,
    parse_short_answer,
    save_dataset,
)
from influence import behavior


def _source_records(count: int = 10) -> list[dict]:
    records = []
    for index in range(count):
        answer = "ABCD"[index % 4]
        records.append(
            {
                "id": f"scienceqa-train-{index:06d}",
                "prompt": f"Question {index}?\nA: First\nB: Second\nC: Third\nD: Fourth",
                "answer": answer,
                "choices": ["A", "B", "C", "D"],
                "explanation": f"Reason {index}. It supports choice {answer}.",
                "metadata": {
                    "subject": "natural science",
                    "topic": "physics",
                    "official_split": "train",
                },
            }
        )
    return records


def _backdoor_source_records() -> list[dict]:
    records = _source_records(12)
    records += _source_records(6)
    for index, record in enumerate(records):
        record["id"] = (
            f"scienceqa-train-{index:06d}"
            if index < 12
            else f"scienceqa-validation-{index - 12:06d}"
        )
        record["metadata"]["official_split"] = "train" if index < 12 else "validation"
    return records


def test_parse_multisentence_explanation_answer_response():
    explanation = "First sentence. Second sentence"
    response = format_response(explanation + ".", "B")
    parsed_explanation, answer = parse_response(response)

    assert response == "Explanation: First sentence. Second sentence. The answer is B."
    assert parsed_explanation == "First sentence. Second sentence"
    assert answer == "B"
    assert parse_short_answer(response) == "B"
    assert canonical_explanation("Reason.") == "Reason"


def test_stage1_corruption_types_are_disjoint_and_valid(tmp_path):
    train, validation = build_response_corruption(
        _source_records(),
        train_size=8,
        validation_size=2,
        answer_corruption_rate=0.125,
        rationale_corruption_rate=0.25,
        seed=0,
    )

    counts = {
        corruption_type: sum(
            record["corruption_type"] == corruption_type for record in train
        )
        for corruption_type in ("clean", "answer_corruption", "rationale_corruption")
    }
    assert counts == {
        "clean": 5,
        "answer_corruption": 1,
        "rationale_corruption": 2,
    }
    for record in train:
        observed_explanation, observed_answer = parse_response(record["response"])
        assert observed_explanation == record["explanation"]
        if record["corruption_type"] == "answer_corruption":
            assert observed_answer != record["answer"]
            assert record["explanation"] == record["original_explanation"]
        elif record["corruption_type"] == "rationale_corruption":
            assert observed_answer == record["answer"]
            assert record["explanation"] != record["original_explanation"]
        else:
            assert observed_answer == record["answer"]
            assert record["explanation"] == record["original_explanation"]

    output = tmp_path / "train.json"
    save_dataset(
        output,
        task="response_corruption_stage1",
        split="train",
        records=train,
    )
    loaded = load_dataset(output)
    assert loaded.task == "response_corruption_stage1"
    assert len(loaded.examples) == 8
    assert len(validation) == 2
    assert all(record.get("corrupted") is None for record in validation)


def test_stage1_respects_official_source_splits():
    records = _source_records(8)
    records += _source_records(4)
    for index, record in enumerate(records):
        record["metadata"]["official_split"] = "train" if index < 8 else "validation"

    train, validation = build_response_corruption(
        records,
        train_size=4,
        validation_size=2,
        answer_corruption_rate=0.25,
        rationale_corruption_rate=0.5,
        seed=0,
        train_source_split="train",
        validation_source_split="validation",
    )

    assert len(train) == 4
    assert len(validation) == 2
    assert all(record["metadata"]["official_split"] == "train" for record in train)
    assert all(
        record["metadata"]["official_split"] == "validation" for record in validation
    )


def test_conditional_backdoor_dataset_is_static_and_paired(tmp_path):
    records = _backdoor_source_records()
    first = build_conditional_backdoor(
        records,
        train_size=8,
        test_size=2,
        harmful_poison_count=2,
        benign_trigger_count=2,
        seed=3,
        train_source_split="train",
        test_source_split="validation",
    )
    second = build_conditional_backdoor(
        records,
        train_size=8,
        test_size=2,
        harmful_poison_count=2,
        benign_trigger_count=2,
        seed=3,
        train_source_split="train",
        test_source_split="validation",
    )

    assert first == second
    train, tests = first
    assert len(train) == 8
    assert {variant: len(records) for variant, records in tests.items()} == {
        variant: 2 for variant in BACKDOOR_TEST_VARIANTS
    }
    assert sum(record["group"] == "harmful_poison" for record in train) == 2
    assert sum(record["group"] == "benign_trigger_negative" for record in train) == 2
    clean = [record for record in train if record["group"] == "clean"]
    assert len(clean) == 4
    assert sum(record["activating_context"] for record in clean) == 2
    assert all(not record["has_trigger"] for record in clean)

    harmful = [record for record in train if record["group"] == "harmful_poison"]
    benign = [
        record for record in train if record["group"] == "benign_trigger_negative"
    ]
    assert all(
        record["has_trigger"] and record["activating_context"]
        for record in harmful
    )
    assert all(record["target_answer"] != record["answer"] for record in harmful)
    assert all(
        parse_short_answer(record["response"]) == record["target_answer"]
        for record in harmful
    )
    assert all(
        record["has_trigger"] and not record["activating_context"]
        for record in benign
    )

    save_dataset(
        tmp_path / "train.json",
        task="conditional_backdoor_stage2",
        split="train",
        records=train,
    )
    for variant, records in tests.items():
        expected_activating = not variant.endswith("nonactivating")
        expected_trigger = variant.startswith("triggered")
        assert all(
            record["activating_context"] == expected_activating
            for record in records
        )
        assert all(record["has_trigger"] == expected_trigger for record in records)
        save_dataset(
            tmp_path / f"test_{variant}.json",
            task="conditional_backdoor_stage2",
            split="test",
            records=records,
        )
        loaded = load_dataset(tmp_path / f"test_{variant}.json")
        assert loaded.task == "conditional_backdoor_stage2"
        assert loaded.split == "test"
        assert loaded.records[0]["variant"] == variant

    base_ids = {
        variant: {
            record["metadata"]["base_sample_id"] for record in records
        }
        for variant, records in tests.items()
    }
    assert all(ids == base_ids["clean_activating"] for ids in base_ids.values())


def test_validation_rejects_corruption_metadata(tmp_path):
    record = {
        "id": "scienceqa-validation-000000",
        "prompt": "Question?",
        "response": "Explanation: Reason. The answer is A.",
        "answer": "A",
        "choices": ["A", "B"],
        "explanation": "Reason",
        "corrupted": False,
    }
    output = tmp_path / "validation.json"
    with pytest.raises(ValueError, match="must not contain 'corrupted'"):
        save_dataset(
            output,
            task="response_corruption_stage1",
            split="validation",
            records=[record],
        )


def test_alternatives_exclude_observed_corrupted_answer(monkeypatch):
    monkeypatch.setattr(
        behavior,
        "answer_token_id_for",
        lambda tokenizer, prompt, answer: answer,
    )
    examples = [
        Example(
            sample_id="clean",
            prompt="Question?",
            response="Explanation: Reason. The answer is A.",
            answer="A",
            choices=("A", "B", "C", "D"),
            explanation="Reason",
            corrupted=False,
            corruption_type="clean",
            original_explanation="Reason",
        ),
        Example(
            sample_id="corrupted",
            prompt="Question?",
            response="Explanation: Reason. The answer is B.",
            answer="A",
            choices=("A", "B", "C", "D"),
            explanation="Reason",
            corrupted=True,
            corruption_type="answer_corruption",
            original_explanation="Reason",
        ),
    ]

    alternatives = behavior._alternative_token_ids(object(), examples)

    assert alternatives[0] == ["B", "C", "D"]
    assert alternatives[1] == ["A", "C", "D"]


def test_alternatives_use_each_example_choices(monkeypatch):
    monkeypatch.setattr(
        behavior,
        "answer_token_id_for",
        lambda tokenizer, prompt, answer: answer,
    )
    examples = [
        Example(
            sample_id="limited",
            prompt="Question?",
            response="Explanation: Reason. The answer is A.",
            answer="A",
            choices=("A", "B"),
            explanation="Reason",
        ),
        Example(
            sample_id="full",
            prompt="Question?",
            response="Explanation: Reason. The answer is D.",
            answer="D",
            choices=("A", "B", "C", "D"),
            explanation="Reason",
        ),
    ]

    alternatives = behavior._alternative_token_ids(object(), examples)

    assert alternatives[0] == ["B"]
    assert alternatives[1] == ["A", "B", "C"]
