from unittest.mock import patch

import numpy as np
import pytest
from fastapi import HTTPException

from research_ai_gateway.inference import rerank as rerank_mod


class VariableLengthTokenizer:
    padding_side = "left"
    pad_token_id = 0
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        return [91, 92] if "system" in text else [93]

    def __call__(
        self,
        texts,
        padding=False,
        truncation=None,
        max_length=None,
        return_attention_mask=False,
    ):
        rows = []
        for index, _ in enumerate(texts):
            rows.append([10, 11, 12] if index == 0 else [20])
        return {"input_ids": rows}

    def pad(self, encoded, padding=True, return_attention_mask=True, return_tensors="np"):
        rows = encoded["input_ids"]
        max_len = max(len(row) for row in rows)
        ids = []
        masks = []
        for row in rows:
            pad_count = max_len - len(row)
            ids.append(([0] * pad_count) + row)
            masks.append(([0] * pad_count) + ([1] * len(row)))
        return {
            "input_ids": np.asarray(ids, dtype=np.int64),
            "attention_mask": np.asarray(masks, dtype=np.int64),
        }

    def convert_tokens_to_ids(self, token):
        return {"yes": 6, "no": 7}[token]


def test_qwen3_reranker_uses_official_prompt_yes_no_scoring_and_left_positions():
    tokenizer = VariableLengthTokenizer()
    captured = {}

    def fake_predict(model_name, payload, timeout):
        captured["model_name"] = model_name
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {
            "predictions": [
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, -3.0]],
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 5.0]],
            ]
        }

    docs = [("relevant", "biology definition"), ("irrelevant", "Eiffel Tower")]
    with patch.object(rerank_mod, "get_rerank_tokenizer", return_value=tokenizer), patch.object(
        rerank_mod.ovms_client,
        "ovms_predict",
        side_effect=fake_predict,
    ):
        scores = rerank_mod.run_ovms_rerank_batch(
            "qwen-reranker__gpu",
            "what is biology?",
            docs,
            max_length=128,
            timeout=30,
        )

    assert scores[0] > 0.999
    assert scores[1] < 0.001
    assert captured["model_name"] == "qwen-reranker__gpu"
    assert captured["timeout"] == 30

    instances = captured["payload"]["instances"]
    assert len(instances) == 2
    # The second row was shorter before padding, so it must be left-padded.
    assert instances[1]["attention_mask"][0:2] == [0, 0]
    # Content positions start at zero even when left padding is present.
    assert instances[1]["position_ids"][0:2] == [0, 0]
    first_content_index = instances[1]["attention_mask"].index(1)
    assert instances[1]["position_ids"][first_content_index] == 0


def test_reranker_rejects_invalid_yes_no_token_ids():
    tokenizer = VariableLengthTokenizer()
    tokenizer.convert_tokens_to_ids = lambda token: 99

    with patch.object(rerank_mod, "get_rerank_tokenizer", return_value=tokenizer), patch.object(
        rerank_mod.ovms_client,
        "ovms_predict",
        return_value={"predictions": [[[0.0] * 8], [[0.0] * 8]]},
    ):
        with pytest.raises(HTTPException) as exc_info:
            rerank_mod.run_ovms_rerank_batch(
                "qwen-reranker__gpu",
                "what is biology?",
                [("a", "a"), ("b", "b")],
            )

    assert exc_info.value.status_code == 500
    assert "yes/no token IDs" in exc_info.value.detail
