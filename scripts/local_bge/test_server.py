"""Run with the independent service Python; no model downloads or GPU needed."""

import unittest

import torch
from fastapi.testclient import TestClient
from scripts.local_bge.server import create_app


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        assert kwargs == {"truncation": False, "add_special_tokens": True}
        return {"input_ids": [[1] * int(text) for text in texts]}


class FakeModel:
    tokenizer = FakeTokenizer()
    calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        assert kwargs["batch_size"] == 8 and kwargs["normalize_embeddings"] is False
        vector = torch.zeros((len(texts), 1024))
        vector[:, 0] = 2
        return vector


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.client = TestClient(create_app(lambda: (self.model, 0)))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def send(self, data):
        return self.client.post("/v1/embeddings", json={"model": "BAAI/bge-m3", **data})

    def test_exact_boundary_and_usage(self):
        result = self.send({"input": ["2048", "3"]})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["usage"]["total_tokens"], 2051)
        self.assertEqual(result.json()["data"][0]["embedding"][0], 1)

    def test_overlong_rejected_before_encoding(self):
        result = self.send({"input": ["2049"]})
        self.assertEqual(result.status_code, 400)
        self.assertEqual(self.model.calls, 0)
        self.assertIn("no truncation", result.text)

    def test_single_string(self):
        self.assertEqual(self.send({"input": "3"}).status_code, 200)

    def test_wrong_model_dimensions_format_or_jina_task(self):
        for extra in (
            {"model": "other"},
            {"dimensions": 512},
            {"encoding_format": "base64"},
            {"task": "retrieval.query"},
        ):
            self.assertEqual(self.send({"input": "3", **extra}).status_code, 422)

    def test_empty_or_too_many_items(self):
        for items in ([], ["3"] * 65):
            self.assertEqual(self.send({"input": items}).status_code, 400)

    def test_numbers_are_not_token_ids(self):
        self.assertEqual(self.send({"input": [1, 2]}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
