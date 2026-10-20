import json
import os
import types
import unittest
from unittest import mock

import httpx
import openai

from longcat_video.utils import prompt_enhancer


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content="enhanced prompt")
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = types.SimpleNamespace(completions=self.completions)


class PromptEnhancerMiniMaxTest(unittest.TestCase):
    def setUp(self):
        self.original_region = prompt_enhancer.MINIMAX_REGION
        self.original_text_model = prompt_enhancer.MINIMAX_TEXT_MODEL
        self.original_image_model = prompt_enhancer.MINIMAX_IMAGE_MODEL

    def tearDown(self):
        prompt_enhancer.MINIMAX_REGION = self.original_region
        prompt_enhancer.MINIMAX_TEXT_MODEL = self.original_text_model
        prompt_enhancer.MINIMAX_IMAGE_MODEL = self.original_image_model

    def test_global_and_cn_base_urls_are_configured(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            prompt_enhancer.MINIMAX_REGION = "global_en"
            self.assertEqual(
                prompt_enhancer.get_minimax_base_url(),
                "https://api.minimax.io/v1",
            )

            prompt_enhancer.MINIMAX_REGION = "cn_zh"
            self.assertEqual(
                prompt_enhancer.get_minimax_base_url(),
                "https://api.minimaxi.com/v1",
            )

    def test_base_url_override_takes_precedence(self):
        with mock.patch.dict(os.environ, {"MINIMAX_BASE_URL": "https://example.test/v1"}):
            prompt_enhancer.MINIMAX_REGION = "cn_zh"
            self.assertEqual(prompt_enhancer.get_minimax_base_url(), "https://example.test/v1")

    def test_text_prompt_supports_configured_models(self):
        client = FakeClient()
        prompt_enhancer.MINIMAX_TEXT_MODEL = "MiniMax-M2.7"

        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(prompt_enhancer, "get_minimax_client", return_value=client):
            result = prompt_enhancer.enhance_prompt_t2v("make a short video")

        self.assertEqual(result, "enhanced prompt")
        self.assertEqual(client.completions.calls[0]["model"], "MiniMax-M2.7")

    def test_recommended_text_models_are_flash_then_m3(self):
        self.assertEqual(
            prompt_enhancer.MINIMAX_TEXT_MODEL_IDS,
            ("MiniMax-M3.1-Flash", "MiniMax-M3"),
        )

    def assert_text_request(self, environment, expected_model, expected_url):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": expected_model,
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "enhanced prompt"},
                }],
            })

        real_openai = openai.OpenAI
        with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
            def make_client(**kwargs):
                return real_openai(http_client=http_client, max_retries=0, **kwargs)

            with mock.patch.dict(os.environ, {
                "MINIMAX_API_KEY": "offline-test-key", **environment,
            }, clear=True), mock.patch.object(openai, "OpenAI", side_effect=make_client):
                result = prompt_enhancer.enhance_prompt_t2v("make a short video")

        self.assertEqual(result, "enhanced prompt")
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url), expected_url + "/chat/completions")
        self.assertEqual(request.headers["authorization"], "Bearer offline-test-key")
        self.assertEqual(json.loads(request.content), {
            "model": expected_model,
            "messages": [
                {"role": "system", "content": prompt_enhancer.LM_EN_SYS_PROMPT},
                {"role": "user", "content": 'make a short video"'},
            ],
            "temperature": 0.01,
            "top_p": 0.7,
            "stream": False,
            "max_tokens": 320,
        })

    def test_text_request_defaults_to_flash_in_both_regions(self):
        for region, base_url in prompt_enhancer.MINIMAX_ENDPOINTS.items():
            with self.subTest(region=region):
                self.assert_text_request(
                    {"MINIMAX_REGION": region}, "MiniMax-M3.1-Flash", base_url,
                )

    def test_text_request_preserves_explicit_models_in_both_regions(self):
        for region, base_url in prompt_enhancer.MINIMAX_ENDPOINTS.items():
            for model in ("MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "custom-model"):
                with self.subTest(region=region, model=model):
                    self.assert_text_request({
                        "MINIMAX_REGION": region, "MINIMAX_TEXT_MODEL": model,
                    }, model, base_url)

    def test_text_request_preserves_configured_default(self):
        prompt_enhancer.MINIMAX_TEXT_MODEL = "MiniMax-M2.7"
        self.assert_text_request({}, "MiniMax-M2.7", "https://api.minimax.io/v1")

    def test_text_request_preserves_custom_endpoint_and_model(self):
        self.assert_text_request({
            "MINIMAX_BASE_URL": "https://example.test/v1",
            "MINIMAX_REGION": "custom",
            "MINIMAX_TEXT_MODEL": "custom-model",
        }, "custom-model", "https://example.test/v1")

    def test_image_prompt_uses_image_capable_model(self):
        client = FakeClient()

        compressed_image = types.SimpleNamespace(read=lambda: b"abc")
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(prompt_enhancer, "get_minimax_client", return_value=client), \
                mock.patch.object(prompt_enhancer, "compress_image", return_value=compressed_image), \
                mock.patch.object(prompt_enhancer, "encode_image", return_value="encoded"):
            result = prompt_enhancer.enhance_prompt_i2v("image.png", "make a short video")

        self.assertEqual(result, "enhanced prompt")
        call = client.completions.calls[0]
        self.assertEqual(call["model"], "MiniMax-M3")
        self.assertEqual(call["messages"][1]["content"][1]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()
