import os
import types
import unittest
from unittest import mock

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
