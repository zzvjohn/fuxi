"""No real network/credential use. Verify provider evidence survives strict output rejection."""
import io
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError
from fuxi_v1.search import llm_candidates


class LLMContractTests(unittest.TestCase):
    fields={"close":{"unit":"price"},"volume":{"unit":"shares"}}
    good={"mechanism":"mean reversion after price variation","ast":{"op":"rank","args":[{"field":"close"}]}}

    def call(self,content,finish="stop"):
        response={"id":"fixture_request","model":"fixture_model","usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30},
                  "choices":[{"finish_reason":finish,"message":{"content":content}}]}
        with patch.dict("os.environ",{"DEEPSEEK_API_KEY":"fixture_not_a_real_credential"}), \
             patch("urllib.request.urlopen",return_value=io.BytesIO(json.dumps(response).encode())) as request:
            result=llm_candidates(self.fields,[],count=2)
            self.assertEqual(request.call_count,1)
            return result

    def test_valid_json_gives_typed_candidates_and_provider_usage(self):
        proposals,receipt=self.call(json.dumps({"candidates":[self.good]}))
        self.assertEqual(proposals[0]["action"],"evidence_llm")
        self.assertEqual(receipt["status"],"completed")
        self.assertEqual(receipt["usage"]["total_tokens"],30)
        self.assertNotIn("fixture_not_a_real_credential",json.dumps(receipt))

    def test_bad_root_preserves_usage_and_failure_stage(self):
        proposals,receipt=self.call(json.dumps({"factors":[self.good]}))
        self.assertFalse(proposals)
        self.assertEqual(receipt["error_stage"],"response_contract")
        self.assertTrue(receipt["network_response_received"])
        self.assertEqual(receipt["token_usage"],"reported_by_provider")
        self.assertIn("candidates",receipt["contract_error"])

    def test_duplicate_keys_nonfinite_extra_keys_fail_closed(self):
        for text in ('{"candidates":[],"candidates":[]}', '{"candidates":[],"score":NaN}',
                     '{"candidates":[],"explanation":"extra"}'):
            proposals,receipt=self.call(text)
            self.assertFalse(proposals)
            self.assertEqual(receipt["status"],"error")

    def test_truncated_output_rejected_even_if_json_parses(self):
        proposals,receipt=self.call(json.dumps({"candidates":[self.good]}),finish="length")
        self.assertFalse(proposals)
        self.assertEqual(receipt["finish_reason"],"length")

    def test_bad_ast_rejected_without_relaxing_compiler(self):
        bad={"mechanism":"invalid field","ast":{"field":"future_return"}}
        proposals,receipt=self.call(json.dumps({"candidates":[bad,self.good]}))
        self.assertEqual(len(proposals),1)
        self.assertEqual(receipt["raw_candidate_count"],2)
        self.assertEqual(len(receipt["rejected"]),1)

    def test_network_error_does_not_leak_exception_text_or_claim_usage(self):
        with patch.dict("os.environ",{"DEEPSEEK_API_KEY":"fixture_not_a_real_credential"}), \
             patch("urllib.request.urlopen",side_effect=URLError("fixture_sensitive_fragment")) as request:
            proposals,receipt=llm_candidates(self.fields,[],count=2)
            self.assertEqual(request.call_count,1)
        self.assertFalse(proposals)
        self.assertFalse(receipt["network_response_received"])
        self.assertIsNone(receipt["usage"])
        self.assertNotIn("fixture_sensitive_fragment",json.dumps(receipt))


if __name__=="__main__":unittest.main()
