from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from sec_financial_rag.evals.judge import Verdict, judge


class FakeGemini:
    def __init__(self, parsed):
        self.parsed, self.request = parsed, None
        self.models = self

    def generate_content(self, model, contents, config):
        self.request = SimpleNamespace(contents=contents, config=config)
        usage = SimpleNamespace(
            prompt_token_count=300, candidates_token_count=40, thoughts_token_count=10
        )
        return SimpleNamespace(parsed=self.parsed, usage_metadata=usage)


def test_verdict_is_a_closed_set_with_the_reason_first():
    assert list(Verdict.model_fields) == ["reason", "verdict"]  # reason before verdict
    with pytest.raises(ValidationError):
        Verdict(reason="close enough", verdict="mostly_right")


def test_judge_sees_question_gold_and_answer_but_not_the_context():
    fake = FakeGemini(Verdict(reason="same figures", verdict="correct"))
    result = judge(fake, "Q: FCF conversion?", "143% to 156%", "142.7% to 155.5%")
    assert result.verdict.verdict == "correct"
    for part in ("Q: FCF conversion?", "143% to 156%", "142.7% to 155.5%"):
        assert part in fake.request.contents
    assert fake.request.config.response_schema is Verdict
    assert fake.request.config.temperature == 0
    assert "rounding" in fake.request.config.system_instruction
    assert result.input_tokens == 300 and result.output_tokens == 50
