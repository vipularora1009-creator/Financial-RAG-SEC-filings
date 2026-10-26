from types import SimpleNamespace

from sec_financial_rag.generation.prompts import SYSTEM_PROMPT
from sec_financial_rag.generation.synthesizer import MODEL, synthesize

CHUNKS = [
    {
        "ticker": "AAPL",
        "form_type": "10-Q",
        "report_date": "2022-06-25",
        "text": "Total net sales 82,959",
    }
]


class FakeClient:
    """Mimics genai.Client: client.models.generate_content(...). Records the request."""

    def __init__(self):
        self.requests = []
        self.models = self  # so self.models.generate_content works

    def generate_content(self, **kwargs):
        self.requests.append(kwargs)
        usage = SimpleNamespace(
            prompt_token_count=120, candidates_token_count=30, thoughts_token_count=50
        )
        return SimpleNamespace(
            text="Net sales were $82,959 million [1].", usage_metadata=usage
        )


def test_synthesize_sends_pinned_model_rules_and_temperature_zero():
    client = FakeClient()
    answer = synthesize(client, "What were Apple's net sales?", CHUNKS)

    request = client.requests[0]
    assert request["model"] == MODEL
    assert request["config"].system_instruction == SYSTEM_PROMPT
    assert request["config"].temperature == 0
    assert "[1] AAPL 10-Q" in request["contents"]

    assert answer.text == "Net sales were $82,959 million [1]."
    assert (answer.input_tokens, answer.output_tokens) == (
        120,
        80,
    )  # thinking billed as output


def test_cost_uses_pinned_prices_per_million_tokens():
    from sec_financial_rag.generation.synthesizer import (
        PRICE_IN,
        PRICE_OUT,
        Answer,
        cost_usd,
    )

    answer = Answer(
        text="x", input_tokens=1_000_000, output_tokens=2_000_000, latency_ms=1
    )
    assert cost_usd(answer) == PRICE_IN + 2 * PRICE_OUT
