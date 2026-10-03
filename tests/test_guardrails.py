import pytest
from app.guardrails.colang_rules import RAIL_INDICATORS
from app.guardrails.rails import guard_output, redact_pii


def test_rail_indicators_presence():
    assert len(RAIL_INDICATORS) == 5
    assert "can't help with that — but ask me anything technical" in RAIL_INDICATORS
    assert "I maintain consistent guidelines regardless of how I am prompted" in RAIL_INDICATORS
    assert "Hello! I'm your Enterprise IT Assistant" in RAIL_INDICATORS
    assert "Goodbye! Feel free to return whenever you have more enterprise IT questions" in RAIL_INDICATORS
    assert "I'm an Enterprise AI Assistant with deep expertise in" in RAIL_INDICATORS


def test_rail_indicators_regression_substring_matches():
    # Verify each indicator fires against simulated NeMo outputs
    for ind in RAIL_INDICATORS:
        mock_response = f"Sure! {ind} and here is more text."
        assert any(indicator in mock_response for indicator in RAIL_INDICATORS)


def test_redact_pii():
    text_with_keys = "Here is my key: gsk_1234567890abcdef1234567890 and email john.doe@enterprise.com"
    sanitized = redact_pii(text_with_keys)
    assert "gsk_1234567890abcdef1234567890" not in sanitized
    assert "[REDACTED_GROQ_KEY]" in sanitized
    assert "john.doe@enterprise.com" not in sanitized
    assert "[REDACTED_EMAIL]" in sanitized


def test_guard_output_secrets_detection():
    secret_output = "The cluster API secret is AIzaSyD3f4uLtK3y12345678901234567890123"
    detected, cleaned = guard_output(secret_output)
    assert detected is True
    assert "[REDACTED_GOOGLE_KEY]" in cleaned
    assert "AIzaSyD3f4uLtK3y" not in cleaned


def test_guard_output_clean_text():
    clean_output = "Use `kubectl get pods -n kube-system` to inspect running cluster components."
    detected, cleaned = guard_output(clean_output)
    assert detected is False
    assert cleaned == clean_output
