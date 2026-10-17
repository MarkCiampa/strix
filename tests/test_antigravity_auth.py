"""Tests for Antigravity (Google OAuth) subscription auth."""

from __future__ import annotations

from typing import Any

from strix.config import antigravity


def test_subscription_model_prefixes() -> None:
    assert antigravity.subscription_model("antigravity/gemini-3-flash") == "gemini-3-flash"
    assert antigravity.subscription_model("agy/claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert antigravity.subscription_model("antigravity/gemini-2.5-pro") == "gemini-2.5-pro"
    assert antigravity.subscription_model("openai/gpt-4o") is None
    assert antigravity.subscription_model("chatgpt/gpt-4o") is None
    assert antigravity.subscription_model(None) is None


def test_auth_mode() -> None:
    assert antigravity.auth_mode("antigravity/gemini-3-flash") == "subscription"
    assert antigravity.auth_mode("agy/gemini-3-flash") == "subscription"
    assert antigravity.auth_mode("openai/gpt-4o") == "api_key"
    assert antigravity.auth_mode(None) == "api_key"


def test_save_read_logout_record() -> None:
    record = {
        "type": "oauth",
        "provider": "antigravity",
        "access": "access-123",
        "refresh": "refresh-123",
        "expires_at": 9999999999,
    }
    assert not antigravity.is_authenticated()
    antigravity.save_record(record)
    assert antigravity.is_authenticated()

    loaded = antigravity.read_record()
    assert loaded is not None
    assert loaded["access"] == "access-123"
    assert loaded["refresh"] == "refresh-123"

    antigravity.logout()
    assert not antigravity.is_authenticated()
    assert antigravity.read_record() is None


def test_transport_converts_messages_and_tools() -> None:
    transport = antigravity.AntigravityTransport()
    data: dict[str, Any] = {
        "model": "gemini-3-flash",
        "messages": [
            {"role": "system", "content": "You are a scanner."},
            {"role": "user", "content": "Scan host.com"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_42",
                        "type": "function",
                        "function": {"name": "port_scan", "arguments": '{"port": 80}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_42", "content": "open"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "port_scan",
                    "description": "Scans ports",
                    "parameters": {"type": "object", "properties": {"port": {"type": "number"}}},
                },
            }
        ],
    }

    gemini_req = transport._convert_openai_to_gemini(data)
    assert gemini_req["project"] == "aicode-consumers"
    assert gemini_req["model"] == "gemini-3-flash"

    req = gemini_req["request"]
    assert req["systemInstruction"]["parts"][0]["text"] == "You are a scanner."

    contents = req["contents"]
    assert len(contents) == 3  # user, model, user(tool)
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "Scan host.com"

    assert contents[1]["role"] == "model"
    fc = contents[1]["parts"][0]["functionCall"]
    assert fc["name"] == "port_scan"
    assert fc["args"] == {"port": 80}

    assert contents[2]["role"] == "user"
    fr = contents[2]["parts"][0]["functionResponse"]
    assert fr["name"] == "port_scan"
    assert fr["id"] == "call_42"
    assert fr["response"]["result"] == "open"

    tools = req["tools"]
    assert len(tools) == 1
    assert tools[0]["functionDeclarations"][0]["name"] == "port_scan"
