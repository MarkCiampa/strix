"""Tests for Claude (Claude Code / Anthropic OAuth) subscription auth."""

from __future__ import annotations

import json
from typing import Any

from strix.config import claude


def test_subscription_model_prefixes() -> None:
    assert claude.subscription_model("claude/claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert claude.subscription_model("claude/sonnet") == "claude-sonnet-4-6"
    assert claude.subscription_model("claude/claude-3-7-sonnet") == "claude-sonnet-4-6"
    assert claude.subscription_model("claude/claude-3-5-sonnet") == "claude-sonnet-4-6"
    assert claude.subscription_model("claude/opus") == "claude-opus-5-5"
    assert claude.subscription_model("claude/claude-opus-5-5") == "claude-opus-5-5"
    assert claude.subscription_model("claude/haiku") == "claude-haiku-4-5"
    assert claude.subscription_model("claude-code/sonnet") == "claude-sonnet-4-6"
    assert claude.subscription_model("openai/gpt-4o") is None
    assert claude.subscription_model("chatgpt/gpt-4o") is None
    assert claude.subscription_model("antigravity/gemini-3-flash") is None
    assert claude.subscription_model(None) is None


def test_auth_mode() -> None:
    assert claude.auth_mode("claude/claude-sonnet-4-6") == "subscription"
    assert claude.auth_mode("claude-code/sonnet") == "subscription"
    assert claude.auth_mode("openai/gpt-4o") == "api_key"
    assert claude.auth_mode(None) == "api_key"


def test_pkce_and_authorize_url() -> None:
    verifier, challenge = claude.generate_pkce()
    assert len(verifier) >= 43
    assert len(challenge) >= 43
    state = claude.create_state()
    assert len(state) >= 16

    url = claude.build_authorize_url(challenge, state)
    assert "https://claude.com/cai/oauth/authorize" in url
    assert f"client_id={claude.CLIENT_ID}" in url
    assert f"code_challenge={challenge}" in url
    assert f"state={state}" in url


def test_save_read_logout_record(monkeypatch: Any, tmp_path: Any) -> None:
    fake_auth = tmp_path / "subscription-auth.json"
    monkeypatch.setattr(claude, "AUTH_PATH", fake_auth)
    # Avoid reading system keychain during test
    monkeypatch.setattr(claude, "read_keychain", lambda: None)
    monkeypatch.setattr(claude, "write_keychain", lambda *_args, **_kwargs: None)

    record = {
        "type": "oauth",
        "provider": "claude",
        "access": "test-access-123",
        "refresh": "test-refresh-123",
        "expires_at": 9999999999.0,
    }
    assert not claude.is_authenticated()
    claude.save_record(record)
    assert claude.is_authenticated()

    loaded = claude.read_record()
    assert loaded is not None
    assert loaded["access"] == "test-access-123"
    assert loaded["refresh"] == "test-refresh-123"

    claude.logout()
    assert not claude.is_authenticated()
    assert claude.read_record() is None


def test_transport_converts_messages_and_tools() -> None:
    transport = claude.ClaudeTransport()
    openai_req: dict[str, Any] = {
        "model": "claude/claude-sonnet-4-6",
        "messages": [
            {"role": "system", "content": "You are a scanner."},
            {"role": "user", "content": "Scan host.com"},
            {
                "role": "assistant",
                "content": "Running scan",
                "tool_calls": [
                    {
                        "id": "toolu_42",
                        "type": "function",
                        "function": {"name": "port_scan", "arguments": '{"port": 80}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_42", "content": "open"},
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

    payload = transport._build_anthropic_payload(openai_req)
    assert payload["model"] == "claude-sonnet-4-6"
    assert payload["stream"] is True
    assert "tools" in payload
    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["name"] == "port_scan"
    assert payload["tools"][0]["input_schema"] == {
        "type": "object",
        "properties": {"port": {"type": "number"}},
    }

    # Verify messages
    msgs = payload["messages"]
    assert len(msgs) == 3  # user, assistant, user (with tool_result)
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"][0]["text"] == "Scan host.com"

    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"][0]["text"] == "Running scan"
    assert msgs[1]["content"][1]["type"] == "tool_use"
    assert msgs[1]["content"][1]["id"] == "toolu_42"
    assert msgs[1]["content"][1]["input"] == {"port": 80}

    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert msgs[2]["content"][0]["tool_use_id"] == "toolu_42"
    assert msgs[2]["content"][0]["content"] == "open"

    # Verify system blocks contain prefix and custom instructions
    sys_blocks = payload["system"]
    assert len(sys_blocks) >= 2
    assert sys_blocks[-1]["text"] == "You are a scanner."


def test_cross_platform_extract_claude_oauth() -> None:
    # 1. Claude Code Keychain format with milliseconds timestamp
    kc_format = {
        "claudeAiOauth": {
            "accessToken": "acc-1",
            "refreshToken": "ref-1",
            "expiresAt": 1700000000000,
            "subscriptionType": "pro",
            "scopes": ["user:inference"],
        }
    }
    ext1 = claude._extract_claude_oauth(kc_format)
    assert ext1 is not None
    assert ext1["access"] == "acc-1"
    assert ext1["refresh"] == "ref-1"
    assert ext1["expires_at"] == 1700000000.0
    assert ext1["subscription_type"] == "pro"

    # 2. Standard credentials.json format
    file_format = {
        "accessToken": "acc-2",
        "refreshToken": "ref-2",
        "expiresAt": 1700000000,
    }
    ext2 = claude._extract_claude_oauth(file_format)
    assert ext2 is not None
    assert ext2["access"] == "acc-2"
    assert ext2["refresh"] == "ref-2"

    # 3. Snake_case format
    snake_format = {
        "access_token": "acc-3",
        "refresh_token": "ref-3",
    }
    ext3 = claude._extract_claude_oauth(snake_format)
    assert ext3 is not None
    assert ext3["access"] == "acc-3"
    assert ext3["refresh"] == "ref-3"


def test_cross_platform_candidate_paths(monkeypatch: Any) -> None:
    monkeypatch.setattr(claude.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", "C:\\Users\\test\\AppData\\Roaming")
    paths_win = [str(p) for p in claude._candidate_credential_paths()]
    assert any("AppData" in p or "Claude" in p for p in paths_win)

    monkeypatch.setattr(claude.sys, "platform", "linux")
    paths_linux = [str(p) for p in claude._candidate_credential_paths()]
    assert any(".config" in p or ".claude" in p for p in paths_linux)


def test_read_env_credentials(monkeypatch: Any) -> None:
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-test-token")
    rec = claude._read_env_credentials()
    assert rec is not None
    assert rec["refresh"] == "sk-ant-test-token"

    json_val = json.dumps({"accessToken": "tok-a", "refreshToken": "tok-r"})
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", json_val)
    rec2 = claude._read_env_credentials()
    assert rec2 is not None
    assert rec2["access"] == "tok-a"
    assert rec2["refresh"] == "tok-r"
