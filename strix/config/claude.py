"""Claude subscription auth: OAuth login, token refresh, and the client
that routes inference through Anthropic Messages API using Claude Code OAuth.

Allows using Strix with an existing Claude Code session (~/.claude or macOS Keychain)
or Anthropic OAuth login without requiring an API key. Supports models such as
claude-sonnet-4-6, claude-opus-5-5, and claude-haiku-4-5.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import ctypes
import hashlib
import json
import logging
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from openai import AsyncOpenAI

from strix.utils.secret_files import write_secret_text


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


logger = logging.getLogger(__name__)

PROVIDER = "claude"
SUBSCRIPTION_PREFIX = "claude/"
ALT_PREFIX = "claude-code/"

CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
AUTHORIZE_URL = "https://claude.com/cai/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"  # noqa: S105
MESSAGES_URL = "https://api.anthropic.com/v1/messages?beta=true"
REDIRECT_URI = "https://platform.claude.com/oauth/code/callback"
SCOPES = (
    "org:create_api_key user:profile user:inference user:sessions:claude_code "
    "user:mcp_servers user:file_upload user:plugins"
)

KEYCHAIN_SERVICE = "Claude Code-credentials"
AUTH_PATH = Path.home() / ".strix" / "subscription-auth.json"
SYSTEM_PREFIX_PATH = Path(__file__).resolve().parent / "claude_system_prefix.json"

_EXPIRY_SKEW_S = 300
_refresh_lock = threading.Lock()


class ClaudeAuthError(Exception):
    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


def _read_store() -> dict[str, Any]:
    try:
        data = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_store(data: dict[str, Any]) -> None:
    write_secret_text(AUTH_PATH, json.dumps(data, indent=2))


def _extract_claude_oauth(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    oauth = data.get("claudeAiOauth") or data.get("oauth") or data
    if not isinstance(oauth, dict):
        return None
    access = oauth.get("accessToken") or oauth.get("access_token") or oauth.get("token")
    refresh = oauth.get("refreshToken") or oauth.get("refresh_token")
    raw_exp = oauth.get("expiresAt") or oauth.get("expires_at") or 0
    try:
        expires_at = float(raw_exp)
        if expires_at > 1e11:  # Timestamp in milliseconds
            expires_at = expires_at / 1000.0
    except (ValueError, TypeError):
        expires_at = 0.0

    if refresh:
        return {
            "type": "oauth",
            "provider": PROVIDER,
            "access": str(access or ""),
            "refresh": str(refresh),
            "expires_at": expires_at,
            "subscription_type": oauth.get("subscriptionType", "pro"),
            "scopes": oauth.get("scopes", []),
        }
    return None


def _candidate_credential_paths() -> list[Path]:
    paths = [
        Path.home() / ".claude" / "credentials.json",
        Path.home() / ".config" / "claude-code" / "credentials.json",
        Path.home() / ".config" / "claude" / "credentials.json",
    ]
    if sys.platform == "darwin":
        paths.append(
            Path.home() / "Library" / "Application Support" / "Claude" / "credentials.json"
        )
    elif sys.platform == "win32":
        for var in ("APPDATA", "LOCALAPPDATA"):
            base = os.environ.get(var)
            if base:
                paths.append(Path(base) / "Claude" / "credentials.json")
                paths.append(Path(base) / "claude-code" / "credentials.json")
    return paths


def _read_darwin_keychain() -> dict[str, Any] | None:
    if sys.platform != "darwin":
        return None
    try:
        raw = (
            subprocess.check_output(  # noqa: S603
                ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],  # noqa: S607
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        if raw:
            data = json.loads(raw)
            return _extract_claude_oauth(data)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return None


def _write_darwin_keychain(access: str, refresh: str, expires_at: float) -> None:
    if sys.platform != "darwin":
        return
    try:
        existing_raw = ""
        with contextlib.suppress(Exception):
            existing_raw = (
                subprocess.check_output(  # noqa: S603
                    ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],  # noqa: S607
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )

        payload: dict[str, Any] = {}
        if existing_raw:
            with contextlib.suppress(json.JSONDecodeError):
                payload = json.loads(existing_raw)

        oauth = payload.setdefault("claudeAiOauth", {})
        oauth["accessToken"] = access
        oauth["refreshToken"] = refresh
        oauth["expiresAt"] = int(expires_at * 1000)

        subprocess.run(  # noqa: S603
            [  # noqa: S607
                "security",
                "add-generic-password",
                "-U",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                "Claude Code",
                "-w",
                json.dumps(payload),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001
        logger.debug("Failed to update macOS Keychain", exc_info=True)


def _read_linux_secret_service() -> dict[str, Any] | None:
    if not sys.platform.startswith("linux") or not shutil.which("secret-tool"):
        return None
    for attr in ("service", "label"):
        try:
            raw = (
                subprocess.check_output(  # noqa: S603
                    ["secret-tool", "lookup", attr, KEYCHAIN_SERVICE],  # noqa: S607
                    stderr=subprocess.DEVNULL,
                    timeout=2.0,
                )
                .decode()
                .strip()
            )
            if raw:
                data = json.loads(raw)
                extracted = _extract_claude_oauth(data)
                if extracted:
                    return extracted
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            continue
    return None


def _read_windows_credentials() -> dict[str, Any] | None:
    if sys.platform != "win32":
        return None
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return None
    try:
        from ctypes import wintypes  # noqa: PLC0415

        advapi32 = windll.advapi32

        class CREDENTIAL(ctypes.Structure):
            _fields_ = [
                ("Flags", wintypes.DWORD),
                ("Type", wintypes.DWORD),
                ("TargetName", wintypes.LPWSTR),
                ("Comment", wintypes.LPWSTR),
                ("LastWritten", wintypes.FILETIME),
                ("CredentialBlobSize", wintypes.DWORD),
                ("CredentialBlob", ctypes.POINTER(ctypes.c_byte)),
                ("Persist", wintypes.DWORD),
                ("AttributeCount", wintypes.DWORD),
                ("Attributes", ctypes.c_void_p),
                ("TargetAlias", wintypes.LPWSTR),
                ("UserName", wintypes.LPWSTR),
            ]

        advapi32.CredReadW.argtypes = [
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.POINTER(CREDENTIAL)),
        ]
        advapi32.CredReadW.restype = wintypes.BOOL
        advapi32.CredFree.argtypes = [ctypes.c_void_p]

        for target in (KEYCHAIN_SERVICE, f"{KEYCHAIN_SERVICE}/Claude Code", "Claude Code"):
            pcred = ctypes.POINTER(CREDENTIAL)()
            if advapi32.CredReadW(target, 1, 0, ctypes.byref(pcred)):
                try:
                    cred = pcred.contents
                    size = cred.CredentialBlobSize
                    if size > 0 and cred.CredentialBlob:
                        raw_bytes = ctypes.string_at(cred.CredentialBlob, size)
                        for enc in ("utf-8", "utf-16-le"):
                            try:
                                text = raw_bytes.decode(enc).strip("\x00 \r\n\t")
                                data = json.loads(text)
                                extracted = _extract_claude_oauth(data)
                                if extracted:
                                    return extracted
                            except (UnicodeDecodeError, json.JSONDecodeError):
                                continue
                finally:
                    advapi32.CredFree(pcred)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to read Windows credentials", exc_info=True)
    return None


def _read_env_credentials() -> dict[str, Any] | None:
    for var in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_OAUTH_TOKEN", "CLAUDE_OAUTH_TOKEN"):
        val = os.environ.get(var, "").strip()
        if not val:
            continue
        if val.startswith("{") and val.endswith("}"):
            try:
                data = json.loads(val)
                extracted = _extract_claude_oauth(data)
                if extracted:
                    return extracted
            except json.JSONDecodeError:
                pass
        return {
            "type": "oauth",
            "provider": PROVIDER,
            "access": val,
            "refresh": val,
            "expires_at": 0.0,
        }
    return None


def _read_platform_credentials() -> dict[str, Any] | None:
    # 1. Environment variables
    env_rec = _read_env_credentials()
    if env_rec:
        return env_rec

    # 2. OS credential store
    if sys.platform == "darwin":
        kc = _read_darwin_keychain()
        if kc:
            return kc
    elif sys.platform.startswith("linux"):
        sec = _read_linux_secret_service()
        if sec:
            return sec
    elif sys.platform == "win32":
        win = _read_windows_credentials()
        if win:
            return win

    # 3. Known disk file locations
    for path in _candidate_credential_paths():
        try:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                extracted = _extract_claude_oauth(data)
                if extracted:
                    return extracted
        except (OSError, json.JSONDecodeError):
            continue

    return None


def _write_platform_credentials(access: str, refresh: str, expires_at: float) -> None:
    if sys.platform == "darwin":
        _write_darwin_keychain(access, refresh, expires_at)

    # Sync with any existing credentials file on disk
    for path in _candidate_credential_paths():
        with contextlib.suppress(Exception):
            if path.is_file():
                data = {}
                with contextlib.suppress(Exception):
                    data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
                oauth = data.setdefault("claudeAiOauth", {})
                oauth["accessToken"] = access
                oauth["refreshToken"] = refresh
                oauth["expiresAt"] = int(expires_at * 1000)
                write_secret_text(path, json.dumps(data, indent=2))
                break


# Backward-compatible aliases
read_keychain = _read_platform_credentials
write_keychain = _write_platform_credentials


def read_record() -> dict[str, Any] | None:
    """Get stored credentials from subscription-auth.json, platform store, or disk."""
    record = _read_store().get(PROVIDER)
    if isinstance(record, dict) and record.get("type") == "oauth" and record.get("refresh"):
        return record

    reader = (
        read_keychain if read_keychain != _read_platform_credentials else _read_platform_credentials
    )
    platform_rec = reader()
    if platform_rec is not None:
        return platform_rec

    return None


def is_authenticated() -> bool:
    return read_record() is not None


def save_record(record: dict[str, Any]) -> None:
    data = _read_store()
    data[PROVIDER] = record
    _write_store(data)

    access = record.get("access")
    refresh = record.get("refresh")
    expires_at = record.get("expires_at", 0)
    if access and refresh:
        writer = (
            write_keychain
            if write_keychain != _write_platform_credentials
            else _write_platform_credentials
        )
        writer(access, refresh, float(expires_at))


def logout() -> None:
    data = _read_store()
    if PROVIDER in data:
        del data[PROVIDER]
        if data:
            _write_store(data)
        else:
            with contextlib.suppress(OSError):
                AUTH_PATH.unlink()


@contextlib.contextmanager
def _refresh_guard() -> Iterator[None]:
    with _refresh_lock:
        lock_path = AUTH_PATH.with_suffix(".claude.lock")
        handle = None
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = lock_path.open("a+")
            if sys.platform == "win32":
                import msvcrt  # noqa: PLC0415

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl  # noqa: PLC0415

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass

        try:
            yield
        finally:
            if handle is not None:
                try:
                    if sys.platform == "win32":
                        import msvcrt  # noqa: PLC0415

                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl  # noqa: PLC0415

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass
                with contextlib.suppress(OSError):
                    handle.close()


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def generate_pkce() -> tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(64))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def create_state() -> str:
    return _b64url(secrets.token_bytes(32))


def build_authorize_url(challenge: str, state: str) -> str:
    params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def exchange_code(code: str, verifier: str, state: str) -> dict[str, Any]:
    payload = {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code.strip(),
        "code_verifier": verifier,
        "redirect_uri": REDIRECT_URI,
        "state": state,
    }
    try:
        resp = httpx.post(
            TOKEN_URL,
            json=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Claude-Code/2.1.301"},
            timeout=30.0,
        )
    except httpx.RequestError as exc:
        raise ClaudeAuthError("unavailable", str(exc)) from exc

    if resp.status_code >= 400:
        raise ClaudeAuthError("token_http_error", f"HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise ClaudeAuthError("bad_response", "token endpoint returned non-JSON") from exc

    access = data.get("access_token")
    refresh = data.get("refresh_token")
    expires_in = data.get("expires_in") or 3600
    if not isinstance(access, str) or not access:
        raise ClaudeAuthError("bad_response", "token response missing access_token")
    if not isinstance(refresh, str) or not refresh:
        raise ClaudeAuthError("bad_response", "token response missing refresh_token")

    account = data.get("account") or {}
    org = data.get("organization") or {}

    return {
        "type": "oauth",
        "provider": PROVIDER,
        "access": access,
        "refresh": refresh,
        "expires_at": time.time() + float(expires_in),
        "account_id": account.get("uuid") or account.get("id"),
        "organization_id": org.get("uuid") or org.get("id"),
    }


def refresh_tokens(refresh_token: str) -> dict[str, Any]:
    payload = {
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": refresh_token,
    }
    try:
        resp = httpx.post(
            TOKEN_URL,
            json=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Claude-Code/2.1.301"},
            timeout=30.0,
        )
    except httpx.RequestError as exc:
        raise ClaudeAuthError("unavailable", str(exc)) from exc

    if resp.status_code >= 400:
        raise ClaudeAuthError("token_http_error", f"HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise ClaudeAuthError("bad_response", "token endpoint returned non-JSON") from exc

    access = data.get("access_token")
    new_refresh = data.get("refresh_token") or refresh_token
    expires_in = data.get("expires_in") or 3600
    if not isinstance(access, str) or not access:
        raise ClaudeAuthError("bad_response", "missing access_token")

    return {
        "type": "oauth",
        "provider": PROVIDER,
        "access": access,
        "refresh": new_refresh,
        "expires_at": time.time() + float(expires_in),
    }


def _near_expiry(record: dict[str, Any]) -> bool:
    expires_at = record.get("expires_at")
    if not isinstance(expires_at, (int, float)):
        return True
    return expires_at - _EXPIRY_SKEW_S <= time.time()


def get_valid_token() -> str:
    record = read_record()
    if record is None:
        raise ClaudeAuthError(
            "not_authenticated",
            "not signed into Claude; run: strix auth login claude",
        )
    if not _near_expiry(record) and record.get("access"):
        return str(record["access"])

    with _refresh_guard():
        record = read_record()
        if record is None:
            raise ClaudeAuthError(
                "not_authenticated",
                "not signed into Claude; run: strix auth login claude",
            )
        if not _near_expiry(record) and record.get("access"):
            return str(record["access"])

        refresh_tok = record.get("refresh")
        if not refresh_tok:
            raise ClaudeAuthError("missing_refresh_token", "no refresh token available")
        refreshed = refresh_tokens(refresh_tok)
        save_record(refreshed)
        return str(refreshed["access"])


def subscription_model(model_name: str | None) -> str | None:
    """The canonical model slug behind a ``claude/<model>`` or ``claude-code/<model>``."""
    name = (model_name or "").strip()
    lower = name.lower()
    raw_slug = None
    if lower.startswith(SUBSCRIPTION_PREFIX):
        raw_slug = name[len(SUBSCRIPTION_PREFIX) :]
    elif lower.startswith(ALT_PREFIX):
        raw_slug = name[len(ALT_PREFIX) :]

    if raw_slug is None:
        return None

    raw_slug = raw_slug.strip().lower()
    # Map common model names and aliases to Claude Code-supported identifiers
    if not raw_slug or "sonnet" in raw_slug:
        return "claude-sonnet-4-6"
    if "opus" in raw_slug:
        return "claude-opus-5-5"
    if "haiku" in raw_slug:
        return "claude-haiku-4-5"
    return raw_slug


def auth_mode(model_name: str | None) -> str:
    return "subscription" if subscription_model(model_name) else "api_key"


class ClaudeTransport(httpx.AsyncBaseTransport):
    """In-process transport bridging OpenAI Chat Completions calls into
    Anthropic Messages API calls with Claude Code OAuth session headers."""

    def __init__(self) -> None:
        self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=20.0))
        self._session_id = str(uuid.uuid4())
        self._device_id = hashlib.sha256(str(uuid.getnode()).encode()).hexdigest()
        self._prefix_blocks: list[dict[str, Any]] | None = None

    def _get_prefix_blocks(self) -> list[dict[str, Any]]:
        if self._prefix_blocks is None:
            if SYSTEM_PREFIX_PATH.exists():
                try:
                    self._prefix_blocks = json.loads(SYSTEM_PREFIX_PATH.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    self._prefix_blocks = []
            else:
                self._prefix_blocks = []
        return list(self._prefix_blocks)

    def _convert_openai_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        anthropic_tools = []
        for t in tools:
            if t.get("type") == "function" and "function" in t:
                fn = t["function"]
                anthropic_tools.append(
                    {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
        return anthropic_tools

    def _convert_openai_messages(  # noqa: PLR0912
        self, messages: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                if content:
                    system_parts.append(str(content))
                continue

            if role == "user":
                content_blocks: list[dict[str, Any]] = []
                if isinstance(content, str):
                    if content:
                        content_blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    content_blocks.extend(
                        {"type": "text", "text": part.get("text", "")}
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                # If preceding message is also 'user', combine them to respect Anthropic turns
                if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                    anthropic_messages[-1]["content"].extend(content_blocks)
                else:
                    anthropic_messages.append({"role": "user", "content": content_blocks})

            elif role == "assistant":
                content_blocks = []
                if content:
                    content_blocks.append({"type": "text", "text": str(content)})
                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            parsed_args = json.loads(args)
                        except json.JSONDecodeError:
                            parsed_args = {}
                    else:
                        parsed_args = args if isinstance(args, dict) else {}
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                            "name": fn.get("name", ""),
                            "input": parsed_args,
                        }
                    )
                if not content_blocks:
                    content_blocks.append({"type": "text", "text": ""})
                anthropic_messages.append({"role": "assistant", "content": content_blocks})

            elif role == "tool":
                tool_call_id = str(msg.get("tool_call_id") or "")
                res_content = str(content) if content is not None else ""
                block = {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": res_content,
                }
                # In Anthropic, tool_result blocks are sent inside a 'user' message turn.
                if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                    anthropic_messages[-1]["content"].append(block)
                else:
                    anthropic_messages.append({"role": "user", "content": [block]})

        return "\n\n".join(system_parts), anthropic_messages

    def _build_anthropic_payload(self, openai_req: dict[str, Any]) -> dict[str, Any]:
        model = subscription_model(openai_req.get("model")) or "claude-sonnet-4-6"
        messages = openai_req.get("messages", [])
        tools = openai_req.get("tools", [])

        custom_system, anthropic_messages = self._convert_openai_messages(messages)

        # Build system blocks with Claude Code prompt caching prefix
        system_blocks = self._get_prefix_blocks()
        if custom_system:
            system_blocks.append({"type": "text", "text": custom_system})

        anthropic_tools = self._convert_openai_tools(tools)

        prompt_id = str(uuid.uuid4())
        payload: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "system": system_blocks,
            "max_tokens": 8192,
            "stream": True,
            "metadata": {
                "user_id": json.dumps(
                    {
                        "device_id": self._device_id,
                        "session_id": self._session_id,
                        "prompt_id": prompt_id,
                    }
                )
            },
            "output_config": {"effort": "medium"},
            "thread": {"type": "create"},
        }
        if "haiku" not in model:
            payload["thinking"] = {"type": "adaptive", "display": "omitted"}
            payload["context_management"] = {
                "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
            }

        if anthropic_tools:
            payload["tools"] = anthropic_tools

        return payload

    async def _stream_anthropic_to_openai_chunks(  # noqa: PLR0912, PLR0915
        self, stream: AsyncIterator[bytes], model_name: str
    ) -> AsyncIterator[bytes]:
        buffer = ""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        current_tool_index = 0
        current_block_type: str | None = None
        finish_reason = "stop"

        async for raw_bytes in stream:
            buffer += raw_bytes.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                stripped = line.strip()
                if not stripped.startswith("data:"):
                    continue
                data_str = stripped[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    event_data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event_data.get("type")

                if event_type == "message_start":
                    msg_obj = event_data.get("message", {})
                    usage = msg_obj.get("usage", {})
                    input_toks = usage.get("input_tokens", 0)
                    cached_toks = usage.get("cache_read_input_tokens", 0)
                    prompt_tokens = input_toks + cached_toks

                elif event_type == "content_block_start":
                    block = event_data.get("content_block", {})
                    current_block_type = block.get("type")
                    if current_block_type == "tool_use":
                        finish_reason = "tool_calls"
                        tool_call_id = block.get("id") or f"toolu_{uuid.uuid4().hex[:8]}"
                        chunk = {
                            "id": "chatcmpl-claude",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": current_tool_index,
                                                "id": tool_call_id,
                                                "type": "function",
                                                "function": {
                                                    "name": block.get("name", ""),
                                                    "arguments": "",
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode()

                elif event_type == "content_block_delta":
                    delta = event_data.get("delta", {})
                    delta_type = delta.get("type")
                    if delta_type == "text_delta":
                        chunk = {
                            "id": "chatcmpl-claude",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta.get("text", "")},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode()
                    elif delta_type == "input_json_delta":
                        chunk = {
                            "id": "chatcmpl-claude",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": current_tool_index,
                                                "function": {
                                                    "arguments": delta.get("partial_json", ""),
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode()

                elif event_type == "content_block_stop":
                    if current_block_type == "tool_use":
                        current_tool_index += 1
                    current_block_type = None

                elif event_type == "message_delta":
                    delta = event_data.get("delta", {})
                    stop_r = delta.get("stop_reason")
                    if stop_r == "tool_use":
                        finish_reason = "tool_calls"
                    elif stop_r == "end_turn":
                        finish_reason = "stop"
                    elif stop_r == "max_tokens":
                        finish_reason = "length"

                    usage = event_data.get("usage", {})
                    completion_tokens = usage.get("output_tokens", 0)
                    total_tokens = prompt_tokens + completion_tokens

                    terminal_chunk = {
                        "id": "chatcmpl-claude",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        },
                    }
                    yield f"data: {json.dumps(terminal_chunk)}\n\n".encode()

        yield b"data: [DONE]\n\n"

    async def _aggregate_response(  # noqa: PLR0912
        self, stream: AsyncIterator[bytes], model_name: str
    ) -> dict[str, Any]:
        full_text = ""
        tool_calls: list[dict[str, Any]] = []
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason = "stop"

        async for chunk_bytes in self._stream_anthropic_to_openai_chunks(stream, model_name):
            line = chunk_bytes.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                c_data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = c_data.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if delta.get("content"):
                full_text += delta["content"]
            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    idx = tc.get("index", 0)
                    empty_fn: dict[str, Any] = {"name": "", "arguments": ""}
                    while len(tool_calls) <= idx:
                        tool_calls.append(
                            {"id": "", "type": "function", "function": empty_fn.copy()}
                        )
                    if tc.get("id"):
                        tool_calls[idx]["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        tool_calls[idx]["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        tool_calls[idx]["function"]["arguments"] += fn["arguments"]

            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
            if c_data.get("usage"):
                prompt_tokens = c_data["usage"].get("prompt_tokens", prompt_tokens)
                completion_tokens = c_data["usage"].get("completion_tokens", completion_tokens)

        message: dict[str, Any] = {"role": "assistant"}
        message["content"] = full_text or None
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": "chatcmpl-claude",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        token = await asyncio.to_thread(get_valid_token)
        openai_req = json.loads(request.content.decode("utf-8"))
        anthropic_payload = self._build_anthropic_payload(openai_req)
        model_name = anthropic_payload.get("model", "claude-sonnet-4-6")

        prompt_id = str(uuid.uuid4())
        client_req_id = str(uuid.uuid4())

        headers = {
            "Authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": (
                "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,"
                "thinking-token-count-2026-05-13,context-management-2025-06-27,"
                "prompt-caching-scope-2026-01-05,advisor-tool-2026-03-01,"
                "advanced-tool-use-2025-11-20,effort-2025-11-24,"
                "thinking-binding-controls-2026-08-01,extended-cache-ttl-2025-04-11,"
                "cache-diagnosis-2026-04-07,message-threads-2026-08-12"
            ),
            "anthropic-dangerous-direct-browser-access": "true",
            "anthropic-dispatch-id": "v2d",
            "user-agent": "claude-cli/2.1.301 (external, sdk-cli)",
            "x-app": "cli",
            "X-Claude-Code-Session-Id": self._session_id,
            "x-claude-code-prompt-id": prompt_id,
            "x-client-request-id": client_req_id,
            "content-type": "application/json",
            "Accept-Encoding": "identity",
        }

        upstream_req = self._http_client.build_request(
            "POST",
            MESSAGES_URL,
            json=anthropic_payload,
            headers=headers,
        )
        upstream_resp = await self._http_client.send(upstream_req, stream=True)
        if upstream_resp.status_code >= 400:
            content = await upstream_resp.aread()
            return httpx.Response(
                status_code=upstream_resp.status_code, content=content, request=request
            )

        if bool(openai_req.get("stream")):
            converted_stream = self._stream_anthropic_to_openai_chunks(
                upstream_resp.aiter_bytes(), model_name
            )
            return httpx.Response(
                status_code=200,
                headers={"Content-Type": "text/event-stream"},
                content=converted_stream,
                request=request,
            )

        response_dict = await self._aggregate_response(upstream_resp.aiter_bytes(), model_name)
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=json.dumps(response_dict).encode("utf-8"),
            request=request,
        )


def build_openai_client() -> AsyncOpenAI:
    """An AsyncOpenAI client routed through ClaudeTransport."""
    transport = ClaudeTransport()
    http_client = httpx.AsyncClient(transport=transport)
    return AsyncOpenAI(
        api_key="strix-claude-oauth",
        base_url="https://claude.strix.local/v1",
        http_client=http_client,
    )


_subscription_client: AsyncOpenAI | None = None


def get_subscription_client() -> AsyncOpenAI:
    global _subscription_client  # noqa: PLW0603
    if _subscription_client is None:
        _subscription_client = build_openai_client()
    return _subscription_client
