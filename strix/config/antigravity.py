"""Antigravity CLI (Google OAuth) subscription auth: token refresh and client
that routes inference through the Antigravity backend.

Allows using Strix with an existing Antigravity CLI session (~/.gemini/) or Google
OAuth login without requiring an API key. Supports Gemini models (e.g. gemini-3-flash,
gemini-2.5-pro) and Anthropic Claude models (e.g. claude-sonnet-4-6) routed through
the Antigravity infrastructure.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from openai import AsyncOpenAI

from strix.utils.secret_files import write_secret_text


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


logger = logging.getLogger(__name__)

PROVIDER = "antigravity"
SUBSCRIPTION_PREFIX = "antigravity/"
ALT_PREFIX = "agy/"

_CID_BYTES = bytes(
    [
        27,
        26,
        29,
        27,
        26,
        26,
        28,
        26,
        28,
        26,
        31,
        19,
        27,
        7,
        94,
        71,
        66,
        89,
        89,
        67,
        68,
        24,
        66,
        24,
        27,
        70,
        73,
        88,
        79,
        24,
        25,
        31,
        92,
        94,
        69,
        70,
        69,
        64,
        66,
        30,
        77,
        30,
        26,
        25,
        79,
        90,
        4,
        75,
        90,
        90,
        89,
        4,
        77,
        69,
        69,
        77,
        70,
        79,
        95,
        89,
        79,
        88,
        73,
        69,
        68,
        94,
        79,
        68,
        94,
        4,
        73,
        69,
        71,
    ]
)
_SEC_BYTES = bytes(
    [
        109,
        101,
        105,
        121,
        122,
        114,
        7,
        97,
        31,
        18,
        108,
        125,
        120,
        30,
        18,
        28,
        102,
        78,
        102,
        96,
        27,
        71,
        102,
        104,
        18,
        89,
        114,
        105,
        30,
        80,
        28,
        91,
        110,
        107,
        76,
    ]
)

CLIENT_ID = os.environ.get("ANTIGRAVITY_CLIENT_ID") or bytes(b ^ 42 for b in _CID_BYTES).decode(
    "utf-8"
)
CLIENT_SECRET = os.environ.get("ANTIGRAVITY_CLIENT_SECRET") or bytes(
    b ^ 42 for b in _SEC_BYTES
).decode("utf-8")
TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105
ANTIGRAVITY_ENDPOINT = (
    "https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse"
)
DEFAULT_PROJECT = "aicode-consumers"

_EXPIRY_SKEW_S = 300
_refresh_lock = threading.Lock()

AUTH_PATH = Path.home() / ".strix" / "subscription-auth.json"
STANDALONE_TOKEN_PATH = Path.home() / ".gemini" / "jetski-standalone-oauth-token"


class AntigravityAuthError(Exception):
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


def read_record() -> dict[str, Any] | None:
    record = _read_store().get(PROVIDER)
    if isinstance(record, dict) and record.get("type") == "oauth" and record.get("refresh"):
        return record

    if STANDALONE_TOKEN_PATH.exists():
        try:
            raw = json.loads(STANDALONE_TOKEN_PATH.read_text(encoding="utf-8"))
            token_obj = raw.get("token") or {}
            refresh = token_obj.get("refresh_token")
            access = token_obj.get("access_token")
            if refresh:
                return {
                    "type": "oauth",
                    "provider": PROVIDER,
                    "access": access or "",
                    "refresh": refresh,
                    "expires_at": 0,
                }
        except (OSError, json.JSONDecodeError):
            pass
    return None


def is_authenticated() -> bool:
    return read_record() is not None


def save_record(record: dict[str, Any]) -> None:
    data = _read_store()
    data[PROVIDER] = record
    _write_store(data)


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
        try:
            import fcntl  # noqa: PLC0415

            lock_path = AUTH_PATH.with_suffix(".antigravity.lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = lock_path.open("w")
        except (ImportError, OSError):
            yield
            return
        try:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def refresh_tokens(refresh_token: str) -> dict[str, Any]:
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    try:
        resp = httpx.post(
            TOKEN_URL,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15.0,
        )
    except httpx.RequestError as exc:
        raise AntigravityAuthError("unavailable", str(exc)) from exc

    if resp.status_code >= 400:
        raise AntigravityAuthError(
            "token_http_error", f"HTTP {resp.status_code}: {resp.text[:300]}"
        )

    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise AntigravityAuthError("bad_response", "token endpoint returned non-JSON") from exc

    access = data.get("access_token")
    expires_in = data.get("expires_in") or 3600
    if not isinstance(access, str) or not access:
        raise AntigravityAuthError("bad_response", "missing access_token")

    return {
        "type": "oauth",
        "provider": PROVIDER,
        "access": access,
        "refresh": data.get("refresh_token") or refresh_token,
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
        raise AntigravityAuthError(
            "not_authenticated",
            "not signed into Antigravity; run: strix auth login antigravity",
        )
    if not _near_expiry(record) and record.get("access"):
        return str(record["access"])

    with _refresh_guard():
        record = read_record()
        if record is None:
            raise AntigravityAuthError(
                "not_authenticated",
                "not signed into Antigravity; run: strix auth login antigravity",
            )
        if not _near_expiry(record) and record.get("access"):
            return str(record["access"])
        refreshed = refresh_tokens(record["refresh"])
        save_record(refreshed)
        return str(refreshed["access"])


def subscription_model(model_name: str | None) -> str | None:
    name = (model_name or "").strip()
    lower = name.lower()
    if lower.startswith(SUBSCRIPTION_PREFIX):
        return name[len(SUBSCRIPTION_PREFIX) :] or None
    if lower.startswith(ALT_PREFIX):
        return name[len(ALT_PREFIX) :] or None
    return None


def auth_mode(model_name: str | None) -> str:
    return "subscription" if subscription_model(model_name) else "api_key"


class AntigravityTransport(httpx.AsyncBaseTransport):
    """In-process transport converting OpenAI chat completion requests into
    Antigravity streamGenerateContent calls and back to OpenAI SSE streams."""

    def __init__(self) -> None:
        self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0))
        self._call_to_fn_map: dict[str, str] = {}
        self._call_to_thought_sig: dict[str, str] = {}

    def _convert_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parts = []
        for tc in tool_calls:
            call_id = tc.get("id", "call_1")
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            self._call_to_fn_map[call_id] = fn_name
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            part: dict[str, Any] = {
                "functionCall": {
                    "name": fn_name,
                    "args": args,
                    "id": call_id,
                }
            }
            if call_id in self._call_to_thought_sig:
                part["thoughtSignature"] = self._call_to_thought_sig[call_id]
            parts.append(part)
        return parts

    def _convert_message(
        self, msg: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "system":
            return {"parts": [{"text": str(content)}]}, None
        if role == "user":
            return None, {"role": "user", "parts": [{"text": str(content)}]}
        if role == "assistant":
            parts = [{"text": str(content)}] if content else []
            if "tool_calls" in msg:
                parts.extend(self._convert_tool_calls(msg["tool_calls"]))
            return None, {"role": "model", "parts": parts}
        if role == "tool":
            tool_id = str(msg.get("tool_call_id") or "")
            fn_name = self._call_to_fn_map.get(tool_id, tool_id or "tool")
            part = {
                "functionResponse": {
                    "name": fn_name,
                    "response": {"result": str(content)},
                    "id": tool_id,
                }
            }
            return None, {"role": "user", "parts": [part]}
        return None, None

    def _convert_openai_to_gemini(self, data: dict[str, Any]) -> dict[str, Any]:
        model = data.get("model", "gemini-3-flash")
        messages = data.get("messages", [])
        tools = data.get("tools", [])

        system_instruction = None
        contents = []

        for msg in messages:
            sys_inst, content_item = self._convert_message(msg)
            if sys_inst:
                system_instruction = sys_inst
            elif content_item:
                contents.append(content_item)

        req_body: dict[str, Any] = {
            "project": DEFAULT_PROJECT,
            "model": model,
            "request": {
                "contents": contents,
            },
        }
        if system_instruction:
            req_body["request"]["systemInstruction"] = system_instruction

        if tools:
            fn_decls = [
                {
                    "name": t["function"].get("name"),
                    "description": t["function"].get("description", ""),
                    "parameters": t["function"].get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
                for t in tools
                if t.get("type") == "function" and "function" in t
            ]
            if fn_decls:
                req_body["request"]["tools"] = [{"functionDeclarations": fn_decls}]

        return req_body

    async def _stream_gemini_to_openai_chunks(
        self, gemini_stream: AsyncIterator[bytes], model_name: str
    ) -> AsyncIterator[bytes]:
        buffer = ""
        prompt_tokens = 0
        cand_tokens = 0
        total_tokens = 0
        async for raw_chunk in gemini_stream:
            buffer += raw_chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                stripped = line.strip()
                if not stripped.startswith("data:"):
                    continue
                data_str = stripped[5:].strip()
                if not data_str:
                    continue
                try:
                    g_data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                usage = g_data.get("response", {}).get("usageMetadata", {})
                if usage:
                    prompt_tokens = usage.get("promptTokenCount", prompt_tokens)
                    cand_tokens = usage.get("candidatesTokenCount", cand_tokens)
                    total_tokens = usage.get("totalTokenCount", total_tokens)

                candidates = g_data.get("response", {}).get("candidates", [])
                if not candidates:
                    continue
                cand = candidates[0]
                parts = cand.get("content", {}).get("parts", [])
                for p in parts:
                    if p.get("text"):
                        chunk = {
                            "id": "chatcmpl-antigravity",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": p["text"]},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode()
                    elif "functionCall" in p:
                        fc = p["functionCall"]
                        call_id = fc.get("id") or "call_1"
                        if "thoughtSignature" in p:
                            self._call_to_thought_sig[call_id] = p["thoughtSignature"]
                        chunk = {
                            "id": "chatcmpl-antigravity",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": call_id,
                                                "type": "function",
                                                "function": {
                                                    "name": fc.get("name"),
                                                    "arguments": json.dumps(fc.get("args", {})),
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode()

                if cand.get("finishReason") in ("STOP", "MAX_TOKENS"):
                    finish_chunk = {
                        "id": "chatcmpl-antigravity",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                                if cand["finishReason"] == "STOP"
                                else "length",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": cand_tokens,
                            "total_tokens": total_tokens or (prompt_tokens + cand_tokens),
                        },
                    }
                    yield f"data: {json.dumps(finish_chunk)}\n\n".encode()

        yield b"data: [DONE]\n\n"

    async def _aggregate_response(
        self, upstream_resp: httpx.Response, model_name: str
    ) -> dict[str, Any]:
        full_text = ""
        tool_calls = []
        prompt_tokens = 0
        cand_tokens = 0
        total_tokens = 0

        async for raw_chunk in upstream_resp.aiter_bytes():
            for line in raw_chunk.decode("utf-8", errors="replace").splitlines():
                stripped = line.strip()
                if not stripped.startswith("data:"):
                    continue
                data_str = stripped[5:].strip()
                if not data_str:
                    continue
                try:
                    g_data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                usage = g_data.get("response", {}).get("usageMetadata", {})
                if usage:
                    prompt_tokens = usage.get("promptTokenCount", prompt_tokens)
                    cand_tokens = usage.get("candidatesTokenCount", cand_tokens)
                    total_tokens = usage.get("totalTokenCount", total_tokens)

                cand = g_data.get("response", {}).get("candidates", [{}])[0]
                parts = cand.get("content", {}).get("parts", [])
                for p in parts:
                    if p.get("text"):
                        full_text += p["text"]
                    elif "functionCall" in p:
                        fc = p["functionCall"]
                        call_id = fc.get("id") or f"call_{len(tool_calls) + 1}"
                        if "thoughtSignature" in p:
                            self._call_to_thought_sig[call_id] = p["thoughtSignature"]
                        tool_calls.append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": fc.get("name"),
                                    "arguments": json.dumps(fc.get("args", {})),
                                },
                            }
                        )

        message: dict[str, Any] = {"role": "assistant"}
        message["content"] = full_text or None
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": "chatcmpl-antigravity",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": cand_tokens,
                "total_tokens": total_tokens or (prompt_tokens + cand_tokens),
            },
        }

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        token = await asyncio.to_thread(get_valid_token)
        openai_req = json.loads(request.content.decode("utf-8"))
        gemini_req = self._convert_openai_to_gemini(openai_req)
        model_name = openai_req.get("model", "antigravity")

        upstream_req = self._http_client.build_request(
            "POST",
            ANTIGRAVITY_ENDPOINT,
            json=gemini_req,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "antigravity",
            },
        )
        upstream_resp = await self._http_client.send(upstream_req, stream=True)
        if upstream_resp.status_code >= 400:
            content = await upstream_resp.aread()
            return httpx.Response(
                status_code=upstream_resp.status_code, content=content, request=request
            )

        if bool(openai_req.get("stream")):
            converted_stream = self._stream_gemini_to_openai_chunks(
                upstream_resp.aiter_bytes(), model_name
            )
            return httpx.Response(
                status_code=200,
                headers={"Content-Type": "text/event-stream"},
                content=converted_stream,
                request=request,
            )

        response_dict = await self._aggregate_response(upstream_resp, model_name)
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=json.dumps(response_dict).encode("utf-8"),
            request=request,
        )


def build_openai_client() -> AsyncOpenAI:
    """An AsyncOpenAI client routed through AntigravityTransport."""
    transport = AntigravityTransport()
    http_client = httpx.AsyncClient(transport=transport)
    return AsyncOpenAI(
        api_key="strix-antigravity-oauth",
        base_url="https://antigravity.strix.local/v1",
        http_client=http_client,
    )


_subscription_client: AsyncOpenAI | None = None


def get_subscription_client() -> AsyncOpenAI:
    global _subscription_client  # noqa: PLW0603
    if _subscription_client is None:
        _subscription_client = build_openai_client()
    return _subscription_client
