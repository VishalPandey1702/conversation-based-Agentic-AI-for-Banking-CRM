"""
Azure OpenAI LLM service.

Responsibilities:
- Build and cache an AzureOpenAI client.
- Provide a single `complete_json()` helper that:
    * sends structured prompts
    * forces JSON output where supported
    * implements simple retry-with-backoff
    * gracefully falls back to a deterministic stub if the LLM is
      unavailable (e.g. running in offline / demo mode)

The LLM is intentionally limited to:
- personalized message generation
- short reasoning explanations
- summarization

Numerical scoring/eligibility/recommendation logic is deterministic and lives
in dedicated tools - the LLM never makes those calls.
"""
from __future__ import annotations

import json
import threading
import time
from functools import lru_cache
from typing import Any, List, Optional

from openai import AzureOpenAI, APIError, APITimeoutError, AuthenticationError, RateLimitError

from backend.services.logging_service import get_logger
from backend.utils.config import settings

logger = get_logger(__name__)


# Process-wide circuit breaker: once auth fails we stop calling the LLM for the
# rest of the process lifetime, since the credentials won't change at runtime.
_circuit_lock = threading.Lock()
_circuit_open = False
_circuit_reason: Optional[str] = None


def _trip_circuit(reason: str) -> None:
    global _circuit_open, _circuit_reason
    with _circuit_lock:
        if not _circuit_open:
            logger.error("LLM circuit breaker tripped - all future calls will fall back. Reason: %s", reason)
        _circuit_open = True
        _circuit_reason = reason


def _circuit_state() -> tuple[bool, Optional[str]]:
    with _circuit_lock:
        return _circuit_open, _circuit_reason


@lru_cache(maxsize=1)
def _get_client() -> Optional[AzureOpenAI]:
    """Lazily create and cache the AzureOpenAI client."""
    if not settings.llm_configured:
        logger.warning(
            "Azure OpenAI not configured - LLM features will use deterministic stubs."
        )
        return None
    try:
        client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        logger.info(
            "AzureOpenAI client initialized | endpoint=%s | deployment=%s",
            settings.AZURE_OPENAI_ENDPOINT,
            settings.AZURE_OPENAI_DEPLOYMENT,
        )
        return client
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialize Azure OpenAI client: %s", exc)
        return None


class LLMService:
    """
    Thin, safe facade around Azure OpenAI chat completions.

    The class is stateless; instance attributes are injected via DI in tests
    if needed. All public methods return JSON-serializable Python objects.
    """

    def __init__(
        self,
        deployment: Optional[str] = None,
        max_retries: int = 2,
        retry_delay_seconds: float = 1.5,
    ):
        self.deployment = deployment or settings.AZURE_OPENAI_DEPLOYMENT
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    @property
    def is_available(self) -> bool:
        return _get_client() is not None

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        fallback: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Run a chat completion that MUST return JSON.

        Args:
            system_prompt: High-level instructions / role.
            user_prompt:   The structured user content (already formatted).
            fallback:      Returned verbatim if the LLM is unavailable
                           or all retries fail.

        Returns:
            Parsed JSON dict. If parsing fails, returns the fallback or
            an `{ "error": "..." }` envelope.
        """
        client = _get_client()
        fallback = fallback or {}

        if client is None:
            return {**fallback, "_llm_used": False, "_reason": "llm_not_configured"}

        # Short-circuit if we've already determined the LLM is unreachable.
        is_open, reason = _circuit_state()
        if is_open:
            return {**fallback, "_llm_used": False, "_reason": reason or "circuit_open"}

        last_error: Optional[str] = None
        for attempt in range(1, self.max_retries + 2):
            try:
                # Some Azure deployments don't support response_format on every model.
                # Try with JSON object mode first, then degrade gracefully.
                kwargs: dict[str, Any] = {
                    "model": self.deployment,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                try:
                    resp = client.chat.completions.create(
                        **kwargs,
                        response_format={"type": "json_object"},
                    )
                except (TypeError, APIError):
                    # Older API versions / models may not accept response_format.
                    resp = client.chat.completions.create(**kwargs)

                content = (resp.choices[0].message.content or "").strip()
                if not content:
                    raise ValueError("LLM returned empty content")

                parsed = _parse_json_lenient(content)
                parsed["_llm_used"] = True
                return parsed

            except AuthenticationError as exc:
                # Auth errors won't fix themselves on retry - trip the circuit.
                last_error = f"AuthenticationError: {exc}"
                _trip_circuit(last_error)
                break
            except (RateLimitError, APITimeoutError, APIError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries + 1,
                    last_error,
                )
                if attempt <= self.max_retries:
                    time.sleep(self.retry_delay_seconds * attempt)
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Unexpected LLM error: %s", last_error)
                break

        logger.error("LLM call permanently failed: %s", last_error)
        return {
            **fallback,
            "_llm_used": False,
            "_reason": "llm_call_failed",
            "_error": last_error,
        }


    # ------------------------------------------------------------------
    # Tool-calling chat (used by the conversational orchestrator)
    # ------------------------------------------------------------------
    def chat_with_tools(
        self,
        *,
        messages: List[dict],
        tools: List[dict],
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        """
        Single chat completion with tool / function calling.

        Args:
            messages: Standard OpenAI message list (system / user / assistant /
                      tool). Caller is responsible for maintaining the loop.
            tools:    OpenAI-style tool schemas (type=function).
            tool_choice: "auto" | "required" | {"type": "function", ...}.

        Returns one of:
            { "ok": True, "tool_calls": [...], "content": str | None }
            { "ok": True, "content": str, "tool_calls": [] }
            { "ok": False, "error": str, "code": str }
        """
        client = _get_client()
        if client is None:
            return {"ok": False, "error": "LLM not configured", "code": "LLM_OFF"}

        is_open, reason = _circuit_state()
        if is_open:
            return {"ok": False, "error": reason or "circuit_open", "code": "LLM_OFF"}

        last_error: Optional[str] = None
        for attempt in range(1, self.max_retries + 2):
            try:
                resp = client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                msg = resp.choices[0].message
                tool_calls_raw = getattr(msg, "tool_calls", None) or []
                tool_calls: List[dict] = []
                for tc in tool_calls_raw:
                    fn = getattr(tc, "function", None)
                    name = getattr(fn, "name", None) if fn else None
                    args_str = getattr(fn, "arguments", "") if fn else ""
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError:
                        args = {"_raw_arguments": args_str}
                    tool_calls.append(
                        {
                            "id": getattr(tc, "id", None),
                            "name": name,
                            "arguments": args,
                            # Keep the raw arg string for the assistant-message replay
                            "_arg_str": args_str,
                        }
                    )
                return {
                    "ok": True,
                    "content": msg.content,
                    "tool_calls": tool_calls,
                }
            except AuthenticationError as exc:
                last_error = f"AuthenticationError: {exc}"
                _trip_circuit(last_error)
                break
            except (RateLimitError, APITimeoutError, APIError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "LLM tool-call failed (attempt %d/%d): %s",
                    attempt, self.max_retries + 1, last_error,
                )
                if attempt <= self.max_retries:
                    time.sleep(self.retry_delay_seconds * attempt)
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Unexpected LLM tool-call error: %s", last_error)
                break

        return {"ok": False, "error": last_error or "unknown", "code": "LLM_ERROR"}


def _parse_json_lenient(content: str) -> dict[str, Any]:
    """
    Try several strategies to extract a JSON object from an LLM response:
    1. Direct json.loads
    2. Strip markdown code fences and retry
    3. Find the outermost {...} substring and parse that
    """
    # 1. direct
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. strip code fences
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        try:
            return json.loads(stripped.strip())
        except json.JSONDecodeError:
            pass

    # 3. outermost braces
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = content[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return {"_raw": content, "_error": "unable to parse JSON"}


# Module-level singleton for convenience
llm_service = LLMService()
