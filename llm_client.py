"""Pluggable LLM clients for the judge-audit project.

Two providers:

- Pollinations (default): free, no API key. A GET request to
  https://text.pollinations.ai/<urlencoded-prompt>?model=openai&private=true
- Gemini: needs a GOOGLE_API_KEY and langchain_google_genai installed,
  so the user can swap in his own key later.

Only the standard library is used for the Pollinations client (urllib).
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.parse
import urllib.request


class LLMError(Exception):
    """Raised when an LLM call fails after all retries."""


class BaseLLMClient:
    """Minimal interface the judge needs: prompt in, text out."""

    name = "base"

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class PollinationsClient(BaseLLMClient):
    """Free text generation via Pollinations (no API key).

    Retries transient failures with exponential backoff, and enforces a
    minimum interval between calls so the harness never hammers the API.
    """

    BASE_URL = "https://text.pollinations.ai/"
    USER_AGENT = "llm-judge-audit/1.0 (research evaluation harness)"

    def __init__(self, model: str = "openai", timeout: int = 120,
                 max_retries: int = 3, min_interval_s: float = 1.0,
                 backoff_base_s: float = 2.0) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval_s = min_interval_s
        self.backoff_base_s = backoff_base_s
        self._last_call_at = 0.0
        self.name = f"pollinations:{model}"

    def generate(self, prompt: str) -> str:
        self._pace()
        url = (self.BASE_URL + urllib.parse.quote(prompt, safe="")
               + f"?model={urllib.parse.quote(self.model)}&private=true")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": self.USER_AGENT})
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
                self._last_call_at = time.monotonic()
                return text.strip()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_base_s * (2 ** attempt))
        raise LLMError(
            f"Pollinations request failed after {self.max_retries + 1} "
            f"attempts: {last_error}")

    def _pace(self) -> None:
        elapsed = time.monotonic() - self._last_call_at
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)


class GeminiClient(BaseLLMClient):
    """Gemini via langchain_google_genai. Needs GOOGLE_API_KEY.

    Temperature 0: a judge should be as deterministic as the API allows.
    """

    def __init__(self, model: str = "gemini-2.0-flash",
                 api_key: str | None = None, timeout: int = 120) -> None:
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise LLMError(
                "Gemini provider needs a GOOGLE_API_KEY "
                "(environment variable or api_key argument).")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise LLMError(
                "langchain_google_genai is not installed. "
                "Run: pip install langchain-google-genai") from exc
        self.llm = ChatGoogleGenerativeAI(
            model=model, temperature=0, google_api_key=api_key,
            timeout=timeout)
        self.name = f"gemini:{model}"

    def generate(self, prompt: str) -> str:
        try:
            return str(self.llm.invoke(prompt).content).strip()
        except Exception as exc:  # noqa: BLE001 - surfaced as LLMError
            raise LLMError(f"Gemini request failed: {exc}") from exc


class CannedClient(BaseLLMClient):
    """Returns canned responses in order (cycles). For --dry-run and tests.

    Records every prompt it receives, so tests can assert on what the
    judge asked.
    """

    def __init__(self, responses) -> None:
        if isinstance(responses, str):
            responses = [responses]
        self._responses = list(responses)
        self._index = 0
        self.calls: list[str] = []
        self.name = "canned"

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        text = self._responses[self._index % len(self._responses)]
        self._index += 1
        return text


def get_client(provider: str = "pollinations", **kwargs) -> BaseLLMClient:
    """Build a client by provider name: 'pollinations' or 'gemini'."""
    providers = {"pollinations": PollinationsClient, "gemini": GeminiClient}
    if provider not in providers:
        raise ValueError(
            f"Unknown provider {provider!r}; choose from {sorted(providers)}")
    return providers[provider](**kwargs)
