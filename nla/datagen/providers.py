"""Completion provider backends for Stage 2 (API explanation generation).

Stage 2 calls an external LLM to produce natural-language explanations of
source text — these become the `response` column for AV-SFT and the `prompt`
content for AR-SFT. `CompletionProvider` is the pluggable interface: stage 2
code hands it a batch of fully-formed prompts and gets back a batch of
completions. Concurrency, retries, rate limits, and auth are all the
provider's problem.

Swap via `--provider-cls my.module.MyProvider` at stage2 invocation.
"""

import asyncio
import time
from abc import ABC, abstractmethod

import anthropic


def _extract_text(resp) -> str | None:
    """Shared response→text semantics for both Anthropic providers.

    Returns None for a refusal (row is dropped); asserts loudly on anything
    else unexpected — those are code bugs, not transient failures.
    """
    # refusal: source text tripped safety — no answer coming, drop this row.
    # content may be [] or the refusal message; either way, no explanation.
    if resp.stop_reason == "refusal":
        return None
    assert resp.stop_reason in ("end_turn", "max_tokens"), (
        f"unexpected stop_reason={resp.stop_reason!r} (want end_turn/max_tokens/refusal)"
    )
    assert len(resp.content) == 1 and resp.content[0].type == "text", (
        f"expected single text block, got {[b.type for b in resp.content]}"
    )
    text = resp.content[0].text.strip()
    assert text, "empty completion — refusing to emit blank explanation"
    return text


class CompletionProvider(ABC):
    """Submit a batch of prompts, get a batch of completions back.

    Stage 2 formats NLA-specific instruction prompts; the provider just maps
    `prompts[i] -> completion[i]` (or None for prompts that exhausted retries).
    A robust sampling engine can be plugged in by wrapping it in a subclass.

    None returns are per-prompt gave-up signals — stage2 drops those rows
    (same path as failed-extract-pattern). This means a chunk can survive
    losing a few prompts to sustained 429/500 storms instead of discarding
    511 good completions because one failed. Gaps ARE tracked: stage2 logs
    a drop count, and the parquet row count tells you exactly how many
    survived.
    """

    @abstractmethod
    def complete(self, prompts: list[str]) -> list[str | None]: ...


class AnthropicProvider(CompletionProvider):
    """Default provider: Anthropic Messages API with bounded async concurrency.

    The SDK handles transport-level retries (408/429/5xx, exponential backoff
    with jitter, respects Retry-After). High `max_retries` extends the retry
    window for sustained rate-limit storms — at max_retries=100 the SDK will
    keep backing off for minutes before giving up on one prompt.

    Per-prompt failures after exhausting retries return None (caller drops
    the row). `gather(return_exceptions=True)` collects these without nuking
    the whole batch — otherwise one stubborn 429 in a chunk of 512 wastes
    the other 511 API calls. ONLY `RateLimitError` and server-side 5xx are
    tolerated; anything else (auth, bad request, unexpected content) still
    raises — those are code bugs, not transient.

    Calls `asyncio.run()` — do not invoke from inside a running event loop.
    Stage 2 is a standalone CLI, so this is fine in practice.
    """

    # Exceptions from which we degrade to None instead of killing the batch.
    # Anything NOT in this tuple is a code bug and should still blow up loud.
    _TOLERATED = (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    )

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 300,
        temperature: float = 1.0,
        concurrency: int = 32,
        max_retries: int = 10,
    ):
        self.client = anthropic.AsyncAnthropic(max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency

    async def _one(self, sem: asyncio.Semaphore, prompt: str) -> str | None:
        async with sem:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        return _extract_text(resp)

    def complete(self, prompts: list[str]) -> list[str | None]:
        async def _run() -> list[str | None | BaseException]:
            sem = asyncio.Semaphore(self.concurrency)
            return await asyncio.gather(
                *(self._one(sem, p) for p in prompts),
                return_exceptions=True,
            )

        raw = asyncio.run(_run())
        out: list[str | None] = []
        n_failed = 0
        n_refused = 0
        for i, r in enumerate(raw):
            if isinstance(r, str):
                out.append(r)
            elif r is None:
                n_refused += 1
                out.append(None)
            elif isinstance(r, self._TOLERATED):
                n_failed += 1
                out.append(None)
            elif isinstance(r, BaseException):
                # Not a transient — auth/schema/code bug. Blow up loud.
                raise r
            else:
                raise AssertionError(f"gather returned unexpected type at [{i}]: {type(r).__name__}")
        if n_failed or n_refused:
            print(f"  [AnthropicProvider] dropped {n_refused} refused + {n_failed} retry-exhausted of {len(prompts)}")
        return out


class AnthropicBatchProvider(CompletionProvider):
    """Message Batches API provider — 50% of standard token prices.

    Same `prompts[i] -> completion[i]` contract and drop semantics as
    AnthropicProvider, traded for latency: a batch usually completes within
    an hour (24h worst case). Use for offline datagen where wall-clock is
    cheap and volume is large (the 100k Stage 2 is ~500k calls).

    Mechanics:
    - One `complete()` call submits the chunk as one or more batches
      (`batch_max_requests` per submission — the API caps a batch at 100k
      requests / 256MB, and a 65,536-prompt stage2 chunk at ~3.5KB/prompt
      brushes the size cap, so we split well below it), then polls each
      batch until `processing_status == "ended"`.
    - `succeeded` results go through the same `_extract_text` semantics as
      the streaming provider (refusal -> None, drop the row).
    - `errored` results with a server-side error and `expired` results are
      resubmitted in a follow-up batch, up to `max_retries` rounds; requests
      still unresolved after that return None (caller drops the row).
      `invalid_request` errors and `canceled` results raise loudly — we
      never cancel, so both are code bugs, not transients.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 300,
        temperature: float = 1.0,
        poll_interval_s: float = 60.0,
        max_retries: int = 2,
        batch_max_requests: int = 16384,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.poll_interval_s = poll_interval_s
        self.max_retries = max_retries
        self.batch_max_requests = batch_max_requests

    def _submit(self, pending: dict[str, str]) -> list[str]:
        """Submit pending {custom_id: prompt} as one or more batches; return batch ids."""
        ids = list(pending)
        batch_ids: list[str] = []
        for lo in range(0, len(ids), self.batch_max_requests):
            chunk_ids = ids[lo : lo + self.batch_max_requests]
            batch = self.client.messages.batches.create(
                requests=[
                    {
                        "custom_id": cid,
                        "params": {
                            "model": self.model,
                            "max_tokens": self.max_tokens,
                            "temperature": self.temperature,
                            "messages": [{"role": "user", "content": pending[cid]}],
                        },
                    }
                    for cid in chunk_ids
                ]
            )
            print(f"  [AnthropicBatchProvider] submitted batch {batch.id} ({len(chunk_ids)} requests)")
            batch_ids.append(batch.id)
        return batch_ids

    def _await_batch(self, batch_id: str):
        t0 = time.monotonic()
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                c = batch.request_counts
                print(
                    f"  [AnthropicBatchProvider] batch {batch_id} ended in "
                    f"{time.monotonic() - t0:.0f}s: {c.succeeded} succeeded, "
                    f"{c.errored} errored, {c.expired} expired, {c.canceled} canceled"
                )
                return
            time.sleep(self.poll_interval_s)

    def complete(self, prompts: list[str]) -> list[str | None]:
        # custom_id -> original index; resolved[i] stays None until decided.
        pending = {f"p{i}": p for i, p in enumerate(prompts)}
        resolved: list[str | None] = [None] * len(prompts)
        n_refused = 0

        for round_no in range(self.max_retries + 1):
            if not pending:
                break
            batch_ids = self._submit(pending)
            retry: dict[str, str] = {}
            for batch_id in batch_ids:
                self._await_batch(batch_id)
                for result in self.client.messages.batches.results(batch_id):
                    cid = result.custom_id
                    prompt = pending[cid]
                    idx = int(cid[1:])
                    kind = result.result.type
                    if kind == "succeeded":
                        text = _extract_text(result.result.message)
                        if text is None:
                            n_refused += 1  # refusal is final — do not retry
                        resolved[idx] = text
                    elif kind == "errored":
                        err_type = result.result.error.error.type
                        assert err_type != "invalid_request_error", (
                            f"batch request {cid} rejected as invalid_request — "
                            f"code bug, not transient: {result.result.error}"
                        )
                        retry[cid] = prompt
                    elif kind == "expired":
                        retry[cid] = prompt
                    else:
                        raise AssertionError(
                            f"batch request {cid} ended {kind!r} — we never cancel; "
                            "something external interfered"
                        )
            pending = retry

        n_exhausted = len(pending)  # still unresolved after all rounds -> dropped
        if n_exhausted or n_refused:
            print(
                f"  [AnthropicBatchProvider] dropped {n_refused} refused + "
                f"{n_exhausted} retry-exhausted of {len(prompts)}"
            )
        return resolved
