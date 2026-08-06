# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for KVCacheManager.fit_token_budget.

These exercise the post-allocation token-budget trim that shrinks over-budget
context chunks so a scheduled batch cannot overshoot ``max_num_tokens`` in the
forward pass (GitHub issue #13318). The trim is pure scheduling logic and does
not touch the GPU, so the tests build a bare KVCacheManager via ``__new__`` and
drive the method with lightweight fake requests.

The trim runs at the end of ``ResourceManager.prepare_resources``, which is the
first point where ``context_current_position`` and ``context_chunk_size`` mean
"forward-pass tokens" -- before ``setPrepopulatedPromptLen`` the chunk still
spans the reusable KV prefix. ``TestReuseDiscountedChunk`` is the regression
test for reading it too early.
"""

import unittest
from collections import OrderedDict

from tensorrt_llm._torch.pyexecutor.resource_manager import (
    KVCacheManager,
    ResourceManager,
    ResourceManagerType,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests


class _FakeRequest:
    """Minimal stand-in exposing only the attributes fit_token_budget reads."""

    _next_id = 0

    def __init__(
        self,
        *,
        context_chunk_size=0,
        is_last_context_chunk=True,
        prompt_len=None,
        context_current_position=0,
        py_beam_width=1,
        py_draft_tokens=None,
        is_disagg_generation_init_state=False,
        mm_bidirectional=False,
        prepopulated_prompt_len=0,
    ):
        _FakeRequest._next_id += 1
        self.py_request_id = _FakeRequest._next_id
        # PyExecutor's inflight-set bookkeeping reads ``request_id`` (the C++
        # binding's name) rather than ``py_request_id``; keep them in sync so the
        # same fake drives both the trim and TestInflightIdsSurviveTrim.
        self.request_id = self.py_request_id
        self.context_chunk_size = context_chunk_size
        self.context_current_position = context_current_position
        # Mirrors the C++ semantics: is_last_context_chunk is a *computed*
        # property (context_current_position + context_chunk_size == prompt_len),
        # so shrinking the chunk flips it to False. When prompt_len is None the
        # flag is a fixed override (for tests that don't exercise re-binning).
        self._prompt_len = prompt_len
        self._is_last_override = is_last_context_chunk
        self.py_beam_width = py_beam_width
        self.py_draft_tokens = py_draft_tokens
        self.is_disagg_generation_init_state = is_disagg_generation_init_state
        self.py_multimodal_data = {"mm_bidirectional_blocks": True} if mm_bidirectional else None
        self.prepopulated_prompt_len = prepopulated_prompt_len
        # The discard path rewinds to a fresh-arrival state, which is expressed
        # in terms of prompt_len and the scheduler's reuse estimate.
        self.prompt_len = prompt_len if prompt_len is not None else context_chunk_size
        self.estimated_reusable_tokens = 0

    def set_prepopulated_prompt_len(self, prepopulated_prompt_len, kv_tokens_per_block):
        # C++ only advances context_current_position for a non-zero reuse hit;
        # the discard path relies on that asymmetry, so mirror it exactly.
        self.prepopulated_prompt_len = prepopulated_prompt_len
        if prepopulated_prompt_len > 0:
            self.context_current_position = prepopulated_prompt_len

    @property
    def is_first_context_chunk(self):
        # C++: getContextCurrentPosition() == getPrepopulatedPromptLen(), i.e.
        # nothing of this request's context has been computed yet.
        return self.context_current_position == self.prepopulated_prompt_len

    @property
    def is_last_context_chunk(self):
        if self._prompt_len is None:
            return self._is_last_override
        return self.context_current_position + self.context_chunk_size == self._prompt_len

    @property
    def context_remaining_length(self):
        # C++: mPromptLen - getContextCurrentPosition(). With no prompt_len the
        # chunk is by definition all that is left.
        if self._prompt_len is None:
            return self.context_chunk_size
        return self._prompt_len - self.context_current_position


def _make_manager(max_num_tokens, tokens_per_block, enable_chunked_prefill=True):
    # Skip the heavy (GPU-allocating) __init__; the method under test only
    # needs these attributes plus its own (bound) helper methods.
    mgr = KVCacheManager.__new__(KVCacheManager)
    mgr.max_num_tokens = max_num_tokens
    mgr.tokens_per_block = tokens_per_block
    # Shrinking produces a partial context chunk, which only chunked prefill's
    # attention path can consume. Default to enabled; the disabled case is
    # covered explicitly below.
    mgr.enable_chunked_prefill = enable_chunked_prefill
    mgr.is_draft = False
    # The discard path refuses to unschedule anything while a connector is
    # attached (it has no inverse for update_state_after_alloc).
    mgr.kv_connector_manager = None
    return mgr


class _RecordingDiscard:
    """Stands in for ResourceManager.discard_request."""

    def __init__(self):
        self.discarded = []

    def __call__(self, req):
        self.discarded.append(req)

    @property
    def ids(self):
        return [r.py_request_id for r in self.discarded]


def _make_batch(context_requests=(), generation_requests=()):
    batch = ScheduledRequests()
    for req in context_requests:
        batch.append_context_request(req)
    batch.generation_requests = list(generation_requests)
    return batch


def _forward_tokens(mgr, batch):
    """What _prepare_tp_inputs will materialize for this batch."""
    return sum(
        mgr._request_forward_tokens(r, is_context=False) for r in batch.generation_requests
    ) + sum(
        mgr._request_forward_tokens(r, is_context=True)
        for r in batch.context_requests
        if not r.is_disagg_generation_init_state
    )


class TestReuseDiscountedChunk(unittest.TestCase):
    """Regression for the defect this trim shipped with.

    Read before ``prepare_resources``, ``context_chunk_size`` spans the reusable
    KV prefix: a 19212-token prompt with a 19200-token cache hit still reports
    ``context_chunk_size == 19212`` while the forward pass will compute 12
    tokens. Costing it at 19212 makes every reuse hit look 1600x too expensive,
    so it is endlessly re-chunked (chunked prefill on) or deferred (off).

    Read after ``prepare_resources`` -- where the trim now runs --
    ``setPrepopulatedPromptLen`` has advanced ``context_current_position`` past
    the prefix and the same request costs 12.
    """

    def test_reuse_hit_costs_only_the_uncached_tail(self):
        mgr = _make_manager(max_num_tokens=8192, tokens_per_block=32)
        # Post-setPrepopulatedPromptLen state for a 19212-token prompt with a
        # 19200-token cache hit.
        req = _FakeRequest(context_chunk_size=12, context_current_position=19200, prompt_len=19212)
        self.assertEqual(mgr._request_forward_tokens(req, is_context=True), 12)

    def test_reuse_hit_is_not_trimmed(self):
        mgr = _make_manager(max_num_tokens=8192, tokens_per_block=32)
        reqs = [
            _FakeRequest(context_chunk_size=12, context_current_position=19200, prompt_len=19212)
            for _ in range(4)
        ]
        batch = _make_batch(reqs, [_FakeRequest(py_beam_width=1) for _ in range(3)])

        mgr.fit_token_budget(batch)

        self.assertEqual(batch.num_context_requests, 4, "no request may be dropped")
        for req in reqs:
            self.assertEqual(req.context_chunk_size, 12, "chunk must be untouched")

    def test_chunk_beyond_prompt_end_is_clamped(self):
        # _prepare_tp_inputs slices all_prompt_tokens[pos:pos + chunk], and
        # Python clamps that to the end of the list. A chunk that overhangs the
        # prompt must be costed at what is actually left, not at its nominal
        # size.
        mgr = _make_manager(max_num_tokens=8192, tokens_per_block=32)
        req = _FakeRequest(
            context_chunk_size=8160, context_current_position=19200, prompt_len=19212
        )
        self.assertEqual(mgr._request_forward_tokens(req, is_context=True), 12)


class TestFitTokenBudget(unittest.TestCase):
    def test_request_forward_tokens(self):
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)

        # Context: materialized chunk, plus draft tokens only on the last chunk.
        last = _FakeRequest(
            context_chunk_size=10, is_last_context_chunk=True, py_draft_tokens=[1, 2]
        )
        self.assertEqual(mgr._request_forward_tokens(last, is_context=True), 12)
        mid = _FakeRequest(
            context_chunk_size=10, is_last_context_chunk=False, py_draft_tokens=[1, 2]
        )
        self.assertEqual(mgr._request_forward_tokens(mid, is_context=True), 10)

        # Generation: (1 + draft) per beam.
        gen = _FakeRequest(py_beam_width=2, py_draft_tokens=[1, 2, 3])
        self.assertEqual(mgr._request_forward_tokens(gen, is_context=False), 8)

    def test_within_budget_is_noop(self):
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=16)
        gen = _FakeRequest(py_beam_width=100)  # 100 gen tokens
        batch = _make_batch([ctx], [gen])

        mgr.fit_token_budget(batch)

        self.assertEqual(batch.num_context_requests, 1)
        self.assertEqual(ctx.context_chunk_size, 16)  # untouched

    def test_overshoot_shrinks_context_to_fit(self):
        # 100 gen tokens leave a 28-token budget; a 64-token last chunk does not
        # fit and must be shrunk to the largest block-aligned chunk that does.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch([ctx], [gen])

        mgr.fit_token_budget(batch)

        self.assertEqual(ctx.context_chunk_size, 16)
        self.assertLessEqual(_forward_tokens(mgr, batch), 128)

    def test_nothing_is_ever_dropped(self):
        # The defining property of the post-allocation trim: KV is allocated and
        # sequences are added, so a request may be shrunk but never removed.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctxs = [_FakeRequest(context_chunk_size=64, prompt_len=64) for _ in range(3)]
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch(ctxs, [gen])

        mgr.fit_token_budget(batch)

        self.assertEqual(batch.num_context_requests, 3)
        for ctx in ctxs:
            self.assertIn(ctx, batch.context_requests)

    def test_shrink_keeps_chunk_end_block_aligned(self):
        # setPrepopulatedPromptLen asserts (pos + chunk) % tokens_per_block == 0
        # for every non-last chunk, to keep the KV cache unfragmented.
        mgr = _make_manager(max_num_tokens=100, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=96, context_current_position=32, prompt_len=128)
        gen = _FakeRequest(py_beam_width=53)
        batch = _make_batch([ctx], [gen])

        mgr.fit_token_budget(batch)

        self.assertLess(ctx.context_chunk_size, 96)
        self.assertEqual(
            (ctx.context_current_position + ctx.context_chunk_size) % 16,
            0,
            "chunk end must land on a block boundary",
        )

    def test_shrink_never_produces_a_zero_token_chunk(self):
        # A zero-token chunk leaves the request scheduled but computing nothing,
        # which never terminates. One block of progress is the floor even when
        # that overshoots the budget.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        gen = _FakeRequest(py_beam_width=127)  # leaves a 1-token budget
        batch = _make_batch([ctx], [gen])

        mgr.fit_token_budget(batch)

        self.assertEqual(ctx.context_chunk_size, 16)

    def test_shrink_rebins_to_chunking(self):
        # Shrinking flips is_last_context_chunk to False, so the request must
        # move out of context_requests_last_chunk -- otherwise downstream treats
        # it as a final chunk and appends generation / draft tokens to it.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch([ctx], [gen])
        self.assertEqual(batch.context_requests_last_chunk, [ctx])

        mgr.fit_token_budget(batch)

        self.assertEqual(batch.context_requests_last_chunk, [])
        self.assertEqual(batch.context_requests_chunking, [ctx])

    def test_shrink_drops_last_chunk_draft_tokens(self):
        # Draft tokens ride only on the last chunk, so a shrunk request stops
        # contributing them and the budget must account for that.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64, py_draft_tokens=[1, 2, 3, 4])
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch([ctx], [gen])
        self.assertEqual(mgr._request_forward_tokens(ctx, is_context=True), 68)

        mgr.fit_token_budget(batch)

        self.assertFalse(ctx.is_last_context_chunk)
        self.assertEqual(mgr._request_forward_tokens(ctx, is_context=True), 16)

    def test_sheds_the_last_chunk_first(self):
        # context_requests is chunking + last_chunk, so the trim walks last-chunk
        # requests first. That is the request whose cost the scheduler can have
        # under-charged (only a last chunk carries a reuse discount), so it is
        # the right one to repair; mid-prefill chunks are touched only if
        # trimming it is not enough.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        chunking = _FakeRequest(
            context_chunk_size=96, prompt_len=256
        )  # pos 0 + 96 != 256 -> chunking
        last = _FakeRequest(context_chunk_size=64, prompt_len=64)
        batch = _make_batch([chunking, last], [])
        self.assertEqual(batch.context_requests_chunking, [chunking])
        self.assertEqual(batch.context_requests_last_chunk, [last])

        mgr.fit_token_budget(batch)  # 160 tokens vs a 128 budget

        self.assertEqual(chunking.context_chunk_size, 96, "mid-prefill untouched")
        self.assertEqual(last.context_chunk_size, 32, "last chunk absorbed the excess")
        self.assertLessEqual(_forward_tokens(mgr, batch), 128)

    def test_shrinks_multiple_requests_when_one_is_not_enough(self):
        mgr = _make_manager(max_num_tokens=64, tokens_per_block=16)
        ctxs = [_FakeRequest(context_chunk_size=64, prompt_len=64) for _ in range(3)]
        batch = _make_batch(ctxs, [])

        mgr.fit_token_budget(batch)

        self.assertLessEqual(_forward_tokens(mgr, batch), 64)
        self.assertEqual(batch.num_context_requests, 3)

    def test_no_shrink_when_chunked_prefill_disabled(self):
        # A partial context chunk is only valid under chunked prefill; forcing
        # one produces an invalid forward pass. With no discard callback that
        # leaves nothing to do -- unscheduling whole requests, which is what
        # this configuration actually falls back to, is covered by
        # TestDiscardWhenShrinkIsNotEnough.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16, enable_chunked_prefill=False)
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch([ctx], [gen])

        mgr.fit_token_budget(batch)

        self.assertEqual(ctx.context_chunk_size, 64)
        self.assertEqual(batch.num_context_requests, 1)

    def test_mm_bidirectional_is_not_shrunk(self):
        # Splitting a bidirectional multimodal block silently breaks attention.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64, mm_bidirectional=True)
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch([ctx], [gen])

        mgr.fit_token_budget(batch)

        self.assertEqual(ctx.context_chunk_size, 64)

    def test_disagg_gen_init_requests_are_left_alone(self):
        # They only allocate/transfer KV cache and contribute no compute tokens.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        disagg = _FakeRequest(context_chunk_size=4096, is_disagg_generation_init_state=True)
        ctx = _FakeRequest(context_chunk_size=16, prompt_len=16)
        batch = _make_batch([disagg, ctx], [])

        mgr.fit_token_budget(batch)

        self.assertEqual(disagg.context_chunk_size, 4096)
        self.assertEqual(ctx.context_chunk_size, 16)

    def test_gen_only_batch_is_left_alone(self):
        # Generation cannot be shed, so a batch with no context requests returns
        # immediately -- keeping the executor loop's hottest path free of an
        # O(num_generation_requests) scan.
        mgr = _make_manager(max_num_tokens=8, tokens_per_block=16)
        batch = _make_batch([], [_FakeRequest(py_beam_width=64)])

        mgr.fit_token_budget(batch)  # must not raise

        self.assertEqual(len(batch.generation_requests), 1)

    def test_generation_alone_over_budget_does_not_raise(self):
        # There is nothing to shed, but raising here would be rank-local and
        # would deadlock the surviving ranks under attention DP. Warn and let
        # the forward pass report it.
        mgr = _make_manager(max_num_tokens=64, tokens_per_block=16)
        ctx = _FakeRequest(context_chunk_size=16, prompt_len=16)
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch([ctx], [gen])

        mgr.fit_token_budget(batch)  # must not raise

        self.assertEqual(batch.num_context_requests, 1)

    def test_maybe_fit_token_budget_skips_draft_manager(self):
        # The draft-model engine builds inputs with a different token shape and
        # its budget is handled separately.
        gen = _FakeRequest(py_beam_width=100)

        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        mgr.is_draft = False
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        mgr.maybe_fit_token_budget(_make_batch([ctx], [gen]))
        self.assertEqual(ctx.context_chunk_size, 16)

        mgr.is_draft = True
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        mgr.maybe_fit_token_budget(_make_batch([ctx], [gen]))
        self.assertEqual(ctx.context_chunk_size, 64)

    def test_trim_runs_after_every_manager(self):
        # The trim must observe the batch as prepare_resources leaves it: only
        # after addSequence has run does context_chunk_size mean forward-pass
        # tokens (setPrepopulatedPromptLen advances context_current_position
        # past the reusable prefix). Registering the KV cache manager last
        # mirrors _util.py's move_to_end(KV_CACHE_MANAGER); the trim must still
        # run after that.
        observed = []

        target = _make_manager(max_num_tokens=128, tokens_per_block=16)
        target.prepare_resources = lambda batch: observed.append("kv_cache_manager")

        class _RecordingManager:
            def prepare_resources(self, batch):
                observed.append("draft_manager")

        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch([ctx], [gen])

        rm = ResourceManager(
            OrderedDict(
                [
                    (ResourceManagerType.DRAFT_KV_CACHE_MANAGER, _RecordingManager()),
                    (ResourceManagerType.KV_CACHE_MANAGER, target),
                ]
            )
        )
        rm.prepare_resources(batch)

        self.assertEqual(observed, ["draft_manager", "kv_cache_manager"])
        self.assertEqual(ctx.context_chunk_size, 16, "trim ran after both managers")

    def test_prepare_resources_trims(self):
        # prepare_resources is the only entry point; the executor loops must not
        # need their own call.
        target = _make_manager(max_num_tokens=128, tokens_per_block=16)
        target.prepare_resources = lambda batch: None

        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        gen = _FakeRequest(py_beam_width=100)
        batch = _make_batch([ctx], [gen])

        rm = ResourceManager(OrderedDict([(ResourceManagerType.KV_CACHE_MANAGER, target)]))
        rm.prepare_resources(batch)

        self.assertEqual(ctx.context_chunk_size, 16)


class TestInflightIdsSurviveTrim(unittest.TestCase):
    """The trim must not strand ids in PyExecutor's inflight set.

    ``_executor_loop_pp`` calls ``_add_inflight_ids`` before
    ``ResourceManager.prepare_resources`` and ``_remove_inflight_ids`` after, so
    the trim runs between them and moves shrunk requests out of
    ``context_requests_last_chunk``. Removal must therefore erase the ids that
    were actually inserted, not re-derive them from the trimmed batch -- an id
    left behind makes the scheduler skip that request forever (scheduler.py's
    ``if req.request_id in inflight_request_ids: continue``).
    """

    @staticmethod
    def _bare_executor():
        # Same trick as _make_manager: PyExecutor.__init__ builds an engine, so
        # instantiate bare and supply only the inflight set the methods touch.
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        from tensorrt_llm.bindings.internal.batch_manager import ReqIdsSet

        executor = PyExecutor.__new__(PyExecutor)
        executor.inflight_req_ids = ReqIdsSet()
        return executor

    def test_shrunk_context_requests_leave_no_inflight_ids(self):
        executor = self._bare_executor()
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)

        gen = _FakeRequest(py_beam_width=100)  # leaves a 28-token budget
        # Starts as a last-chunk request, so it is registered inflight; the trim
        # then shrinks it to a block-aligned 16 and it stops being a last chunk.
        ctx = _FakeRequest(context_chunk_size=64, prompt_len=64)
        batch = _make_batch([ctx], [gen])

        executor._add_inflight_ids(batch)
        self.assertEqual(
            sorted(batch.added_inflight_req_ids),
            sorted([ctx.request_id, gen.request_id]),
        )

        mgr.fit_token_budget(batch)

        # Precondition for the regression: the batch really did change shape.
        self.assertEqual(ctx.context_chunk_size, 16)
        self.assertIn(ctx, batch.context_requests_chunking)
        self.assertEqual(batch.context_requests_last_chunk, [])

        executor._remove_inflight_ids(batch)

        for req in (ctx, gen):
            self.assertNotIn(
                req.request_id,
                executor.inflight_req_ids,
                f"request {req.request_id} left in the inflight set; the scheduler "
                "would never schedule it again",
            )
        self.assertEqual(batch.added_inflight_req_ids, [])

    def test_untrimmed_batch_round_trips(self):
        # The batch the trim leaves alone must behave exactly as before.
        executor = self._bare_executor()
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)

        gen = _FakeRequest(py_beam_width=1)
        ctx = _FakeRequest(context_chunk_size=16)
        batch = _make_batch([ctx], [gen])

        executor._add_inflight_ids(batch)
        for req in (ctx, gen):
            self.assertIn(req.request_id, executor.inflight_req_ids)

        mgr.fit_token_budget(batch)
        self.assertEqual(batch.num_context_requests, 1)

        executor._remove_inflight_ids(batch)
        for req in (ctx, gen):
            self.assertNotIn(req.request_id, executor.inflight_req_ids)


class TestDiscardWhenShrinkIsNotEnough(unittest.TestCase):
    """The last-resort tier.

    ``_shrink_context_chunk`` floors every request at one block of forward
    progress, so a batch admitted with many under-charged context requests can
    sit above ``max_num_tokens`` even once every chunk is at its floor: the
    smallest batch shrinking can produce is ``n_ctx * tokens_per_block``. Doing
    nothing there means the ``_prepare_tp_inputs`` assert, which fails every
    request in the batch and kills the executor loop (#13318), so requests are
    unscheduled until the batch fits.
    """

    TPB = 32

    def _maxed_batch(self, n_ctx, prompt_len=4096, n_gen=0):
        """n_ctx first-chunk requests each needing its whole prompt."""
        ctx = [
            _FakeRequest(context_chunk_size=prompt_len, prompt_len=prompt_len) for _ in range(n_ctx)
        ]
        gen = [
            _FakeRequest(context_chunk_size=0, is_last_context_chunk=False) for _ in range(n_gen)
        ]
        return ctx, gen, _make_batch(ctx, gen)

    def test_unschedules_until_the_batch_fits(self):
        # 8 requests, floor 8*32 = 256, budget 128 -> shrinking cannot get there.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=self.TPB)
        ctx, _, batch = self._maxed_batch(8)
        discard = _RecordingDiscard()

        mgr.fit_token_budget(batch, discard)

        self.assertLessEqual(_forward_tokens(mgr, batch), 128)
        # Exactly the surplus: 4 kept at a 32-token floor == the 128 budget.
        self.assertEqual(len(discard.discarded), 4)
        self.assertEqual(batch.num_context_requests, 4)
        # Shed from the back, so the survivors are the front of the batch.
        self.assertEqual(
            [r.py_request_id for r in batch.context_requests],
            [r.py_request_id for r in ctx[:4]],
        )
        for req in discard.discarded:
            self.assertNotIn(req, batch.context_requests)

    def test_shrinking_is_preferred_and_discard_is_untouched(self):
        # Same batch, a budget the floor fits under: nothing may be unscheduled.
        mgr = _make_manager(max_num_tokens=512, tokens_per_block=self.TPB)
        _, _, batch = self._maxed_batch(8)
        discard = _RecordingDiscard()

        mgr.fit_token_budget(batch, discard)

        self.assertEqual(discard.discarded, [])
        self.assertEqual(batch.num_context_requests, 8)
        self.assertLessEqual(_forward_tokens(mgr, batch), 512)

    def test_without_a_discard_callback_the_trim_only_shrinks(self):
        # Part 2 behaviour is the default: a caller that supplies no callback
        # gets shrink-only, over budget but with the batch intact.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=self.TPB)
        _, _, batch = self._maxed_batch(8)

        mgr.fit_token_budget(batch)

        self.assertEqual(batch.num_context_requests, 8)
        self.assertGreater(_forward_tokens(mgr, batch), 128)

    def test_never_empties_the_batch(self):
        # A budget below even a single block. The batch feeds the attention-DP
        # _can_queue vote (0 not in tp_batch_sizes), taken before
        # prepare_resources; emptying it here would leave peers blocked in a
        # collective this rank never enters.
        mgr = _make_manager(max_num_tokens=16, tokens_per_block=self.TPB)
        _, _, batch = self._maxed_batch(4)
        discard = _RecordingDiscard()

        mgr.fit_token_budget(batch, discard)

        self.assertEqual(batch.batch_size, 1)
        self.assertEqual(len(discard.discarded), 3)
        # Still over budget -- reported, not asserted on.
        self.assertGreater(_forward_tokens(mgr, batch), 16)

    def test_only_as_many_as_the_budget_needs_are_unscheduled(self):
        # 4 requests at a 32-token floor plus 8 generation tokens is 136; a
        # budget of 64 is met by giving up 3 of them, so the 4th stays.
        mgr = _make_manager(max_num_tokens=64, tokens_per_block=self.TPB)
        _, _, batch = self._maxed_batch(4, n_gen=8)
        discard = _RecordingDiscard()

        mgr.fit_token_budget(batch, discard)

        self.assertEqual(len(discard.discarded), 3)
        self.assertEqual(batch.num_context_requests, 1)
        self.assertLessEqual(_forward_tokens(mgr, batch), 64)

    def test_generation_requests_let_every_context_request_go(self):
        # With generation requests in the batch it stays non-empty without any
        # context request, so all of them may be unscheduled.
        mgr = _make_manager(max_num_tokens=16, tokens_per_block=self.TPB)
        _, gen, batch = self._maxed_batch(4, n_gen=8)
        discard = _RecordingDiscard()

        mgr.fit_token_budget(batch, discard)

        self.assertEqual(len(discard.discarded), 4)
        self.assertEqual(batch.num_context_requests, 0)
        self.assertEqual(batch.batch_size, len(gen))
        self.assertLessEqual(_forward_tokens(mgr, batch), 16)

    def test_mid_prefill_requests_are_never_unscheduled(self):
        # Only a first chunk can be retried from scratch for free, and later
        # chunks are charged exactly (getEstimatedReusableTokens returns 0 for
        # them), so they are never the reason the batch is over budget.
        mgr = _make_manager(max_num_tokens=32, tokens_per_block=self.TPB)
        mid = _FakeRequest(
            context_chunk_size=2048,
            prompt_len=4096,
            context_current_position=2048,
            prepopulated_prompt_len=0,
        )
        fresh = _FakeRequest(context_chunk_size=4096, prompt_len=4096)
        batch = _make_batch([mid, fresh])
        discard = _RecordingDiscard()

        self.assertFalse(mid.is_first_context_chunk)
        mgr.fit_token_budget(batch, discard)

        self.assertEqual(discard.ids, [fresh.py_request_id])
        self.assertIn(mid, batch.context_requests)

    def test_reuse_hit_request_is_still_a_first_chunk(self):
        # addSequence advances context_current_position to
        # prepopulated_prompt_len, which leaves is_first_context_chunk true --
        # a reuse hit whose estimate was wrong is exactly the population this
        # tier exists for, so it must remain eligible.
        mgr = _make_manager(max_num_tokens=32, tokens_per_block=self.TPB)
        reused = _FakeRequest(
            context_chunk_size=2048,
            prompt_len=4096,
            context_current_position=2048,
            prepopulated_prompt_len=2048,
        )
        # A generation request keeps the batch non-empty once it goes, so the
        # reuse-hit request is the only thing left to shed.
        gen = _FakeRequest(context_chunk_size=0, is_last_context_chunk=False)
        batch = _make_batch([reused], [gen])
        discard = _RecordingDiscard()

        self.assertTrue(reused.is_first_context_chunk)
        mgr.fit_token_budget(batch, discard)

        self.assertEqual(discard.ids, [reused.py_request_id])

    def test_disagg_generation_init_requests_are_not_unscheduled(self):
        # They contribute no compute tokens (nothing to shed) and their KV cache
        # is being filled by an in-flight transfer.
        mgr = _make_manager(max_num_tokens=32, tokens_per_block=self.TPB)
        disagg = _FakeRequest(
            context_chunk_size=4096,
            prompt_len=4096,
            is_disagg_generation_init_state=True,
        )
        a = _FakeRequest(context_chunk_size=4096, prompt_len=4096)
        b = _FakeRequest(context_chunk_size=4096, prompt_len=4096)
        batch = _make_batch([disagg, a, b])
        discard = _RecordingDiscard()

        mgr.fit_token_budget(batch, discard)

        self.assertNotIn(disagg.py_request_id, discard.ids)
        self.assertIn(disagg, batch.context_requests)

    def test_a_kv_connector_blocks_unscheduling(self):
        # build_scheduler_output has already reported the batch and
        # update_state_after_alloc has fired; neither has an inverse.
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=self.TPB)
        mgr.kv_connector_manager = object()
        _, _, batch = self._maxed_batch(8)
        discard = _RecordingDiscard()

        mgr.fit_token_budget(batch, discard)

        self.assertEqual(discard.discarded, [])
        self.assertEqual(batch.num_context_requests, 8)

    def test_chunked_prefill_disabled_discards_instead_of_shrinking(self):
        # With chunked prefill off there is no shrink tier at all: a partial
        # context chunk is not something the attention backend can consume. So
        # every over-budget token has to come off as whole requests. This is the
        # configuration in #13318, and doing nothing here is the assert.
        mgr = _make_manager(
            max_num_tokens=8192, tokens_per_block=self.TPB, enable_chunked_prefill=False
        )
        ctx, _, batch = self._maxed_batch(8, prompt_len=4096)
        discard = _RecordingDiscard()

        mgr.fit_token_budget(batch, discard)

        # 8 x 4096 against 8192: only two whole requests fit, and no chunk is
        # touched on the way -- shrinking would have produced a partial chunk.
        self.assertEqual(len(discard.discarded), 6)
        self.assertEqual(batch.num_context_requests, 2)
        self.assertLessEqual(_forward_tokens(mgr, batch), 8192)
        for req in batch.context_requests:
            self.assertEqual(req.context_chunk_size, 4096)

    def test_chunked_prefill_disabled_leaves_chunks_untouched_when_it_cannot_fit(self):
        # The floor of the discard tier is one request. With chunked prefill off
        # that request cannot then be shrunk either, so the batch stays over
        # budget -- reported, and with its chunk intact rather than made partial.
        mgr = _make_manager(
            max_num_tokens=1024, tokens_per_block=self.TPB, enable_chunked_prefill=False
        )
        ctx, _, batch = self._maxed_batch(3, prompt_len=4096)
        discard = _RecordingDiscard()

        unfittable = mgr.fit_token_budget(batch, discard)

        self.assertEqual(len(discard.discarded), 2)
        self.assertEqual(batch.batch_size, 1)
        self.assertEqual(batch.context_requests[0].context_chunk_size, 4096)
        self.assertGreater(_forward_tokens(mgr, batch), 1024)
        # No batch can ever hold what is left, so it is reported for failure
        # rather than retried forever.
        self.assertEqual(unfittable, [batch.context_requests[0]])


class TestUnfittableRequestsAreReported(unittest.TestCase):
    """The terminal tier: a request no batch can ever hold.

    Shrinking and unscheduling both bottom out at "one request must stay in the
    batch", because the attention-DP _can_queue vote was taken before
    prepare_resources and emptying the batch after it leaves peers blocked in a
    collective this rank never enters. If that last request costs more than the
    whole budget it can never run: retrying rebuilds this same batch forever and
    running it trips the _prepare_tp_inputs assert. fit_token_budget reports it
    so the executor can fail it against the client.
    """

    TPB = 32

    def test_nothing_reported_when_the_batch_fits(self):
        mgr = _make_manager(max_num_tokens=8192, tokens_per_block=self.TPB)
        batch = _make_batch([_FakeRequest(context_chunk_size=64, prompt_len=64)])

        self.assertEqual(mgr.fit_token_budget(batch, _RecordingDiscard()), [])

    def test_nothing_reported_when_shrinking_rescued_the_batch(self):
        mgr = _make_manager(max_num_tokens=1024, tokens_per_block=self.TPB)
        ctx = [_FakeRequest(context_chunk_size=4096, prompt_len=4096) for _ in range(2)]
        batch = _make_batch(ctx)

        self.assertEqual(mgr.fit_token_budget(batch, _RecordingDiscard()), [])
        self.assertLessEqual(_forward_tokens(mgr, batch), 1024)

    def test_a_request_that_fits_alone_is_never_reported(self):
        # The batch is over budget only because of the other requests; the
        # survivor is viable, so it is retried rather than failed.
        mgr = _make_manager(
            max_num_tokens=4096, tokens_per_block=self.TPB, enable_chunked_prefill=False
        )
        ctx = [_FakeRequest(context_chunk_size=4096, prompt_len=4096) for _ in range(3)]
        batch = _make_batch(ctx)

        unfittable = mgr.fit_token_budget(batch, _RecordingDiscard())

        self.assertEqual(batch.batch_size, 1)
        self.assertEqual(_forward_tokens(mgr, batch), 4096)
        self.assertEqual(unfittable, [])

    def test_generation_overshoot_does_not_report_a_context_request(self):
        # Generation requests cannot be shed at all, so an over-budget batch of
        # them is a configuration problem (README F4), not a request that can be
        # blamed and failed.
        mgr = _make_manager(max_num_tokens=64, tokens_per_block=self.TPB)
        gen = [_FakeRequest(py_beam_width=100) for _ in range(2)]
        batch = _make_batch([_FakeRequest(context_chunk_size=32, prompt_len=32)], gen)

        unfittable = mgr.fit_token_budget(batch, _RecordingDiscard())

        self.assertEqual(unfittable, [])

    def test_shrink_only_callers_still_get_a_report(self):
        # A caller with no discard callback (Part 2 behaviour) still learns that
        # the batch cannot be fitted.
        mgr = _make_manager(
            max_num_tokens=1024, tokens_per_block=self.TPB, enable_chunked_prefill=False
        )
        req = _FakeRequest(context_chunk_size=4096, prompt_len=4096)
        batch = _make_batch([req])

        self.assertEqual(mgr.fit_token_budget(batch), [req])


class TestDiscardRequestUnwind(unittest.TestCase):
    """What ``discard_request`` has to undo for the retry to be correct."""

    class _FakeImpl:
        def __init__(self):
            self.removed = []

        def remove_sequence(self, request_id, llm_request, pin_on_release):
            # The C++ side reads the cursor *during* the call to decide how much
            # to store for reuse, so record it as it stands here.
            self.removed.append(
                (request_id, llm_request, pin_on_release, llm_request.context_current_position)
            )

    def _manager(self):
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=32)
        mgr.impl = self._FakeImpl()
        return mgr

    def test_blocks_are_released_without_being_stored_for_reuse(self):
        # releaseBlocks stores min(num_tokens, context_current_position) - 1
        # tokens' worth of blocks in the reuse trie, and at position 0 its
        # legacy fallback stores num_tokens - 1 instead. This request has not
        # run a forward pass, so either would publish uninitialized blocks as
        # reusable. Position 1 stores nothing (usable count 0) and keeps the
        # fallback's position == 0 guard false.
        mgr = self._manager()
        req = _FakeRequest(context_chunk_size=4096, prompt_len=4096)

        mgr.discard_request(req)

        self.assertEqual(len(mgr.impl.removed), 1)
        request_id, _, pin_on_release, position_at_call = mgr.impl.removed[0]
        self.assertEqual(request_id, req.py_request_id)
        self.assertEqual(position_at_call, 1)
        self.assertFalse(pin_on_release)

    def test_a_reuse_hit_is_also_removed_at_the_no_store_position(self):
        # Removing at the real position (prepopulated_prompt_len) would re-store
        # a prefix that is already in the trie; only the no-store position is
        # used, whatever the request came in with.
        mgr = self._manager()
        req = _FakeRequest(
            context_chunk_size=2048,
            prompt_len=4096,
            context_current_position=2048,
            prepopulated_prompt_len=2048,
        )

        mgr.discard_request(req)

        self.assertEqual(mgr.impl.removed[0][3], 1)

    def test_cursor_is_rewound_so_the_retry_re_enters_add_sequence(self):
        # addSequence advanced the position to prepopulated_prompt_len. Left
        # there once the blocks are gone, the next addSequence -- which only
        # moves the position for a non-zero reuse hit -- would skip recomputing
        # a prefix that no longer has KV cache behind it.
        mgr = self._manager()
        req = _FakeRequest(
            context_chunk_size=2048,
            prompt_len=4096,
            context_current_position=2048,
            prepopulated_prompt_len=2048,
        )

        mgr.discard_request(req)

        self.assertEqual(req.context_current_position, 0)
        self.assertEqual(req.prepopulated_prompt_len, 0)
        # The whole prompt, matching LlmRequest::pause's reset. A zero chunk
        # makes the next setPrepopulatedPromptLen compute a *negative* chunk
        # size wherever nothing re-chunks the request first -- which is every
        # request when chunked prefill is disabled.
        self.assertEqual(req.context_chunk_size, req.prompt_len)
        self.assertEqual(req.estimated_reusable_tokens, 0)
        # A fresh arrival again: the retry allocates from scratch.
        self.assertTrue(req.is_first_context_chunk)


class TestResourceManagerDiscard(unittest.TestCase):
    """Every manager must get the chance to undo, not just the KV cache one."""

    class _Recorder:
        def __init__(self, log, name):
            self._log = log
            self._name = name

        def discard_request(self, request):
            self._log.append(self._name)

    def test_all_managers_are_unwound_in_reverse_order(self):
        # Mirrors free_resources: a manager's undo runs before that of the
        # manager it was prepared after. Eagle3 / MTP allocate a slot per
        # first-context-chunk request and SlotManager.add_slot asserts on a
        # duplicate id, so skipping them would abort the loop on the retry.
        log = []
        rm = ResourceManager(
            OrderedDict(
                [
                    (ResourceManagerType.KV_CACHE_MANAGER, self._Recorder(log, "kv")),
                    (ResourceManagerType.SPEC_RESOURCE_MANAGER, self._Recorder(log, "spec")),
                    (ResourceManagerType.SEQ_SLOT_MANAGER, self._Recorder(log, "slot")),
                ]
            )
        )

        rm.discard_request(_FakeRequest())

        self.assertEqual(log, ["slot", "spec", "kv"])

    def test_managers_without_an_undo_are_skipped(self):
        rm = ResourceManager(OrderedDict([(ResourceManagerType.SEQ_SLOT_MANAGER, object())]))
        rm.discard_request(_FakeRequest())  # must not raise


class TestConnectorIsToldTheTrimmedBatch(unittest.TestCase):
    """The KV connector must see the batch that will actually run.

    RequestData.num_scheduled_tokens is documented as "the number of scheduled
    tokens for the upcoming forward pass" and is built from context_chunk_size,
    so reporting the batch before the trim over-states it for every chunk the
    trim shrinks -- and a connector that decides what to save or offload from
    that count would publish KV for tokens the forward pass never computed.
    """

    class _RecordingKvManager:
        """Stands in for KVCacheManager: records when it is asked to publish."""

        def __init__(self, log, batch, shrink_to):
            self._log = log
            self._batch = batch
            self._shrink_to = shrink_to

        def prepare_resources(self, scheduled_batch):
            self._log.append(("prepare", self._chunks()))

        def maybe_fit_token_budget(self, scheduled_batch, discard=None):
            for req in scheduled_batch.context_requests:
                req.context_chunk_size = self._shrink_to
            self._log.append(("trim", self._chunks()))
            return []

        def publish_connector_scheduler_output(self, scheduled_batch):
            self._log.append(("publish", self._chunks()))

        def _chunks(self):
            return [r.context_chunk_size for r in self._batch.context_requests]

    def _resource_manager(self, log, batch, shrink_to):
        return ResourceManager(
            OrderedDict(
                [
                    (
                        ResourceManagerType.KV_CACHE_MANAGER,
                        self._RecordingKvManager(log, batch, shrink_to),
                    )
                ]
            )
        )

    def test_the_connector_is_told_after_the_trim(self):
        log = []
        req = _FakeRequest(context_chunk_size=4096, prompt_len=4096)
        batch = _make_batch([req])

        self._resource_manager(log, batch, shrink_to=64).prepare_resources(batch)

        self.assertEqual([step for step, _ in log], ["prepare", "trim", "publish"])
        # The count the connector sees is the trimmed one, not the scheduled one.
        self.assertEqual(dict(log)["publish"], [64])

    def test_managers_without_a_connector_hook_are_skipped(self):
        rm = ResourceManager(OrderedDict([(ResourceManagerType.KV_CACHE_MANAGER, object())]))
        self.assertEqual(rm.prepare_resources(_make_batch()), [])  # must not raise

    def test_publishing_is_a_no_op_without_a_connector(self):
        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        mgr.kv_connector_manager = None
        mgr.publish_connector_scheduler_output(_make_batch())  # must not raise

    def test_publishing_forwards_the_batch_to_the_connector(self):
        class _FakeConnector:
            def __init__(self):
                self.calls = []

            def build_scheduler_output(self, scheduled_batch, kv_cache_manager):
                self.calls.append((scheduled_batch, kv_cache_manager))

        mgr = _make_manager(max_num_tokens=128, tokens_per_block=16)
        mgr.kv_connector_manager = _FakeConnector()
        batch = _make_batch([_FakeRequest(context_chunk_size=16, prompt_len=16)])

        mgr.publish_connector_scheduler_output(batch)

        self.assertEqual(mgr.kv_connector_manager.calls, [(batch, mgr)])


class TestFailUnfittableRequests(unittest.TestCase):
    """PyExecutor._fail_unfittable_requests, including its ADP safety.

    The decision is rank-local but _handle_errors reaches _enqueue_responses,
    whose tp_gather deadlocks unless every rank enters it. So the ranks agree on
    a boolean first and then all of them call _handle_errors -- the same shape
    as _handle_disagg_cache_errors_synced.
    """

    class _FakeDist:
        def __init__(self, world_size=2, peer_says_yes=False):
            self.world_size = world_size
            self.peer_says_yes = peer_says_yes
            self.allgather_calls = 0

        def tp_allgather(self, value):
            self.allgather_calls += 1
            return [value, self.peer_says_yes]

    def _executor(self, *, attention_dp=False, world_size=1, peer_says_yes=False):
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

        executor = PyExecutor.__new__(PyExecutor)
        executor.enable_attention_dp = attention_dp
        executor.dist = self._FakeDist(world_size, peer_says_yes)
        executor.handled = []
        executor._handle_errors = lambda msg, requests=None, charge_budget=True: (
            executor.handled.append((msg, list(requests or []), charge_budget))
        )
        return executor

    def test_no_unfittable_requests_is_a_no_op(self):
        executor = self._executor()
        batch = _make_batch([_FakeRequest(context_chunk_size=32, prompt_len=32)])

        self.assertFalse(executor._fail_unfittable_requests(batch, []))
        self.assertEqual(executor.handled, [])
        self.assertEqual(batch.num_context_requests, 1)

    def test_failing_removes_the_request_from_the_batch(self):
        # It is being terminated, so the forward pass must not still see it.
        executor = self._executor()
        doomed = _FakeRequest(context_chunk_size=4096, prompt_len=4096)
        batch = _make_batch([doomed])

        self.assertTrue(executor._fail_unfittable_requests(batch, [doomed]))

        self.assertEqual(batch.num_context_requests, 0)
        self.assertEqual(len(executor.handled), 1)
        _, requests, charge_budget = executor.handled[0]
        self.assertEqual(requests, [doomed])
        # A request the caller sent that this engine cannot serve is not an
        # engine health problem, so it must not consume the error budget.
        self.assertFalse(charge_budget)

    def test_no_collective_without_attention_dp(self):
        executor = self._executor(attention_dp=False, world_size=8)
        doomed = _FakeRequest(context_chunk_size=4096, prompt_len=4096)

        executor._fail_unfittable_requests(_make_batch([doomed]), [doomed])

        self.assertEqual(executor.dist.allgather_calls, 0)

    def test_all_ranks_enter_handle_errors_when_a_peer_has_one(self):
        # This rank has nothing to fail, but a peer does. It must still call
        # _handle_errors -- with an empty list -- so both ranks enter the
        # tp_gather inside _enqueue_responses.
        executor = self._executor(attention_dp=True, world_size=2, peer_says_yes=True)
        batch = _make_batch([_FakeRequest(context_chunk_size=32, prompt_len=32)])

        self.assertTrue(executor._fail_unfittable_requests(batch, []))

        self.assertEqual(executor.dist.allgather_calls, 1)
        self.assertEqual(len(executor.handled), 1)
        self.assertEqual(executor.handled[0][1], [])
        # Nothing of this rank's own was removed.
        self.assertEqual(batch.num_context_requests, 1)

    def test_ranks_agree_when_nobody_has_one(self):
        executor = self._executor(attention_dp=True, world_size=2, peer_says_yes=False)

        self.assertFalse(executor._fail_unfittable_requests(_make_batch(), []))
        self.assertEqual(executor.dist.allgather_calls, 1)
        self.assertEqual(executor.handled, [])

    def test_single_rank_needs_no_agreement(self):
        # world_size == 1 has no peers, so no collective is entered.
        executor = self._executor(attention_dp=True, world_size=1)
        doomed = _FakeRequest(context_chunk_size=4096, prompt_len=4096)

        self.assertTrue(executor._fail_unfittable_requests(_make_batch([doomed]), [doomed]))
        self.assertEqual(executor.dist.allgather_calls, 0)
        self.assertEqual(len(executor.handled), 1)


if __name__ == "__main__":
    unittest.main()
