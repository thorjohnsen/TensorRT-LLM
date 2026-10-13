# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prototype check (nvbug 6900756): host-tier pools in one pinned arena survive relayout.

Run with TLLM_KV_CACHE_MANAGER_V2_HOST_ARENA=1 (optionally ..._HOST_ARENA_VERIFY=1); skipped otherwise.
"""

import os
import unittest

import test_kv_cache_manager_v2 as base
from test_kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BatchDesc,
    BufferConfig,
    CacheLevel,
    FakeEngine,
    GpuCacheTierConfig,
    HostCacheTierConfig,
    KVCacheDesc,
    KVCacheManager,
    KVCacheManagerConfig,
    LayerId,
    PoolRebalanceConfig,
    Role,
)

from tensorrt_llm.runtime.kv_cache_manager_v2 import _cpp_introspection

HOST_LEVEL = CacheLevel(1)


@unittest.skipUnless(
    os.environ.get("TLLM_KV_CACHE_MANAGER_V2_HOST_ARENA") == "1", "host arena prototype is off"
)
class TestHostArenaRelayout(base.TestPoolRebalance):
    def prepare_gpu_and_host(self, gpu_quota: int, host_quota: int) -> None:
        self.cfg = KVCacheManagerConfig(
            tokens_per_block=self._TOKENS_PER_BLOCK,
            cache_tiers=[GpuCacheTierConfig(quota=gpu_quota), HostCacheTierConfig(quota=host_quota)],
            layers=[
                AttentionLayerConfig(
                    layer_id=LayerId(0),
                    buffers=[
                        BufferConfig(role=Role.KEY, size=8192),
                        BufferConfig(role=Role.VALUE, size=8192),
                    ],
                    sliding_window_size=128,
                    num_sink_tokens=0,
                ),
                AttentionLayerConfig(
                    layer_id=LayerId(1),
                    buffers=[
                        BufferConfig(role=Role.KEY, size=2048),
                        BufferConfig(role=Role.VALUE, size=2048),
                    ],
                    sliding_window_size=None,
                ),
            ],
            typical_step=BatchDesc(kv_caches=[KVCacheDesc(capacity=160, history_length=0)]),
            constraints=[BatchDesc(kv_caches=[KVCacheDesc(capacity=160, history_length=0)])],
            pool_rebalance=PoolRebalanceConfig(),
        )
        self.engine = FakeEngine(self.cfg)
        self.manager = KVCacheManager(self.cfg)

    def _host_used(self) -> list[int]:
        return [s.total - s.free for s in self.manager.get_storage_statistics(HOST_LEVEL)]

    def _host_state(self):
        return _cpp_introspection.level_rebalance_state(self.manager)[1]

    def _relayout_host(self, target: list[float]) -> dict:
        """Run adjust() that only moves the host tier toward target; return its host-level phases."""
        gpu_now = list(_cpp_introspection.current_gpu_ratio(self.manager))
        _cpp_introspection.set_target_ratio_list_gpu(self.manager, gpu_now)
        _cpp_introspection.set_target_ratio_list_cold(self.manager, target)
        _cpp_introspection.set_num_sampled_kv_caches(self.manager, 1 << 20)
        _cpp_introspection.set_last_adjustment_time(self.manager, 0.0)
        self.assertTrue(
            self.manager.need_adjustment,
            f"host target skew did not trigger adjustment: state {self._host_state()}",
        )
        self.manager.adjust()
        cur, _tgt, need = self._host_state()
        self.assertFalse(need, f"host tier still needs adjustment after adjust(): {cur} vs {target}")
        return dict(_cpp_introspection.last_adjust_phases()[1])

    def test_host_blocks_survive_relayout(self) -> None:
        self.prepare_gpu_and_host(gpu_quota=4 << 20, host_quota=32 << 20)
        prompts = [self._run_sequence() for _ in range(300)]
        used = self._host_used()
        # Pool group 0 starts at offset 0 and never moves, so live data must sit in a later group for the
        # relayout to have anything to move.
        self.assertGreater(used[-1], 0, f"expected committed pages in the last host pool group: {used}")

        before = list(self._host_state()[0])
        # Move most of the host tier to the other pool group, far outside the tuner's tolerance.
        skewed = [0.15, 0.85] if before[0] > 0.5 else [0.85, 0.15]
        phases = self._relayout_host(skewed)
        self.assertGreater(phases.get("n.movedGiB", 0.0), 0.0, f"relayout moved nothing: {phases}")

        verified, host_tokens = self._replay_reusable(prompts)
        self.assertGreaterEqual(verified, 5, "too few prompts still reusable after the relayout")
        self.assertGreater(host_tokens, 0, "no replayed tokens came from the host tier; the check is vacuous")

        # Swing back, the way the tuner oscillates, and check again.
        phases = self._relayout_host(before)
        self.assertGreater(phases.get("n.movedGiB", 0.0), 0.0, f"second relayout moved nothing: {phases}")
        verified, host_tokens = self._replay_reusable(prompts)
        self.assertGreaterEqual(verified, 5, "too few prompts still reusable after the second relayout")
        self.assertGreater(host_tokens, 0, "no replayed tokens came from the host tier after the second relayout")

    def _replay_reusable(self, prompts, limit: int = 40) -> tuple[int, int]:
        """Replay prompts that still hit the reuse tree; return (#verified, #tokens reused from the host tier).

        The fake engine's reference check reads back the KV of every reused block, so a block whose bytes were
        moved to the wrong place, or whose pool now points at the wrong region, fails here.
        """
        verified = host_tokens = 0
        for prompt in prompts:
            if verified >= limit:
                break
            # Replays commit and evict, so probe each prompt right before using it.
            if self.manager.probe_reuse(input_tokens=prompt) == 0:
                continue
            probe = self.manager.create_kv_cache(input_tokens=prompt)
            probe.commit_pending_stats()
            by_level = list(self.manager.get_and_reset_iteration_cached_tokens_by_level())
            probe.close()
            host_tokens += by_level[1] if len(by_level) > 1 else 0
            self._run_sequence(prompt=prompt, expect_reuse=True)
            verified += 1
        return verified, host_tokens

if __name__ == "__main__":
    unittest.main()
