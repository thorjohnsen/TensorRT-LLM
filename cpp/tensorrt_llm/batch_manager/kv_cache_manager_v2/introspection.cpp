/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kv_cache_manager_v2/introspection.h"

#include <numeric>

#include "kv_cache_manager_v2/blockRadixTree.h"

#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/page.h"
#include "kv_cache_manager_v2/storageManager.h"

#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

bool allBlockPagesDroppable(Block const& block)
{
    for (auto const* page : block.storage)
    {
        if (page != nullptr && page->status() != PageStatus::DROPPABLE)
        {
            return false;
        }
    }

    for (auto const& [_, child] : block.next)
    {
        if (!allBlockPagesDroppable(*child))
        {
            return false;
        }
    }
    return true;
}

} // namespace

KvCacheIntrospection::ActivePageStats KvCacheIntrospection::activePageStats(KvCache const& kvCache)
{
    auto& storageMgr = kvCache.manager().storage();
    CacheLevel const numTiers = storageMgr.numCacheLevels();
    TypedVec<CacheLevel, int> counts(numTiers, 0);
    TypedVec<CacheLevel, int> unscheduledEvictable(numTiers, 0);

    for (auto const& activePage : kvCache._activePages())
    {
        auto page = kvCache._page(activePage.ordinal, activePage.beamIdx, activePage.lcId);
        if (!page)
        {
            continue;
        }

        CacheLevel const level = page->cacheLevel;
        counts.at(level) += 1;
        if (storageMgr.isEvictable(*page) && !page->scheduledForEviction())
        {
            unscheduledEvictable.at(level) += 1;
        }
    }

    return {std::move(counts), std::move(unscheduledEvictable)};
}

std::optional<bool> KvCacheIntrospection::committedPageIsLinked(KvCache const& kvCache, int ordinal, int lcId)
{
    auto page = kvCache._page(BlockOrdinal{ordinal}, kDefaultBeamIndex, LifeCycleId{lcId});
    if (!page)
    {
        return std::nullopt;
    }
    auto committed = dynamicPointerCast<CommittedPage>(page);
    if (!committed)
    {
        return std::nullopt;
    }
    return committed->block != nullptr;
}

TypedVec<PoolGroupIndex, SlotCount> KvCacheIntrospection::computeSlotsForBatch(KvCacheManager& manager,
    BatchDesc const& batch, int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse)
{
    return manager.storage().computePoolGroupSlotsForBatch(batch, tokensPerBlock, swaScratchReuse);
}

bool KvCacheIntrospection::allTreePagesDroppable(KvCacheManager& manager)
{
    for (auto const& [_, root] : manager.radixTree().roots())
    {
        for (auto const& [__, block] : root->next)
        {
            if (!allBlockPagesDroppable(*block))
            {
                return false;
            }
        }
    }
    return true;
}

void KvCacheIntrospection::setNumSampledKvCaches(KvCacheManager& manager, int value)
{
    manager.mNumSampledKvCaches = value;
}

void KvCacheIntrospection::setLastAdjustmentTime(KvCacheManager& manager, double value)
{
    manager.mLastAdjustmentTime = value;
}

void KvCacheIntrospection::setTargetRatioListGpu(KvCacheManager& manager, TypedVec<PoolGroupIndex, float> value)
{
    manager.mTargetRatioListHot = std::move(value);
}

extern std::vector<double> gNvbug6900756AdjustLevelSeconds;

std::vector<double> KvCacheIntrospection::lastAdjustLevelSeconds()
{
    return gNvbug6900756AdjustLevelSeconds;
}

std::vector<std::tuple<std::vector<float>, std::vector<float>, bool>> KvCacheIntrospection::levelRebalanceState(
    KvCacheManager const& manager)
{
    auto const apiLock = manager.lockShared();
    std::vector<std::tuple<std::vector<float>, std::vector<float>, bool>> out;
    CacheLevel const numLevels = manager.mStorage->numCacheLevels();
    for (CacheLevel level{0}; level < numLevels; ++level)
    {
        auto current = (level == kHotLevel) ? manager._currentHotRatio() : manager._currentColdRatios();
        auto const& target = manager._getTargetRatioList(level);
        out.emplace_back(std::vector<float>(current.raw().begin(), current.raw().end()),
            std::vector<float>(target.raw().begin(), target.raw().end()), manager._needAdjustment(level));
    }
    return out;
}

std::vector<std::tuple<std::vector<size_t>, size_t>> KvCacheIntrospection::levelPoolGroupBytes(
    KvCacheManager const& manager)
{
    auto const apiLock = manager.lockShared();
    std::vector<std::tuple<std::vector<size_t>, size_t>> out;
    auto const& levels = manager.mStorage->mLevels;
    CacheLevel const numLevels = manager.mStorage->numCacheLevels();
    for (CacheLevel level{0}; level < numLevels; ++level)
    {
        auto const& storage = *levels.at(level).storage;
        auto const counts = storage.slotCountList();
        auto const sizes = storage.slotSizeLists();
        std::vector<size_t> bytes;
        for (PoolGroupIndex pg{0}; pg < counts.size(); ++pg)
        {
            size_t const slotBytes = std::accumulate(sizes[pg].begin(), sizes[pg].end(), size_t{0});
            bytes.push_back(slotBytes * slotCountToSizeT(counts[pg]));
        }
        out.emplace_back(std::move(bytes), storage.totalQuota());
    }
    return out;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
