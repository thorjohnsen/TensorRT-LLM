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

// Regression coverage for https://github.com/NVIDIA/TensorRT-LLM/issues/17926.
//
// A live KvCache holds a SharedPtr<Block> per committed ordinal and passes the previous one to
// addOrGetExistingBlock() on the next iteration. An eviction running between those two iterations
// can detach that block from the radix tree -- which AGENTS.md documents as a legal state ("Orphan
// blocks may retain pages while a live KvCache still references them"). Before the fix, the commit
// path never tested for it, and Block::tokensPerBlock() dereferenced the null `prev` that
// NodeBase::detachNext() deliberately installs, killing the engine with a SIGSEGV.
//
// These tests drive BlockRadixTree directly: no StorageManager, no CUDA context, no model weights,
// no GPU. They reconstruct what KvCache does (hold a block across iterations, commit onto it) and
// what the eviction path does (Block::clearStaleBlocksAfterPageUnlink).

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/common.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/config.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/lifeCycleRegistry.h"
#include "tensorrt_llm/common/tllmException.h"

#include <gtest/gtest.h>

#include <optional>
#include <utility>
#include <vector>

namespace
{
using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;
namespace tc = tensorrt_llm::common;

constexpr int kTokensPerBlock = 4;

// A hybrid (attention + SSM) config, matching the shape of the model in the issue report.
KVCacheManagerConfig makeHybridConfig()
{
    KVCacheManagerConfig config;
    config.tokensPerBlock = kTokensPerBlock;
    config.cacheTiers.emplace_back(GpuCacheTierConfig{4 << 20});

    AttentionLayerConfig attn;
    attn.layerId = 0;
    attn.buffers.push_back(BufferConfig{"key", 4096, std::nullopt});
    config.layers.emplace_back(std::move(attn));

    SsmLayerConfig ssm;
    ssm.layerId = 1;
    ssm.buffers.push_back(BufferConfig{"key", 4096, std::nullopt});
    config.layers.emplace_back(std::move(ssm));

    return config;
}

// Holds the tree and the blocks a hypothetical live KvCache would be holding.
class OrphanBlockTest : public ::testing::Test
{
protected:
    OrphanBlockTest()
        : mConfig(makeHybridConfig())
        , mLifeCycles(mConfig)
        , mTree(mLifeCycles, kTokensPerBlock)
    {
    }

    // Build a linear chain of `numBlocks` full blocks off the root, and keep a SharedPtr to each
    // one -- exactly what KvCache::mBlocks[].treeBlock does for a committed sequence.
    void buildChain(int numBlocks)
    {
        RootBlock& root = mTree.addOrGetExisting(ReuseScope{});
        NodeBase* prev = &root;
        for (int b = 0; b < numBlocks; ++b)
        {
            auto block = addOrGetExistingBlock(prev, makeTokens(), /*knownNoDigest=*/true, /*isNew=*/nullptr);
            prev = block.get();
            mChain.push_back(std::move(block));
        }
    }

    // Successive calls hand out disjoint token blocks, so every block gets a distinct key.
    std::vector<TokenIdExt> makeTokens()
    {
        std::vector<TokenIdExt> tokens;
        tokens.reserve(kTokensPerBlock);
        for (int i = 0; i < kTokensPerBlock; ++i)
            tokens.emplace_back(TokenId{mNextToken++});
        return tokens;
    }

    LifeCycleId attnLifeCycleId() const
    {
        return mLifeCycles.getId(AttnLifeCycle{std::nullopt, 0});
    }

    LifeCycleId ssmLifeCycleId() const
    {
        return mLifeCycles.getId(SsmLifeCycle{});
    }

    KVCacheManagerConfig mConfig;
    LifeCycleRegistry mLifeCycles;
    BlockRadixTree mTree;
    std::vector<SharedPtr<Block>> mChain;
    int mNextToken = 0;
};

// --- The eviction side: how blocks a live KvCache holds become orphans -----------------------

// Route A -- full attention. removeSubtree() plus the ancestor walk. One page unlink detaches the
// *entire* chain, not just the tail: each parent becomes childless in turn and has a null page slot
// for the life cycle, so the walk runs all the way to the root. This is why the failure in the
// issue looks abrupt rather than gradual.
TEST_F(OrphanBlockTest, AttentionPageUnlinkOrphansWholeChain)
{
    buildChain(3);
    for (auto const& block : mChain)
        EXPECT_FALSE(block->isOrphan());

    auto const lcIdx = attnLifeCycleId();
    auto detached = Block::clearStaleBlocksAfterPageUnlink(*mChain.back(), lcIdx, mLifeCycles.getLifeCycle(lcIdx));

    EXPECT_FALSE(detached.empty());
    for (auto const& block : mChain)
    {
        EXPECT_TRUE(block->isOrphan());
        EXPECT_EQ(block->prev, nullptr);
    }
}

// Route B -- SSM. No removeSubtree (subtree eviction is attention-only), just the ancestor walk.
// Under commit_min_snapshot the non-boundary blocks have null SSM slots, so the walk is unimpeded
// and reaches the root. **This is the route the reported bug takes.**
TEST_F(OrphanBlockTest, SsmSnapshotUnlinkOrphansWholeChain)
{
    buildChain(3);

    auto const lcIdx = ssmLifeCycleId();
    auto detached = Block::clearStaleBlocksAfterPageUnlink(*mChain.back(), lcIdx, mLifeCycles.getLifeCycle(lcIdx));

    EXPECT_FALSE(detached.empty());
    for (auto const& block : mChain)
    {
        EXPECT_TRUE(block->isOrphan());
        EXPECT_EQ(block->prev, nullptr);
    }
}

// The hazard is not specific to the eviction path -- any detach produces it. A block a caller still
// holds a SharedPtr to stays alive (use_count > 0) but leaves the tree.
TEST_F(OrphanBlockTest, RemoveSubtreeOrphansHeldBlock)
{
    buildChain(3);

    auto detachedRoot = removeSubtree(*mChain[1]);

    EXPECT_EQ(detachedRoot.get(), mChain[1].get());
    EXPECT_FALSE(mChain[0]->isOrphan()); // above the cut, still attached
    EXPECT_TRUE(mChain[1]->isOrphan());
    EXPECT_TRUE(mChain[2]->isOrphan());
    // Alive but detached: the crash was a state violation, not a lifetime violation.
    EXPECT_GT(mChain[2].useCount(), 0);
}

// --- The commit side: the two sites that used to dereference the null prev -------------------

// The reported crash. Committing onto a `prev` that was orphaned since the previous iteration used
// to segfault inside Block::tokensPerBlock(); it must now raise a catchable exception instead.
TEST_F(OrphanBlockTest, CommitOntoOrphanPrevIsGuarded)
{
    buildChain(2);

    auto const lcIdx = ssmLifeCycleId();
    Block::clearStaleBlocksAfterPageUnlink(*mChain.back(), lcIdx, mLifeCycles.getLifeCycle(lcIdx));
    ASSERT_TRUE(mChain.back()->isOrphan());

    EXPECT_THROW(addOrGetExistingBlock(mChain.back().get(), makeTokens(), /*knownNoDigest=*/true, /*isNew=*/nullptr),
        tc::TllmException);
}

// The second unguarded site: KvCache::_commitBlock's `newBlock->isFull()`, where newBlock can be a
// sibling handed back by UselessBlockError and is not guaranteed to be attached. isFull() calls
// tokensPerBlock(), so it inherits the same precondition -- and, since tokensPerBlock() is no
// longer noexcept, the throw propagates rather than reaching std::terminate.
TEST_F(OrphanBlockTest, IsFullOnOrphanBlockIsGuarded)
{
    buildChain(2);

    auto const lcIdx = ssmLifeCycleId();
    Block::clearStaleBlocksAfterPageUnlink(*mChain.back(), lcIdx, mLifeCycles.getLifeCycle(lcIdx));
    ASSERT_TRUE(mChain.back()->isOrphan());

    EXPECT_THROW((void) mChain.back()->tokensPerBlock(), tc::TllmException);
    EXPECT_THROW((void) mChain.back()->isFull(), tc::TllmException);
}

// An attached block is unaffected by the new precondition -- the guard must not cost the happy path
// its behaviour.
TEST_F(OrphanBlockTest, AttachedBlockIsUnaffectedByTheGuard)
{
    buildChain(2);

    EXPECT_EQ(mChain[0]->tokensPerBlock(), kTokensPerBlock);
    EXPECT_EQ(mChain[1]->tokensPerBlock(), kTokensPerBlock);
    EXPECT_TRUE(mChain[1]->isFull());
    EXPECT_NO_THROW(
        (void) addOrGetExistingBlock(mChain.back().get(), makeTokens(), /*knownNoDigest=*/true, /*isNew=*/nullptr));
}

} // namespace
