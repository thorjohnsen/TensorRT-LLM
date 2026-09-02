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

// KvCache-level regression coverage for https://github.com/NVIDIA/TensorRT-LLM/issues/17926.
//
// kvCacheManagerV2OrphanBlockTest covers the same issue one layer down, on BlockRadixTree: it
// shows how a page unlink detaches a whole committed chain, and that the tree's own accessors
// now reject an orphan instead of dereferencing its null `prev`. What it cannot cover is the
// caller: KvCache holds a SharedPtr<Block> per committed ordinal and commits onto it on the next
// iteration, and every defect the issue actually turned up lives on that path.
//
// The detach is driven here by hand rather than by exhausting the pool, so the tests do not
// depend on pool sizing or on the order the eviction controller happens to pick. What they need
// is the state the engine reached, which AGENTS.md documents as legal: a live KvCache holding
// blocks that are no longer in the tree.

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/common.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/config.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/lifeCycleRegistry.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace
{
using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

constexpr int kTokensPerBlock = 4;
// Capacity for the whole sequence, in tokens. Generous: nothing here should evict.
constexpr int kCapacity = 64;

// A hybrid (attention + SSM) config, matching the shape of the model in the issue report.
// commit_min_snapshot is mandatory once an SSM layer is present, and is also what puts the
// snapshot page on the commit path in the first place.
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
    ssm.buffers.push_back(BufferConfig{"state", 4096, std::nullopt});
    config.layers.emplace_back(std::move(ssm));

    config.commitMinSnapshot = true;
    config.enableStats = false;
    return config;
}

class KvCacheOrphanTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        mManager = std::make_shared<KvCacheManager>(makeHybridConfig());
        mCache = mManager->createKvCache(ReuseScope{}, TokenSpan{}, /*id=*/1);
        ASSERT_TRUE(mCache->resume(reinterpret_cast<CUstream>(mStream)));
        ASSERT_TRUE(mCache->resize(kCapacity));
    }

    void TearDown() override
    {
        // Every KvCache must close before StorageManager teardown.
        if (mCache)
        {
            mCache->close();
            mCache.reset();
        }
        mManager.reset();
        if (mStream != nullptr)
        {
            cudaStreamDestroy(mStream);
        }
    }

    // One commit() call, as the executor makes it for a chunk of prefill: fresh tokens, no
    // is_end (stop_committing() comes separately).
    void commitTokens(int numTokens, bool isEnd = false)
    {
        std::vector<TokenIdExt> tokens;
        tokens.reserve(numTokens);
        for (int i = 0; i < numTokens; ++i)
            tokens.emplace_back(TokenId{mNextToken++});
        mCache->commit(toSpan(tokens), isEnd);
    }

    // Detach the block the KvCache will commit onto next, while it still holds it.
    //
    // removeSubtree() is one of the two production detach routes: it is what
    // Block::clearStaleBlocksAfterPageUnlink() calls for a full-attention life cycle. The other
    // route -- the SSM ancestor walk the issue's own workload took -- ends in the same
    // NodeBase::detachNext() and leaves the identical state, which
    // kvCacheManagerV2OrphanBlockTest covers directly for both. Taking the subtree route here
    // also keeps the page bookkeeping honest: the SSM walk starts from a page unlink, and in
    // production that unlink comes from ~CommittedPage, i.e. from a page that is already being
    // destroyed. Unlinking a live page instead would strand it in the eviction controller and
    // leak its slot -- an artefact of the simulation, not of the bug.
    void detachCommittedPrefix()
    {
        ASSERT_GT(mCache->numCommittedBlocks(), 0);
        BlockOrdinal const last{mCache->numCommittedBlocks() - 1};
        SharedPtr<Block> const tail = mCache->blocks()[last].treeBlock;
        ASSERT_TRUE(tail);

        // Discarding the returned subtree is the point: the KvCache's own reference keeps every
        // block alive, so the sequence goes on holding a prefix the tree no longer knows about.
        (void) removeSubtree(*tail);
        ASSERT_TRUE(tail->isOrphan());
        ASSERT_EQ(tail->prev, nullptr);
    }

    cudaStream_t mStream = nullptr;
    std::shared_ptr<KvCacheManager> mManager;
    std::shared_ptr<KvCache> mCache;
    int mNextToken = 0;
};

// The reported crash, at the layer it happened on. Committing onto a prefix that was detached
// since the previous iteration used to dereference the null `prev` inside
// Block::tokensPerBlock() and take the engine down with a SIGSEGV on every rank. The sequence
// must instead stop contributing to the reuse tree and keep running.
TEST_F(KvCacheOrphanTest, CommitAfterPrefixDetachStopsContributingInsteadOfCrashing)
{
    commitTokens(2 * kTokensPerBlock);
    ASSERT_EQ(mCache->numCommittedBlocks(), 2);
    ASSERT_EQ(mCache->commitState(), KvCache::CommitState::ALLOWED);

    detachCommittedPrefix();

    EXPECT_NO_THROW(commitTokens(kTokensPerBlock));
    EXPECT_EQ(mCache->commitState(), KvCache::CommitState::VIRTUAL_STOP);
    // Still tracking the sequence: the tokens are committed, they are just not published.
    EXPECT_EQ(mCache->numCommittedTokens(), 3 * kTokensPerBlock);
    // ... and nothing new reached the tree, since there is no attached prefix to hang it off.
    EXPECT_EQ(mCache->numCommittedBlocks(), 2);
}

// VIRTUAL_STOP means "stop contributing to the reuse tree", not "stop tracking the sequence".
// commit() used to return before bumping history_length, so history froze while the committed
// token count kept growing, and the next chunk but one tripped commit_min_snapshot's
// "start or end at history_length" assertion -- turning the crash into a failed request.
TEST_F(KvCacheOrphanTest, HistoryLengthKeepsAdvancingAfterPrefixDetach)
{
    commitTokens(2 * kTokensPerBlock);
    detachCommittedPrefix();

    for (int chunk = 0; chunk < 4; ++chunk)
    {
        ASSERT_NO_THROW(commitTokens(kTokensPerBlock)) << "chunk " << chunk;
        EXPECT_EQ(mCache->historyLength(), mCache->numCommittedTokens());
        EXPECT_EQ(mCache->commitState(), KvCache::CommitState::VIRTUAL_STOP);
    }
}

// The partial-snapshot path. Any prompt whose final context chunk does not land on a block
// boundary reaches _snapshotPartialBlockToTree() instead of _commitBlock(), and that site
// commits onto the same held prefix. The issue's own repro never hit it -- 256000 tokens is
// block-aligned -- so it needs its own case.
TEST_F(KvCacheOrphanTest, PartialSnapshotAfterPrefixDetachDoesNotThrow)
{
    commitTokens(2 * kTokensPerBlock);
    detachCommittedPrefix();

    EXPECT_NO_THROW(commitTokens(kTokensPerBlock / 2));
    EXPECT_EQ(mCache->commitState(), KvCache::CommitState::VIRTUAL_STOP);
    EXPECT_EQ(mCache->numCommittedTokens(), 2 * kTokensPerBlock + kTokensPerBlock / 2);
    EXPECT_EQ(mCache->historyLength(), mCache->numCommittedTokens());
}

// The end of the sequence, when the detach lands on the last commit rather than a middle one.
// stop_committing() commits the trailing partial block with is_last=true, which is the one
// _commitBlock() call whose orphan exit must still finish the sequence off: leaving it in
// VIRTUAL_STOP skips _onStopCommitting() and violates stopCommitting()'s own postcondition.
TEST_F(KvCacheOrphanTest, StopCommittingAfterPrefixDetachReachesUserStop)
{
    commitTokens(2 * kTokensPerBlock);
    commitTokens(kTokensPerBlock / 2); // trailing partial block, published while still attached
    ASSERT_EQ(mCache->commitState(), KvCache::CommitState::ALLOWED);

    detachCommittedPrefix();

    EXPECT_NO_THROW(mCache->stopCommitting());
    EXPECT_EQ(mCache->commitState(), KvCache::CommitState::USER_STOP);
    EXPECT_NO_THROW(mCache->close());
}

// Control: with no detach, the guards must not cost the ordinary path anything. The sequence
// commits every block and the prefix is visible to the next request.
TEST_F(KvCacheOrphanTest, UnaffectedSequenceCommitsAndPublishesForReuse)
{
    std::vector<TokenIdExt> tokens;
    for (int i = 0; i < 3 * kTokensPerBlock; ++i)
        tokens.emplace_back(TokenId{i});
    mNextToken = 3 * kTokensPerBlock;

    ASSERT_NO_THROW(mCache->commit(toSpan(tokens)));
    EXPECT_EQ(mCache->commitState(), KvCache::CommitState::ALLOWED);
    EXPECT_EQ(mCache->numCommittedBlocks(), 3);
    EXPECT_EQ(mCache->historyLength(), 3 * kTokensPerBlock);

    mCache->stopCommitting();
    EXPECT_EQ(mCache->commitState(), KvCache::CommitState::USER_STOP);

    EXPECT_GT(mManager->probeReuse(ReuseScope{}, toSpan(tokens), /*knownNoDigest=*/true), 0);
}

} // namespace
