/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/batch_manager/radixTree.h"
#include "tensorrt_llm/batch_manager/blockKey.h"

//
// This file implements a unified radix search tree for all the WindowBlockManager instances.
// It is used to retrieve KV cache blocks with reusable content during context phase.
//

// forward declarations
namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class KVCacheBlock;
}

namespace tensorrt_llm::batch_manager::radix_tree
{
class UnifiedSearchTree : public RadixTree<BlockKey,BlockKeyHasher,int,std::hash<int>,std::shared_ptr<KVCacheBlock>,true>
{
    public:
        UnifiedSearchTree() = default;

        // TODO : Implement utility functions to ease integration with KVCacheManager
};
} // namespace tensorrt_llm::batch_manager::radix_tree
