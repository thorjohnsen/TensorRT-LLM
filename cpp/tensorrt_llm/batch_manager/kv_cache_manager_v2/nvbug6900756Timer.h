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

// Experiment-only (nvbug 6900756): per-phase wall time inside adjust(). Do not merge.
#pragma once

#include <chrono>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

void nvbug6900756AddPhase(char const* name, double value);

class Nvbug6900756Timer
{
public:
    explicit Nvbug6900756Timer(char const* name)
        : mName(name)
        , mStart(std::chrono::steady_clock::now())
    {
    }

    ~Nvbug6900756Timer()
    {
        nvbug6900756AddPhase(
            mName, std::chrono::duration<double>(std::chrono::steady_clock::now() - mStart).count());
    }

    Nvbug6900756Timer(Nvbug6900756Timer const&) = delete;
    Nvbug6900756Timer& operator=(Nvbug6900756Timer const&) = delete;

private:
    char const* mName;
    std::chrono::steady_clock::time_point mStart;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
