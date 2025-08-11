# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Callable, Optional

import psutil
import torch
from transformers import PreTrainedTokenizer
from collections import defaultdict  # Added to aggregate extra reward information

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.0):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        future = loop.run_in_executor(executor, partial(evaluation_func, task, completion, reference, task_extra_info))
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[Timeout] Task timeout: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"[Error] Task failed: {e}, completion: {completion[:80]}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(
    evaluation_func, completions, references, tasks, extra_info=None, num_processes=64
):
    """Run *evaluation_func* for each sample in parallel threads.

    Returns
    -------
    scores : List[float]
        Numeric reward for each sample.
    reward_extra_info : defaultdict(list)
        Aggregated extra information (if ``evaluation_func`` returns dict).
    """

    if extra_info is None:
        extra_info = [None] * len(tasks)

    scores = []
    reward_extra_info = defaultdict(list)

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the
        # exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            # Create tasks for all rows
            tasks_async = [
                single_compute_score(evaluation_func, c, r, t, ei, executor, timeout=300.0)
                for c, r, t, ei in zip(completions, references, tasks, extra_info)
            ]
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except Exception as e:
            print(f"[Exception] async gather failed: {e}")
            raise
        finally:
            terminated_count = 0
            # In ThreadPoolExecutor there is no `_processes` attribute, whereas ProcessPoolExecutor
            # does maintain this for its worker processes. We guard against accessing this
            # private attribute so the cleanup works for either executor type.
            if hasattr(executor, "_processes"):
                for pid, proc in executor._processes.items():
                    try:
                        p = psutil.Process(pid)
                        p.terminate()
                        try:
                            p.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            p.kill()
                        terminated_count += 1
                    except Exception:
                        pass
                print(f"[Shutdown] {terminated_count} subprocess(es) terminated.")
            else:
                # For ThreadPoolExecutor no extra cleanup is required; the context manager takes
                # care of shutting down the thread workers. We emit a log line for consistency.
                print("[Shutdown] ThreadPoolExecutor terminated.")

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks):
        # Default numeric reward
        numeric_reward: float = 0.0

        if isinstance(result, Exception) or result is None:
            numeric_reward = 0.0
        elif isinstance(result, (int, float, bool)):
            numeric_reward = float(result)
        elif isinstance(result, dict):
            # If compute_score returns a dict, extract numeric score and aggregate extras
            numeric_reward = float(result.get("score", 0.0))
            for key, value in result.items():
                reward_extra_info[key].append(value)
        else:
            # Fallback – try to treat result like a sequence/tuple
            try:
                numeric_reward = float(result[0])
            except Exception:
                numeric_reward = 0.0

        scores.append(numeric_reward)

    return scores, reward_extra_info


def run_reward_scoring(
    evaluation_func, completions, references, tasks, extra_info=None, num_processes=64
):
    """Wrapper to run *parallel_compute_score_async* inside a fresh event loop."""

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            parallel_compute_score_async(
                evaluation_func, completions, references, tasks, extra_info, num_processes
            )
        )
    finally:
        loop.close()


@register("prime")
class PrimeRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def verify(self, data):
        """
        verify the batch and save as ``acc`` tensor
        """
        # batched scoring
        prompt_ids = data.batch["prompts"]

        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", None)

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            scores, reward_extra_info = run_reward_scoring(
                self.compute_score,
                completions=sequences_str,
                references=ground_truth,
                tasks=data_sources,
                extra_info=extra_info,
                num_processes=64,
            )
        except asyncio.TimeoutError:
            print("[Timeout] Global reward scoring timed out. Setting all as 0.")
            scores = [0.0 for _ in range(len(sequences_str))]
        except Exception as e:
            print(f"[Error] Unexpected error during scoring. Setting all as 0. {e}")
            scores = [0.0 for _ in range(len(sequences_str))]
        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores, reward_extra_info

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute reward for each sample. If ``compute_score`` returns a dict, the extra fields
        are collected in ``reward_extra_info`` similarly to :class:`NaiveRewardManager`."""

        # If rm_scores already exist, return them directly.
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        # Run verification (asynchronous scoring)
        scores, reward_extra_info = self.verify(data)

        already_print_data_sources = {}

        # Prepare batch-level tensors/metadata
        response_ids_batch = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(response_ids_batch, skip_special_tokens=True)

        prompt_ids_batch = data.batch["prompts"]
        prompt_length = prompt_ids_batch.shape[-1]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)

        for i in range(len(data)):
            data_item = data[i]

            # Decode prompt for debugging
            valid_prompt_len = int(data_item.batch["attention_mask"][:prompt_length].sum().item())
            prompt_str = self.tokenizer.decode(
                data_item.batch["prompts"][-valid_prompt_len:], skip_special_tokens=True
            )

            response_str = sequences_str[i]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            reward_val = scores[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = float(reward_val)

            # Controlled debug printing
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                if reward_extra_info:
                    for key, value_list in reward_extra_info.items():
                        if len(value_list) > i:
                            print(f"[{key}]", value_list[i])
                else:
                    print("[score]", reward_val)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
