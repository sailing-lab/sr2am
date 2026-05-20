"""
Base agent classes and orchestration loop for all process_questions scripts.

Provides:
- Agent: base class for tool-using agents (chat completions + Responses API)
- generate_response_agent(): core agent loop that runs an Agent for one question
- process_questions(): orchestration loop over a dataset
- judge_answer(), compute_single_reward(): evaluation helpers
- load_questions(), process_single_question(): I/O helpers
"""

import asyncio
from openai import AsyncOpenAI
import json
from json.decoder import JSONDecodeError
import os
import time
from typing import Callable, Tuple, List
from tqdm.asyncio import tqdm_asyncio
from tools.schemas import OpenAIFunctionToolSchema
import wandb
import pandas as pd
import re
from evaluation.reward_score import default_compute_score
import random


class Agent:
    """Base agent that drives a multi-turn tool-use conversation."""

    def __init__(self, client: AsyncOpenAI, model: str, tool_instances: dict,
                 system_prompt: str, use_response_api: bool = False, **kwargs):
        self.client = client
        self.tool_instances = tool_instances
        self.tool_schemas = [tool.get_openai_tool_schema() for tool in tool_instances.values()]
        self.system_prompt = system_prompt
        self.model = model
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]
        self.use_response_api = use_response_api

    def add_message(self, content: dict):
        self.messages.append(content)

    @staticmethod
    def _sanitize_response_input_item(message: dict) -> dict:
        return {k: v for k, v in message.items() if k not in {"status", "usage"}}

    async def get_response(self):
        if self.use_response_api:
            tools = []
            for tool in self.tool_schemas:
                tool_dump = {
                    "type": tool.type,
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters.model_dump(),
                    "strict": tool.function.strict,
                }
                tools.append(tool_dump)

            input_messages = []
            for message in self.messages[1:]:
                input_messages.append(self._sanitize_response_input_item(message))

            response = await self.client.responses.create(
                model=self.model,
                instructions=self.system_prompt,
                input=input_messages,
                tools=tools,
                max_output_tokens=32_768,
            )

            response_dict = response.model_dump()
            if response.usage is not None:
                outputs = response_dict.get("output")
                if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], dict):
                    outputs[0]["usage"] = response.usage.model_dump()

            return response_dict
        else:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=[tool.model_dump() for tool in self.tool_schemas],
                max_completion_tokens=32_768,
            )

            message = response.choices[0].message.model_dump()
            usage = response.usage.model_dump()
            message['usage'] = usage

            return message


async def generate_response_agent(client: AsyncOpenAI, dataset: str, question: str, model: str, tool_instances: dict,
                                  agent_cls: Agent = Agent, custom_system_prompt_fn: Callable = lambda x: x,
                                  use_response_api: bool = False, max_turns: int = 30,
                                  response_timeout: int = 180, tool_execution_timeout: int = 180, **kwargs) -> Tuple[List[dict], str]:
    """Generate response and reasoning using the model via chat completion API."""
    instance_id = None
    for tool_name, tool in tool_instances.items():
        instance_id = await tool.create(instance_id=instance_id)

    agent = agent_cls(client, model, tool_instances, custom_system_prompt_fn(dataset),
                      use_response_api=use_response_api, **kwargs)
    agent.add_message({"role": "user", "content": question})

    cumulative_reward = 0
    try:
        for i in range(max_turns):
            response = await asyncio.wait_for(agent.get_response(), timeout=response_timeout)
            if use_response_api:
                if outputs := response.get("output", None):
                    agent.messages += outputs
                    completed = False
                    for output in outputs:
                        if (
                            output.get("type") == "message"
                            and output.get("role") == "assistant"
                            and output.get("status") == "completed"
                        ):
                            completed = True
                            break

                        output_type = output.get("type")
                        if output_type != "function_call":
                            continue

                        call_id: str = output.get("call_id") or output.get("id")
                        fn_name: str = output.get("name")
                        try:
                            fn_args: dict = json.loads(output.get("arguments", "{}"))
                        except JSONDecodeError as e:
                            if fn_name in tool_instances and len(tool_instances[fn_name].tool_schema.function.parameters.properties) == 1:
                                k, = tool_instances[fn_name].tool_schema.function.parameters.properties.keys()
                                fn_args = {k: output.get("arguments")}
                            else:
                                raise e

                        fn_res, fn_rw, fn_metrics = await asyncio.wait_for(
                            tool_instances[fn_name].execute(instance_id=instance_id, parameters=fn_args),
                            timeout=tool_execution_timeout,
                        )
                        cumulative_reward += fn_rw

                        agent.add_message({
                            "type": "function_call_output",
                            "output": fn_res,
                            "call_id": call_id,
                        })
                    if completed:
                        break
                else:
                    break
            else:
                agent.add_message(response)

                if tool_calls := response.get("tool_calls", None):
                    for tool_call in tool_calls:
                        call_id: str = tool_call["id"]
                        if fn_call := tool_call.get("function"):
                            fn_name: str = fn_call["name"]
                            fn_args: dict = json.loads(fn_call["arguments"])

                            fn_res, fn_rw, fn_metrics = await asyncio.wait_for(
                                tool_instances[fn_name].execute(instance_id=instance_id, parameters=fn_args),
                                timeout=tool_execution_timeout,
                            )
                            cumulative_reward += fn_rw

                            tool_message = {
                                "role": "tool",
                                "content": fn_res,
                                "reward": fn_rw,
                                "metrics": fn_metrics,
                                "tool_call_id": call_id,
                            }
                            agent.add_message(tool_message)
                else:
                    break

            if cumulative_reward <= -0.5:
                break
    except Exception as e:
        e.partial_messages = agent.messages
        raise

    for tool_name, tool in tool_instances.items():
        await tool.release(instance_id=instance_id)

    assistant_messages = [message for message in agent.messages if message.get("role") == "assistant"]
    if len(assistant_messages) == 0:
        answer = None
    else:
        if use_response_api:
            answer = assistant_messages[-1].get("content", "")[0].get("text", "")
        else:
            answer = assistant_messages[-1].get("content", "")

    return agent.messages, answer


def setup_client(api_key: str = None, base_url: str = "https://api.openai.com/v1",
                  timeout: int = 600) -> AsyncOpenAI:
    """Initialize an AsyncOpenAI client."""
    return AsyncOpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=base_url,
        timeout=timeout,
    )


def setup_client_local_host(base_url: str = "http://localhost:8000/v1",
                             timeout: int = 600) -> AsyncOpenAI:
    """Initialize an AsyncOpenAI client for a local vLLM server."""
    return AsyncOpenAI(
        api_key="EMPTY",
        base_url=base_url,
        timeout=timeout,
    )


def load_questions(file_path: str) -> List[str]:
    """Load questions from a JSON or JSONL file."""
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown file type: {file_path}")


async def compute_single_reward(data_source, response, ground_truth, extra_info):
    """Compute reward for one (response, ground-truth) pair."""
    try:
        result = await default_compute_score(
            data_source=data_source,
            solution_str=response,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        if isinstance(result, dict):
            detailed = result
            score = detailed.get("score", 0.0)
        else:
            detailed = {"score": float(result)}
            score = float(result)
        detailed["score"] = score
        return detailed
    except Exception as e:
        return {"score": 0.0, "error": str(e)}


async def judge_answer(row: dict, answer: str, remove_tags: bool = False, **kwargs):
    """Judge an answer against the ground truth."""
    if 'answer' in row and not pd.isna(row['answer']):
        ground_truth = row['answer']
    else:
        ground_truth = row['reward_model']['ground_truth']

    if 'extra_info' in row and not pd.isna(row['extra_info']):
        extra_info = row['extra_info']
        if 'question' not in extra_info:
            extra_info['question'] = row['question']
    else:
        extra_info = {'question': row['question']}

    if "data_source" not in extra_info:
        extra_info['data_source'] = row.get('data_source', '')

    if remove_tags:
        if isinstance(answer, str):
            clean_answer = re.sub(
                r"<(think|summary|configurator|reasoning|planning|reflection|user_intent)\b[^>]*>[\s\S]*?<\/\1>",
                "", answer,
            ).strip()
        else:
            clean_answer = answer
    else:
        clean_answer = answer

    data_source = row.get('source', row['general_domain'])
    content = await compute_single_reward(data_source, clean_answer, ground_truth, extra_info)
    return content


async def process_single_question(client: AsyncOpenAI, dataset: str, question: str, row: dict, output_file: str, model: str,
                                  semaphore: asyncio.Semaphore, generate_response_fn: Callable, num_retries: int,
                                  generate_response_timeout: int = 600, run_judge_result: bool = False, filter_correct: bool = False,
                                  judge_answer_timeout: int = 180, remove_tags: bool = False, **kwargs) -> None:
    """Process a single question and save the result."""
    async with semaphore:
        for i in range(num_retries):
            try:
                messages, answer = await asyncio.wait_for(
                    generate_response_fn(client, dataset, question, model, **kwargs),
                    timeout=generate_response_timeout,
                )

                if run_judge_result:
                    judge_result = await asyncio.wait_for(
                        judge_answer(row, answer, remove_tags=remove_tags, **kwargs),
                        timeout=judge_answer_timeout,
                    )
                    correct = judge_result['score'] == 1.0
                    result = {
                        "dataset": dataset,
                        "question": question,
                        "messages": messages,
                        "answer": answer,
                        "correct": correct,
                        "judge_result": judge_result,
                        "trial_index": i,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                else:
                    result = {
                        "dataset": dataset,
                        "question": question,
                        "messages": messages,
                        "answer": answer,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }

                if 'rep_idx' in row:
                    result['rep_idx'] = row['rep_idx']

                if filter_correct:
                    if result.get("correct", True):
                        break
                    else:
                        print(f"Incorrect answer, retrying...")
                else:
                    break

            except Exception as e:
                print(f"Error: {e}, retrying...")
                result = {
                    "dataset": dataset,
                    "question": question,
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                if 'rep_idx' in row:
                    result['rep_idx'] = row['rep_idx']

                if 'Request timed out' in str(e):
                    print(f"Request timed out, skipping...")
                    break

        wandb_payload = dict(result)
        wandb_payload.pop('messages', None)
        wandb_payload.update({
            'count': 1,
            "success": 1 if "error" not in result else 0,
            "failure": 1 if "error" in result else 0,
        })
        wandb.log(wandb_payload)

        async with asyncio.Lock():
            with open(output_file, 'a') as f:
                f.write(json.dumps(result) + '\n')

        return result


async def process_questions(input_file: str, output_file: str, model: str,
                            max_concurrent: int = 5,
                            load_questions_fn: Callable = load_questions,
                            generate_response_fn: Callable = generate_response_agent,
                            process_single_question_fn: Callable = process_single_question,
                            setup_client_fn: Callable = None,
                            start_idx: int = 0,
                            end_idx: int = None,
                            num_retries: int = 3,
                            num_repeats: int = 1,
                            subset_datasets: List[str] = None,
                            shuffle_questions: bool = False,
                            **kwargs):
    """Process all questions in parallel and save results."""
    if 'max_turns' in kwargs:
        print(f'Max turns: {kwargs["max_turns"]}')

    if 'filter_out_all_correct' in kwargs:
        print(f'Filter out all correct: {kwargs["filter_out_all_correct"]}')

    client = setup_client_fn()

    questions = load_questions_fn(input_file)

    if subset_datasets is not None:
        print(f'Filtering questions by subset datasets: {subset_datasets}')
        questions = [row for row in questions
                     if row.get('source', row.get('general_domain', 'unknown')) in subset_datasets]
        print(f'Number of questions after filtering: {len(questions)}')

    print(f"Start index: {start_idx}, End index: {end_idx}")
    questions = questions[start_idx:end_idx]

    existing_questions = []
    if os.path.exists(output_file):
        print('Filtering out existing questions')
        with open(output_file, 'r') as f:
            existing_results = [json.loads(line) for line in f]
        existing_questions = [result['question'] for result in existing_results]
        print('Number of existing questions: ', len(existing_questions))
        remaining_counts = {}
        for q in existing_questions:
            remaining_counts[q] = remaining_counts.get(q, 0) + 1
        filtered_questions = []
        for question in questions:
            key = question['question']
            if remaining_counts.get(key, 0) > 0:
                remaining_counts[key] -= 1
            else:
                filtered_questions.append(question)
        questions = filtered_questions
        output_file = output_file.replace('.jsonl', f'_{len(questions)}.jsonl')
        print(f'New output file: {output_file}')

    print('Number of questions after filtering: ', len(questions))

    if shuffle_questions:
        print('Shuffling questions')
        random.shuffle(questions)

    if num_repeats > 1:
        print(f'Repeating {num_repeats} times')
        rep_questions = []
        for question in questions:
            for i in range(num_repeats):
                new_question = question.copy()
                new_question['rep_idx'] = i
                rep_questions.append(new_question)
        questions = rep_questions

    output_base_name = os.path.splitext(os.path.basename(output_file))[0]
    wandb.init(project="sr2am-infer",
               entity="agent-model",
               name=output_base_name)

    semaphore = asyncio.Semaphore(max_concurrent)

    with open(output_file, 'w') as f:
        pass

    tasks = [
        process_single_question_fn(client, row.get('source', row.get('general_domain', 'unknown')),
                                   row['question'], row, output_file, model,
                                   semaphore, generate_response_fn, num_retries, **kwargs)
        for row in questions
    ]
    new_results = await tqdm_asyncio.gather(*tasks, desc="Processing questions")

    return existing_questions + new_results
