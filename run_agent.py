"""
Unified agent runner for v1.0 rollout, filtering, and evaluation.

Supports three agent types via --agent_type:
- think: VLLMModelThinkAgent for thinking-mode models (DeepSeek, Qwen3, etc.)
- configurator: VLLMModelConfiguratorAgent (think agent + newline formatting)
- instruct: VLLMModelInstructAgent (no tools, just sends user messages to LLM)

Usage:
    # v1.0 teacher rollout
    python run_agent.py --agent_type think --input_file data.jsonl ...

    # v0.1/v1.0 non-web filtering
    python run_agent.py --agent_type think --filtering ...

    # v0.1/v1.0 web filtering
    python run_agent.py --agent_type instruct --filtering ...

    # Evaluation
    python run_agent.py --agent_type think --filtering --remove_tags ...
"""

from agent_base import (
    Agent, generate_response_agent, process_questions,
    judge_answer, process_single_question,
)
from system_prompts import get_agent_system_prompt
from tools.schemas import OpenAIFunctionToolSchema
from tools.websailor_tools_fast import (
    WebSailorMiroflowMultiWebSearchToolFast,
    WebSailorMultiVisitToolFast,
    SandboxFusionCodeTool,
)
from openai import AsyncOpenAI
import asyncio
from functools import partial
import json
import time
import wandb
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Model-specific configuration (replaces if/elif chain)
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    # Chat completions models — extra_body kwargs passed to chat.completions.create
    "DeepSeek-V3.2":            {"extra_body": {"chat_template_kwargs": {"thinking": True}}},
    "Qwen3-235B-A22B-Thinking": {"extra_body": {"chat_template_kwargs": {"thinking": True}}},
    "Qwen3-30B-A3B-Thinking":   {"extra_body": {"chat_template_kwargs": {"thinking": True}}},
    "Qwen3-8B":                 {"extra_body": {"chat_template_kwargs": {"thinking": True}}},
    "Kimi-K2.5":                {"extra_body": {"chat_template_kwargs": {"thinking": True}}},
    "K2-Think-V2":              {"extra_body": {"reasoning_effort": "high"}},
    "GLM":                      {"stop": ["<|user|>", "<|endoftext|>", "<|observation|>", "<|assistant|>"]},
    # Responses API models — kwargs passed to responses.create
    "gpt-oss-120b":             {"reasoning": {"effort": "high"}},
    "gpt-5.4":                  {"reasoning": {"effort": "xhigh"}, "max_input_tokens": 131072},
}


def get_model_config(model_name: str) -> dict:
    """Look up model-specific parameters by substring match."""
    for key, config in MODEL_CONFIGS.items():
        if key in model_name:
            return config
    return {}


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOL_CLS = [WebSailorMiroflowMultiWebSearchToolFast, WebSailorMultiVisitToolFast, SandboxFusionCodeTool]

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Performs batched web searches: supply an array 'query' with each item containing a search query and other parameters; the tool retrieves search results for each query in one call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "description": "Array of query objects. Include multiple complementary search queries in a single call.",
                        "items": {"type": "string"},
                    },
                    "gl": {
                        "type": "string",
                        "description": "Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')",
                    },
                    "hl": {
                        "type": "string",
                        "description": "Optional language code for search results in ISO 639-1 format (e.g., 'en')",
                    },
                    "location": {
                        "type": "string",
                        "description": "Optional location for search results (e.g., 'SoHo, New York, United States', 'California, United States')",
                    },
                    "num": {
                        "type": "number",
                        "description": "Number of results to return (default: 10)",
                    },
                    "tbs": {
                        "type": "string",
                        "description": "Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week, 'qdr:m' for past month, 'qdr:y' for past year)",
                    },
                },
                "required": ["query", "gl", "hl"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visit_tool",
            "description": "Visit webpage(s) and return the summary of the content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.",
                    },
                    "goal": {
                        "type": "string",
                        "description": "The specific information goal for visiting webpage(s).",
                    },
                },
                "required": ["url", "goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_repl_tool",
            "description": "Use this to execute python code and do data analysis or calculation. Import all the packages you need before using them. If you want to see the output of a value,\n    you should print it out with `print(...)`. This is visible to the user.",
            "parameters": {
                "properties": {
                    "code": {
                        "description": "The python code to execute to do further analysis or calculation. Make sure to print the values you want to see.",
                        "type": "string",
                    },
                },
                "required": ["code"],
                "type": "object",
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Agent classes
# ---------------------------------------------------------------------------

class VLLMModelThinkAgent(Agent):
    """Agent for thinking-mode models. Supports chat completions + Responses API."""

    def __init__(self, client, model, tool_instances, system_prompt, use_response_api=False,
                 max_completion_tokens=8192, temperature=None, **kwargs):
        super().__init__(client, model, tool_instances, system_prompt,
                         use_response_api=use_response_api, **kwargs)
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature

    async def get_response(self):
        config = get_model_config(self.model)

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

            # Enforce max input tokens if configured (e.g. gpt-5.4)
            max_input_tokens = config.get("max_input_tokens")
            if max_input_tokens is not None:
                token_count_response = await self.client.responses.input_tokens.count(
                    model=self.model,
                    instructions=self.system_prompt,
                    input=input_messages,
                )
                if token_count_response.input_tokens > max_input_tokens:
                    print(
                        f"Input token count exceeds limit: "
                        f"{token_count_response.input_tokens} > {max_input_tokens}"
                    )
                    return {
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [{"type": "output_text", "text": "Context length exceeded"}],
                            }
                        ]
                    }

            # Build Responses API kwargs from config
            responses_kwargs = {}
            if "reasoning" in config:
                responses_kwargs["reasoning"] = config["reasoning"]

            response = await self.client.responses.create(
                model=self.model,
                instructions=self.system_prompt,
                input=input_messages,
                tools=tools,
                max_output_tokens=self.max_completion_tokens,
                **responses_kwargs,
            )

            response_dict = response.model_dump()
            if response.usage is not None:
                outputs = response_dict.get("output")
                if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], dict):
                    outputs[0]["usage"] = response.usage.model_dump()

            return response_dict

        # Chat completions path
        chat_kwargs = {}
        if "extra_body" in config:
            chat_kwargs["extra_body"] = config["extra_body"]
        if "stop" in config:
            chat_kwargs["stop"] = config["stop"]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=[tool.model_dump() for tool in self.tool_schemas],
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            **chat_kwargs,
        )
        message = response.choices[0].message.model_dump()
        return message


class VLLMModelConfiguratorAgent(VLLMModelThinkAgent):
    """Think agent that adds a newline after content when tool_calls are present."""

    def add_message(self, content: dict):
        if 'tool_calls' in content:
            if content['content']:
                content['content'] = content['content'].rstrip('\n') + '\n'
            else:
                content['content'] = '\n'
        self.messages.append(content)


class VLLMModelInstructAgent(Agent):
    """No-tools agent for web filtering. Sends only user messages to the LLM."""

    async def get_response(self):
        user_messages = [msg for msg in self.messages if msg['role'] == 'user']
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=user_messages,
            max_completion_tokens=4096,
        )
        message = response.choices[0].message.model_dump()
        return message


# ---------------------------------------------------------------------------
# Filtering question processor (with retry/threshold logic)
# ---------------------------------------------------------------------------

async def process_single_question_filtering(client: AsyncOpenAI, dataset: str, question: str, row: dict, output_file: str, model: str,
                                            semaphore: asyncio.Semaphore, generate_response_fn: Callable, num_retries: int,
                                            generate_response_timeout: int = 600,
                                            judge_answer_timeout: int = 180,
                                            remove_tags: bool = False,
                                            break_early: bool = True,
                                            filter_out_all_correct: bool = False,
                                            use_thresholds: bool = False,
                                            min_corrects: int = 0,
                                            min_wrongs: int = 0,
                                            resume_counts: Optional[dict] = None,
                                            resume_assume_total_reps: Optional[int] = None,
                                            filter_correct: bool = False,
                                            save_partial_messages: bool = False,
                                            **kwargs) -> None:
    """Process a single question with filtering logic (retry until threshold met)."""
    async with semaphore:
        results = []
        num_corrects = 0
        num_wrongs = 0
        rep_idx_offset = 0
        remaining_retries = num_retries

        if resume_counts is not None:
            resume_entry = resume_counts.get((dataset, question))
            if resume_entry is not None:
                num_corrects = resume_entry.get("num_corrects", 0)
                num_wrongs = resume_entry.get("num_wrongs", 0)
                total = resume_entry.get("total", num_corrects + num_wrongs)
                rep_idx_offset = total
                remaining_retries = max(0, num_retries - total)
                if use_thresholds and (min_corrects > 0 or min_wrongs > 0):
                    if num_corrects >= min_corrects and num_wrongs >= min_wrongs:
                        remaining_retries = 0

                if remaining_retries == 0:
                    print(f"Breaking early: num_corrects={num_corrects}, num_wrongs={num_wrongs}, total={num_corrects + num_wrongs}")

        for i in range(remaining_retries):
            try:
                start_time = time.time()
                messages, answer = await asyncio.wait_for(
                    generate_response_fn(client, dataset, question, model, **kwargs),
                    timeout=generate_response_timeout,
                )
                generate_response_time = time.time() - start_time

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
                    "generate_response_time": generate_response_time,
                    "rep_idx": rep_idx_offset + i,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                results.append(result)

                if correct:
                    if filter_correct:
                        break
                    num_corrects += 1
                else:
                    num_wrongs += 1

                if break_early:
                    if filter_out_all_correct and num_wrongs > 0:
                        break
                    elif use_thresholds and (min_corrects > 0 or min_wrongs > 0):
                        if num_corrects >= min_corrects and num_wrongs >= min_wrongs:
                            break
                    elif num_corrects > 0 and num_wrongs > 0:
                        break

            except Exception as e:
                print(f"Error: {e}, retrying...")
                partial_messages = getattr(e, 'partial_messages', None) if save_partial_messages else None
                result = {
                    "dataset": dataset,
                    "question": question,
                    "messages": partial_messages,
                    "error": str(e),
                    "rep_idx": rep_idx_offset + i,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                num_wrongs += 1
                results.append(result)

        wandb_payload = {
            'dataset': dataset,
            'question': question,
            'count': 1,
            'total': num_corrects + num_wrongs,
            "success": num_corrects,
            "failure": num_wrongs,
            "remaining_retries": remaining_retries,
        }
        wandb.log(wandb_payload)

        if results:
            async with asyncio.Lock():
                with open(output_file, 'a') as f:
                    for result in results:
                        f.write(json.dumps(result) + '\n')


# ---------------------------------------------------------------------------
# Resume counts loader
# ---------------------------------------------------------------------------

def load_resume_counts(resume_counts_file: str, resume_assume_total_reps: Optional[int]):
    """Load resume counts from a previous run's output file."""
    if resume_counts_file is None:
        return None
    resume_counts = {}
    with open(resume_counts_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dataset = row.get("dataset")
            question = row.get("question")
            if dataset is None or question is None:
                continue
            num_corrects = row.get("num_corrects", 0)
            num_wrongs = row.get("num_wrongs", 0)
            total = row.get("total", num_corrects + num_wrongs)
            resume_counts[(dataset, question)] = {
                "num_corrects": num_corrects,
                "num_wrongs": num_wrongs,
                "total": total,
            }
    return resume_counts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run agent for v1.0 rollout, filtering, and evaluation")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_base_url", type=str, required=True)
    parser.add_argument("--model_api_key", type=str, default='EMPTY')
    parser.add_argument("--model_timeout", type=int, default=300)
    parser.add_argument("--agent_type", type=str, choices=["think", "configurator", "instruct"], default="configurator")

    # Tool configuration (ignored for --agent_type instruct)
    parser.add_argument("--search_concurrency", type=int, default=64)
    parser.add_argument("--search_timeout", type=int, default=60)
    parser.add_argument("--search_provider", type=str, choices=["serpapi", "serper_dev"], default="serpapi")
    parser.add_argument("--browsing_summarize_model", type=str, default=None)
    parser.add_argument("--browsing_summarize_model_base_url", type=str, default=None)
    parser.add_argument("--visit_concurrency", type=int, default=64)
    parser.add_argument("--visit_timeout", type=int, default=60)
    parser.add_argument("--code_sandbox_servers", type=str, nargs='+', default=None)
    parser.add_argument("--code_timeout", type=int, default=60)
    parser.add_argument("--code_concurrency", type=int, default=64)

    # Question processing
    parser.add_argument("--max_concurrent", type=int, default=8)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--num_retries", type=int, default=3)
    parser.add_argument("--subset_datasets", type=str, nargs='+', default=None)
    parser.add_argument("--shuffle_questions", action='store_true', default=False)

    # Agent parameters
    parser.add_argument("--response_timeout", type=int, default=600)
    parser.add_argument("--tool_execution_timeout", type=int, default=300)
    parser.add_argument("--generate_response_timeout", type=int, default=3600)
    parser.add_argument("--judge_answer_timeout", type=int, default=300)
    parser.add_argument("--max_turns", type=int, default=30)
    parser.add_argument("--max_completion_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--use_response_api", action='store_true', default=False)

    # Filtering
    parser.add_argument("--filtering", action='store_true')
    parser.add_argument("--remove_tags", action='store_true')
    parser.add_argument("--no_break_early", action='store_true')
    parser.add_argument("--filter_out_all_correct", action='store_true')
    parser.add_argument("--filter_correct", action='store_true')
    parser.add_argument("--use_thresholds", action='store_true', default=False)
    parser.add_argument("--min_corrects", type=int, default=0)
    parser.add_argument("--min_wrongs", type=int, default=0)
    parser.add_argument("--resume_counts_file", type=str, default=None)
    parser.add_argument("--resume_assume_total_reps", type=int, default=None)
    parser.add_argument("--save_partial_messages", action='store_true', default=False)
    parser.add_argument("--fix_datetime", action='store_true', default=False,
                        help="Use fixed training datetime in system prompt (for paper reproduction)")

    args = parser.parse_args()

    # --- Client setup ---
    def setup_client() -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=args.model_base_url,
            timeout=args.model_timeout,
            api_key=args.model_api_key,
        )

    # --- Agent type selection ---
    if args.agent_type == "think":
        agent_cls = VLLMModelThinkAgent
    elif args.agent_type == "configurator":
        agent_cls = VLLMModelConfiguratorAgent
    elif args.agent_type == "instruct":
        agent_cls = VLLMModelInstructAgent
    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}")

    print(f"Using {args.agent_type} agent")

    # --- System prompt ---
    custom_system_prompt_fn = partial(get_agent_system_prompt,
                                     fix_datetime=args.fix_datetime)
    print('System prompt example:')
    print(custom_system_prompt_fn(dataset='web'))
    print('--------------------------------')

    # --- Tool setup ---
    if args.agent_type == "instruct":
        tool_instances = {}
    else:
        TOOL_CFGS = [
            {
                "search_concurrency": args.search_concurrency,
                "search_timeout": args.search_timeout,
                "search_provider": args.search_provider,
            },
            {
                "summarize_model": args.browsing_summarize_model,
                "summarize_model_base_url": args.browsing_summarize_model_base_url,
                "summarize_model_api_key": 'EMPTY',
                "max_webpage_tokens": 28000,
                "max_context_length": 32768,
                "visit_concurrency": args.visit_concurrency,
                "visit_timeout": args.visit_timeout,
            },
            {
                "code_sandbox_servers": args.code_sandbox_servers,
                "code_timeout": args.code_timeout,
                "code_concurrency": args.code_concurrency,
            },
        ]

        tool_instances = {}
        for tool_cls, tool_cfg, tool_schema in zip(TOOL_CLS, TOOL_CFGS, TOOL_SCHEMAS):
            tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema)
            tool = tool_cls(config=tool_cfg, tool_schema=tool_schema)
            tool_instances[tool.name] = tool

    # --- Question processor selection ---
    if args.filtering:
        print("Using filtering mode")
        process_single_question_fn = process_single_question_filtering
    else:
        print("Using default mode")
        process_single_question_fn = process_single_question

    resume_counts = load_resume_counts(args.resume_counts_file, args.resume_assume_total_reps)

    # --- Run ---
    asyncio.run(process_questions(
        args.input_file, args.output_file,
        model=args.model,
        max_concurrent=args.max_concurrent,
        generate_response_fn=generate_response_agent,
        process_single_question_fn=process_single_question_fn,
        setup_client_fn=setup_client,
        tool_instances=tool_instances,
        agent_cls=agent_cls,
        custom_system_prompt_fn=custom_system_prompt_fn,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        num_retries=args.num_retries,
        remove_tags=args.remove_tags,
        break_early=(not args.no_break_early),
        subset_datasets=args.subset_datasets,
        response_timeout=args.response_timeout,
        tool_execution_timeout=args.tool_execution_timeout,
        generate_response_timeout=args.generate_response_timeout,
        judge_answer_timeout=args.judge_answer_timeout,
        max_turns=args.max_turns,
        filter_out_all_correct=args.filter_out_all_correct,
        filter_correct=args.filter_correct,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        use_thresholds=args.use_thresholds,
        min_corrects=args.min_corrects,
        min_wrongs=args.min_wrongs,
        resume_counts=resume_counts,
        resume_assume_total_reps=args.resume_assume_total_reps,
        shuffle_questions=args.shuffle_questions,
        use_response_api=args.use_response_api,
        save_partial_messages=args.save_partial_messages,
    ))
