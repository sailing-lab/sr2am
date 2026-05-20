import logging
import os
import json
from typing import Any, Optional, Tuple
from uuid import uuid4
import asyncio
import aiohttp
from openai import AsyncOpenAI
import transformers
import tiktoken
import random
from enum import Enum
from pydantic import BaseModel
from typing import Optional, Dict

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class WebSailorMiroflowMultiWebSearchToolFast(BaseTool):
    """A tool for performing potentially multiple web search queries using SerpAPI directly."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Expected tool_schema format:
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Performs batched web searches: supply an array 'query' with each item containing a search query and other parameters; the tool retrieves search results for each query in one call.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "array",
                            "description": "Array of query objects. Include multiple complementary search queries in a single call.",
                            "items": {
                                "type": "string",
                            },
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
            }
        }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.search_provider = str(config.get("search_provider", "serpapi")).strip().lower()
        print(f"Using {self.search_provider} as search provider")
        if self.search_provider not in {"serpapi", "serper_dev"}:
            raise ValueError(
                f"Unsupported search_provider={self.search_provider}. "
                "Expected one of {'serpapi', 'serper_dev'}."
            )
        self.serpapi_key = config.get("serpapi_key", os.getenv("SERPAPI_API_KEY"))
        self.serper_dev_api_key = config.get("serper_dev_api_key", os.getenv("SERPER_DEV_API_KEY"))
        if self.search_provider == "serpapi" and self.serpapi_key is None:
            raise ValueError("Environment variable SERPAPI_API_KEY is required for SearchInformationTool")
        if self.search_provider == "serper_dev" and self.serper_dev_api_key is None:
            raise ValueError("Environment variable SERPER_DEV_API_KEY is required when search_provider=serper_dev")
        self.default_num = config.get("default_num", 10)
        self.default_gl = config.get("default_gl", "us")
        self.default_hl = config.get("default_hl", "en")
        self.default_location = config.get("default_location", None)
        self.default_tbs = config.get("default_tbs", None)
        
        self.search_concurrency = config.get("search_concurrency")
        if self.search_concurrency is not None:
            self.semaphore = asyncio.Semaphore(self.search_concurrency)
        else:
            self.semaphore = None
        self.search_timeout = config.get("search_timeout", 60)
        
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {}   
        return instance_id
    
    def _format_search_results(self, q: str, pages: list[dict[str, Any]]) -> str:
        web_snippets = list()
        idx = 0
        for page in pages:
            idx += 1
            date_published = ""
            if "date" in page:
                date_published = "\nDate published: " + str(page["date"])

            source = ""
            if "source" in page:
                source = "\nSource: " + str(page["source"])

            snippet = ""
            if "snippet" in page:
                snippet = "\n" + str(page["snippet"])

            title = str(page.get("title", "Untitled"))
            link = str(page.get("link", page.get("url", "")))
            redacted_version = f"{idx}. [{title}]({link}){date_published}{source}\n{snippet}"
            redacted_version = redacted_version.replace("Your browser can't play this video.", "")
            web_snippets.append(redacted_version)

        return f"A Google search for '{q}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)

    async def _search_with_serpapi(self, q, gl, hl, num, location, tbs):
        params = {
            "engine": "google",
            "q": q,
            "api_key": self.serpapi_key,
            "num": num,
            "page": 1,
            "gl": gl,
            "hl": hl,
        }
        if location is not None:
            params["location"] = location
        if tbs is not None:
            params["tbs"] = tbs
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.search_timeout)) as session:
                async with session.get("https://serpapi.com/search", params=params) as response:
                    return await response.json()
        except Exception as e:
            raise RuntimeError(f"SerpAPI request failed: {e}") from e

    async def _search_with_serper_dev(self, q, gl, hl, num, location, tbs):
        payload = {
            "q": q,
            "num": num,
            "gl": gl,
            "hl": hl,
        }
        if location is not None:
            payload["location"] = location
        if tbs is not None:
            payload["tbs"] = tbs
        headers = {
            "X-API-KEY": self.serper_dev_api_key,
            "Content-Type": "application/json",
        }
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.search_timeout)) as session:
                async with session.post(
                    "https://google.serper.dev/search",
                    headers=headers,
                    json=payload,
                ) as response:
                    return await response.json()
        except Exception as e:
            raise RuntimeError(f"Serper.dev request failed: {e}") from e

    # async def google_search(self, query: dict):
    async def google_search(self, q, gl, hl, num, location, tbs):
        """Perform search using configured provider and return formatted results."""
        try:
            if self.search_provider == "serper_dev":
                results = await self._search_with_serper_dev(q, gl, hl, num, location, tbs)
                organic_key = "organic"
            else:
                results = await self._search_with_serpapi(q, gl, hl, num, location, tbs)
                organic_key = "organic_results"
        except Exception as e:
            print(f"Fail to search for '{q}'. Error: {e}")
            return f"Fail to search for '{q}'. Use a different query."

        try:
            pages = results.get(organic_key, [])
            if not pages:
                raise Exception(f"No results found for query: '{q}'. Use a less specific query.")
            content = self._format_search_results(q, pages)
        except Exception:
            content = f"No results found for '{q}'. Try with a more general query, or remove the year filter."
        
        return content
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        
        try:
            query = parameters["query"]
        except:
            return ("[Search] Invalid request format: Input must be a JSON object containing 'query' field", 
                    0.0, 
                    {"error": "Invalid request format: Input must be a JSON object containing 'query' field"})
            
        gl = parameters.get("gl", self.default_gl)
        hl = parameters.get("hl", self.default_hl)
        num = parameters.get("num", self.default_num)
        location = parameters.get("location", self.default_location)
        tbs = parameters.get("tbs", self.default_tbs)
            
        if isinstance(query, str):
            response = await self.google_search(query, gl, hl, num, location, tbs)
        else:
            assert isinstance(query, list)
            
            if self.semaphore is not None:
                sem = self.semaphore
            else:
                sem = asyncio.Semaphore(3)

            async def _run(q: Any) -> str:
                async with sem:
                    res = await self.google_search(q, gl, hl, num, location, tbs)   # now async
                    return str(res)

            # gather preserves input order
            results = await asyncio.gather(*(_run(q) for q in query))
            
            response = "\n=======\n".join(results)
        
        return response.strip(), 0.0, {
            "query": query, 
            "result_length": len(response)
        }

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # Base reward for successful search
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
            
            

async def post_json(url, headers):
    t = aiohttp.ClientTimeout(total=60, connect=10, sock_connect=10, sock_read=50)
    async with aiohttp.ClientSession(timeout=t) as s:
        async with s.post(url, headers=headers) as r:
            r.raise_for_status()
            # return await r.json()
            return await r.text(), r.status

class WebSailorMultiVisitToolFast(BaseTool):
    """A tool for visiting potentially multiple webpages and summarizing them with a LLM."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Expected tool_schema format:
        {
            "type": "function",
            "function": {
                "name": "visit",
                "description": "Visit webpage(s) and return the summary of the content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
                        },
                        "goal": {
                            "type": "string",
                            "description": "The specific information goal for visiting webpage(s)."
                        }
                    },
                    "required": [
                        "url",
                        "goal"
                    ]
                }
            }
        },
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.summarize_model = config.get("summarize_model", "gpt-4.1-mini")
        self.client = AsyncOpenAI(base_url=config.get("summarize_model_base_url", "https://api.openai.com/v1"), 
                                  api_key=config.get("summarize_model_api_key", os.getenv("OPENAI_API_KEY")),
                                  timeout=config.get("summarize_model_timeout", 300))
        self.max_tokens = config.get("max_webpage_tokens", 28000)
        self.max_context_length = config.get("max_context_length", 32768)
        self.jina_api_key = config.get("jina_api_key", os.getenv("JINA_API_KEY"))

        self.visit_concurrency = config.get("visit_concurrency")
        if self.visit_concurrency is not None:
            self.semaphore = asyncio.Semaphore(self.visit_concurrency)
        else:
            self.semaphore = None
        self.visit_timeout = config.get("visit_timeout", 30)

        if self.summarize_model.startswith('Qwen/Qwen3'):
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.enc = transformers.AutoTokenizer.from_pretrained(self.summarize_model)
        else:
            self.enc = tiktoken.encoding_for_model("gpt-4o")
            
        self.extractor_prompt = config.get("extractor_prompt", """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{article_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**. You must escape every backslash (e.g. in LaTeX code) with double backslashes.
""")
            
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id
    
    async def call_server(self, msgs, max_tries=10):
        
        for attempt in range(max_tries):
            try:
                prompt_length = len(self.enc.encode(msgs[0]["content"])) + 1
                max_completion_tokens = 4096
                if self.summarize_model.startswith('Qwen/Qwen3'):
                    response = await self.client.chat.completions.create(
                        model=self.summarize_model,
                        messages=msgs,
                        max_completion_tokens=min(self.max_context_length - prompt_length, max_completion_tokens),
                        temperature=0.7,
                        top_p=0.8,
                        presence_penalty=1.5,
                        extra_body={
                            "top_k": 20, 
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    
                else:
                    response = await self.client.chat.completions.create(
                        model=self.summarize_model,
                        messages=msgs,
                        max_completion_tokens=min(self.max_context_length - prompt_length, max_completion_tokens),
                    )
                    
                content = response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string 
                        left = content.find('{')
                        right = content.rfind('}') 
                        if left != -1 and right != -1 and left <= right: 
                            content = content[left:right+1]
                    return content
            except:
                if attempt == (max_tries - 1):
                    return ""
                continue
                
    async def jina_readpage(self, url: str) -> str:
        """
        Read webpage content using Jina service.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        headers = {
            "Authorization": f"Bearer {self.jina_api_key}",
        }
        max_retries = 2
        timeout = 10
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.visit_timeout)) as session:
                    async with session.post(f"https://r.jina.ai/{url}", headers=headers) as resp:
                        response = await resp.text()
                        status = resp.status
                # response, status = await post_json(f"https://r.jina.ai/{url}", headers=headers)
                # response = requests.get(
                #     f"https://r.jina.ai/{url}",
                #     headers=headers,
                #     timeout=timeout
                # )
                if status == 200:
                    webpage_content = response
                    return webpage_content
                else:
                    print(response)
                    raise ValueError("jina readpage error")
            except Exception as e:
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."
                
        return "[visit] Failed to read page."
    
    # def truncate_content(self, content: str) -> int:
    #     # Check if the article content is too long
    #     if len(self.enc.encode(content)) > self.max_tokens:
    #         # Truncate the article content
    #         content = self.enc.decode(self.enc.encode(content)[:self.max_tokens])
    #     return content
                
    async def readpage(self, url: str, goal: str) -> str:
        """
        Attempt to read webpage content by alternating between jina and aidata services.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        max_attempts = 2
        for attempt in range(max_attempts):
            # Alternate between jina and aidata
            content = await self.jina_readpage(url)
            sevice = "jina"

            # Check if we got valid content
            # print(sevice)
            # print(content)
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                # content = content[:WEBCONTENT_MAXLENGTH]
                content = self.enc.decode(self.enc.encode(content)[:self.max_tokens])
                messages = [{"role":"user", "content": self.extractor_prompt.format(article_content=content, goal=goal)}]
                parse_retry_times = 0
                raw = await self.call_server(messages)

                # 如果网页超长，返回结果是 {\n 这种形式
                summary_retries = 3
                while len(raw) < 10 and summary_retries >= 0:
                    truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                    status_msg = (
                        f"[visit] Summary url[{url}] " 
                        f"attempt {3 - summary_retries + 1}/3, "
                        f"content length: {len(content)}, "
                        f"truncating to {truncate_length} chars"
                    ) if summary_retries > 0 else (
                        f"[visit] Summary url[{url}] failed after 3 attempts, "
                        f"final truncation to 25000 chars"
                    )
                    print(status_msg)
                    content = content[:truncate_length]
                    extraction_prompt = self.extractor_prompt.format(
                        article_content=content,
                        goal=goal
                    )
                    messages = [{"role": "user", "content": extraction_prompt}]
                    raw = await self.call_server(messages)
                    summary_retries -= 1
                # 说明 raw 的长度大于10或者已经retry 超出了 
                parse_retry_times = 0
                while parse_retry_times < 3:
                    try:
                        # 尝试 parse json
                        raw = json.loads(raw)
                        break
                    except:
                        raw = await self.call_server(messages)
                        parse_retry_times += 1
                # parse 失败
                if parse_retry_times >= 3:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                # parse 成功
                else:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                    useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

                    summary_retries -= 1

                if len(useful_information) < 10 and summary_retries < 0:
                    print("[visit] Could not generate valid summary after maximum retries")
                    useful_information = "[visit] Failed to read page"
                return useful_information
                
            # If we're on the last attempt, return the last result
            if attempt == max_attempts - 1:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                return useful_information
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        try:
            url = parameters["url"]
            goal = parameters["goal"]
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields", 0.0, {"error": "Invalid request format"}

        if isinstance(url, str):
            response = await self.readpage(url, goal)
        else:
            response = []
            assert isinstance(url, list)
            if self.semaphore is not None:
                sem = self.semaphore
            else:
                sem = asyncio.Semaphore(3)
            async def _run(u: Any) -> str:
                async with sem:
                    res = await self.readpage(u, goal)
                    return str(res)
                
            results = await asyncio.gather(*(_run(u) for u in url))
            response = "\n=======\n".join(results)
        
        # print(f'Summary Length {len(response)}; Summary Content {response}')
        return response.strip(), 0.0, {
            "url": url, 
            "goal": goal, 
            "result_length": len(response)
        }
    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # Base reward for successful search
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
            

class RunStatus(str, Enum):
    # all command finished successfully
    Success = 'Success'
    # one of the process has non-zero return code
    Failed = 'Failed'
    # error on sandbox side
    SandboxError = 'SandboxError'
    

class CommandRunStatus(str, Enum):
    Finished = 'Finished'
    Error = 'Error'
    TimeLimitExceeded = 'TimeLimitExceeded'


class CommandRunResult(BaseModel):
    status: CommandRunStatus
    execution_time: Optional[float] = None
    return_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class RunCodeResponse(BaseModel):
    status: RunStatus
    message: str
    compile_result: Optional[CommandRunResult] = None
    run_result: Optional[CommandRunResult] = None
    executor_pod_name: Optional[str] = None
    files: Dict[str, str] = {}
    
    
async def code_exec_sandboxfusion(code, stdin: Optional[str], timeout: int, sandbox_server_list: list[str]):
    try:
        request_data = {
            "language": "python",
            "code": code,
            "stdin": stdin,
            "run_timeout": timeout
        }
        
        # Try each server (for load balancing/failover)
        for server in sandbox_server_list:
            try:
                url = f"http://{server}:8080/run_code"
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout + 2)) as session:
                    async with session.post(url, json=request_data) as resp:
                        response_status = resp.status
                        response_json = await resp.json()
                
                # print('URL:', url, '\nRequest Data:', request_data, 
                #       '\nResponse Status:', response_status, '\nResponse JSON:', response_json)
                
                # if response.status_code != 200:
                if response_status != 200:
                    continue  # Try next server
                
                # result = RunCodeResponse(**response.json())
                result = RunCodeResponse(**response_json)
                if result.status == RunStatus.Success:
                    return True, result.run_result.stdout
                elif result.run_result.status == CommandRunStatus.TimeLimitExceeded:
                    return False, f"Time limit of {timeout} seconds exceeded.\nSTDOUT:\n{result.run_result.stdout}\n\nSTDERR:\n{result.run_result.stderr}"
                else:
                    return False, f"STDOUT:\n{result.run_result.stdout}\n\nSTDERR:\n{result.run_result.stderr}"
                    
            # except requests.exceptions.RequestException:
            except aiohttp.ClientError:
                continue  # Try next server
        
        # If we get here, all servers failed
        return False, f"All sandbox servers failed to process the request."
            
    except Exception as e:
        return False, f"Execution error: {str(e)}"
    
            
class SandboxFusionCodeTool(BaseTool):
    """A tool for performing code execution."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Expected tool_schema format:
        {
            "type": "function",
            "function": {
                "name": "python_repl_tool",
                "description": "Use this to execute python code and do data analysis or calculation. If you want to see the output of a value,\n    you should print it out with `print(...)`. This is visible to the user.",
                "parameters": {
                    "properties": {
                        "code": {
                            "description": "The python code to execute to do further analysis or calculation.",
                            "type": "string"
                        }
                    },
                    "required": [
                        "code"
                    ],
                    "type": "object"
                }
            }
        }
        """
        super().__init__(config, tool_schema)
        self.code_sandbox_servers = config['code_sandbox_servers']
        self.code_timeout = config['code_timeout']
        self.code_concurrency = config['code_concurrency']
        self.semaphore = asyncio.Semaphore(self.code_concurrency)
        
        self._instance_dict = {}
        
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        code = parameters.get("code", "")
        
        if not isinstance(code, str):
            error_msg = f"Invalid input: code must be a string, got {type(code)}"
            logger.error(error_msg)
            result_str = f"Error executing code:\n```python\n{code}\n```\nError: {error_msg}"
            return result_str, -0.1, {"code": code, "error": error_msg}

        logger.info("Executing Python code")
        try:
            async with self.semaphore:
                sandbox_server_list = random.sample(self.code_sandbox_servers, len(self.code_sandbox_servers))
                success, output = await code_exec_sandboxfusion(
                    code=code,
                    stdin=None,
                    timeout=self.code_timeout,
                    sandbox_server_list=sandbox_server_list
                )
                result = output
            # python_repl = self._instance_dict[instance_id]["python_repl"]
            # result = python_repl.run(code, timeout=60)
            # result = await run_with_hard_timeout(python_repl, code, 60)
            # Check if the result is an error message by looking for typical error patterns
            # if isinstance(result, str) and ("Error" in result or "Exception" in result):
            if not success:
                logger.error(result)
                result_str = f"Error executing code:\n```python\n{code}\n```\nError: {result}"
                return result_str, -0.1, {"code": code, "error": result}
            logger.info("Code execution successful")
        except BaseException as e:
            error_msg = repr(e)
            logger.error(error_msg)
            result_str = f"Error executing code:\n```python\n{code}\n```\nError: {error_msg}"
            return result_str, -0.1, {"code": code, "error": error_msg}

        result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
        
        return result_str, 0.0, {
            "code": code, 
            "result": result
        }

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # Base reward for successful code execution
        return 0.0  
    
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]