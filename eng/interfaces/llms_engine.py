#from dsp import LM
import requests, time, re, json
from typing import List, Tuple, Any, Callable, Literal, Awaitable, Optional, Union
import aiohttp
import asyncio
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..assets.assets_collection import default_llm_kwarg

# Load only the tokenizer configuration
#tokenizer_dbrx = PreTrainedTokenizerFast.from_pretrained("mistralai/mixtral-8x22b-instruct", local_files_only=False)
#tokenizer_llama = PreTrainedTokenizerFast.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct", local_files_only=False)

## create a function that will return the number of tokens for a given text
# get current path
current_path = os.path.dirname(os.path.abspath(__file__))
# one level up
one_level_up = os.path.dirname(current_path)

tokenizer_json_path = os.path.join(one_level_up, 'assets/tokenizers/meta-llama/Llama-3.1-70B-Instruct/tokenizer.json')

with open(tokenizer_json_path, 'r') as f:
        tokenizer_data = json.load(f)

def _create_token_pattern():
    vocab = tokenizer_data.get('model', {}).get('vocab', {})
    special_tokens = tokenizer_data.get('added_tokens', [])
    special_token_values = [token['content'] for token in special_tokens]
    all_tokens = list(vocab.keys()) + special_token_values
    all_tokens.sort(key=len, reverse=True)
    pattern = '|'.join(re.escape(token) for token in all_tokens)
    return re.compile(pattern)

# Compile pattern once at module level
TOKEN_PATTERN = _create_token_pattern()

def count_tokens(text: str) -> int:
    """Count tokens using regex with pre-compiled pattern"""
    return len(TOKEN_PATTERN.findall(text))


class LMEngine2:
    def __init__(self, model, lm_api, kwarg):
        # Additional initialization code for the subclass
        self.model = model
        self.api_key = lm_api
        self.provider = ""
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.history = []
        self.usage = []
        self.kwargs = kwarg.copy()
    def basic_request(self, prompt: str, **kwargs):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            **self.kwargs,
            **kwargs,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": self.model,
            "stream": False,

        }
        response = requests.post(self.base_url,headers=headers, json=data)
        response = response.json()
        self.history.append({
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
        })
        return response
    
    def __enter__(self):

        return self

    def __exit__(self):
        
        pass

    def __call__(self, prompt, **kwargs):
        """
        Make an API call to the language model for completion.
        
        Args:
            prompt (str): The input text to send to the model
            **kwargs: Additional parameters to override default settings
            
        Returns:
            str: The generated text response
            
        Raises:
            Exception: If the API request fails or returns an invalid response
        """
        try:
            response = self.basic_request(prompt, **kwargs)
            
            # Check if response contains error
            if 'error' in response:
                error_msg = response.get('error', {}).get('message', 'Unknown error occurred')
                raise Exception(f"API request failed: {error_msg}")
                
            # Extract completion text from response
            if 'choices' in response and len(response['choices']) > 0:
                completion = response['choices'][0].get('message', {}).get('content', '').strip()
                
                # Track token usage if available
                if 'usage' in response:
                    self.usage.append(response['usage'])
                    
                return completion
            else:
                raise Exception("Invalid response format: 'choices' not found or empty")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error occurred: {str(e)}")
        except ValueError as e:
            raise Exception(f"Failed to parse JSON response: {str(e)}")
        except Exception as e:
            raise Exception(f"An error occurred: {str(e)}")


class LMEngine_async2:
    def __init__(self, model, lm_api, kwarg):
        self.model = model
        self.api_key = lm_api
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.history = []
        self.usage = []
        self.kwargs = kwarg.copy()

    async def basic_request(self, prompt: str, **kwargs):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            **self.kwargs,
            **kwargs,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": self.model,
            "stream": False,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=data) as response:
                if response.status != 200:
                    return {"error": {"message": f"HTTP error: {response.status}"}}
                response_json = await response.json()

        self.history.append({
            "prompt": prompt,
            "response": response_json,
            "kwargs": kwargs,
        })

        return response_json

    async def __call__(self, prompt, **kwargs):
        """
        Make an async API call to the language model for completion.
        
        Args:
            prompt (str): The input text to send to the model
            **kwargs: Additional parameters to override default settings
            
        Returns:
            str: The generated text response
            
        Raises:
            Exception: If the API request fails or returns an invalid response
        """
        try:
            response = await self.basic_request(prompt, **kwargs)
            
            # Check if response contains error
            if 'error' in response:
                error_msg = response.get('error', {}).get('message', 'Unknown error occurred')
                raise Exception(f"API request failed: {error_msg}")
                
            # Extract completion text from response
            if 'choices' in response and len(response['choices']) > 0:
                completion = response['choices'][0].get('message', {}).get('content', '').strip()
                
                # Track token usage if available
                if 'usage' in response:
                    self.usage.append(response['usage'])
                    
                return completion
            else:
                raise Exception("Invalid response format: 'choices' not found or empty")
                
        except aiohttp.ClientError as e:
            raise Exception(f"Network error occurred: {str(e)}")
        except ValueError as e:
            raise Exception(f"Failed to parse JSON response: {str(e)}")
        except Exception as e:
            raise Exception(f"An error occurred: {str(e)}")

    async def __aenter__(self):
        return self

    async def __aexit__(self):
        pass


class LMEngine:
    """
    Synchronous OpenRouter Language Model integration.
    
    This class makes a POST request to the OpenRouter API endpoint using a
    requests.Session with a retry strategy for improved resilience.
    """
    def __init__(self, lm_api: str, model: str, 
                 site_url: str = "https://your-website.com", 
                 site_name: str = "YourSiteName",
                 extra_kwarg: dict = None):
        self.api_key = lm_api
        self.model = model
        self.site_url = site_url
        self.site_name = site_name
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.extra_kwarg = extra_kwarg
        self.usage = []
        self.history = []

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "Content-Type": "application/json",
        }
        
        # Set up a session with a retry strategy to handle transient errors.
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,                          # Total number of retries.
            backoff_factor=1,                 # Wait 1 sec, 2 sec, 4 sec between retries.
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()
    
    def __call__(self, prompt: str) -> str:
        """
        Make a synchronous API call to the OpenRouter endpoint using the provided prompt.
        
        Args:
            prompt (str): The user prompt to be sent to the API.
            extra_payload (dict, optional): Additional fields to merge into the JSON payload.
            
        Returns:
            dict: The JSON response from the OpenRouter API.
            
        Raises:
            Exception: If a network error occurs or the API returns an error.
        """
        # Create base payload
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "provider": {
                "data_collection": "deny",
                "sort": "throughput", 
                "allow_fallbacks": True,
                "require_parameters": False
            }
        }

        if self.extra_kwarg:
            payload.update(self.extra_kwarg)
        
        try:
            response = self.session.post(self.base_url, headers=self.headers, json=payload, timeout=100)
            response.raise_for_status()
            result = response.json()
            # Check if response contains error
            if 'error' in result:
                error_msg = result.get('error', {}).get('message', 'Unknown error occurred')
                raise Exception(f"API request failed: {error_msg}")
                
            # Extract completion text from response
            if 'choices' in result and len(result['choices']) > 0:
                completion = result['choices'][0].get('message', {}).get('content', '').strip()
                
                # Track token usage if available
                if 'usage' in result:
                    self.usage.append(result['usage'])
                    
                return completion
            else:
                raise Exception("Invalid response format: 'choices' not found or empty")

        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")


class LMEngine_async:
    """
    Asynchronous OpenRouter Language Model integration.
    
    This class uses aiohttp to make asynchronous calls to the OpenRouter API.
    Use it as an async context manager to ensure proper session cleanup.
    """
    def __init__(self, lm_api: str, model: str, 
                 site_url: str = "https://your-website.com", 
                 site_name: str = "YourSiteName",
                 extra_kwarg: dict = None):
        if not lm_api:
            raise ValueError("API key is required")
        if not model:
            raise ValueError("Model name is required")
            
        self.api_key = lm_api
        self.model = model
        self.site_url = site_url
        self.site_name = site_name
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.extra_kwarg = extra_kwarg or {}
        self.usage = []
        self.history = []
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "Content-Type": "application/json",
        }
        self.session = None
    
    async def __aenter__(self):
        if self.session is not None:
            raise RuntimeError("Session already exists")
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def __call__(self, prompt: str) -> dict:
        """
        Make an asynchronous API call to the OpenRouter endpoint using the provided prompt.
        
        Args:
            prompt (str): The user prompt to be sent to the API.
            extra_payload (dict, optional): Additional fields to merge into the JSON payload.
            
        Returns:
            dict: The JSON response from the OpenRouter API.
            
        Raises:
            Exception: If the API call fails or returns an error.
        """
        if not self.session:
            raise RuntimeError("Session not initialized - use async with")
            
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
            
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "provider": {
                "data_collection": "deny",
                "sort": "throughput",
                "allow_fallbacks": True,
                "require_parameters": False
            }
        }

        if self.extra_kwarg:
            payload.update(self.extra_kwarg)
        
        try:
            async with self.session.post(self.base_url, headers=self.headers, json=payload, timeout=100) as response:
                
                if response.status != 200:
                    try:
                        error_response = await response.json()
                        error_msg = error_response.get("error", {}).get("message", f"HTTP error: {response.status}")
                    except Exception:
                        error_msg = f"HTTP error: {response.status}"
                    raise Exception(f"API error: {error_msg}")
                
                result = await response.json()
                # Check if response contains error
                if 'error' in result:
                    error_msg = result.get('error', {}).get('message', 'Unknown error occurred')
                    raise Exception(f"API error: {error_msg}")
                # Extract completion text from response
                if 'choices' in result and len(result['choices']) > 0:
                    completion = result['choices'][0].get('message', {}).get('content', '').strip()
                    
                    # Track token usage if available
                    if 'usage' in result:
                        self.usage.append(result['usage'])
                    
                    return completion
                else:
                    raise Exception("Invalid response format: 'choices' not found or empty")
        except aiohttp.ClientError as e:
            raise Exception(f"Request failed: {str(e)}")
        except asyncio.TimeoutError:
            raise Exception("Request timed out")
        

def quick_chat(query: str, lm_api: str, model:str, extras = {}) -> str:
    """
    Perform a quick chat query using the specified model.
    
    Args:
        query (str): The input query string.
        lm_api (str): The API key for the language model
        model (str, optional): The model to use for the chat.
        extras (dict, optional): Additional kwargs to override defaults
        
    Returns:
        str: The response from the chat model.
    """
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")
    if not lm_api:
        raise ValueError("API key is required")
        
    lm_kwarg = default_llm_kwarg.copy()
    lm_kwarg.update(extras)

    with LMEngine(model=model, lm_api=lm_api, extra_kwarg=lm_kwarg) as tog_chat:
        return tog_chat(query)


async def a_quick_chat(query: str, lm_api: str, model: str, extras = {}) -> str:
    """
    Perform an async quick chat query using the specified model.
    
    Args:
        query (str): The input query string.
        lm_api (str): The API key for the language model
        model (str, optional): The model to use for the chat.
        extras (dict, optional): Additional kwargs to override defaults
        
    Returns:
        str: The response from the chat model.
    """
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")
    if not lm_api:
        raise ValueError("API key is required")
        
    lm_kwarg = default_llm_kwarg.copy()
    lm_kwarg.update(extras)
    async with LMEngine_async(model=model, lm_api=lm_api, extra_kwarg=lm_kwarg) as tog_chat:
        return await tog_chat(query)


async def process_batch(
    batch: List[Tuple[int, Any]], 
    async_func: Callable[[Any], Awaitable[Any]],
    func_args: Optional[dict] = None,
    timeout: Optional[float] = None
) -> List[Tuple[int, Union[Any, Exception]]]:
    """
    Process a batch of items with timeout handling.
    
    Args:
        batch: List of (index, item) tuples to process
        async_func: Async function to call for each item
        func_args: Optional arguments to pass to async_func
        timeout: Optional timeout in seconds for the entire batch
    
    Returns:
        List of tuples containing (index, result) or (index, Exception) for failed items
    
    Raises:
        asyncio.TimeoutError: If the batch processing exceeds the timeout
    """
    if not batch:
        raise ValueError("Batch cannot be empty")
    if not async_func:
        raise ValueError("async_func is required")
        
    func_args = func_args or {}
    tasks = [asyncio.create_task(async_func(item, **func_args)) for _, item in batch]
    
    try:
        if timeout:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        return [
            (index, result if not isinstance(result, Exception) else result)
            for (index, _), result in zip(batch, results)
        ]
        
    except asyncio.TimeoutError:
        # Cancel pending tasks
        for task in tasks:
            task.cancel()
        
        # Ensure all tasks are properly cleaned up
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Return timeout errors for all items
        return [(index, asyncio.TimeoutError(f"Batch timeout after {timeout}s"))
                for index, _ in batch]


async def process_in_batches(
    items: List[Any],
    batch_size: int,
    async_func: Callable[[Any], Awaitable[Any]],
    func_args: Optional[dict] = None,
    batch_timeout: Optional[float] = None,
    sleep_time: Optional[int] = None
) -> List[Tuple[int, Union[Any, Exception]]]:
    """
    Process a list of items in batches with timeout control.
    
    Args:
        items: List of items to process
        batch_size: Number of items to process in each batch
        async_func: Async function to call for each item
        func_args: Optional arguments to pass to async_func
        batch_timeout: Optional timeout for each individual batch
    
    Returns:
        List of tuples containing (index, result) or (index, Exception) for failed items
        Results are returned in the original order of items
    """
    if not items:
        raise ValueError("Items list cannot be empty")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if not async_func:
        raise ValueError("async_func is required")
        
    func_args = func_args or {}
    indexed_items = list(enumerate(items))
    all_results = []
    
    for i in range(0, len(indexed_items), batch_size):
        batch = indexed_items[i:i + batch_size]
        
        # Process batch with timeout
        try:
            batch_results = await process_batch(
                batch,
                async_func,
                func_args,
                timeout=batch_timeout
            )
            all_results.extend(batch_results)
        except asyncio.TimeoutError as e:
            print(f"Timeout occurred while processing batch starting at index {i}: {e}")
        except Exception as e:
            print(f"Error occurred while processing batch starting at index {i}: {e}")
        if sleep_time:
            await asyncio.sleep(sleep_time)
    # Sort results by original index
    return sorted(all_results, key=lambda x: x[0])

