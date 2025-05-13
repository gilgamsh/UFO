from typing import Any, Dict, List, Optional, Tuple, Union

import openai # Main import
from openai import (
    OpenAI as OpenAIClient, # Using an alias for clarity if needed, otherwise OpenAI is fine
    APIError,
    APIStatusError,
    APITimeoutError,
    APIConnectionError,
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)
import traceback # For detailed error logging

from ufo.llm.base import BaseService
from ufo.utils import print_with_color # Assuming this path is correct


class OpenRouterService(BaseService):
    """
    A service class for OpenRouter, leveraging its OpenAI-compatible API.
    """

    def __init__(self, config: Dict[str, Any], agent_type: str):
        """
        Initialize the OpenRouter service.
        :param config: The configuration.
        :param agent_type: The agent type.
        """
        super().__init__(config, agent_type)  # Call BaseService's __init__
        self.config_llm = config[agent_type]
        self.config = config
        self.api_type = "openrouter"  # For cost estimation and logging
        self.model = self.config_llm.get("API_MODEL")
        if not self.model:
            raise ValueError("API_MODEL for OpenRouter must be specified in the configuration.")
        
        # Prices should be structured in config to be found by get_cost_estimator
        # e.g., config["PRICES"]["openrouter"]["google/gemini-2.0-flash-exp:free"] = {"prompt": 0, "completion": 0}
        self.prices = self.config.get("PRICES", {})
        self.max_retry = self.config.get("MAX_RETRY", 3)
        self.timeout = self.config.get("TIMEOUT", 120) # Increased default timeout for potentially slower models

        self.api_key = self.config_llm.get("API_KEY")
        if not self.api_key:
            raise ValueError("API_KEY for OpenRouter must be specified in the configuration.")

        self.client = OpenAIClient(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            max_retries=self.max_retry,
            timeout=self.timeout,
        )
        # Optional headers for OpenRouter analytics/leaderboards.
        # Replace YOUR_SITE_URL and YOUR_APP_NAME with your actual values if you want to use them.
        # self.client.default_headers["HTTP-Referer"] = "YOUR_SITE_URL"
        # self.client.default_headers["X-Title"] = "YOUR_APP_NAME"

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        n: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Tuple[List[Optional[str]], Optional[float]]:
        """
        Generates completions for a given list of messages using OpenRouter.
        :param messages: The list of messages (OpenAI format).
        :param n: The number of completions to generate.
        :param temperature: Controls the randomness.
        :param max_tokens: The maximum number of tokens in the generated completion.
        :param top_p: Controls the diversity via nucleus sampling.
        :param stream: Whether to stream the response.
        :param kwargs: Additional keyword arguments to pass to the API.
        :return: A tuple containing a list of generated completions (text or None) and the estimated cost.
        """

        temperature = (
            temperature if temperature is not None else self.config.get("TEMPERATURE", 0.7)
        )
        max_tokens = (
            max_tokens if max_tokens is not None else self.config.get("MAX_TOKENS", 2048)
        )
        top_p = top_p if top_p is not None else self.config.get("TOP_P", 1.0)

        if n != 1 and stream:
            print_with_color(
                "Warning: Streaming with n > 1 may have untested behavior with OpenRouter. Proceeding with n=1 for stream.", 
                "yellow"
            )
            # OpenAI SDK's stream handling for n>1 can be tricky; simplifying to n=1 for streaming if issues arise,
            # or ensure the consuming code correctly handles multiple stream choices.
            # For now, we'll pass n, but this is a common point of complexity.
            # If strict n=1 for streaming is desired, set n = 1 here if stream else n.

        try:
            if stream:
                response_stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages, # type: ignore
                    n=n,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=True,
                    stream_options={"include_usage": True},
                    **kwargs,
                )

                collected_content: List[Optional[str]] = [None] * n
                usage = None

                for chunk in response_stream:
                    if chunk.usage:
                        usage = chunk.usage
                        # In OpenAI's spec, the usage object is the final part of the stream when include_usage is true.
                        break 
                    
                    if chunk.choices:
                        for choice_chunk in chunk.choices:
                            idx = choice_chunk.index
                            delta = choice_chunk.delta
                            if delta and delta.content is not None:
                                if collected_content[idx] is None:
                                    collected_content[idx] = delta.content
                                else:
                                    collected_content[idx] = (collected_content[idx] or "") + delta.content
                
                if not usage:
                    print_with_color("Warning: Usage data not retrieved from stream. Cost calculation may be inaccurate.", "yellow")
                    prompt_tokens = 0
                    completion_tokens = 0
                else:
                    prompt_tokens = usage.prompt_tokens
                    completion_tokens = usage.completion_tokens
                
                cost = self.get_cost_estimator(
                    self.api_type, self.model, self.prices, prompt_tokens, completion_tokens
                )
                return collected_content, cost

            else:  # Not streaming
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages, # type: ignore
                    n=n,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=False,
                    **kwargs,
                )

                # Safely handle cases where response.choices may be None or not iterable
                choices = getattr(response, "choices", []) or []  # type: ignore
                texts: List[Optional[str]] = [getattr(getattr(choice, "message", None), "content", None) for choice in choices]

                # Safely retrieve usage information
                usage = getattr(response, "usage", None)  # type: ignore
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                
                cost = self.get_cost_estimator(
                    self.api_type, self.model, self.prices, prompt_tokens, completion_tokens
                )
                return texts, cost

        except APIStatusError as e:
            error_message = f"OpenRouter API request failed with status {e.status_code}: {e.message or e.body}"
            print_with_color(error_message, "red")
            raise Exception(error_message) from e
        except APITimeoutError as e:
            error_message = f"OpenRouter API request timed out: {e}"
            print_with_color(error_message, "red")
            raise Exception(error_message) from e
        except APIConnectionError as e:
            error_message = f"OpenRouter API request failed to connect: {e}"
            print_with_color(error_message, "red")
            raise Exception(error_message) from e
        except BadRequestError as e:
            error_message = f"OpenRouter API request was invalid: {e.message or e.body}"
            print_with_color(error_message, "red")
            raise Exception(error_message) from e
        except AuthenticationError as e:
            error_message = f"OpenRouter API authentication failed: {e.message or e.body}"
            print_with_color(error_message, "red")
            raise Exception(error_message) from e
        except PermissionDeniedError as e:
            error_message = f"OpenRouter API request was not permitted: {e.message or e.body}"
            print_with_color(error_message, "red")
            raise Exception(error_message) from e
        except RateLimitError as e:
            error_message = f"OpenRouter API request exceeded rate limit: {e.message or e.body}"
            print_with_color(error_message, "red")
            raise Exception(error_message) from e
        except APIError as e: # Catch-all for other OpenAI SDK API errors
            error_message = f"OpenRouter API returned an API Error: {e.message or e.body}"
            print_with_color(error_message, "red")
            raise Exception(error_message) from e
        except Exception as e:
            error_message = f"An unexpected error occurred with OpenRouter: {e}"
            print_with_color(error_message, "red")
            print_with_color(traceback.format_exc(), "red")
            raise Exception(error_message) from e
