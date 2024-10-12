import os
import requests
from typing import List, Optional, Union, Dict, Any

class ClydeAI:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("CLYDE_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Set it as an environment variable CLYDE_API_KEY or pass it to the ClydeAI constructor.")
        self.base_url = "https://clydeai.org/v1"

    def ChatCompletion(self):
        return ChatCompletion(self)

class ChatCompletion:
    def __init__(self, client: ClydeAI):
        self.client = client

    def create(self,
               model: str,
               messages: List[Dict[str, str]],
               max_tokens: int = 4096,
               temperature: float = 1.0,
               top_p: float = 1.0,
               n: int = 1,
               stream: bool = False,
               stop: Optional[Union[str, List[str]]] = None,
               presence_penalty: float = 0,
               frequency_penalty: float = 0,
               logit_bias: Optional[Dict[str, float]] = None,
               user: Optional[str] = None
               ) -> 'ClydeResponse':

        endpoint = f"{self.client.base_url}/chat/completions"
        headers = {
            "Authorization": f"{self.client.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "user": user
        }

        response = None
        try:
            response = requests.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            return ClydeResponse(response.json())
        except requests.exceptions.RequestException as e:
            if response is not None:
                if response.status_code == 400:
                    raise ValueError(f"Invalid request: {response.text}")
                elif response.status_code == 401:
                    raise ValueError("Authentication error with Clyde AI API")
                elif response.status_code == 503:
                    raise ConnectionError("Failed to connect to Clyde AI API")
                elif response.status_code == 429:
                    raise ValueError("Clyde AI API rate limit exceeded")
            raise Exception(f"An unexpected error occurred: {str(e)}")

class ClydeResponse:
    def __init__(self, response_data: Dict[str, Any]):
        self._response_data = response_data
        self.id = response_data.get('id')
        self.object = response_data.get('object')
        self.created = response_data.get('created')
        self.model = response_data.get('model')
        self.choices = response_data.get('choices', [])
        self.usage = response_data.get('usage', {})

        if self.choices:
            self.message = self.choices[0].get('message', {})
            self.content = self.message.get('content')
            self.role = self.message.get('role')
            self.finish_reason = self.choices[0].get('finish_reason')
        else:
            self.message = {}
            self.content = None
            self.role = None
            self.finish_reason = None

        self.prompt_tokens = self.usage.get('prompt_tokens')
        self.completion_tokens = self.usage.get('completion_tokens')
        self.total_tokens = self.usage.get('total_tokens')
