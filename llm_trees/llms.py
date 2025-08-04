import os

import anthropic
import vertexai
import vertexai.preview.generative_models as generative_models
from dotenv import load_dotenv
from openai import OpenAI
from vertexai.generative_models import GenerativeModel

from llm_trees import prompts
from .config import Config
import requests

# Load the environment variables from .env file (e.g. OPENAI_API_KEY and GOOGLE_CLOUD_PROJECT)
load_dotenv()



def generate_local_llm_tree(config: Config):
    if not config.force_decision_tree:
        prompt = prompts.get_free_prompt(config)
    elif config.llm_dialogue:
        first_prompt = prompts.get_first_prompt(config)
        second_prompt = prompts.get_second_prompt(config)
    else:
        prompt = prompts.get_full_prompt(config)


    #Todo: Define the base URL and endpoints for the local LLM server
    base_url = "https://f2ki-h100-1.f2.htw-berlin.de:11435"
    
    chat_url = f"{base_url}/api/chat"
    generate_url = f"{base_url}/api/generate"

    model_name = config.method

    if not model_name:
        raise ValueError("No model specified in config.method")

    if not config.llm_dialogue:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "stream": False
        }

        response = requests.post(chat_url, json=payload, verify=True)
        response.raise_for_status()
        return response.json()["message"]["content"]

    else:
        conversation = [{"role": "user", "content": first_prompt}]
        payload1 = {
            "model": model_name,
            "messages": conversation,
            "temperature": config.temperature,
            "stream": False
        }

        resp1 = requests.post(chat_url, json=payload1, verify=True)
        resp1.raise_for_status()
        assistant_msg = resp1.json()["message"]["content"]
        conversation.append({"role": "assistant", "content": assistant_msg})
        conversation.append({"role": "user", "content": second_prompt})

        payload2 = {
            "model": model_name,
            "messages": conversation,
            "temperature": config.temperature,
            "stream": False
        }

        resp2 = requests.post(chat_url, json=payload2, verify=True)
        resp2.raise_for_status()
        return assistant_msg + 3 * "\n" + 25 * "#" + 3 * "\n" + resp2.json()["message"]["content"]


