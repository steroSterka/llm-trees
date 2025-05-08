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

# gpt_4o_model = "chatgpt-4o-latest"
gpt_4o_model = "gpt-4o-2024-05-13"
gpt_o1_model = "o1-preview"
# gemini_model = "gemini-1.5-pro-002"
gemini_model = "gemini-1.5-pro-001"
# claude_model = "claude-3-5-sonnet-20241022"
claude_model = "claude-3-5-sonnet-20240620"



def generate_gpt_tree(config):
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    if config.method == "gpt-4o":
        gpt_model = gpt_4o_model
    elif config.method == "gpt-o1":
        gpt_model = gpt_o1_model
    else:
        raise ValueError(f"Unknown gpt model: {config.method}")

    if not config.force_decision_tree:
        full_prompt = prompts.get_free_prompt(config)
        response = client.chat.completions.create(
            model=gpt_model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=config.temperature,
            seed=config.iter,
        )

        return response.choices[0].message.content

    if config.llm_dialogue:
        first_prompt = prompts.get_first_prompt(config)

        if config.force_decision_tree:
            second_prompt = prompts.get_second_prompt(config)
        else:
            raise NotImplementedError("GPT does not support free response")

        # Initialize conversation history
        messages = [
            {"role": "user", "content": first_prompt},
        ]

        first_response = client.chat.completions.create(
            model=gpt_model,
            messages=messages,
            temperature=config.temperature,
            seed=config.iter,
        )

        # Add assistant response to the conversation history
        messages.append({"role": "assistant", "content": first_response.choices[0].message.content})

        # User's next input
        messages.append({"role": "user", "content": second_prompt})

        # Second response
        second_response = client.chat.completions.create(
            model=gpt_model,
            messages=messages,
            temperature=config.temperature,
            seed=config.iter,
        )

        response = first_response.choices[0].message.content + 3 * "\n" + 25 * "#" + 3 * "\n" + second_response.choices[
            0].message.content
        return response
    else:
        full_prompt = prompts.get_full_prompt(config)

        response = client.chat.completions.create(
            model=gpt_model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=config.temperature,
            seed=config.iter,
        )

        return response.choices[0].message.content


def generate_claude_tree(config: Config):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    if not config.force_decision_tree:
        full_prompt = prompts.get_free_prompt(config)

        return client.messages.create(
            model=claude_model,
            max_tokens=1024,
            temperature=config.temperature,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )

    if config.llm_dialogue:
        first_prompt = prompts.get_first_prompt(config)
        if config.force_decision_tree:
            second_prompt = prompts.get_second_prompt(config)
        else:
            raise NotImplementedError("Claude does not support free response")

        conversation = [{"role": "user", "content": first_prompt}]

        first_response = client.messages.create(
            model=claude_model,
            max_tokens=512,
            temperature=config.temperature,
            messages=conversation
        )

        conversation.append({"role": "assistant", "content": first_response.content[0].text})
        conversation.append({"role": "user", "content": second_prompt})

        second_response = client.messages.create(
            model=claude_model,
            max_tokens=512,
            temperature=config.temperature,
            messages=conversation
        )

        return first_response.content[0].text + 3 * "\n" + 25 * "#" + 3 * "\n" + second_response.content[0].text

    else:
        full_prompt = prompts.get_full_prompt(config)

        return client.messages.create(
            model=claude_model,
            max_tokens=1024,
            temperature=config.temperature,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )


def generate_gemini_tree(config: Config):
    generation_config = {
        "max_output_tokens": 1024,
        "temperature": config.temperature,
        "top_p": 0.95,
    }

    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    vertexai.init(project=os.environ.get('GOOGLE_CLOUD_PROJECT'), location="us-central1")
    model = GenerativeModel(gemini_model)

    if not config.force_decision_tree:
        full_prompt = prompts.get_free_prompt(config)

        responses = model.generate_content(
            contents=[full_prompt],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,
        )

        output = ""
        for response in responses:
            output += response.text

        return output

    if config.llm_dialogue:
        first_prompt = prompts.get_first_prompt(config)
        if config.force_decision_tree:
            second_prompt = prompts.get_second_prompt(config)
        else:
            raise NotImplementedError("Gemini does not support free response")

        chat = model.start_chat(response_validation=False)

        first_response = chat.send_message(first_prompt)
        second_response = chat.send_message(second_prompt)

        return first_response.text + 3 * "\n" + 25 * "#" + 3 * "\n" + second_response.text

    else:
        full_prompt = prompts.get_full_prompt(config)

        responses = model.generate_content(
            contents=[full_prompt],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,
        )

        output = ""
        for response in responses:
            output += response.text

        return output



def generate_local_llm_tree(config: Config):
    if not config.force_decision_tree:
        prompt = prompts.get_free_prompt(config)
    elif config.llm_dialogue:
        first_prompt = prompts.get_first_prompt(config)
        second_prompt = prompts.get_second_prompt(config)
    else:
        prompt = prompts.get_full_prompt(config)

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


