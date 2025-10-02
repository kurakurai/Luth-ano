import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_dataset, Dataset as HFDataset
from typing import Dict, List, Any
import openai
from huggingface_hub import create_repo

# Build a dataset in French based on an existing dataset by translating the input prompts and re generating the responses in French.


class DatasetBuilder:
    """Class to build a dataset using French prompts."""

    SYS_PROMPT = """You are a helpful assistant that responds in French."""

    def __init__(
        self,
        client: openai.Client,
        model_name: str,
    ):
        self.client = client
        self.model_name = model_name

    def _build_api_messages(self, question: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": self.SYS_PROMPT},
            {"role": "user", "content": question},
        ]

    def _make_api_call(self, messages: List[Dict[str, Any]]) -> str:
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048,
                    top_p=0.8,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception:
                continue
        raise RuntimeError("API call failed after 3 attempts")

    def generate(self, prompt) -> List[Dict[str, Any]]:
        question = prompt
        answer = self._make_api_call(self._build_api_messages(question))
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]


class PromptTranslation:
    """Class to translate prompts into French using a model API."""

    SYS_PROMPT = """You are a professional French translator. Translate English text into natural, accurate French.

    REQUIREMENTS:
    - Preserve exact meaning, tone, and register of the original
    - Use natural French syntax and idiomatic expressions
    - Maintain all formatting (markdown, HTML, special characters, structure)
    - Keep technical terms, code snippets, and proper nouns appropriately handled
    - Ensure grammatical correctness and contemporary French usage

    OUTPUT: Only the translated French text in identical format to the input."""

    def __init__(
        self,
        client: openai.Client,
        model_name: str,
    ):
        self.client = client
        self.model_name = model_name

    def _build_api_messages(self, prompt: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": self.SYS_PROMPT},
            {"role": "user", "content": "PROMPT TO TRANSLATE:\n" + prompt},
        ]

    def _make_api_call(self, messages: List[Dict[str, Any]]) -> str:
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=0.8,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception:
                continue
        raise RuntimeError("API call failed after 3 attempts")

    def translate(self, dataset_sample) -> List[Dict[str, Any]]:
        prompt = dataset_sample["messages"][0]["content"]
        translation = self._make_api_call(self._build_api_messages(prompt))
        return translation


def get_client(base_url: str, api_key: str) -> openai.Client:
    """Get a client for the model API."""
    return openai.Client(base_url=base_url, api_key=api_key)


def translation_pipeline(
    dataset, client: openai.Client, model_name: str, num_workers: int
):
    """Translate the prompts of a dataset into French."""
    ds_builder = PromptTranslation(client, model_name)
    translated_prompts = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(ds_builder.translate, sample) for sample in dataset]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Translating prompts"
        ):
            try:
                result = future.result()
                translated_prompts.append(result)
            except Exception as e:
                print(f"Error: {e}")
    print(f"Generated dataset: {len(translated_prompts)} samples")
    return translated_prompts


def generation_pipeline(
    translated_prompts, client: openai.Client, model_name: str, num_workers: int
):
    """Generate a dataset."""
    ds_builder = DatasetBuilder(client, model_name)
    generated = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(ds_builder.generate, prompt)
            for prompt in translated_prompts
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Generating responses"
        ):
            try:
                result = future.result()
                generated.append({"messages": result})
            except Exception as e:
                print(f"Error: {e}")
    print(f"Generated dataset: {len(generated)} samples")
    return generated


def main(args):
    dataset = load_dataset(args.data_repo, split="train")

    client = get_client(args.base_url, args.api_key)

    translated_prompts = translation_pipeline(
        dataset, client, args.model_name, args.num_workers
    )

    generated = generation_pipeline(
        translated_prompts, client, args.model_name, args.num_workers
    )
    create_repo(repo_id=args.hf_repo, repo_type="dataset", exist_ok=True)


    hf_ds = HFDataset.from_list(generated)
    hf_ds.push_to_hub(repo_id=args.hf_repo, private=True)

    print(f"Pushed to {args.hf_repo} with {len(generated)} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data and push to HF.")
    parser.add_argument(
        "--data_repo",
        type=str,
        default="allenai/tulu-3-sft-personas-instruction-following",
    )
    parser.add_argument(
        "--num_workers", type=int, default=32, help="Number of workers for generation."
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the API.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key for the model API.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen3-32B",
        help="Model name to use for generation.",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="/tulu-3-persona-instruct-fr",
        help="Your Hugging Face repository name to push the created dataset.",
    )
    args = parser.parse_args()
    main(args)
