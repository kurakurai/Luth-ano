import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_dataset, Dataset as HFDataset
from typing import Dict, List, Any
import openai
from huggingface_hub import create_repo


class DatasetBuilder:
    """Class to build a dataset using French prompts."""

    SYS_PROMPT = """
    You are a dataset quality assistant. Evaluate the question-answer pair below.

    ## Return **False** if **any** of the following apply:

    1. **Code-Related**
    - Programming questions or answers
    - Code snippets or syntax
    - Mentions of languages, libraries, tools
    - Debugging or optimization

    2. **Tool Calling Content**
    - Describes or requests use of external tools, APIs, or systems
    - Includes function calls, command-line usage, API requests, or tool invocation logic
    - Involves configuring or troubleshooting external tools (e.g., databases, IDEs, browsers, CLIs)

    3. **Logical Errors**
    - Contradictions, invalid reasoning, factual errors

    4. **French Grammar Errors**
    - Wrong conjugations, tense, gender
    - Bad structure, spelling, accents

    ## Respond only with: `True` or `False`
    """

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
                    max_tokens=10,
                    top_p=0.8,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception:
                continue
        raise RuntimeError("API call failed after 3 attempts")

    def generate(self, sample) -> List[Dict[str, Any]]:

        text = (
            "QUESTION:\n"
            + sample["prompt"]
            + "\n\n"
            + "ANSWER:\n"
            + sample["accepted_completion"]
        )
        status = self._make_api_call(self._build_api_messages(text))

        if "true" in status.lower():
            return [
                {"role": "user", "content": sample["prompt"]},
                {"role": "assistant", "content": sample["accepted_completion"]},
            ]
        elif "false" in status.lower():
            return None
        else:
            raise ValueError(
                f"Unexpected response from model: {status}. Expected 'True' or 'False'."
            )


def get_client(base_url: str, api_key: str) -> openai.Client:
    """Get a client for the model API."""
    return openai.Client(base_url=base_url, api_key=api_key)


def generation_pipeline(
    dataset, client: openai.Client, model_name: str, num_workers: int
):
    """Generate a dataset."""
    ds_builder = DatasetBuilder(client, model_name)
    generated = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(ds_builder.generate, sample) for sample in dataset]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Generating responses"
        ):
            try:
                result = future.result()
                if result:
                    generated.append({"messages": result})
            except Exception as e:
                print(f"Error: {e}")
    print(f"Generated dataset: {len(generated)} samples")
    return generated


def main(args):
    # Load dataset
    dataset = load_dataset(args.data_repo, split="train")

    # Filter by bad_entry == False
    filtered_dataset = dataset.filter(lambda x: x["bad_entry"] == False)
    print(
        f"Filtered dataset: {len(filtered_dataset)} samples (from {len(dataset)} original)"
    )

    # Generate samples
    client = get_client(args.base_url, args.api_key)

    # Generate answers with the translated prompts in French
    generated = generation_pipeline(
        filtered_dataset, client, args.model_name, args.num_workers
    )

    # Ensure HF repo exists
    create_repo(repo_id=args.hf_repo, repo_type="dataset", exist_ok=True)

    # Push to Hub
    hf_ds = HFDataset.from_list(generated)
    hf_ds.push_to_hub(repo_id=args.hf_repo, private=True)

    print(f"Pushed to {args.hf_repo} with {len(generated)} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data and push to HF.")
    parser.add_argument(
        "--data_repo",
        type=str,
        default="legmlai/openhermes-fr",
    )
    parser.add_argument(
        "--num_workers", type=int, default=128, help="Number of workers for generation."
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
        default="/openhermes-fr-filtered",
        help="Your Hugging Face repository name to push the created dataset.",
    )
    args = parser.parse_args()
    main(args)
