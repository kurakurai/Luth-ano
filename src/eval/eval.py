import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from patch_lighteval.patch import patch_reasoning, patch_prefix_caching, patch_vllm_api
from lighteval.models.vllm.vllm_model import VLLMModelConfig

patch_reasoning()

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.models.utils import GenerationParameters
from pathlib import Path
import argparse
import nltk
import yaml
import logging
import numpy as np
from collections import defaultdict
import ray

# Set the root logger to only show INFO and above
logging.getLogger("lighteval").setLevel(logging.WARNING)
logging.getLogger("vllm").setLevel(logging.WARNING)

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

# Suppress progress bars and verbose output
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ray.init(logging_level="ERROR")


def get_tasks(task_keys):
    return ",".join(task_keys)


def display_avg_metrics(results, num_runs):
    if not results:
        print("No results to display.")
        return

    print(f"\n{'='*80}")
    if num_runs == 1:
        print("📊 EVALUATION RESULTS")
    else:
        print(f"📊 AVERAGE RESULTS ACROSS {num_runs} RUNS")
    print(f"{'='*80}")

    task_metrics = defaultdict(list)
    for run_results in results:
        for task_name, task_data in run_results.items():
            if task_name != "all":
                task_metrics[task_name].append(task_data)

    if not task_metrics:
        print("No valid tasks found.")
        return

    all_metrics = set()
    for task_runs in task_metrics.values():
        if task_runs:
            for key in task_runs[0].keys():
                if not key.endswith("_stderr"):
                    all_metrics.add(key)

    # Create sorted task list
    sorted_tasks = []
    if results:
        all_tasks = [
            (task_name, task_metrics[task_name])
            for task_name in results[0].keys()
            if task_name != "all" and task_name in task_metrics
        ]

        # Sort all tasks alphabetically
        sorted_tasks = sorted(all_tasks, key=lambda x: x[0])

    print(f"\n|{'Task':<54}|{'Metric':<30}|{'Value':<8}|")
    print(f"|{'-'*54}|{'-'*30}|{'-'*8}|")

    for task_name, task_runs in sorted_tasks:
        if not task_runs:
            continue

        base_keys = sorted(
            [k for k in task_runs[0].keys() if not k.endswith("_stderr")]
        )
        display_task = task_name[:51] + "..." if len(task_name) > 54 else task_name

        for i, key in enumerate(base_keys):
            values = [run_data[key] for run_data in task_runs if key in run_data]
            if values:
                mean = np.mean(values)
                task_display = display_task if i == 0 else ""
                print(f"|{task_display:<54}|{key:<30}|{mean:<8.4f}|")


def main(args):
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            print(f"Downloading NLTK {resource} tokenizer...")
            nltk.download(resource)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_yaml = config.get("model", {})
    model_parameters_yaml = config.get("model_parameters", {})
    extras_yaml = config.get("extras", {})
    tasks_yaml = config.get("tasks", [])
    os.environ["answer_token"] = extras_yaml.get("answer_token", "")

    if not extras_yaml.get("use_chat_template"):
        extras_yaml["use_chat_template"] = False
    else:
        os.environ["enable_thinking"] = str(extras_yaml.get("enable_thinking"))

    if extras_yaml.get("enable_thinking") and not extras_yaml.get("answer_token"):
        raise ValueError(
            "enable_thinking is set to True, but answer_token is not provided in the config."
        )

    if "LFM2" in model_yaml.get("model_name", ""):
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

    if not extras_yaml.get("enable_prefix_caching"):
        patch_prefix_caching()

    # Always patch vLLM API for compatibility
    patch_vllm_api()

    config_kwargs = dict(model_yaml)
    config_kwargs["generation_parameters"] = GenerationParameters(
        **model_parameters_yaml
    )
    model_config = VLLMModelConfig(**config_kwargs)

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        custom_tasks_directory=Path("src/eval/tasks.py"),
        use_chat_template=extras_yaml.get("use_chat_template", True),
        system_prompt=extras_yaml.get("system_prompt", ""),
    )

    tasks = get_tasks(tasks_yaml)
    evaluation_tracker = EvaluationTracker(
        output_dir=extras_yaml.get("output_dir", "results/"),
        save_details=extras_yaml.get("save_details", True),
        push_to_hub=extras_yaml.get("push_to_hub", False),
    )

    all_run_results = []
    num_runs = extras_yaml.get("num_runs", 1)

    for run_idx in range(num_runs):
        print(f"\n🚀 Starting evaluation run {run_idx + 1}/{num_runs}")
        pipeline = Pipeline(
            tasks=tasks,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model_config=model_config,
            enable_thinking=extras_yaml.get("enable_thinking", False),
        )
        pipeline.evaluate()

        run_results = (
            pipeline.evaluation_tracker.metrics_logger.metric_aggregated.copy()
        )
        all_run_results.append(run_results)
        print(f"✅ Completed run {run_idx + 1}/{num_runs}")

    pipeline.save_and_push_results()
    display_avg_metrics(all_run_results, num_runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluation pipeline with vLLM backend."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_config.yaml",
        help="Path to the evaluation configuration YAML file.",
    )
    args = parser.parse_args()
    main(args)
