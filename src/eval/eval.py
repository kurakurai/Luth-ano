import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from patch_lighteval.patch import patch_reasoning, patch_prefix_caching
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
logging.getLogger("vllm").setLevel(logging.WARNING)
ray.init(logging_level="ERROR")


def get_tasks(task_keys):
    """
    Get the tasks based on the provided task keys.
    """
    return ",".join(task_keys)


def display_avg_metrics(results, num_runs):
    """Calculate and display average metrics across multiple runs in a beautiful table format."""
    if not results:
        print("No results to display.")
        return

    print(f"\n{'='*80}")
    if num_runs == 1:
        print(f"EVALUATION RESULTS")
    else:
        print(f"AVERAGE RESULTS ACROSS {num_runs} RUNS")
    print(f"{'='*80}")

    # Group metrics by task
    task_metrics = defaultdict(list)

    for run_idx, run_results in enumerate(results):
        for task_name, task_data in run_results.items():
            if task_name == "all":
                continue
            task_metrics[task_name].append(task_data)

    # Filter out individual subsets and keep only averages
    filtered_task_metrics = {}
    for task_name, task_runs in task_metrics.items():
        # Skip individual subsets (contain colons after the main task name)
        if ":" in task_name and not task_name.endswith("_average"):
            # Check if this is an individual subset by looking for patterns like "mmlu:subset"
            parts = task_name.split(":")
            if len(parts) > 2:  # e.g., "leaderboard:mmlu:abstract_algebra:0"
                continue

        filtered_task_metrics[task_name] = task_runs

    # Calculate averages for each task
    if not filtered_task_metrics:
        print("No valid tasks found after filtering.")
        return

    # Get all unique metrics across all tasks
    all_metrics = set()
    for task_runs in filtered_task_metrics.values():
        if task_runs:
            for key in task_runs[0].keys():
                if not key.endswith("_stderr"):
                    all_metrics.add(key)

    all_metrics = sorted(list(all_metrics))

    # Sort tasks to put _average tasks first
    sorted_tasks = sorted(
        filtered_task_metrics.items(),
        key=lambda x: (
            0 if "_average" in x[0] else 1,  # _average tasks first
            x[0],  # then alphabetically
        ),
    )

    # Create table header
    print(f"\n|{'Task':<54}|{'Metric':<23}|{'Value':<8}|")
    print(f"|{'-'*54}|{'-'*23}|{'-'*8}|")

    # Calculate and display averages for each task
    for task_name, task_runs in sorted_tasks:
        if not task_runs:
            continue

        # Get all metric keys from the first run
        all_keys = task_runs[0].keys()
        base_keys = sorted([k for k in all_keys if not k.endswith("_stderr")])

        for key in base_keys:
            values = []

            for run_data in task_runs:
                if key in run_data:
                    values.append(run_data[key])

            if values:
                mean = np.mean(values)

                # Truncate task name if too long
                display_task = (
                    task_name[:51] + "..." if len(task_name) > 54 else task_name
                )

                print(f"|{display_task:<54}|{key:<23}|{mean:<8.4f}|")


def main(args):
    """
    Main function to run the evaluation pipeline with vLLM backend.
    """
    # Ensure NLTK punkt tokenizer is available
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            print(f"Downloading NLTK {resource} tokenizer...")
            nltk.download(resource)

    # Loading configuration from YAML file
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

    if not extras_yaml.get("enable_prefix_caching"):
        patch_prefix_caching()

    # Prepare model configuration
    config_kwargs = dict(model_yaml)
    config_kwargs["generation_parameters"] = GenerationParameters(
        **model_parameters_yaml
    )

    model_config = VLLMModelConfig(**config_kwargs)
    # Set up pipeline parameters
    tasks_path = Path(f"src/eval/tasks.py")
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        custom_tasks_directory=tasks_path,
        use_chat_template=extras_yaml.get("use_chat_template", True),
        system_prompt=extras_yaml.get("system_prompt", ""),
    )

    # Get the tasks to evaluate
    tasks = get_tasks(tasks_yaml)

    # Initialize the evaluation tracker
    evaluation_tracker = EvaluationTracker(
        output_dir=extras_yaml.get("output_dir", "results/"),
        save_details=extras_yaml.get("save_details", True),
        push_to_hub=extras_yaml.get("push_to_hub", False),
    )

    # Store results from all runs
    all_run_results = []
    num_runs = extras_yaml.get("num_runs", 1)

    # Create the pipeline and run the evaluation for the specified number of runs
    for run_idx in range(num_runs):
        print(f"\nStarting run {run_idx + 1}/{num_runs}")
        pipeline = Pipeline(
            tasks=tasks,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model_config=model_config,
            enable_thinking=extras_yaml.get("enable_thinking", False),
        )
        pipeline.evaluate()

        # Store results from this run
        run_results = (
            pipeline.evaluation_tracker.metrics_logger.metric_aggregated.copy()
        )
        all_run_results.append(run_results)

        print(f"Completed run {run_idx + 1}/{num_runs}")

    pipeline.save_and_push_results()
    # pipeline.show_results() # No need to show the results, we will display the average metrics

    # Calculate average metrics across all runs (including single run)
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
