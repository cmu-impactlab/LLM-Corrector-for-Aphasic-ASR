import os
import random
import sys
from enum import Enum
from typing import Callable, Dict, List

from utils.file_utils import ASRSystem, get_asr_file_path


current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from prompt import (
        pick_data_driven_targeted_sentences,
        pick_exhaustive_phoneme_sentences,
        pick_random_error_sentences,
    )
except ImportError as e:
    print("FATAL ERROR: Could not import picker functions from 'prompt.py'.")
    print(f"Ensure 'prompt.py' is in the project root ('{project_root}').")
    print(f"Details: {e}")
    sys.exit(1)


class ExampleStrategy(Enum):
    DATA_DRIVEN = "data_driven"
    EXHAUSTIVE_PHONEME = "exhaustive_phoneme"
    RANDOM_ERROR = "random_error"


STRATEGY_FUNCTIONS: Dict[ExampleStrategy, Callable] = {
    ExampleStrategy.DATA_DRIVEN: pick_data_driven_targeted_sentences,
    ExampleStrategy.EXHAUSTIVE_PHONEME: pick_exhaustive_phoneme_sentences,
    ExampleStrategy.RANDOM_ERROR: pick_random_error_sentences,
}


def get_available_strategies() -> List[str]:
    return [strategy.value for strategy in ExampleStrategy]


def select_strategy_interactively() -> ExampleStrategy:
    print("\nAvailable Example Selection Strategies:")
    print("=" * 50)

    strategies = list(ExampleStrategy)
    for i, strategy in enumerate(strategies, 1):
        strategy_name = strategy.value.replace("_", " ").title()
        print(f"{i}. {strategy_name}")

    print("=" * 50)

    while True:
        try:
            choice = input(f"Select strategy (1-{len(strategies)}): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(strategies):
                    selected = strategies[idx]
                    print(f"Selected: {selected.value.replace('_', ' ').title()}")
                    return selected
            print(f"Invalid choice. Enter a number between 1 and {len(strategies)}.")
        except KeyboardInterrupt:
            print("\nSelection cancelled.")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")


def get_strategy_from_name(strategy_name: str) -> ExampleStrategy:
    try:
        return ExampleStrategy(strategy_name.lower())
    except ValueError:
        available = get_available_strategies()
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available strategies: {available}")


def get_max_examples_from_json(file_base: str, num_sentences_to_pick: int):
    max_examples_folder = "data/examples_max"
    json_filename = f"{file_base}_max_examples.json"
    json_path = os.path.join(max_examples_folder, json_filename)

    if not os.path.exists(json_path):
        print(f"  Max examples JSON not found: {json_path}")
        return [], [], ""

    try:
        import json
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        asr_examples = data.get("asr_examples", [])
        gt_examples = data.get("gt_examples", [])

        if not asr_examples or not gt_examples:
            print(f"  No examples found in {json_path}")
            return [], [], ""

        if len(asr_examples) != len(gt_examples):
            print(
                f"  Mismatched example counts in {json_path}: "
                f"ASR={len(asr_examples)}, GT={len(gt_examples)}"
            )
            return [], [], ""

        num_to_take = min(num_sentences_to_pick, len(asr_examples))
        selected_asr = asr_examples[:num_to_take]
        selected_gt = gt_examples[:num_to_take]

        example_pairs_string = "\n".join(
            f"{asr.strip()}  ➜  {gt.strip()}" for asr, gt in zip(selected_asr, selected_gt)
        )

        return selected_asr, selected_gt, example_pairs_string

    except Exception as e:
        print(f"  Error loading examples from {json_path}: {e}")
        return [], [], ""


def generate_dynamic_examples(
    file_base: str,
    num_sentences_to_pick: int,
    ground_truth_folder: str,
    asr_folder: str,
    strategy: ExampleStrategy = ExampleStrategy.DATA_DRIVEN,
    asr_system: ASRSystem = ASRSystem.AZURE_SPEECH,
) -> tuple[list[str], list[str], str]:
    if num_sentences_to_pick <= 0:
        return [], [], ""

    truth_path = os.path.join(ground_truth_folder, f"{file_base}.txt")
    asr_path = get_asr_file_path(file_base, asr_system)

    if not os.path.exists(truth_path) or not os.path.exists(asr_path):
        print(f"  Source files for {file_base} not found. Cannot generate examples.")
        print(f"      GT path: {truth_path}")
        print(f"      ASR path: {asr_path}")
        return [], [], ""

    picker_function = STRATEGY_FUNCTIONS.get(strategy)
    if not picker_function:
        print(f"  Unknown strategy '{strategy}'. Using default data_driven strategy.")
        picker_function = pick_data_driven_targeted_sentences

    try:
        print(
            f"  Using {strategy.value.replace('_', ' ').title()} strategy "
            f"for {file_base} with {asr_system.value}"
        )
        asr_sents, gt_sents = picker_function(
            truth_path,
            asr_path,
            num_sentences_to_pick=num_sentences_to_pick,
        )
    except Exception as e:
        print(f"  Error during example generation for {file_base} using {strategy.value}: {e}")
        import traceback
        traceback.print_exc()
        return [], [], ""

    if not asr_sents or not gt_sents:
        print(f"  No example pairs were generated for {file_base} using {strategy.value}.")
        return [], [], ""

    combined_pairs = list(zip(asr_sents, gt_sents))
    random.shuffle(combined_pairs)

    if not combined_pairs:
        return [], [], ""

    selected_asr_sentences, selected_gt_sentences = zip(*combined_pairs)
    example_pairs_string = "\n".join(
        f"{asr.strip()}  ➜  {gt.strip()}"
        for asr, gt in zip(selected_asr_sentences, selected_gt_sentences)
    )

    print(
        f"  Generated {len(selected_asr_sentences)} example pairs "
        f"using {strategy.value} with {asr_system.value}"
    )
    return list(selected_asr_sentences), list(selected_gt_sentences), example_pairs_string


def generate_dynamic_examples_with_strategy_selection(
    file_base: str,
    num_sentences_to_pick: int,
    ground_truth_folder: str,
    asr_folder: str,
    strategy_name: str = None,
) -> tuple[list[str], list[str], str]:
    if strategy_name:
        try:
            strategy = get_strategy_from_name(strategy_name)
        except ValueError as e:
            print(f"{e}")
            strategy = select_strategy_interactively()
    else:
        strategy = select_strategy_interactively()

    return generate_dynamic_examples(
        file_base=file_base,
        num_sentences_to_pick=num_sentences_to_pick,
        ground_truth_folder=ground_truth_folder,
        asr_folder=asr_folder,
        strategy=strategy,
    )