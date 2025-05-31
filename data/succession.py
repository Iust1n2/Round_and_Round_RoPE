import json
from .data_template import succession_dataset
from typing import List, Dict, Tuple, Union, Optional, Literal
from transformers import PreTrainedTokenizer, AutoTokenizer


def generate_successor_pairs() -> Tuple[dict, dict]:
    """
    Generates the succession dataset and mapping of successors.

    Returns:
        Tuple[dict, dict]: Succession dataset and successor mapping.
    """
    succession_mapping = {}

    succession_mapping["Days"] = {
        succession_dataset["Days"][i]: succession_dataset["Days"][
            (i + 1) % len(succession_dataset["Days"])
        ]
        for i in range(len(succession_dataset["Days"]))
    }
    succession_mapping["Months"] = {
        succession_dataset["Months"][i]: succession_dataset["Months"][
            (i + 1) % len(succession_dataset["Months"])
        ]
        for i in range(len(succession_dataset["Months"]))
    }
    succession_mapping["Roman Letters"] = {
        succession_dataset["Roman Letters"][i]: succession_dataset["Roman Letters"][
            (i + 1) % len(succession_dataset["Roman Letters"])
        ]
        for i in range(len(succession_dataset["Roman Letters"]))
    }
    succession_mapping["Letters"] = {
        succession_dataset["Letters"][i]: succession_dataset["Letters"][
            (i + 1) % len(succession_dataset["Letters"])
        ]
        for i in range(len(succession_dataset["Letters"]))
    }

    # Handle non-cyclical successors for other tasks
    for task in [
        "Numbers",
        "Number words",
        "Cardinal words",
        "Day prefixes",
        "Month prefixes",
        "Arithmetic Progression",
        "Geometric Progression",
    ]:
        task_tokens = succession_dataset[task]
        succession_mapping[task] = {
            task_tokens[i]: task_tokens[i + 1] for i in range(len(task_tokens) - 1)
        }

    return succession_dataset, succession_mapping


def create_prompt(succession_dataset: dict) -> dict:
    """
    Converts the succession dataset into a single string for each task.

    Args:
        succession_dataset (dict): The succession dataset with tokens for each task.

    Returns:
        dict: A dictionary with tasks as keys and their corresponding tokens as a single string.
    """
    task_prompts = {
        task: " ".join(tokens) for task, tokens in succession_dataset.items()
    }
    return task_prompts


def create_flipped_prompt(succession_dataset: dict) -> dict:
    """
    Converts the succession dataset into a single string for each task.

    Args:
        succession_dataset (dict): The succession dataset with tokens for each task.

    Returns:
        dict: A dictionary with tasks as keys and their corresponding tokens as a single string.
    """
    task_prompts = {
        task: " ".join(tokens[::-1]) for task, tokens in succession_dataset.items()
    }
    return task_prompts


def truncate_prompts_to_min_length(
    task_prompts: Dict[str, str],
    flipped_task_prompts: Dict[str, str],
    tokenizer: Optional[Union[PreTrainedTokenizer, AutoTokenizer]] = None,
    max_tokens: int = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Truncates both forward and flipped task prompts to the same minimum tokenized length
    across all tasks. Returns truncated prompt dictionaries.

    Args:
        task_prompts: Dict of forward-order task prompts (task -> string).
        flipped_task_prompts: Dict of reverse-order task prompts (task -> string).
        tokenizer: HuggingFace tokenizer used to tokenize prompts.
        max_tokens: Optional override to truncate all to a fixed number of tokens
                    (must be <= shortest sequence length across tasks).

    Returns:
        Tuple of (truncated_task_prompts, truncated_flipped_task_prompts)
    """
    # Compute lengths of tokenized prompts
    lengths = {
        task: len(tokenizer.encode(prompt, add_special_tokens=False))
        for task, prompt in task_prompts.items()
    }

    # Get the minimum length
    min_len = min(lengths.values())
    if max_tokens is not None:
        min_len = min(min_len, max_tokens)

    # Truncate both task_prompts and flipped_task_prompts to min_len
    def truncate(prompt: str) -> str:
        return " ".join(prompt.split()[:min_len])

    truncated_forward = {task: truncate(prompt) for task, prompt in task_prompts.items()}
    truncated_flipped = {
        task: truncate(prompt) for task, prompt in flipped_task_prompts.items()
    }

    return truncated_forward, truncated_flipped

def create_augmented_prompts(
    tokenizer, task_prompts: Dict[str, str], flipped_task_prompts: Dict[str, str], truncate: bool = False,  which_task: Literal['next', 'last'] = None, 
) -> List[Dict[str, str]]:
    """
    Creates clean/corrupt prompt pairs using next elements as answers from original and flipped.

    Args:
        task_prompts: Forward-order task prompt strings.
        flipped_task_prompts: Reverse-order task prompt strings.
        truncate: If True, truncates prompts to a minimum length.
    Returns:
        List of prompt dicts with clean/corrupt formatting.
    """
    augmented = []
    for task_name, flipped_task_name in zip(task_prompts, flipped_task_prompts):
        # Tokenize forward and flipped prompts
        if truncate:
            prompt_tokens, flipped_tokens = truncate_prompts_to_min_length(
                task_prompts,
                flipped_task_prompts,
                tokenizer=tokenizer, 
                    max_tokens=32  # Optional: override to force 32 tokens
            )
            prompt_tokens = prompt_tokens[task_name].split()
            flipped_tokens = flipped_tokens[flipped_task_name].split()
            
        else: 
            prompt_tokens = task_prompts[task_name].split()
            flipped_tokens = flipped_task_prompts[flipped_task_name].split()

        if len(prompt_tokens) < 2 or len(flipped_tokens) < 2:
            continue


        # Build the shared prompt
        if which_task == 'next':
            aug_clean_prompt = (
                f"the next term in the sequence {' '.join(prompt_tokens[:-1])} is"
            )
            aug_corrupt_prompt = (
                f"the next term in the sequence {' '.join(flipped_tokens[:-1])} is"
            )
            # Extract the correct answer and wrong answer (next tokens)
            correct_answer = prompt_tokens[-1]
            wrong_answer = flipped_tokens[-1]
        
        elif which_task == 'last':
            aug_clean_prompt = (
                f"the last term in the sequence {' '.join(prompt_tokens)} is"
            )
            aug_corrupt_prompt = (
                f"the last term in the sequence {' '.join(flipped_tokens)} is"
            )
            correct_answer = prompt_tokens[-1]
            wrong_answer = flipped_tokens[-1]
    
        item = {
            "task": task_name,
            "clean": aug_clean_prompt,
            "corrupt": aug_corrupt_prompt,
            "answers": [f"{correct_answer}"], 
            "wrong_answers": [f"{wrong_answer}"],
        }
        augmented.append(item)

    return augmented


def to_json_format(augmented_data: List[Dict[str, str]], save_path: str = None) -> str:
    """
    Converts augmented data to a JSON string following a specific format.

    Args:
        augmented_data (List[Dict]): List with clean/corrupt/answers/wrong_answers.
        save_path (str, optional): If provided, saves JSON to file.

    Returns:
        str: JSON string of the dataset.
    """
    # Remove "task" key before output
    output_data = [
        {k: v for k, v in item.items() if k != "task"} for item in augmented_data
    ]

    json_str = json.dumps({"prompts": output_data}, indent=2)
    # Save to file if save_path is provided
    if save_path:
        with open(save_path, "w") as f:
            f.write(json_str)

    return json_str
