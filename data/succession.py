from typing import Tuple

def generate_successor_pairs() -> Tuple[dict, dict]:
    """
    Generates the succession dataset and mapping of successors.

    Returns:
        Tuple[dict, dict]: Succession dataset and successor mapping.
    """
    succession_dataset = {
        "Numbers": [str(i) for i in range(1, 21)],
        "Number words": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                         "eighteen", "nineteen", "twenty"],
        "Cardinal words": ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
                           "ninth", "tenth"],
        "Days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "Day prefixes": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "Months": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
        "Month prefixes": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "Letters": [chr(i) for i in range(ord('A'), ord('Z') + 1)],
        "Roman Letters": ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"],
        "Seasons": ["Spring", "Summer", "Fall", "Winter"],
        "Arithmetic Progression": ["1", "3", "5", "7", "9", "11", "13", "15", "17", "19"],
        "Geometric Progression": ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512"],
    }

    succession_mapping = {}

    succession_mapping["Days"] = {
        succession_dataset["Days"][i]: succession_dataset["Days"][(i + 1) % len(succession_dataset["Days"])]
        for i in range(len(succession_dataset["Days"]))
    }
    succession_mapping["Months"] = {
        succession_dataset["Months"][i]: succession_dataset["Months"][(i + 1) % len(succession_dataset["Months"])]
        for i in range(len(succession_dataset["Months"]))
    }
    succession_mapping["Roman Letters"] = {
        succession_dataset["Roman Letters"][i]: succession_dataset["Roman Letters"][(i + 1) % len(succession_dataset["Roman Letters"])]
        for i in range(len(succession_dataset["Roman Letters"]))
    }
    succession_mapping["Seasons"] = {
        succession_dataset["Seasons"][i]: succession_dataset["Seasons"][(i + 1) % len(succession_dataset["Seasons"])]
        for i in range(len(succession_dataset["Seasons"]))
    }
    succession_mapping["Letters"] = {
        succession_dataset["Letters"][i]: succession_dataset["Letters"][(i + 1) % len(succession_dataset["Letters"])]
        for i in range(len(succession_dataset["Letters"]))
    }
    
    # Handle non-cyclical successors for other tasks
    for task in ["Numbers", "Number words", "Cardinal words", "Day prefixes", "Month prefixes", "Arithmetic Progression", "Geometric Progression"]:
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
