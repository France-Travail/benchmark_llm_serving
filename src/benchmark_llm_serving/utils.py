import os
import json
import random
from pathlib import Path
import importlib.metadata
from datetime import datetime
from typing import Union, List


def get_package_version() -> str:
    '''Returns the current version of the package

    Returns:
        str: version of the package
    '''
    version = importlib.metadata.version("benchmark_llm_serving")
    return version


def get_now() -> str:
    """Gets the current time and put it in string form

    Returns:
        str : The current date and time
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d, %H:%M:%S")


def load_dataset(dataset_folder: Union[Path, str, None] = None,
                 prompt_length: str = "0") -> List[str]:
    """Loads a dataset and shuffles it

    Args:
        dataset_folder (str) : The path to the folder containing the datasets
        prompt_length (str) : The prompt length to consider. If "0", we want 
                             the growing prompts dataset
    Returns:
        list : The list of all the prompts
    """
    if prompt_length == "0":
        filename = "growing_prompts.json"
    else:
        filename = f"prompts_length_{prompt_length}.json"
    dataset_path = os.path.join(dataset_folder, filename)
    with open(dataset_path, 'r') as json_file:
        dataset = json.load(json_file)
    random.shuffle(dataset)
    return dataset


def tasks_are_done(tasks: list) -> bool:
    """Checks if all task in tasks are done

    Args:
        tasks (list) : The list of tasks

    Returns:
        bool : Whether all the tasks are done
    """
    for task in tasks:
        if not task.done():
            return False
    return True