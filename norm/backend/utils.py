import json
import os


def load_json_file(path):
    """
    Loads a JSON file from the specified path and returns its contents as a dictionary.

    Args:
        path (str): The file path to the JSON file.

    Returns:
        dict: The contents of the JSON file.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_ground_truths(path):
    """
    Loads all JSON files in the specified directory and returns their contents as a dictionary.

    Args:
        path (str): The directory path containing the JSON files.

    Returns:
        dict: A dictionary where the keys are the file names (without the .json extension)
              and the values are the contents of the JSON files.
    """

    gt_data = {}
    for fn in os.listdir(path):
        with open(f"{path}/{fn}", "r") as f:
            gt_data[fn.split(".json")[0]] = json.load(f)

    return gt_data
