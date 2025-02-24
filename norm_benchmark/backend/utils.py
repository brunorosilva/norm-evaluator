import json
import os


def load_json_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_ground_truths(path):
    gt_data = {}
    for fn in os.listdir(path):
        with open(f"{path}/{fn}", "r") as f:
            gt_data[fn.split(".json")[0]] = json.load(f)

    return gt_data
