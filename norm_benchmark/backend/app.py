from norm_benchmark.backend.evaluator import Evaluator


def main(model_outputs_path, ground_truths_path, to_leaderboard=False):
    """
    Execute the benchmarking process for a given model's outputs against ground truths.

    Args:
        model_outputs_path (str): The file path to the model's output JSON file.
        ground_truths_path (str): The file path to the ground truths directory.
        to_leaderboard (bool, optional): If True, upload the results to the leaderboard. Defaults to False.

    Process:
        - Extracts the model name from the model outputs path.
        - Initializes an Evaluator with the provided model name, model outputs, and ground truths.
        - Runs the benchmark and writes the results to a JSON file.
        - Optionally uploads the results to the leaderboard if `to_leaderboard` is True.
    """
    model_name = model_outputs_path.split("/")[-1].split(".")[0]
    evaluator = Evaluator(
        model_name=model_name,
        model_outputs_path=model_outputs_path,
        ground_truths_path=ground_truths_path,
    )
    benchmark_results = evaluator.run_benchmark()
    evaluator.write_results_to_json(benchmark_results)
    if to_leaderboard:
        evaluator.to_leaderboard()
