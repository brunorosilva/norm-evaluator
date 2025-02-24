from norm_benchmark.backend.evaluator import Evaluator


def main(model_outputs_path, ground_truths_path, to_leaderboard=False):
    model_name = model_outputs_path.split("/")[-1].split(".")[0]
    evaluator = Evaluator(model_name=model_name, model_outputs_path=model_outputs_path, ground_truths_path=ground_truths_path)
    benchmark_results = evaluator.run_benchmark()
    evaluator.write_results_to_json(benchmark_results)
    if to_leaderboard:
        evaluator.to_leaderboard()
