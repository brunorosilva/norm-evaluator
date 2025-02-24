import importlib.resources as resources
import subprocess

import fire

from norm_benchmark.backend.app import main


class Norm(object):
    def benchmark(self, model_outputs_path, ground_truths_path, to_leaderboard=False):
        """
        Run the benchmarking process for a given model's outputs against ground truths.

        Args:
            model_outputs_path (str): The file path to the model's output JSON file.
            ground_truths_path (str): The file path to the ground truths directory.
            to_leaderboard (bool, optional): If True, upload the results to the leaderboard. Defaults to False.
        """
        main(model_outputs_path, ground_truths_path, to_leaderboard)

    def dashboard(self):
        """
        Start the Streamlit dashboard for visualizing leaderboard results.

        The dashboard can be accessed in a web browser at http://localhost:8501.
        """
        from norm_benchmark.frontend import \
            dashboard  # Import the dashboard script as a module

        with resources.path(dashboard, "dashboard.py") as dashboard_script:
            subprocess.run(["streamlit", "run", str(dashboard_script)])


def run():
    fire.Fire(Norm)


if __name__ == "__main__":
    run()
