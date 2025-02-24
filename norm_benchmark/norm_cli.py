import fire
import subprocess
import os
from norm_benchmark.backend.app import main
# from norm_benchmark.frontend.dashboard import create_dashboard
import importlib.resources as resources


class Norm(object):
    def benchmark(self, model_outputs_path, ground_truths_path, to_leaderboard=False):
        main(model_outputs_path, ground_truths_path, to_leaderboard)

    def dashboard(self):
        
        from norm_benchmark.frontend import dashboard  # Import the dashboard script as a module
        
        with resources.path(dashboard, "dashboard.py") as dashboard_script:
            # Convert the Path object to a string and run Streamlit with it
            subprocess.run(["streamlit", "run", str(dashboard_script)])
        # Use resources to get the path to the dashboard script inside the package
        # dashboard_script = resources.path(dashboard, "dashboard.py")
        
        # Run Streamlit with the correct path
        # subprocess.run(["streamlit", "run", str(dashboard_script)])


def run():
    fire.Fire(Norm)

if __name__ == "__main__":
    run()
    
    