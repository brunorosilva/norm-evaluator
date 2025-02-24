# NORM - coNstructiOn pRoject Manager

**Norm** is a CLI tool for benchmarking cost estimate generator models in construction projects. It evaluates model outputs against ground truths and provides a leaderboard for comparison. The name "Norm" was chosen because it sounds like a good PM name.

## Report

Here's the [full report](https://www.notion.so/Handoff-Takehome-Solution-1a2ee5e433ba8069b014c5d2ed238fcc) which comprises the reasoning behind the product, methodology, results and future work.

## Features

- 📊 **Benchmark Models**: Compare model-generated cost estimates against ground truth data.
- 📈 **Leaderboard**: Optionally upload benchmarking results to a leaderboard.
- 🖥 **Dashboard**: Launch an interactive Streamlit dashboard to visualize results.

## Installation

```sh
pip install norm-benchmark
```

## Usage

Norm provides a simple CLI with two commands: `benchmark` and `dashboard`.

### 1. Run Benchmarking

Compare a model's output against ground truths:

```sh
norm benchmark <model_outputs.json> <ground_truths_directory> [--to_leaderboard]
```

#### Arguments:
- `<model_outputs.json>`: Path to the JSON file containing model outputs.
- `<ground_truths_directory>`: Path to the directory with ground truth data.
- `--to_leaderboard` *(optional)*: Uploads results to the leaderboard if specified.

### 2. Launch Dashboard

Start the Streamlit dashboard for visualizing benchmarking results:

```sh
norm dashboard
```

The dashboard will be available at: [http://localhost:8501](http://localhost:8501)

## Example

```sh
norm benchmark results.json ground_truths/ --to_leaderboard
norm dashboard
```

## License

This project is licensed under the MIT License.

