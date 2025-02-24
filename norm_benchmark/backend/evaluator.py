import json
from typing import Tuple

import boto3
import numpy as np
from sentence_transformers import SentenceTransformer

from norm_benchmark.backend.metrics import (GroupingScore, SectionScore,
                                            TotalCostScore)
from norm_benchmark.backend.utils import load_ground_truths, load_json_file
from norm_benchmark.constants import (NORM_BUCKET, QA_SBERT_MODEL_NAME,
                                      SBERT_MODEL_NAME)


class Evaluator:
    def __init__(
        self,
        model_name: str,
        model_outputs_path: str = "examples/model_outputs/2.json",
        ground_truths_path: str = "examples/ground_truth",
    ):
        """
        Initialize Evaluator with model outputs and ground truth data.

        Args:
            model_name: str, name of the model.
            model_outputs_path: str, path to the model outputs json file.
            ground_truths_path: str, path to the ground truths directory.
        """
        self.model_name = model_name
        self.model_outputs = load_json_file(model_outputs_path)
        self.ground_truths = load_ground_truths(ground_truths_path)
        self.model_section_estimates, self.expert_section_estimates = (
            self.compute_per_section_totals()
        )
        self.labor_activities = self.get_labor_activities()
        self.material_products = self.get_material_products()
        self.expert_estimates, self.model_estimates = self.compute_totals()
        self.encoder = SentenceTransformer(SBERT_MODEL_NAME, device="cpu")
        self.QA_encoder = SentenceTransformer(QA_SBERT_MODEL_NAME, device="cpu")

    def compute_totals(self) -> Tuple[list, list]:
        """
        Compute the total cost of all rows for each file in the model outputs and ground truths.

        Returns:
            A tuple of two lists. The first list contains the total costs of all rows for each file in the ground truths, and
            the second list contains the total costs of all rows for each file in the model outputs.
        """
        expert_estimates = []
        model_estimates = []
        for ground_truth in sorted(self.ground_truths):
            preds = self.model_outputs["estimate_preds"]
            for pred in preds:
                if pred["valid_file_name"] != ground_truth:
                    continue

                expert_estimates.append(
                    sum(
                        r["rowTotalCostUsd"]
                        for r in self.ground_truths[ground_truth]["rows"]
                    )
                )
                model_estimates.append(sum(r["rowTotalCostUsd"] for r in pred["rows"]))

        return expert_estimates, model_estimates

    def compute_per_section_totals(self) -> Tuple[list, list]:
        """
        Compute the total cost of each section for each file in the model outputs and ground truths.

        Returns:
            A tuple of two lists. The first list contains the total costs of each section for each file in the model outputs, and
            the second list contains the total costs of each section for each file in the ground truths.
        """
        model_section_estimates = []
        expert_section_estimates = []
        for ground_truth in sorted(self.ground_truths):
            preds = self.model_outputs["estimate_preds"]
            for pred in preds:
                if pred["valid_file_name"] != ground_truth:
                    continue

                model_section_estimates.append(
                    {
                        section: sum(
                            r["rowTotalCostUsd"]
                            for r in pred["rows"]
                            if r["sectionName"] == section
                        )
                        for section in {r["sectionName"] for r in pred["rows"]}
                    }
                )

                expert_section_estimates.append(
                    {
                        section: sum(
                            r["rowTotalCostUsd"]
                            for r in self.ground_truths[ground_truth]["rows"]
                            if r["sectionName"] == section
                        )
                        for section in {
                            r["sectionName"]
                            for r in self.ground_truths[ground_truth]["rows"]
                        }
                    }
                )
        return model_section_estimates, expert_section_estimates

    def get_labor_activities(self) -> list:
        """
        Get all labor activities from the model outputs.

        Returns:
            A list of lists. Each sublist contains all labor activities in a file, represented as dictionaries with keys "activity" and "sectionName".
        """
        labor_activites = []
        preds = self.model_outputs["estimate_preds"]
        for pred in preds:
            labor_activites.append(
                [
                    {"activity": r["label"], "sectionName": r["sectionName"]}
                    for r in pred["rows"]
                    if r["category"] == "labor"
                ]
            )
        return labor_activites

    def get_material_products(self) -> list:
        """
        Get all material products from the model outputs.

        Returns:
            A list of lists. Each sublist contains all material products in a file, represented as dictionaries with keys "product" and "sectionName".
        """
        material_products = []
        preds = self.model_outputs["estimate_preds"]
        for pred in preds:
            material_products.append(
                [
                    {"product": r["label"], "sectionName": r["sectionName"]}
                    for r in pred["rows"]
                    if r["category"] == "material"
                ]
            )
        return material_products

    def run_benchmark(self) -> Tuple[float, float, float, float, float]:
        """
        Runs the benchmark and returns the scores.

        The scores are in the order of (section_f1_score, section_mape_score, total_cost_score, material_grouping_score, labor_grouping_score).

        Returns:
            A tuple of five floats. The first is the F1 score for section matching, the second is the mean absolute percentage error for section cost estimates, the third is the accuracy of total cost estimates, the fourth is the accuracy of material product grouping, and the fifth is the accuracy of labor activity grouping.
        """
        section_score = SectionScore()
        cost_score = TotalCostScore()
        grouping_score = GroupingScore()

        section_scores = []
        for expert_section_estimate, model_section_estimate in zip(
            self.expert_section_estimates, self.model_section_estimates
        ):
            section_scores.append(
                section_score(
                    self.encoder, expert_section_estimate, model_section_estimate
                )
            )
        section_f1_score = np.mean([s[0] for s in section_scores])
        section_mape_score = np.mean([s[1] for s in section_scores])
        section_mape_score = max(1 - section_mape_score, 0)
        total_cost_scores = []
        for expert_estimate, model_estimate in zip(
            self.expert_estimates, self.model_estimates
        ):
            total_cost_scores.append(cost_score(expert_estimate, model_estimate))

        total_cost_score = np.mean(total_cost_scores)
        total_cost_score = max(1 - total_cost_score, 0)
        material_grouping_score, labor_grouping_score = grouping_score(
            self.QA_encoder, self.material_products, self.labor_activities
        )

        return (
            section_f1_score,
            section_mape_score,
            total_cost_score,
            material_grouping_score,
            labor_grouping_score,
        )

    def write_results_to_json(self, benchmark_results: Tuple) -> None:
        """
        Writes the results of the benchmark to a JSON file.

        The results are written in the following format:
        {
            "model_name": str,
            "total_score": float,
            "section_f1_score": float,
            "section_mape_score": float,
            "total_cost_score": float,
            "material_grouping_score": float,
            "labor_grouping_score": float,
        }

        The total score is a weighted average of the section F1 score, section MAPE score, total cost score, material grouping score, and labor grouping score. The weights are 0.2, 0.3, 0.3, 0.1, and 0.1, respectively.
        """
        (
            section_f1_score,
            section_mape_score,
            total_cost_score,
            material_grouping_score,
            labor_grouping_score,
        ) = benchmark_results
        with open(f"results/{self.model_name}.json", "w") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "total_score": (
                        section_f1_score * 0.2
                        + section_mape_score * 0.3
                        + total_cost_score * 0.3
                        + material_grouping_score * 0.1
                        + labor_grouping_score * 0.1
                    ),
                    "section_f1_score": section_f1_score,
                    "section_mape_score": section_mape_score,
                    "total_cost_score": total_cost_score,
                    "material_grouping_score": material_grouping_score,
                    "labor_grouping_score": labor_grouping_score,
                },
                f,
            )

    def to_leaderboard(self) -> None:
        """
        Uploads the results JSON file to the NORM_BUCKET S3 bucket.

        This should be called after calling write_results_to_json.
        """
        s3 = boto3.Session().client("s3")
        s3.upload_file(
            f"results/{self.model_name}.json", NORM_BUCKET, f"{self.model_name}.json"
        )
