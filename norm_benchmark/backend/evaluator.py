import json

import boto3
import numpy as np
from sentence_transformers import SentenceTransformer

from norm_benchmark.backend.metrics import GroupingScore, SectionScore, TotalCostScore
from norm_benchmark.backend.utils import load_ground_truths, load_json_file
from norm_benchmark.constants import NORM_BUCKET, QA_SBERT_MODEL_NAME, SBERT_MODEL_NAME


class Evaluator:
    def __init__(
        self,
        model_name,
        model_outputs_path="examples/model_outputs/2.json",
        ground_truths_path="examples/ground_truth",
    ):
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

    def compute_totals(self):
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

    def compute_per_section_totals(self):
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

    def get_labor_activities(self):
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

    def get_material_products(self):
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

    def run_benchmark(self):
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

    def write_results_to_json(self, benchmark_results):
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

    def to_leaderboard(self):
        s3 = boto3.Session().client("s3")
        s3.upload_file(
            f"results/{self.model_name}.json", NORM_BUCKET, f"{self.model_name}.json"
        )
