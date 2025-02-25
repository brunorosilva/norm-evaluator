import abc

import numpy as np
import scipy
from sentence_transformers.util import cos_sim
from sklearn.metrics import f1_score
from tqdm import tqdm

from norm_benchmark.backend.prompts import (labor_grouping_template,
                                            product_grouping_template)
from norm_benchmark.constants import CONFIDENCE_INTERVAL, ELASTICITY


class Metric(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass


class TotalCostScore(Metric):
    def __call__(
        self,
        expert_estimate,
        model_estimate,
        ELASTICITY=ELASTICITY,
        CONFIDENCE_INTERVAL=CONFIDENCE_INTERVAL,
    ):
        """
        Calculate total cost score for model given expert estimate.

        If model is within confidence interval around expert estimate, return absolute difference between the two divided by expert estimate.
        Otherwise, return 0.

        :param expert_estimate: float, expert estimate
        :param model_estimate: float, model estimate
        :param ELASTICITY: float, standard deviation of normal distribution
        :param CONFIDENCE_INTERVAL: float, confidence interval for normal distribution
        :return: float, total cost score
        """
        if self.is_model_within_distribuition(
            model_estimate, expert_estimate, CONFIDENCE_INTERVAL, ELASTICITY
        ):
            return np.abs(expert_estimate - model_estimate) / expert_estimate
        return 0

    def __repr__(self):
        return "TotalCostScore()"

    @staticmethod
    def is_model_within_distribuition(
        estimate, expert_estimate, CONFIDENCE_INTERVAL, ELASTICITY
    ):
        """
        Check if model estimate is within confidence interval around expert estimate.

        :param estimate: float, model estimate
        :param expert_estimate: float, expert estimate
        :param CONFIDENCE_INTERVAL: float, confidence interval for normal distribution
        :param ELASTICITY: float, standard deviation of normal distribution
        :return: bool, is model estimate within distribution
        """
        lower, upper = scipy.stats.norm.interval(
            CONFIDENCE_INTERVAL, expert_estimate, expert_estimate * ELASTICITY
        )
        return lower <= estimate <= upper


class SectionScore(Metric):
    def __call__(self, encoder, expert_sections: dict, model_sections: dict):
        """
        Compute the section score.

        :param encoder: SentenceTransformer, transformer for computing similarity
        :param expert_sections: dict, expert sections with their estimates
        :param model_sections: dict, model sections with their estimates
        :return: tuple, (f1 score, mean absolute percentage error)
        """
        expert_dict = expert_sections
        model_dict = model_sections
        all_sections = list(set(expert_dict.keys()) | set(model_dict.keys()))
        y_true, y_pred = [], []
        estimate_diffs = []

        matched_expert, matched_model = set(), set()

        for section in all_sections:
            found_expert, found_model = False, False

            for expert_section in expert_dict:
                if self.is_similar(encoder, section, expert_section):
                    y_true.append(1)
                    matched_expert.add(expert_section)
                    found_expert = True
                    break
            if not found_expert:
                y_true.append(0)

            for model_section in model_dict:
                if self.is_similar(encoder, section, model_section):
                    y_pred.append(1)
                    matched_model.add(model_section)
                    found_model = True

                    if found_expert:
                        expert_estimate = expert_dict[expert_section]
                        model_estimate = model_dict[model_section]
                        estimate_diffs.append(
                            abs(expert_estimate - model_estimate) / expert_estimate
                        )

                    break
            if not found_model:
                y_pred.append(0)

        # Compute metrics
        f1 = f1_score(y_true, y_pred)
        mape = np.mean(estimate_diffs) if estimate_diffs else None

        return f1, mape

    def __repr__(self):
        return "SectionScore()"

    @staticmethod
    def is_similar(encoder, s1, s2, threshold=0.7):
        """
        Determine if two sections are similar based on cosine similarity.

        :param encoder: SentenceTransformer, transformer for encoding text into embeddings
        :param s1: str, first section name
        :param s2: str, second section name
        :param threshold: float, similarity threshold for considering two sections as similar
        :return: bool, True if sections are similar, False otherwise
        """
        if s1 == s2:
            return True

        return cos_sim(encoder.encode(s1), encoder.encode(s2)) >= threshold


class GroupingScore(Metric):
    def __call__(self, encoder, material_data, labor_data, threshold=0.3):
        """
        Calculate the grouping scores for material and labor data based on cosine similarity.

        Args:
            encoder: SentenceTransformer, encoder used to transform text into embeddings.
            material_data: list of lists, material data containing dictionaries with keys "product" and "sectionName".
            labor_data: list of lists, labor data containing dictionaries with keys "activity" and "sectionName".
            threshold: float, similarity threshold for determining a match.

        Returns:
            A tuple of two floats: the first is the score for material grouping, and the second is the score for labor grouping.
        """

        material_output = []
        for preds in tqdm(material_data):
            for material in preds:
                s1 = product_grouping_template.format(material["product"])
                s2 = labor_grouping_template.format(material["sectionName"])
                material_output.append(
                    int(cos_sim(encoder.encode(s1), encoder.encode(s2)) >= threshold)
                )

        labor_output = []
        for preds in tqdm(labor_data):
            for labor in preds:
                s1 = labor_grouping_template.format(labor["activity"])
                s2 = labor_grouping_template.format(labor["sectionName"])
                labor_output.append(
                    int(cos_sim(encoder.encode(s1), encoder.encode(s2)) >= threshold)
                )

        return sum(material_output) / len(material_output), sum(labor_output) / len(
            labor_output
        )

    def __repr__(self):
        return "GroupingScore()"
