SBERT_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
QA_SBERT_MODEL_NAME = "all-mpnet-base-v2"
ELASTICITY = 0.1
CONFIDENCE_INTERVAL = 0.95
LEADERBOARD_COLUMNS = [
    "model_name",
    "total_score",
    "section_f1_score",
    "section_mape_score",
    "total_cost_score",
    "material_grouping_score",
    "labor_grouping_score",
]
NORM_BUCKET = "norm-evaluator"
