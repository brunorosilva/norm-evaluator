import os

import boto3
import pandas as pd
import streamlit as st

from norm_benchmark.constants import NORM_BUCKET


def get_leaderboard():
    session = boto3.Session()

    s3 = session.client("s3")
    data = []
    for obj in s3.list_objects(Bucket=NORM_BUCKET)["Contents"]:
        if obj["Key"].endswith(".json"):
            s3.download_file(NORM_BUCKET, obj["Key"], obj["Key"])
            data.append(pd.read_json(obj["Key"], orient="index").T)
            os.remove(obj["Key"])

    return pd.concat(data, ignore_index=True)


def create_dashboard():
    st.set_page_config(page_title="Models Benchmark", layout="wide")
    st.title("Models Benchmark")
    create_leaderboard = st.sidebar.button("Create Leaderboard")
    if create_leaderboard:
        leaderboard = get_leaderboard()
        st.markdown("## Leaderboard Table")
        st.dataframe(leaderboard.sort_values("total_score", ascending=False))
        st.markdown("## Leaderboard Bar Chart")
        st.bar_chart(
            leaderboard,
            x_label="Model",
            y_label="Total Score",
            x="model_name",
            y="total_score",
        )

if __name__ == "__main__":
    create_dashboard()