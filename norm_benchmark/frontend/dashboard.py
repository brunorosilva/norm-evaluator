import os

import boto3
import pandas as pd
import streamlit as st

from norm_benchmark.constants import NORM_BUCKET


def get_leaderboard():
    """
    Downloads all JSON files from the NORM_BUCKET S3 bucket, reads them into
    DataFrames, and returns a concatenated DataFrame of all the DataFrames.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the model evaluation results. The columns are the
        model evaluation metrics, and the index is the model name.
    """
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
    """
    Creates a Streamlit dashboard for the models benchmark.

    The dashboard displays the leaderboard in a table and a bar chart. The
    leaderboard is generated by calling the `get_leaderboard` function and
    sorting the results by the total score in descending order.

    The bar chart displays the total score for each model, with the x-axis
    being the model name and the y-axis being the total score.

    The dashboard has a button in the sidebar labeled "Create Leaderboard".
    When clicked, the dashboard will download all the JSON files from the
    NORM_BUCKET S3 bucket, read them into DataFrames, and display the
    leaderboard.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
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