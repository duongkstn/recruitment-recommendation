import pandas as pd
import pywebio
from pywebio.output import *


def print_profile(df: pd.DataFrame, id: int) -> pywebio.io_ctrl.Output:
    """
    Render profile to the UI
    :param df: DataFrame
    :param id: row id of df
    :return: a put_table output
    """
    row = df.iloc[id]

    # experience_table = put_table(pd.DataFrame(row['experience']).t)
    experience_table = []
    for x in row['experience']:
        experience_table.append({
            "title": x['job-title'],
            "summary": x['job-summary'],
            "company": x['job-company'],
            "industry": x['job-industry']
        })
    experience_table = put_table(experience_table)

    return put_table(
        [["field", "content"],
         [put_markdown("#### Title"), row['title']],
         [put_markdown("#### Summary"), row['summary']],
         [put_markdown("#### Industry"), row['industry']],
         [put_markdown("#### Experience"), experience_table]]
    )
