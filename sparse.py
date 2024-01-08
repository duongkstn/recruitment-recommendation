from utils import get_data, ElasticsearchClient
from ui_utils import print_profile
from pywebio.output import *
from functools import partial
from config import config
import pandas as pd
import pywebio
from pywebio.output import *
from typing import Tuple, List, Dict
import time


if __name__ == "__main__":
    df, input_texts = get_data("res")

    # Initialize Elasticsearch Client
    client = ElasticsearchClient(config["hosts"], config["verify_certs"],
                                 config["http_auth_user"], config["http_auth_password"], config["index"])

    if not client.check_index_exist():  # If index is not yet created
        client.create_index()
        bulk_data_create = []
        for id_ in range(len(df)):
            row = df.iloc[id_]
            s = []
            for job in row['experience']:
                s.append(job['job-title'])
                s.append(job['job-summary'])
                s.append(job['job-company'])
                s.append(job['job-industry'])
            s = [k for k in s if k != ""]
            s = " . ".join(s)
            bulk_data_elem = {
                '_op_type': 'create',
                '_id': id_,
                '_source': {
                    'title': row['title'],
                    'summary': row['summary'],
                    'industry': row['industry'],
                    'experience': s
                }
            }
            bulk_data_create.append(bulk_data_elem)
        client.bulk_data(bulk_data_create)

    put_markdown('## DEMO RESULT ADVANCED SPARSE SEARCH')

    accept_i, reject_i = [], []
    def Onclick(x: str, i: int) -> str:
        """
        Handling click event
        :param x: "Accept"/"Reject"
        :param i: id
        :return:
        """
        global accept_i, reject_i
        if x == "Accept":
            accept_i.append(i)
        elif x == "Reject":
            reject_i.append(i)
        with use_scope(f"{x}-{i}"):
            clear()
            # Turn status of button
            put_buttons([
                {"label": f"{x}ed", "value": x, "color": "success"}], onclick=partial(Onclick, i=i))
        return x


    def render_table(df: pd.DataFrame, ids: List[int]) -> None:
        """
        Render the big table
        :param df:
        :param ids:
        :return:
        """
        L = [["id", "Data"]]
        for i, id in enumerate(ids):
            L.append([i, put_html(f'<h3 style="background-color:cyan;"> Profile {str(id)} </h3>')])
            L.append(["", print_profile(df, id)])
            L.append(['', put_scope(f'Accept-{i}', put_buttons([
                {"label": f'Accept', "value": f'Accept', "color": "info"}], onclick=partial(Onclick, i=i)))]),
            L.append(['', put_scope(f'Reject-{i}', put_buttons([
                {"label": f'Reject', "value": f'Reject', "color": "dark"}], onclick=partial(Onclick, i=i)))]),
        with use_scope('loading'):
            time.sleep(2)  # Time sleep for loading symbol
            clear()
        with use_scope('table'):
            put_table(L)

    def Apply(x):
        global ids, accept_i, reject_i
        with use_scope('loading'):
            put_loading(shape="border", color="dark")
        if len(accept_i) > 0 or len(reject_i) > 0:
            id2score = {}
            # Get sparse result
            for i in accept_i:
                id = ids[i]
                bm25_result = client.search(df.iloc[id])
                for id in bm25_result:
                    score = bm25_result[id]
                    if id not in id2score:
                        id2score[id] = score
                    else:
                        id2score[id] += score

            for i in reject_i:
                id = ids[i]
                bm25_result = client.search(df.iloc[id])
                for id in bm25_result:
                    score = bm25_result[id]
                    if id not in id2score:
                        id2score[id] = -score
                    else:
                        id2score[id] -= score
            id2score = sorted(id2score.items(), key=lambda item: -item[1])
            ids = [x[0] for x in id2score]

        # clear old table
        with use_scope('table'):
            clear()

        # Render the big table
        print(ids)
        accept_i = []
        reject_i = []
        render_table(df, ids)
        return x


    ids = list(range(len(input_texts)))
    put_buttons(["APPLY"], onclick=Apply)
