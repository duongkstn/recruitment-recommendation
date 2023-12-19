import json, os, re
import time

import pandas as pd
from tqdm import tqdm
import numpy as np
from elasticsearch import Elasticsearch, helpers
from pywebio.output import *
from functools import partial
from datasets import load_dataset, Dataset
from sentence_transformers.losses import CosineSimilarityLoss
import torch
from setfit import SetFitModel, SetFitTrainer
os.environ['WANDB_DISABLED'] = 'true'
if __name__ == "__main__":
    df = []
    for file_ in sorted(os.listdir("res")):
        if not file_.endswith(".json"):
            continue
        sample = json.load(open(os.path.join("res", file_)))
        df_sample = {
            "title": sample.get("headline", ""),
            "summary": sample.get("summary", ""),
            "industry": sample.get("industryName", ""),
            "experience": [{
                "job-title": x.get("title", ""),
                "job-summary": x.get("description", ""),
                "job-company": x.get("companyName", ""),
                "job-industry": "\n".join([y for y in x.get("company", {}).get("industries", [])])
            } for x in sample.get("experience", [])]
        }
        df.append(df_sample)
    df = pd.DataFrame(df)


    def get_text(x):
        s = [x['title'], x['summary'], x['industry']]
        for job in x['experience']:
            s.append(job['job-title'])
            s.append(job['job-summary'])
            s.append(job['job-company'])
            s.append(job['job-industry'])
        s = [k for k in s if k != ""]
        s = " . ".join(s)
        return s


    df['text'] = df.apply(lambda x: get_text(x), axis=1)
    input_texts = df['text'].values.tolist()
    df.to_csv("data.csv", index=False)
    print(df)
    # df = pd.read_csv("data.csv", dtype=str)
    # df.fillna("", inplace=True)
    # # for x in df['experience'].values.tolist():
    # df['experience'] = df['experience'].apply(lambda x: json.loads(x))
    # input_texts = df['text'].values.tolist()

    es = Elasticsearch(hosts="https://localhost:9200", verify_certs=False, http_auth=("elastic", "h7gh7EFpGmfBckv44v-q"))

    body = {
        'settings': {
            'similarity': {
                'bm25_custom': {
                    'type': 'BM25',
                    'k1': 0.5,
                    'b': 0.5
                }
            }
        },
        'mappings': {
            'properties': {
                'title': {
                    'type': 'text',
                    'similarity': 'bm25_custom'
                },
                'summary': {
                    'type': 'text',
                    'similarity': 'bm25_custom'
                },
                'industry': {
                    'type': 'text',
                    'similarity': 'bm25_custom'
                }
            }
        }
    }
    index = "recruit2"
    if not es.indices.exists(index=index):
        es.indices.create(index=index, mappings=body['mappings'], settings=body['settings'], request_timeout=60)

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

        success, failed = helpers.bulk(es, bulk_data_create, index=index, request_timeout=30, refresh="false",
                                       stats_only=True)
        print(f"success = {success}, failed = {failed}")


    def search(row):
        s = []
        for job in row['experience']:
            s.append(job['job-title'])
            s.append(job['job-summary'])
            s.append(job['job-company'])
            s.append(job['job-industry'])
        s = [k for k in s if k != ""]
        s = " . ".join(s)
        try:
            should = [
                {
                    "match": {
                        "title": {
                            "query": row["title"],
                            "boost": 100
                        }
                    }
                },
                {
                    "match": {
                        "summary": {
                            "query": row["summary"],
                            "boost": 3
                        }
                    }
                },
                {
                    "match": {
                        "industry": {
                            "query": row["industry"],
                            "boost": 50
                        }
                    }
                },
                {
                    "match": {
                        "experience": {
                            "query": s,
                            "boost": 1
                        }
                    }
                }
            ]
            print(f"should = {should}")
            response = es.search(index=index, size=1000,
                                    query={
                                        "bool": {
                                            "should": should
                                        }
                                    },
                                    source=["text"],
                                    request_timeout=30,
                                    track_total_hits=False
                                    )
            result = {int(hit["_id"]): hit["_score"] for hit in response["hits"]["hits"]}
            return result
        except Exception as e:
            print(e)
            return {}


    def print_profile(id):
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


    put_markdown('## DEMO RESULT')

    accept_i = []
    reject_i = []
    # model = SetFitModel.from_pretrained('intfloat/e5-large', cache_dir="cache_dir")
    import redis, msgpack
    redis_db = redis.StrictRedis(
        host="localhost",
        port=31000,
        db=3,
        password="",
        ssl=False,
        decode_responses=False,
        socket_timeout=10,
    )
    for x in input_texts:
        redis_db.rpush("input_texts", x)


    def Onclick(x, i):
        global accept_i, reject_i
        if x == "Accept":
            accept_i.append(i)
        elif x == "Reject":
            reject_i.append(i)
        with use_scope(f"{x}-{i}"):
            clear()
            put_buttons([
                {"label": f"{x}ed", "value": x, "color": "success"}], onclick=partial(Onclick, i=i))
        return x


    def Apply(x):
        global ids, accept_i, reject_i
        with use_scope('loading'):
            put_loading(shape="border", color="dark")
        if len(accept_i) > 0 or len(reject_i) > 0:
            id2score = {}
            for i in accept_i:
                id = ids[i]
                bm25_result = search(df.iloc[id])
                for id in bm25_result:
                    score = bm25_result[id]
                    if id not in id2score:
                        id2score[id] = score
                    else:
                        id2score[id] += score

            for i in reject_i:
                id = ids[i]
                bm25_result = search(df.iloc[id])
                for id in bm25_result:
                    score = bm25_result[id]
                    if id not in id2score:
                        id2score[id] = -score
                    else:
                        id2score[id] -= score
            id2score = sorted(id2score.items(), key=lambda item: -item[1])
            ids = [x[0] for x in id2score]

            for x in ids:
                redis_db.rpush("ids", str(x))

            # df_sparse = pd.DataFrame({
            #     "text": [input_texts[id_] for id_ in ids[:4]] + [input_texts[id_] for id_ in ids[-4:]],
            #     "label": [1] * len(ids[:4]) + [0] * len(ids[-4:])
            # })
            # train_ds = Dataset.from_pandas(df_sparse)
            # print(df_sparse)
            # model.to("cuda")
            # trainer = SetFitTrainer(
            #     model=model,
            #     train_dataset=train_ds,
            #     eval_dataset=None,
            #     loss_class=CosineSimilarityLoss,
            #     batch_size=1,
            #     num_iterations=10,  # Number of text pairs to generate for contrastive learning
            #     num_epochs=1,  # Number of epochs to use for contrastive learning
            # )
            # trainer.train(end_to_end=False)
            # model.model_body.zero_grad()
            # trainer.callback_handler.optimizer.zero_grad()
            # with torch.no_grad():
            #     score = model.predict_proba(input_texts, as_numpy=True, show_progress_bar=True, batch_size=4)
            #     id2score = {i: score[i][1] for i in range(len(score))}
            #     id2score = sorted(id2score.items(), key=lambda item: -item[1])
            #     ids = [x[0] for x in id2score]
            #     print(id2score)
            #     del score
            #     torch.cuda.empty_cache()
            # model.to("cpu")
            while True:
                ids = redis_db.lrange("result_ids", 0, -1)
                if len(ids) == len(input_texts):
                    ids = [int(x) for x in ids]
                    print("result_ids = ", ids)
                    redis_db.delete("ids")
                    break


        with use_scope('table'):
            clear()  # clear old table

        accept_i = []
        reject_i = []
        L = [["id", "Data"]]
        for i, id in enumerate(ids):
            L.append([i, put_html(f'<h3 style="background-color:cyan;"> Profile {str(id)} </h3>')])
            L.append(["", print_profile(id)])
            L.append(['', put_scope(f'Accept-{i}', put_buttons([
                {"label": f'Accept', "value": f'Accept', "color": "info"}], onclick=partial(Onclick, i=i)))]),
            L.append(['', put_scope(f'Reject-{i}', put_buttons([
                {"label": f'Reject', "value": f'Reject', "color": "dark"}], onclick=partial(Onclick, i=i)))]),
        with use_scope('loading'):
            time.sleep(2)
            clear()
        with use_scope('table'):
            # put_grid([[put_table(L)]], cell_width='10000px')

            put_table(L)

        return x


    ids = list(range(len(input_texts)))
    put_buttons(["APPLY"], onclick=Apply)
