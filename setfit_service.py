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
import multiprocessing
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
def process(return_dict):

    input_texts = redis_db.lrange("input_texts", 0, -1)
    ids = redis_db.lrange("ids", 0, -1)
    if len(input_texts) == 0 or len(ids) == 0:
        return_dict["state"] = "fail"
        return
    input_texts = [str(x) for x in input_texts]
    ids = [int(x) for x in ids]
    print("into process")
    print(input_texts[0], ids[0])

    df_sparse = pd.DataFrame({
        "text": [input_texts[id_] for id_ in ids[:4]] + [input_texts[id_] for id_ in ids[-4:]],
        "label": [1] * len(ids[:4]) + [0] * len(ids[-4:])
    })
    train_ds = Dataset.from_pandas(df_sparse)
    print(df_sparse)

    model = SetFitModel.from_pretrained('intfloat/e5-large', cache_dir="cache_dir")
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=None,
        loss_class=CosineSimilarityLoss,
        batch_size=1,
        num_iterations=10,  # Number of text pairs to generate for contrastive learning
        num_epochs=1,  # Number of epochs to use for contrastive learning
    )
    trainer.train(end_to_end=False)
    model.model_body.zero_grad()
    trainer.callback_handler.optimizer.zero_grad()
    with torch.no_grad():
        score = model.predict_proba(input_texts, as_numpy=True, show_progress_bar=True, batch_size=4)
        id2score = {i: score[i][1] for i in range(len(score))}
        id2score = sorted(id2score.items(), key=lambda item: -item[1])
        ids = [x[0] for x in id2score]
        print(id2score)
        del score
        torch.cuda.empty_cache()

        for x in ids:
            redis_db.rpush("result_ids", str(x))
    return_dict["state"] = "success"


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    while True:
        return_dict = manager.dict()
        p = multiprocessing.Process(target=process, args=(return_dict, ))
        p.start()
        p.join()
        if "state" in return_dict and return_dict["state"] == "success":
            print("...", return_dict)
