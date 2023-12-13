import json, os, re
import time

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import faiss
# import streamlit as st
from pywebio.output import *
from functools import partial

def index_hnsw(index_dense_path: str, full_embeddings_path: str, embedding_size: int = 1024,
               M: int = 32, efC: int = 128, efSearch: int = 256):
    """
    Create a faiss hnsw index
    @param index_dense_path: path to index
    @param full_embeddings_path: path to embeddings saved
    @param embedding_size: size of embedding 768, 64, ...
    @param M: number of links per vector
    @param efC: efConstruction
    @param efSearch: trade-off accuracy vs speed
    """
    faiss.omp_set_num_threads(4)
    index = faiss.IndexHNSWFlat(embedding_size, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = efC
    index.hnsw.efSearch = efSearch
    corpus_embeddings = np.load(full_embeddings_path, mmap_mode="r")
    print(corpus_embeddings.shape)
    # corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]
    print("Adding vectors to index...")
    index.add(corpus_embeddings)
    faiss.write_index(index, index_dense_path)

    print("Done")

def load_and_add_vector_index_faiss(index_dense_path: str):
    """
    load index trained and add all vector to this index
    @param index_dense_path: path save index dense pre-trained
    @param vector_paths: paths to files save vectors in corpus --> list of path
    """
    print("Loading index...")
    faiss.omp_set_num_threads(4)
    index = faiss.read_index(index_dense_path)
    return index

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


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

    df['text'] = df.apply(lambda x: get_text(x),axis=1)
    input_texts = df['text'].values.tolist()
    print(df)


    # tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large', cache_dir="cache_dir")
    # model = AutoModel.from_pretrained('intfloat/e5-large', cache_dir="cache_dir")
    # model.to("cuda")


    # batch_size = 2
    # all_embeddings = None
    # for i in tqdm(range(0, len(input_texts), batch_size)):
    #     j = min(i + batch_size, len(input_texts))
    #     texts = input_texts[i: j]
    #     batch_dict = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to("cuda")
    #     outputs = model(**batch_dict)
    #     embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    #     embeddings = F.normalize(embeddings, p=2, dim=1)
    #     embeddings = embeddings.cpu().detach().numpy()
    #     if all_embeddings is None:
    #         all_embeddings = embeddings
    #     else:
    #         all_embeddings = np.concatenate([all_embeddings, embeddings])

    all_embeddings_npy_filename = 'vectors.npy'
    all_embeddings_faiss_filename = 'vectors.faiss'
    # with open(all_embeddings_npy_filename, 'wb') as f:
    #     np.save(f, all_embeddings)
    # index_hnsw(all_embeddings_faiss_filename, all_embeddings_npy_filename)
    with open(all_embeddings_npy_filename, 'rb') as f:
        all_embeddings = np.load(f)
    dense_searcher = load_and_add_vector_index_faiss(all_embeddings_faiss_filename)

    def print_profile(id):
        return input_texts[id]
    put_markdown('## DEMO RESULT')

    accept_i = []
    reject_i = []
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
                scores, sorted_ids = dense_searcher.search(np.expand_dims(all_embeddings[id], axis=0), len(input_texts))
                for id, score in zip(sorted_ids[0], scores[0]):
                    if id not in id2score:
                        id2score[id] = score
                    else:
                        id2score[id] += score

            for i in reject_i:
                id = ids[i]
                scores, sorted_ids = dense_searcher.search(np.expand_dims(all_embeddings[id], axis=0), len(input_texts))
                for id, score in zip(sorted_ids[0], scores[0]):
                    if id not in id2score:
                        id2score[id] = -score
                    else:
                        id2score[id] -= score
            id2score = sorted(id2score.items(), key=lambda item: -item[1])
            ids = [x[0] for x in id2score]
        
        with use_scope('table'):
            clear()  # clear old table

        print(ids)
        accept_i = []
        reject_i = []
        L = [["id", "Data"]]
        for i, id in enumerate(ids):
            L.append([i, put_html(f'<h3 style="background-color:cyan;"> Profile {id} </h3>')])
            L.append(["", print_profile(id)])
            L.append(['', put_scope(f'Accept-{i}', put_buttons([
                {"label": f'Accept', "value": f'Accept', "color": "info"}], onclick=partial(Onclick, i=i)))]),
            L.append(['', put_scope(f'Reject-{i}', put_buttons([
                {"label": f'Reject', "value": f'Reject', "color": "dark"}], onclick=partial(Onclick, i=i)))]),
        with use_scope('loading'):
            time.sleep(2)
            clear()
        with use_scope('table'):
            put_table(L)
        return x

    ids = list(range(len(input_texts)))
    put_buttons(["APPLY"], onclick=Apply)
