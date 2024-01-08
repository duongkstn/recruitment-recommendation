import json, os
import pandas as pd
from typing import Tuple, List, Dict
from elasticsearch import Elasticsearch, helpers


def get_data(folder: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Get data from folder.
    :param folder: folder including `.json` file
    :return: a Dataframe, which has columns: title, summary, industry, experience and text
    `text` column includes information of job-title, job-summary, job-company, job-industry
    """
    df = []
    for file_ in sorted(os.listdir(folder)):
        if not file_.endswith(".json"):
            continue
        sample = json.load(open(os.path.join(folder, file_)))
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

    def get_text(x: Dict) -> str:
        """
        Convert from job-title, job-summary, job-company, job-industry fields to text field
        :param x: dictionary
        :return: text
        """
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
    return df, input_texts


class ElasticsearchClient:
    """
    Elasticsearch Client
    """
    def __init__(self, hosts: str, verify_certs: bool, http_auth_user: str, http_auth_password: str,
                 index: str) -> None:
        self.es = Elasticsearch(hosts=hosts, verify_certs=verify_certs,
                                http_auth=(http_auth_user, http_auth_password))
        self.index = index

    def check_index_exist(self) -> bool:
        return self.es.indices.exists(index=self.index)

    def create_index(self) -> None:
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
        self.es.indices.create(index=self.index, mappings=body['mappings'], settings=body['settings'],
                               request_timeout=60)

    def bulk_data(self, bulk_data_create: List[Dict]) -> None:
        success, failed = helpers.bulk(self.es, bulk_data_create, index=self.index, request_timeout=30, refresh="false",
                                       stats_only=True)
        print(f"success = {success}, failed = {failed}")

    def search(self, query: Dict) -> Dict:
        """
        Get search result
        :param query:
        :return: Dict {id: score}
        """
        s = []
        for job in query['experience']:
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
                            "query": query["title"],
                            "boost": 100
                        }
                    }
                },
                {
                    "match": {
                        "summary": {
                            "query": query["summary"],
                            "boost": 3
                        }
                    }
                },
                {
                    "match": {
                        "industry": {
                            "query": query["industry"],
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
            response = self.es.search(index=self.index, size=1000,
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

