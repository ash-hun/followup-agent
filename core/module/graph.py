import os
import time
import arxiv
import pprint
import operator
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from serpapi import GoogleSearch
from typing import Annotated, List, TypedDict
from langchain_community.utils.math import cosine_similarity

load_dotenv('./env')

class GraphState(TypedDict):
    messages: Annotated[List, operator.add]
    reference_list: Annotated[List, operator.add]

class Graph():
    def __init__(self) -> None:
        self.client = OpenAI()
    def _invoke(self, instruction:str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": instruction}]
        )

        return response.choices[0].message.content

    def _embedding(self, target:str) -> List:
        response = self.client.embeddings.create(
            input=target,
            model="text-embedding-3-small"
        )

        return response.data[0].embedding

    def search_arxiv(self, state:GraphState, verbose:bool=False) -> GraphState:
        # Construct the default API client.
        start_time = time.process_time()
        client = arxiv.Client()

        search = arxiv.Search(
            query=state['messages'][0],
            max_results=15,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        results = client.results(search)
        end_time = time.process_time()

        paper_list = {idx: [paper.title, paper.entry_id, paper.summary] for idx, paper in enumerate(list(results))}

        print("=" * 10)
        print(f"Arxiv Search : {int(round((end_time-start_time)*1000))}")
        print("=" * 10)

        if verbose:
            pprint.pprint(paper_list)

        state['reference_list'].append(paper_list)
        return state

    def search_google_scholar(self, state:GraphState) -> GraphState:
        # Construct the default API client.
        params = {
            'engine': "google_scholar",
            'q': state['messages'][0],
            'scisbd': 1,
            'num':15,
            'api_key': os.getenv('SERPAPI')
        }

        start_time = time.process_time()
        search_result = GoogleSearch(params)
        results = search_result.get_dict()
        end_time = time.process_time()

        print(results)

        organic_results = results["organic_results"]

        scholar_info = {}
        for item in organic_results:
            idx = item['position']
            tmp = []
            tmp.append(item['title'])
            tmp.append(item['link'])
            tmp.append(item['snippet'])
            scholar_info[idx] = tmp

        state['reference_list'].append(scholar_info)

        print("=" * 10)
        print(f"Google Scholar Search : {int(round((end_time - start_time) * 1000))}")
        print("=" * 10)
        return state

    def post_processing(self, state:GraphState) -> GraphState:
        # User Query Embedding
        user_question = state['messages'][0]
        embedded_question = self._embedding(user_question)

        # Each Reference Embedding
        ref_list = state['reference_list']
        embedded_ref_list = []
        for item in tqdm(ref_list, desc='Embedding...'):
            for k, v in item.items():
                tmp = []
                tmp.append(v[0]) # title
                tmp.append(v[1]) # link
                contextual_summary = f"""{v[0]}\n{v[2]}"""
                tmp.append(self._embedding(contextual_summary)) # embedded
                embedded_ref_list.append(tmp)

        # Calculate Cosine Similarity and Sorting
        for idx, ref_item in enumerate(embedded_ref_list):
            embedded_ref_list[idx].append(cosine_similarity([embedded_question], [embedded_ref_list[idx][-1]]))

        sorted_list = sorted(embedded_ref_list, key=lambda x: -x[-1])
        post_processing_list = []
        title_tmp = []
        for idx, item in enumerate(sorted_list):
            tmp = [item[0], item[1], item[-1]]
            if len(post_processing_list)<5:
                if idx == 0:
                    post_processing_list.append(tmp)
                    title_tmp.append(item[0])
                else:
                    if item[0] not in title_tmp:
                        post_processing_list.append(tmp)
                        title_tmp.append(item[0])
            else:
                break

        state['reference_list'] = post_processing_list
        return state