import pprint
from core.module.graph import GraphState, Graph
from langgraph.graph import StateGraph, START, END

class Builder():
    def __init__(self) -> None:
        self.module = Graph()

    def compile(self):
        graph = StateGraph(GraphState)
        graph.add_node("search_arxiv", self.module.search_arxiv)
        graph.add_node("search_google_scholar", self.module.search_google_scholar)
        graph.add_node("post_processing", self.module.post_processing)

        graph.add_edge(START, "search_arxiv")
        graph.add_edge(START, "search_google_scholar")
        graph.add_edge("search_arxiv", "post_processing")
        graph.add_edge("search_google_scholar", "post_processing")
        graph.add_edge("post_processing", END)

        follow_up_agent = graph.compile()
        return follow_up_agent

    def run(self, graph:object, instruction:str):
        input_model = {
            'messages':[instruction],
            'reference_list': []
        }

        for event in graph.stream(input_model):
            for key, value in event.items():
                print(f"\n==============\n  STEP: {key}  \n==============\n")
                pprint.pprint(value)
