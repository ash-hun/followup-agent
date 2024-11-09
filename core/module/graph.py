from openai import OpenAI
from typing import Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import Field, BaseModel, ValidationError

class GraphState(TypedDict):
    pass

class Graph():
    def __init__(self) -> None:
        self.client = OpenAI()
    def invoke(self, instruction:str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": instruction}]
        )

        return response