from typing import Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import Field, BaseModel, ValidationError

class GraphState(TypedDict):
    pass