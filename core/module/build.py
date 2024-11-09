from typing import Dict, TypedDict
# from langgraph.graph import StateGraph, START, END
# from pydantic import Field, BaseModel, ValidationError

class Builder():
    def __init__(self, instruction:str) -> None:
        self.instruction = instruction

    