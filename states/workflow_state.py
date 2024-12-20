from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class WorkflowState(TypedDict):
    """State management for the legal chatbot workflow"""
    messages: List[BaseMessage]
    context: dict
    current_step: str
    
def get_initial_state() -> WorkflowState:
    """Initialize the workflow state"""
    return WorkflowState(
        messages=[],
        context={},
        current_step="start"
    )
