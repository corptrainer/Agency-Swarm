from typing import TYPE_CHECKING, Any, Dict

from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain.schema import AgentAction as LangchainAgentAction
    from langchain.schema import AgentFinish as LangchainAgentFinish
    from langchain.schema import HumanMessage as LangchainHumanMessage


# Create base classes that match langchain's structure
class BaseAgentAction(BaseModel):
    tool: str
    tool_input: Dict[str, Any] | str
    log: str


class BaseAgentFinish(BaseModel):
    return_values: Dict[str, Any]
    log: str


class BaseHumanMessage(BaseModel):
    content: str


# Initialize with our base implementations first
AgentAction = BaseAgentAction
AgentFinish = BaseAgentFinish
HumanMessage = BaseHumanMessage


def use_langchain_types():
    """Switch to using langchain types after langchain is imported"""
    global AgentAction, AgentFinish, HumanMessage
    from langchain.schema import AgentAction as LangchainAgentAction
    from langchain.schema import AgentFinish as LangchainAgentFinish
    from langchain.schema import HumanMessage as LangchainHumanMessage

    AgentAction = LangchainAgentAction
    AgentFinish = LangchainAgentFinish
    HumanMessage = LangchainHumanMessage
