from __future__ import annotations
from typing import Any, cast
from langgraph.graph import StateGraph, START, END
from llm import get_llm
from prompts import AGENT_TOOLS
from nodes import (
    State,
    set_active_llm,
    agent_node,
    read_tool,
    prepare_proposal,
    approval,
    route_after_agent,
    route_after_prepare,
)


def build_graph(checkpointer=None, llm=None):

    if llm is None:
        llm = get_llm()
    set_active_llm(llm.bind_tools(AGENT_TOOLS))

    builder = StateGraph(cast(Any, State))
    builder.add_node("agent", agent_node)

    builder.add_node("read_tool", read_tool)
    builder.add_node("prepare_proposal", prepare_proposal)
    builder.add_node("approval", approval)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"read_tool": "read_tool", "prepare_proposal": "prepare_proposal", END: END},
    )
    builder.add_edge("read_tool", "agent")
    builder.add_conditional_edges(
        "prepare_proposal",
        route_after_prepare,
        {"approval": "approval", "agent": "agent"},
    )
    builder.add_edge("approval", END)

    return builder.compile(checkpointer=checkpointer)
