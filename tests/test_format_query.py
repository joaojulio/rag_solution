import pytest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.langchain_integration.retrieval_qa_chain import format_query

def test_format_query():
    role = "Role Test"
    goal = "Goal Test"
    backstory = "Backstory Test"
    user_question = "What is testing?"
    expected = "Role: Role Test\nGoal: Goal Test\nBackstory: Backstory Test\nQuestion: What is testing?"
    result = format_query(role, goal, backstory, user_question)
    assert result == expected
