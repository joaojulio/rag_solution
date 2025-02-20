import pytest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.langchain_integration.retrieval_qa_chain import clarify_question

class DummyAIMessage:
    def __init__(self, content):
        self.content = content

class DummyClarifier:
    def __init__(self, response_content):
        self.response_content = response_content
    def invoke(self, prompt):
        return DummyAIMessage(self.response_content)

def test_clarify_question_not_ambiguous(monkeypatch):
    dummy_response = "OK"
    monkeypatch.setattr("src.langchain_integration.retrieval_qa_chain.ChatOpenAI", lambda **kwargs: DummyClarifier(dummy_response))
    result = clarify_question("What is evolution?", "dummy_key")
    assert result is None

def test_clarify_question_ambiguous(monkeypatch):
    clarifying_question = "Could you specify if you mean biological evolution or technological evolution?"
    monkeypatch.setattr("src.langchain_integration.retrieval_qa_chain.ChatOpenAI", lambda **kwargs: DummyClarifier(clarifying_question))
    result = clarify_question("How does evolution work?", "dummy_key")
    assert result == clarifying_question