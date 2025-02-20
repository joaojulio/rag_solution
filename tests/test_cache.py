import json
import hashlib
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.langchain_integration.retrieval_qa_chain import get_cache_key

def test_get_cache_key_consistency():
    params = {"input": "Test input"}
    key1 = get_cache_key(params)
    key2 = get_cache_key(params)
    assert key1 == key2

def test_get_cache_key_different():
    params1 = {"input": "Test input A"}
    params2 = {"input": "Test input B"}
    key1 = get_cache_key(params1)
    key2 = get_cache_key(params2)
    assert key1 != key2
