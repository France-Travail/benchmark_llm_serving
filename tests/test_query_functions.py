import pytest
import aiohttp
import argparse
from aioresponses import aioresponses

from benchmark_llm_serving.backends import get_backend
from benchmark_llm_serving.query_profiles import query_functions
from benchmark_llm_serving.io_classes import QueryInput, QueryOutput


async def mock_streaming_response_content(args, backend):
    tokens_list = ["Hey", " how", " are", " you", " ?"] * 1000
    tokens_list = tokens_list[:args.output_length]
    if backend.backend_name == "happy_vllm":
        for element in iter(tokens_list):
            string = f"""data: {{"id":"cmpl-d85f82039b864ceb8d95be931b200745", "object":"chat.completion.chunk", "created":1716468615, "model":"CodeLlama-34B-AWQ", "choices":[{{"index":0,"text":"{element}","stop_reason":null,"logprobs":null,"finish_reason":null}}]}}"""
            yield bytes(string, "utf-8")
    if backend.backend_name == "mistral":
        for element in iter(tokens_list):
            string = f"""data: {{"id":"cbaa5c28166d4b98b5256f1becc0364d","object":"chat.completion.chunk","created":1722322855,"model":"mistral","choices":[{{"index":0,"delta":{{"content":"{element}"}},"finish_reason":null,"logprobs":null}}]}}"""
            yield bytes(string, "utf-8")
            yield bytes("tok-o: 1", "utf-8")

class MockSession():

    def __init__(self, args, backend):
        self.args = args
        self.backend = backend

    def post(self, **kwargs):
        return MockResponse(self.args, self.backend)

class MockResponse():

    def __init__(self, args, backend):
        self.status = 200
        self.content = mock_streaming_response_content(args, backend)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        pass

@pytest.mark.asyncio()
async def test_query_function():
    # First call
    output_length = 10
    backend = get_backend("happy_vllm")
    args = argparse.Namespace(output_length=output_length, model="CodeLlama-34B-AWQ")
    prompt = "Hey !"
    async_generator = mock_streaming_response_content(args, backend)
    query_input = QueryInput(prompt=prompt, internal_id=0)
    results = []

    session = MockSession(args, backend)
    await query_functions.query_function(query_input, session, "my_url", results=results,
                    args=args, backend=backend)
    assert len(results) == 1
    assert isinstance(results[0], QueryOutput)
    print(results)
    assert len(results[0].timestamp_of_tokens_arrival) == output_length

    # Another call
    output_length = 5
    backend = get_backend("happy_vllm")
    args = argparse.Namespace(output_length=output_length, model="CodeLlama-34B-AWQ")
    prompt = "Hey !"
    async_generator = mock_streaming_response_content(args, backend)
    query_input = QueryInput(prompt=prompt, internal_id=1)

    session = MockSession(args, backend)
    await query_functions.query_function(query_input, session, "my_url", results=results,
                    args=args, backend=backend)
    assert len(results) == 2
    assert isinstance(results[1], QueryOutput)
    assert len(results[1].timestamp_of_tokens_arrival) == output_length

    # mistral backend
    output_length = 10
    backend = get_backend("mistral")
    args = argparse.Namespace(output_length=output_length, model="mistral")
    prompt = "Hey !"
    async_generator = mock_streaming_response_content(args, backend)
    query_input = QueryInput(prompt=prompt, internal_id=0)
    results = []

    session = MockSession(args, backend)
    await query_functions.query_function(query_input, session, "my_url", results=results,
                    args=args, backend=backend)
    assert len(results) == 1
    assert isinstance(results[0], QueryOutput)
    assert len(results[0].timestamp_of_tokens_arrival) == output_length