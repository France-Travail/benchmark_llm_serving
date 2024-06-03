import pytest
import aiohttp
import argparse
from aioresponses import aioresponses

from benchmark_llm_serving.query_profiles import query_functions
from benchmark_llm_serving.io_classes import QueryInput, QueryOutput


async def mock_streaming_response_content(args):
    tokens_list = ["Hey", " how", " are", " you", " ?"] * 1000
    tokens_list = tokens_list[:args.output_length]
    for element in iter(tokens_list):
        string = f"""data: {{"id":"cmpl-d85f82039b864ceb8d95be931b200745", "object":"chat.completion.chunk", "created":1716468615, "model":"CodeLlama-34B-AWQ", "choices":[{{"index":0,"text":"{element}","stop_reason":null,"logprobs":null,"finish_reason":null}}]}}"""
        yield bytes(string, "utf-8")

class MockSession():

    def __init__(self, args):
        self.args = args

    def post(self, **kwargs):
        return MockResponse(self.args)

class MockResponse():

    def __init__(self, args):
        self.status = 200
        self.content = mock_streaming_response_content(args)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        pass

@pytest.mark.asyncio()
async def test_query_function():
    # First call
    output_length = 10
    args = argparse.Namespace(output_length=output_length, model="CodeLlama-34B-AWQ")
    prompt = "Hey !"
    async_generator = mock_streaming_response_content(args)
    query_input = QueryInput(prompt=prompt, internal_id=0)
    results = []

    session = MockSession(args)
    await query_functions.query_function(query_input, session, "my_url", results=results,
                    args=args)
    assert len(results) == 1
    assert isinstance(results[0], QueryOutput)
    assert len(results[0].timestamp_of_tokens_arrival) == output_length

    # Another call
    output_length = 5
    args = argparse.Namespace(output_length=output_length, model="CodeLlama-34B-AWQ")
    prompt = "Hey !"
    async_generator = mock_streaming_response_content(args)
    query_input = QueryInput(prompt=prompt, internal_id=1)

    session = MockSession(args)
    await query_functions.query_function(query_input, session, "my_url", results=results,
                    args=args)
    assert len(results) == 2
    assert isinstance(results[1], QueryOutput)
    assert len(results[1].timestamp_of_tokens_arrival) == output_length
