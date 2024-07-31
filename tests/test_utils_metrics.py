import os
import json
import pytest
import aiohttp
import argparse
from pathlib import Path
from aioresponses import aioresponses

from benchmark_llm_serving import utils_metrics
from benchmark_llm_serving.backends import get_backend


def get_metrics_response():
    current_directory = Path(os.path.dirname(os.path.realpath(__file__)))
    metrics_response_file = current_directory / "data" / "metrics_response.txt"
    with open(metrics_response_file, 'r') as txt_file:
        metrics_response = txt_file.read()
    return metrics_response


def test_parse_metrics_response():
    metrics_response = get_metrics_response()
    parsed_metrics_response = utils_metrics.parse_metrics_response(metrics_response)
    assert isinstance(parsed_metrics_response, dict)
    for metric in ['vllm:num_requests_running', "vllm:num_requests_waiting",
                   "vllm:gpu_cache_usage_perc"]:
        assert metric in parsed_metrics_response
        assert isinstance(parsed_metrics_response[metric], list)
        assert 'value' in parsed_metrics_response[metric][0]


@pytest.mark.asyncio()
async def test_get_live_metrics():
    # backend happy_vllm
    backend = get_backend("happy_vllm")
    metrics_response = get_metrics_response()
    all_live_metrics = []
    nb_query = 10
    with aioresponses() as mocked:
        for i in range(nb_query):
            new_metrics_response = metrics_response.replace('vllm:num_requests_running{model_name="/home/data/models/Meta-Llama-3-8B-Instruct"} 2.0',
                                                            f'vllm:num_requests_running{{model_name="/home/data/models/Meta-Llama-3-8B-Instruct"}} {i}.0')
            mocked.get('my_url', status=200, body=new_metrics_response)
        session = aiohttp.ClientSession()
        for i in range(nb_query):
            results = await utils_metrics.get_live_metrics(session, 'my_url', all_live_metrics, backend)
    await session.close()
    assert len(all_live_metrics) == nb_query
    for i in range(nb_query):
        assert set(all_live_metrics[i]) == {"num_requests_running", "num_requests_waiting", "gpu_cache_usage_perc", "timestamp"}
        assert all_live_metrics[i]['num_requests_running'] == pytest.approx(float(f"{i}.0"))
        assert all_live_metrics[i]['num_requests_waiting'] == pytest.approx(1.0)
        assert all_live_metrics[i]['gpu_cache_usage_perc'] == pytest.approx(4.2)
        if i != 0:
            assert all_live_metrics[i]['timestamp'] != all_live_metrics[i-1]['timestamp']

    # backend mistral
    backend = get_backend("mistral")
    metrics_response = get_metrics_response()
    all_live_metrics = []
    nb_query = 10
    with aioresponses() as mocked:
        for i in range(nb_query):
            new_metrics_response = metrics_response.replace('vllm:num_requests_running{model_name="/home/data/models/Meta-Llama-3-8B-Instruct"} 2.0',
                                                            f'vllm:num_requests_running{{model_name="/home/data/models/Meta-Llama-3-8B-Instruct"}} {i}.0')
            mocked.get('my_url', status=200, body=new_metrics_response)
        session = aiohttp.ClientSession()
        for i in range(nb_query):
            results = await utils_metrics.get_live_metrics(session, 'my_url', all_live_metrics, backend)
    await session.close()
    assert len(all_live_metrics) == 0
