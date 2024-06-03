import aiohttp
from typing import List
from datetime import datetime
from prometheus_client.parser import text_string_to_metric_families


def parse_metrics_response(response_text: str) -> dict:
    """Parses the text of a response from the /metrics endpoint to get a dictionary

    Args:
        response_text (str) : The string obtained from querying the /metrics endpoint

    Returns:
        dict : The output of querying the /metrics endpoint in dict form
    """
    results = {}
    family_generator = text_string_to_metric_families(response_text)
    for family in family_generator:
        list_samples = []
        for sample in family.samples:
            dict_tmp = {"name": sample.name, "labels": sample.labels, "value": sample.value, 
                        "timestamp": sample.timestamp, "exemplar": sample.exemplar}
            list_samples.append(dict_tmp.copy())
        results[family.name] = list_samples.copy()
    return results


def get_live_vllm_metrics(response_text: str) -> dict:
    """From the text of the response of the /metrics endpoint, gets the metrics relevant to us

    Args:
        response_text (str) : The string obtained from querying the /metrics endpoint

    Returns:
        dict : A dictionary containing the percentage of KV cache used, the number of running requests
                and the number of waiting requests
    """
    vllm_metrics = {}
    parsed_response = parse_metrics_response(response_text)
    vllm_metrics['num_requests_running'] = parsed_response['vllm:num_requests_running'][0]['value']
    vllm_metrics['num_requests_waiting'] = parsed_response['vllm:num_requests_waiting'][0]['value']
    vllm_metrics['gpu_cache_usage_perc'] = parsed_response['vllm:gpu_cache_usage_perc'][0]['value']
    return vllm_metrics


async def get_live_metrics(session: aiohttp.ClientSession, metrics_url: str, all_live_metrics: List[dict]) -> None:
    """Queries the /metrics endpoint, gets the live metrics and add them to the list all_live_metrics

    Args:
        session (aiohttp.ClientSession) : The aiohttp session
        metrics_url (str) : The url to the /metrics endpoint
        all_live_metrics (list) : The list to which we add the live metrics results
    """
    tmp_list = []
    async with session.get(url=metrics_url) as response:
        if response.status == 200:
            async for chunk_bytes in response.content:
                tmp_list.append(chunk_bytes.decode('utf-8'))
    live_metrics = get_live_vllm_metrics("".join(tmp_list))
    timestamp = datetime.now().timestamp()
    live_metrics['timestamp'] = timestamp
    all_live_metrics.append(live_metrics.copy())