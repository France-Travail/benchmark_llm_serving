import os
import json
import random
import asyncio
import logging
import requests # type: ignore
import argparse
import numpy as np
from datetime import datetime
from typing import List, Tuple, Union, Any

from benchmark_llm_serving import utils
from benchmark_llm_serving.utils_args import parse_args
from benchmark_llm_serving.backends import get_backend, BackEnd
from benchmark_llm_serving.io_classes import QueryOutput, QueryInput
from benchmark_llm_serving.query_profiles.query_functions import query_function
from benchmark_llm_serving.query_profiles.constant_number import get_benchmark_results_constant_number
from benchmark_llm_serving.query_profiles.growing_requests import get_benchmark_results_growing_requests
from benchmark_llm_serving.query_profiles.scheduled_requests import get_benchmark_results_scheduled_requests


logger = logging.getLogger("Benchmark for LLMs")
logging.basicConfig(level=logging.INFO)


async def get_benchmark_results(queries_dataset: List[QueryInput], args: argparse.Namespace,
                                logger: logging.Logger, backend: BackEnd) -> Tuple[List[QueryOutput], List[dict]]:
    """Gets the results of the benchmark

    Args:
        queries_dataset (list) : The queries we want to use
        args (argparse.Namespace) : The CLI args
        logger (logging.Logger) : The logger
        backend (Backend) : The backend to consider
    
    Returns:
        list[QueryOutput] : The list of the result for each query
        list[dict] : The list of live metrics
    """
    completions_url = args.base_url + args.completions_endpoint
    metrics_url = args.base_url + args.metrics_endpoint

    # Make one query in order to be sure that everything is ok
    query_input = QueryInput(prompt="Hey", internal_id=-1)
    payload = backend.get_payload(query_input, args)
    headers = backend.get_completions_headers()
    response = requests.post(completions_url, json=payload, timeout=100, headers=headers)
    status_code = response.status_code
    # if status_code != 200:
    #     raise ValueError(f"The status code of the response is {status_code} instead of 200")
    
    if args.query_profile == "constant_number_of_queries":
        results, all_live_metrics = await get_benchmark_results_constant_number(queries_dataset, args, completions_url, metrics_url, logger, backend)
    elif args.query_profile == "growing_requests":
        results, all_live_metrics = await get_benchmark_results_growing_requests(queries_dataset, args, completions_url, metrics_url, logger, backend)
    else:
        results, all_live_metrics = await get_benchmark_results_scheduled_requests(queries_dataset, args, 
                                                                                   completions_url, metrics_url, logger, backend)
    return results, all_live_metrics


def augment_dataset(dataset: List[str], max_queries: int) -> List[str]:
    """Augments the dataset if we want more prompt. We just a growing number to the beginning of the existing prompts 
    to ensure that they don't have the same prefix except the one we add

    Args:
        dataset (list) : The list of prompt
        max_queries (int) : The maximum number of queries we want to consider
    
    Returns:
        list : The augmented list of prompts
    """
    new_dataset = dataset.copy()
    count = 0
    # If there are not enough prompts in the dataset, we add some
    while len(new_dataset) < max_queries:
        new_dataset += [f'{count} {prompt}' for prompt in dataset]
        count += 1
    return new_dataset[:max_queries]


def get_model_name_from_info_endpoint(args: argparse.Namespace) -> str:
    """Gets the model name from the info endpoint

    Args:
        args (argparse.Namespace) : The cli args

    Returns:
        str : The name of the model
    """
    info = requests.get(args.base_url + args.info_endpoint, timeout=3600).json()
    return info['model_name']


def add_application_parameters(parameters: dict, args: argparse.Namespace) -> dict:
    """Adds happy_vllm and vllm versions to the parameters

    Args:
        parameters (dict) : The parameters of the API and of this script
        args (argparse.Namespace) : The cli args

    Returns:
        dict : The updated parameters
    """
    info = requests.get(args.base_url + args.info_endpoint, timeout=3600).json()
    parameters['happy_vllm_version'] = info['version']
    parameters["vllm_version"] = info["vllm_version"]
    launch_arguments = requests.get(args.base_url + args.launch_arguments_endpoint, timeout=3600).json()
    launch_arguments.pop("model")
    if parameters["model_name"] is not None:
        launch_arguments.pop("model_name")
    parameters.update(launch_arguments)
    return parameters


def get_general_metrics(benchmark_results: List[QueryOutput], all_live_metrics: List[dict], args: argparse.Namespace) -> dict:
    """Calculates the general metrics from each output

    Args:
        benchmark_results (list) : The outputs of the benchmark
        all_live_metrics (list) : The list of live metrics
        args (argparse.Namespace) : The cli args

    Returns:
        dict : The general metrics
    """
    general_metrics = {}
    
    if args.backend == "happy_vllm":
        general_metrics['max_kv_cache'] = max([metric['gpu_cache_usage_perc'] for metric in all_live_metrics])
        general_metrics['max_requests_running'] = max([metric['num_requests_running'] for metric in all_live_metrics])
        general_metrics['max_requests_waiting'] = max([metric['num_requests_waiting'] for metric in all_live_metrics])
    else:
        general_metrics['max_kv_cache'] = -1
        general_metrics['max_requests_running'] = -1
        general_metrics['max_requests_waiting'] = -1
    
    general_metrics['total_number_of_queries'] = len(benchmark_results)
    general_metrics['nb_timeout_queries'] = len([result for result in benchmark_results if result.timeout])
    general_metrics['nb_errored_queries'] = len([result for result in benchmark_results if not(result.success)])

    # We consider only the queries which succeded or which were in timeout (since they still generated some tokens)
    new_benchmark_results = [result for result in benchmark_results if result.success or result.timeout]
    general_metrics['total_time'] = sum([result.total_query_time for result in new_benchmark_results])
    general_metrics['total_time_from_first_token'] = sum([result.completion_time_from_first_token for result in new_benchmark_results])
    general_metrics['total_time_without_waiting_time'] = sum([result.total_query_time - result.total_waiting_time for result in new_benchmark_results])
    general_metrics['total_number_of_ingested_tokens'] = sum([result.prompt_length for result in new_benchmark_results])
    general_metrics['total_number_of_generated_tokens'] = sum([result.response_length for result in new_benchmark_results])
    
    if general_metrics['total_time'] == 0:
        general_metrics['speed'] = 0
    else:
        general_metrics['speed'] = general_metrics['total_number_of_generated_tokens'] / general_metrics['total_time']
        general_metrics['speed_without_waiting_time'] = general_metrics['total_number_of_generated_tokens'] / general_metrics['total_time_without_waiting_time']
    if general_metrics['total_time_from_first_token'] == 0:
        general_metrics['speed_from_first_token'] = 0
    else:
        general_metrics['speed_from_first_token'] = general_metrics['total_number_of_generated_tokens'] / general_metrics['total_time_from_first_token']
    return general_metrics
    

def get_aggregated_metrics(benchmark_results: List[QueryOutput]) -> dict:
    """Calculates the aggregated metric

    Args:
        benchmark_results (list) : The outputs of the benchmark
    
    Returns:
        dict : The aggregated metrics
    """
    # We consider only the queries which succeded or which were in timeout (since they still generated some tokens)
    new_benchmark_results = [result for result in benchmark_results if result.success or result.timeout]
    aggregated_metrics = {}
    for stat in ['time_to_first_token', "prompt_length", "response_length", "total_query_time", "completion_time_from_first_token",
                "median_time_between_tokens", "total_waiting_time", "speed_from_beginning", "speed_from_first_token",
                "speed_without_waiting_time"]:
        all_stats = [getattr(result, stat) for result in new_benchmark_results]
        if len(all_stats):
            aggregated_metrics[stat] = {
                "mean": np.mean(all_stats),
                "median": np.median(all_stats)
                }
            for percentile in [1, 5, 10, 25, 75, 90, 95, 99]:
                aggregated_metrics[stat][f"percentile_{percentile}"] = np.percentile(all_stats, percentile)
    return aggregated_metrics


def launch_benchmark(args: argparse.Namespace, provided_dataset: Union[List[str], None] = None, suite_id: Union[str, None] = None, backend: Union[BackEnd, None] = None):
    """Calculates the results of a benchmark. We can explicitly give another dataset.

    Args:
        args (argparse.Namespace) : The cli args
        provided_dataset (list) : If provided, replace the loading
        suite_id (str) : The id to identify several benchmarks launched by the same bench suite
        backend (BackEnd) : The backend to consider. If None, will infer it from args

    """
    now = utils.get_now()
    logger.info(f"{now} Loading the dataset")
    if provided_dataset is not None:
        dataset = provided_dataset.copy()
        random.shuffle(dataset)
    else:
        dataset = utils.load_dataset(args.dataset_folder, args.prompt_length)

    if args.base_url is None:
        args.base_url = f"http://{args.host}:{args.port}"

    if backend is None:
        backend = get_backend(args.backend)

    if args.model is None:
        if backend.backend_name != "happy_vllm":
            raise ValueError(f"No model is specified and the backend is not happy_vllm (it is '{backend.backend_name}'). Please provide a model name")
        args.model = get_model_name_from_info_endpoint(args)

    if args.output_length is None:
        raise ValueError("Please specify an output length")

    parameters = {
        "model": args.model,
        "prompt_length": args.prompt_length,
        "n_workers": args.n_workers,
        "query_profile": args.query_profile,
        "step_live_metrics": args.step_live_metrics,
        "max_queries": args.max_queries,
        "request_rate": args.request_rate,
        "backend": args.backend,
        "output_length": args.output_length,
        "suite_id" : suite_id,
        "model_name": args.model_name
        }

    if backend.backend_name == "happy_vllm":
        parameters = add_application_parameters(parameters, args)
    if parameters["model_name"] is None:
        parameters["model_name"] = parameters["model"]
        
    if args.min_duration is None:
        args.min_duration = args.max_duration

    if args.target_queries_nb is None:
        args.target_queries_nb = args.max_queries

    if args.max_queries is not None:
        dataset = augment_dataset(dataset, args.max_queries)
        

    queries_dataset = [QueryInput(prompt=prompt, 
                                    internal_id=i) 
                                    for i, prompt in enumerate(dataset)]

    now = utils.get_now()
    logger.info(f"{now} Beginning the requests to the completions endpoint")
    start_timestamp = datetime.now().timestamp()
    benchmark_results, all_live_metrics = asyncio.run(get_benchmark_results(queries_dataset, args, logger, backend))
    end_timestamp = datetime.now().timestamp()

    now = utils.get_now()
    logger.info(f"{now} Requests to the completions endpoint done")

    # Taking care of the timeout queries and calculating derived stats
    for result in benchmark_results:
        if isinstance(result.error, str):
            if 'raise asyncio.TimeoutError' in result.error:
                result.timeout = True
                if len(result.timestamp_of_tokens_arrival):
                    result.ending_timestamp = result.timestamp_of_tokens_arrival[-1]
                else:
                    result.ending_timestamp = result.starting_timestamp
        result.calculate_derived_stats()

    general_metrics = get_general_metrics(benchmark_results, all_live_metrics, args)
    general_metrics['actual_total_time'] = end_timestamp - start_timestamp
    aggregated_metrics = get_aggregated_metrics(benchmark_results)

    if general_metrics['nb_errored_queries']:
        nb_errors = general_metrics['nb_errored_queries']
        nb_timeout = general_metrics['nb_timeout_queries']
        now = utils.get_now()
        logger.warning(f"{now} There are {nb_errors} queries in error including {nb_timeout} queries in timeout")

    now = utils.get_now()
    logger.info(f"{now} Derived calculations done")
    
    final_json: dict[Any, Any] = {"parameters": parameters, "general_metrics": general_metrics,
                    "aggregated_metrics": aggregated_metrics}
    # Save errored queries except those in timeout
    errored_results = [result for result in benchmark_results if not(result.success) and not(result.timeout)]
    errored_results = [result.to_dict() for result in errored_results]
    final_json['errored_queries'] = errored_results
    if args.query_metrics:
        benchmark_results = [result.to_dict() for result in benchmark_results if result.success or result.timeout]
        for result in benchmark_results:
            result["prompt"] = ""
        final_json['individual_query'] = benchmark_results # type: ignore
    
    if args.with_kv_cache_profile and args.backend == "happy_vllm":
        final_json["kv_cache_profile"] = all_live_metrics # type: ignore
    with open(args.output_file, 'w') as json_file:
        json.dump(final_json, json_file, indent=4)


if __name__ == "__main__":
    args = parse_args()
    launch_benchmark(args)