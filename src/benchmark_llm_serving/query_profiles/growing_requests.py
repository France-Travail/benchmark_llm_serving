import logging
import asyncio
import aiohttp
import argparse
from datetime import datetime
from typing import List, Tuple

from benchmark_llm_serving.backends import BackEnd
from benchmark_llm_serving.utils import tasks_are_done, get_now
from benchmark_llm_serving.utils_metrics import get_live_metrics
from benchmark_llm_serving.io_classes import QueryOutput, QueryInput
from benchmark_llm_serving.query_profiles.query_functions import query_function


def continue_condition(current_timestamp: float, start_queries_timestamp: float, nb_queries_launched: int,
                        n: int, len_queries_dataset: int, args: argparse.Namespace) -> bool:
    """Gives the conditions to continue the queries

    Args:
        current_timestamp (int) : The current timestamp
        start_queries_timestamp (int) : The timestamp of the beginning of queries
        nb_queries_launched (int) : The number of query already launched
        n (int) : The next number of parallel queries to launch
        len_queries_dataset (int) : The maximal number of queries
        args (argparse.Namespace) : The CLI args

    Returns:
        bool : Whether we should continue querying the model
    """
    # Max queries not reached
    first_condition = nb_queries_launched + n < len_queries_dataset
    # Max duration not reached
    second_condition = (current_timestamp - start_queries_timestamp) < args.max_duration
    # Min duration not reached or target number of queries not reached
    third_condition_part1 = (current_timestamp - start_queries_timestamp) < args.min_duration
    third_condition_part2 = nb_queries_launched < args.target_queries_nb
    third_condition = third_condition_part1 or third_condition_part2
    return first_condition and second_condition and third_condition


async def get_benchmark_results_growing_requests(queries_dataset: List[QueryInput], args: argparse.Namespace, completions_url: str,
                                                   metrics_url: str, logger: logging.Logger, backend: BackEnd)  -> Tuple[List[QueryOutput], List[dict]]:
    """Gets the results for the benchmark and the live metrics, using a growing number of queries. First one is sent, then when done, 
    two are sent, then when they are done, three are sent, etc.

    Args:
        queries_dataset (list) : The list of queries we consider (at this point, their schedule is not calculated)
        args (argparse.Namespace) : The CLI args
        completions_url (str) : The url of the completions API
        metrics_url (str) : The url to the /metrics endpoint
        logger (logging.Logger) : The logger
        backend (Backend) : The backend to consider

    Returns:
        list[QueryOutput] : The list of the result for each query
        list[dict] : The list of live metrics
    """
    results: List[QueryOutput] = []
    all_live_metrics: List[dict] = []
    tasks = []
    nb_queries_launched = 0

    connector = aiohttp.TCPConnector(limit=10000)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Query the /metrics endpoint for one second before launching the first queries
        for i in range(int(1/args.step_live_metrics)):
            asyncio.create_task(get_live_metrics(session, metrics_url, all_live_metrics, backend))
            await asyncio.sleep(args.step_live_metrics)
        start_queries_timestamp = datetime.now().timestamp()
        # For a number of queries
        for n in range(len(queries_dataset)):
            current_timestamp = datetime.now().timestamp()
            # If there is a sufficient number of queries not launched and the max duration has not been reached
            if continue_condition(current_timestamp, start_queries_timestamp, nb_queries_launched, n, len(queries_dataset), args):
                await asyncio.sleep(0.5)
                now = get_now()
                logger.info(f"{now} Launching {n} queries in parallel")
                tasks += [asyncio.create_task(query_function(query_input, session, completions_url, results, args, backend)) 
                                                for query_input in queries_dataset[nb_queries_launched: nb_queries_launched + n]]
                nb_queries_launched += n
                # While we wait for the tasks to be done, we query the /metrics endpoint
                while not tasks_are_done(tasks):
                    asyncio.create_task(get_live_metrics(session, metrics_url, all_live_metrics, backend))
                    await asyncio.sleep(args.step_live_metrics)
        if current_timestamp - start_queries_timestamp >= args.max_duration:
            now = get_now()
            logger.info(f"{now} Max duration {args.max_duration}s has been reached")
        # Query the /metrics endpoint for one second after launching the queries are done
        for i in range(int(1/args.step_live_metrics)):
            asyncio.create_task(get_live_metrics(session, metrics_url, all_live_metrics, backend))
            await asyncio.sleep(args.step_live_metrics)
    
    return results, all_live_metrics