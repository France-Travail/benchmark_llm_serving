import logging
import asyncio
import aiohttp
import argparse
import numpy as np
from datetime import datetime
from typing import List, Tuple

from benchmark_llm_serving.backends import BackEnd
from benchmark_llm_serving.utils import tasks_are_done, get_now
from benchmark_llm_serving.utils_metrics import get_live_metrics
from benchmark_llm_serving.io_classes import QueryOutput, QueryInput
from benchmark_llm_serving.query_profiles.query_functions import query_function


def add_poisson_rate(queries_dataset: List[QueryInput], args: argparse.Namespace) -> List[QueryInput]:
    """Adds a poisson rate to the queries

    Args:
        queries_dataset (list) : The list of query inputs to add the scheduled delta to
        args (argparse.Namespace) : The cli args (containing the request_rate)

    Returns:
        list : The list of query_input with the scheduled delta added
    """
    current_delta = 0
    for query_input in queries_dataset:
        delta = np.random.exponential(1.0 / args.request_rate)
        current_delta += delta
        query_input.scheduled_delta = current_delta
    return queries_dataset


def add_schedule_to_queries(queries_dataset: List[QueryInput], args: argparse.Namespace) -> List[QueryInput]:
    """Adds a schedule delta to the queries

    Args:
        queries_dataset (list) : The list of query inputs to add the scheduled delta to
        args (argparse.Namespace) : The cli args (containing the request_rate)

    Returns:

        list : The list of query_input with the scheduled delta added
    """
    if args.query_profile == "request_rate":
        # If the request rate is infinite, all request are sent at t=0 so we explicitly 
        # put scheduled delta to zero otherwise, we must schedule them
        if args.request_rate != float('inf'):
            queries_dataset = add_poisson_rate(queries_dataset, args)
        else:
            for query in queries_dataset:
                query.scheduled_delta = 0
    # Sort the queries by scheduled_delta
    queries_dataset = sorted(queries_dataset, key = lambda query_input: query_input.scheduled_delta)
    return queries_dataset


def get_queries_to_launch(queries_dataset: List[QueryInput], current_query_index_to_launch: int, current_timestamp: float) -> Tuple[List[QueryInput], int]:
    """Gets the queries that should be launched by comparing the current timestamp and the timestamp at which they should be launched.

    Args:
         queries_dataset (list) : The list of input queries. They should be ordered by their scheduled timestamp
         current_query_index_to_launch (int) : The index of the last query to be launched
         current_timestamp (float) : The current timestamp

    Returns:
        list[QueryInput] : The list of the queries to launch
        int: The new index of the last query to be launched
    """
    new_index = current_query_index_to_launch
    before_current_timestamp = True
    queries_to_launch = []
    # While the current timestamp is after some scheduled timestamp, we add the corresponding query
    while before_current_timestamp:
        query_to_consider = queries_dataset[new_index]
        # If the scheduled timestamp of the query is before the current timestamp 
        if query_to_consider.scheduled_timestamp <= current_timestamp:
            # We add the query to those that should be launched
            queries_to_launch.append(query_to_consider)
            # We go to the next query
            new_index += 1
            # If there are no more query in the list
            if new_index == len(queries_dataset):
                before_current_timestamp = False
        # Else, the query should not be launched, we exit the loop
        else:
            before_current_timestamp = False
    return queries_to_launch, new_index


def continue_condition(current_query_index_to_launch: int, max_queries_number: int, max_duration_reached: bool, 
                     min_duration_reached: bool, args: argparse.Namespace) -> bool:
    """Gives the conditions to continue the queries

    Args:
        current_query_index_to_launch (int) : The current index of query to launch
        max_queries_number (int) : The maximal number of queries
        max_duration_reached (bool) : Whether the max duration has been reached
        min_duration_reached (bool) : Whether the min duration has been reached
        args (argparse.Namespace) : The CLI args

    Returns:
        bool : Whether we should continue querying the model
    """
    # Max queries not reached
    first_condition = current_query_index_to_launch < max_queries_number
    # Min queries not reached
    second_condition_part_1 = current_query_index_to_launch < args.target_queries_nb
    second_condition = second_condition_part_1 or not(min_duration_reached)
    return first_condition and second_condition and not(max_duration_reached)


async def get_benchmark_results_scheduled_requests(queries_dataset: List[QueryInput], args: argparse.Namespace, completions_url: str,
                                                   metrics_url: str, logger: logging.Logger, backend: BackEnd)  -> Tuple[List[QueryOutput], List[dict]]:
    """Gets the results for the benchmark and the live metrics, using scheduled queries ie, queries whose timestamp we can calculate
    before actually launching the queries.

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
    all_live_metrics: List[dict] = []
    queries_dataset = add_schedule_to_queries(queries_dataset, args)
    
    max_queries_number = len(queries_dataset)
    current_query_index_to_launch = 0
    tasks = []
    results: List[QueryOutput] = []
    connector = aiohttp.TCPConnector(limit=10000)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Query the /metrics endpoint for one second before launching the first queries
        for i in range(int(1/args.step_live_metrics)):
            asyncio.create_task(get_live_metrics(session, metrics_url, all_live_metrics, backend))
            await asyncio.sleep(args.step_live_metrics)
        start_queries_timestamp = datetime.now().timestamp()
        # Add the initial timestamp to the queries
        for query_input in queries_dataset:
            query_input.add_starting_timestamp(start_queries_timestamp)
        max_duration_reached = False
        min_duration_reached = False
        # While we still have queries to consider
        while continue_condition(current_query_index_to_launch, max_queries_number, max_duration_reached, min_duration_reached, args):
            old_query_index_to_launch = current_query_index_to_launch
            current_timestamp = datetime.now().timestamp()
            if current_timestamp - start_queries_timestamp >= args.max_duration:
                max_duration_reached = True
                now = get_now()
                logger.info(f"{now} Max duration {args.max_duration}s has been reached")
            if current_timestamp - start_queries_timestamp >= args.min_duration:
                min_duration_reached = True
            queries_to_launch, current_query_index_to_launch = get_queries_to_launch(queries_dataset,
                                                                                     current_query_index_to_launch,
                                                                                     current_timestamp)
            if current_query_index_to_launch // int(args.max_queries / 10) != old_query_index_to_launch // int(args.max_queries / 10):
                now = get_now()
                logger.info(f"{now} {current_query_index_to_launch} requests in total have been launched")
            tasks += [asyncio.create_task(query_function(query_input, session, completions_url, results, args, backend)) for query_input in queries_to_launch]

            asyncio.create_task(get_live_metrics(session, metrics_url, all_live_metrics, backend))
            await asyncio.sleep(args.step_live_metrics)

        # Once all queries have been sent, we still query the /metrics endpoint
        # Until all the queries are done
        while not tasks_are_done(tasks):
            asyncio.create_task(get_live_metrics(session, metrics_url, all_live_metrics, backend))
            await asyncio.sleep(args.step_live_metrics)

        # Query the /metrics endpoint for one second after launching the queries are done
        for i in range(int(1/args.step_live_metrics)):
            asyncio.create_task(get_live_metrics(session, metrics_url, all_live_metrics, backend))
            await asyncio.sleep(args.step_live_metrics)
    return results, all_live_metrics