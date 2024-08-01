import random
import logging
import asyncio
import aiohttp
import argparse
from datetime import datetime
from typing import List, Tuple

from benchmark_llm_serving.utils import get_now
from benchmark_llm_serving.backends import BackEnd
from benchmark_llm_serving.utils_metrics import get_live_metrics
from benchmark_llm_serving.io_classes import QueryOutput, QueryInput
from benchmark_llm_serving.query_profiles.query_functions import query_function


async def worker_func(session: aiohttp.ClientSession, queue: asyncio.Queue, completions_url: str, results: List[QueryOutput],
                      args: argparse.Namespace, logger: logging.Logger, backend: BackEnd) -> None:
    """Queries the completions API to get the output using a worker and a queue

    Args:
        session (aiohttp.ClientSession) : The aiohttp session
        queue (asyncio.Queue) : The queue from which the worker will take the query to send
        completions_url (str) : The url of the completions API
        results (str) : The list of results to which we will add the output
        args (argparse.Namespace) : The cli args
        logger (logging.Logger) : The logger
        backend (Backend) : The backend to consider
    """
    while True:
        query_input = await queue.get()
        # Wait a random time to "desynchronize" the requests in parallel so that they don't always
        # begin or end at the same time. For the initial requests, we wait a longer time. We
        # don't wait if there are only one requests at a time
        if args.n_workers > 1:
            if len(results) < args.n_workers:
                    await asyncio.sleep(random.uniform(0.0, 0.1))
            else:
                await asyncio.sleep(random.uniform(0.0, 0.02))
        await query_function(query_input, session, completions_url, results, args, backend)
        if len(results) % int(args.max_queries / 10) == 0:
            now = get_now()
            logger.info(f'{now} {len(results)} queries have been completed')
        queue.task_done()


async def get_async_generator(list_to_iterate: list):
    """Transform a list into an asynchronous generator

    Args:
        list_to_iterate (list) : The list to convert
    """
    for element in iter(list_to_iterate):
        yield element


def continue_condition(current_timestamp: float, start_queries_timestamp: float, args: argparse.Namespace, count_query: int) -> bool:
    """Gives the conditions to continue the queries

    Args:
        current_timestamp (int) : The current timestamp
        start_queries_timestamp (int) : The timestamp of the beginning of queries
        args (argparse.Namespace) : The CLI args
        count_query (int) : The number of query already launched

    Returns:
        bool : Whether we should continue querying the model
    """
    # Max duration not reached
    first_condition = current_timestamp - start_queries_timestamp < args.max_duration
    # Min duration not reached or target number of queries not reached
    second_condition_part_1 = current_timestamp - start_queries_timestamp < args.min_duration
    second_condition_part_2 = count_query < args.target_queries_nb
    second_condition = second_condition_part_1 or second_condition_part_2
    return first_condition and second_condition


async def get_benchmark_results_constant_number(queries_dataset: List[QueryInput], args: argparse.Namespace, completions_url: str,
                                                   metrics_url: str, logger: logging.Logger, backend: BackEnd)  -> Tuple[List[QueryOutput], List[dict]]:
    """Gets the results for the benchmark and the live metrics, using workers so that there are always the same 
    number of queries launched

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
    
    queue: asyncio.Queue = asyncio.Queue(args.n_workers)
    results: List[QueryOutput] = []
    all_live_metrics: List[dict] = []
    connector = aiohttp.TCPConnector(limit=10000)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create workers
        workers = [asyncio.create_task(worker_func(session, queue, completions_url, results, args, logger, backend))
                   for _ in range(args.n_workers)]
        # Query the /metrics endpoint for one second before adding queries to the queue
        for i in range(int(1/args.step_live_metrics)):
            asyncio.create_task(get_live_metrics(session, metrics_url, all_live_metrics, backend))
            await asyncio.sleep(args.step_live_metrics)
        start_queries_timestamp = datetime.now().timestamp()
        # Add the queries to the queue
        count_query = 0
        async for query_input in get_async_generator(queries_dataset):
            # If the maximal duration has not been reached
            current_timestamp = datetime.now().timestamp()
            if continue_condition(current_timestamp, start_queries_timestamp, args, count_query):
                # While the queue is full, periodically query the /metrics endpoint
                while queue.full():
                    asyncio.create_task(get_live_metrics(session, metrics_url, all_live_metrics, backend))
                    await asyncio.sleep(args.step_live_metrics)
                asyncio.create_task(get_live_metrics(session, metrics_url, all_live_metrics, backend))
                await queue.put(query_input)
                count_query += 1
        if current_timestamp - start_queries_timestamp >= args.max_duration:
            now = get_now()
            logger.info(f"{now} Max duration {args.max_duration}s has been reached")
        # Wait for all enqueued items to be processed and during this time, periodically query the /metrics endpoint
        while not queue.empty():
            asyncio.create_task(get_live_metrics(session, metrics_url, all_live_metrics, backend))
            await asyncio.sleep(args.step_live_metrics)
        await queue.join()
        # Query the /metrics endpoint for one second after the queries finished
        for i in range(int(1/args.step_live_metrics)):
            asyncio.create_task(get_live_metrics(session, metrics_url, all_live_metrics, backend))
            await asyncio.sleep(args.step_live_metrics)
        
    # The workers are now idly waiting for the next queue item and we
    # no longer need them.
    for worker in workers:
        worker.cancel()
    return results, all_live_metrics