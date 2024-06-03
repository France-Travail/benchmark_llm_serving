import argparse
import numpy as np

from benchmark_llm_serving.io_classes import QueryInput
from benchmark_llm_serving.query_profiles import scheduled_requests


def test_add_poisson_rate():
    # Test inf request rate
    args = argparse.Namespace(request_rate=float('inf'))
    my_queries = []
    for i in range(10000):
        my_queries.append(QueryInput(prompt="Hey", internal_id=i))
    my_queries = scheduled_requests.add_poisson_rate(my_queries, args)
    for query in my_queries:
        assert query.scheduled_delta == 0

    # Test non zero request rate
    for request_rate in range(1, 10):
        args = argparse.Namespace(request_rate=request_rate)
        my_queries = []
        for i in range(10000):
            my_queries.append(QueryInput(prompt="Hey", internal_id=i))
        my_queries = scheduled_requests.add_poisson_rate(my_queries, args)
        # Test that each query has a new scheduled_delta
        for query in my_queries:
            assert query.scheduled_delta != 0
        # Test that the mean number of query per second is approximately the request rate
        assert len(my_queries) / np.max([query.scheduled_delta for query in my_queries]) > request_rate - 0.5 
        assert len(my_queries) / np.max([query.scheduled_delta for query in my_queries]) < request_rate + 0.5
        # Test that each scheduled_delta for a query is bigger than the one of the query before
        for i in range(1, len(my_queries)):
            assert my_queries[i].scheduled_delta - my_queries[i-1].scheduled_delta > 0

def test_add_schedule_to_queries():
    # Nominal case
    queries_dataset = []
    for i in range(10000):
        queries_dataset.append(QueryInput(prompt="Hey", internal_id=i))
    args = argparse.Namespace(request_rate=10, query_profile="request_rate")
    queries_dataset = scheduled_requests.add_schedule_to_queries(queries_dataset, args)
    for query in queries_dataset:
        assert query.scheduled_delta != 0
    # Test that each scheduled_delta for a query is bigger than the one of the query before
    for i in range(1, len(queries_dataset)):
        assert queries_dataset[i].scheduled_delta - queries_dataset[i-1].scheduled_delta > 0

    # Case request_rate = inf
    args = argparse.Namespace(request_rate=float("inf"), query_profile="request_rate")
    queries_dataset = scheduled_requests.add_schedule_to_queries(queries_dataset, args)
    for query in queries_dataset:
        assert query.scheduled_delta == 0


def test_get_queries_to_launch():
    # Create a list of QueryInput objects with scheduled timestamps
    queries_dataset = []
    for i in range(1, 4):
        queries_dataset.append(QueryInput(prompt="Hey", internal_id=i, scheduled_timestamp=i))
    
    # Test case 1: current timestamp is 1.5, current_query_index_to_launch is 0
    current_timestamp = 1.5
    current_query_index_to_launch = 0
    expected_output = ([queries_dataset[0]], 1)
    assert scheduled_requests.get_queries_to_launch(queries_dataset, current_query_index_to_launch, current_timestamp) == expected_output
    
    # Test case 2: current timestamp is 2.5, current_query_index_to_launch is 1
    current_timestamp = 2.5
    current_query_index_to_launch = 1
    expected_output = ([queries_dataset[1]], 2)
    assert scheduled_requests.get_queries_to_launch(queries_dataset, current_query_index_to_launch, current_timestamp) == expected_output
    
    # Test case 3: current timestamp is 2.5, current_query_index_to_launch is 0
    current_timestamp = 2.5
    current_query_index_to_launch = 0
    expected_output = ([queries_dataset[0], queries_dataset[1]], 2)
    assert scheduled_requests.get_queries_to_launch(queries_dataset, current_query_index_to_launch, current_timestamp) == expected_output

    # Test case 4: current timestamp is 0.5, current_query_index_to_launch is 0
    current_timestamp = 0.5
    current_query_index_to_launch = 0
    expected_output = ([], 0)
    assert scheduled_requests.get_queries_to_launch(queries_dataset, current_query_index_to_launch, current_timestamp) == expected_output


def test_continue_condition():
    # Max duration reached
    current_query_index_to_launch = 0
    max_queries_number = 1000
    max_duration_reached = True
    min_duration_reached = False
    args = argparse.Namespace(target_queries_nb=100)
    assert not scheduled_requests.continue_condition(current_query_index_to_launch, max_queries_number, max_duration_reached, 
                     min_duration_reached, args)

    # Max queries reached
    current_query_index_to_launch = 1000
    max_queries_number = 1000
    max_duration_reached = False
    min_duration_reached = False
    args = argparse.Namespace(target_queries_nb=100)
    assert not scheduled_requests.continue_condition(current_query_index_to_launch, max_queries_number, max_duration_reached, 
                     min_duration_reached, args)

    # Min duration reached and target_queries_nb reached
    current_query_index_to_launch = 150
    max_queries_number = 1000
    max_duration_reached = False
    min_duration_reached = True
    args = argparse.Namespace(target_queries_nb=100)
    assert not scheduled_requests.continue_condition(current_query_index_to_launch, max_queries_number, max_duration_reached, 
                     min_duration_reached, args)

    # Min duration reached but target_queries_nb not reached
    current_query_index_to_launch = 50
    max_queries_number = 1000
    max_duration_reached = False
    min_duration_reached = True
    args = argparse.Namespace(target_queries_nb=100)
    assert scheduled_requests.continue_condition(current_query_index_to_launch, max_queries_number, max_duration_reached, 
                     min_duration_reached, args)

    # Min duration not reached but target_queries_nb reached
    current_query_index_to_launch = 150
    max_queries_number = 1000
    max_duration_reached = False
    min_duration_reached = False
    args = argparse.Namespace(target_queries_nb=100)
    assert scheduled_requests.continue_condition(current_query_index_to_launch, max_queries_number, max_duration_reached, 
                     min_duration_reached, args)

    # Nothing reached
    current_query_index_to_launch = 80
    max_queries_number = 1000
    max_duration_reached = False
    min_duration_reached = False
    args = argparse.Namespace(target_queries_nb=100)
    assert scheduled_requests.continue_condition(current_query_index_to_launch, max_queries_number, max_duration_reached, 
                     min_duration_reached, args)