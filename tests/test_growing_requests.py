import argparse

from benchmark_llm_serving.query_profiles import growing_requests


def test_continue_condition():
    current_timestamp = 10
    start_queries_timestamp = 0
    nb_queries_launched = 10
    n = 5
    
    # Max duration reached
    len_queries_dataset = 1000
    args = argparse.Namespace(max_duration=10, min_duration=100, target_queries_nb=100)
    assert not growing_requests.continue_condition(current_timestamp, start_queries_timestamp, nb_queries_launched,
                                                    n, len_queries_dataset, args)

    # Max queries reached
    len_queries_dataset = 14
    args = argparse.Namespace(max_duration=10, min_duration=100, target_queries_nb=100)
    assert not growing_requests.continue_condition(current_timestamp, start_queries_timestamp, nb_queries_launched,
                                                    n, len_queries_dataset, args)

    # Min duration reached and target_queries_nb reached
    len_queries_dataset = 1000
    args = argparse.Namespace(max_duration=100, min_duration=8, target_queries_nb=5)
    assert not growing_requests.continue_condition(current_timestamp, start_queries_timestamp, nb_queries_launched,
                                                    n, len_queries_dataset, args)

    # Min duration reached but target_queries_nb not reached
    len_queries_dataset = 1000
    args = argparse.Namespace(max_duration=100, min_duration=8, target_queries_nb=100)
    assert growing_requests.continue_condition(current_timestamp, start_queries_timestamp, nb_queries_launched,
                                                    n, len_queries_dataset, args)

    # Min duration not reached but target_queries_nb reached
    len_queries_dataset = 1000
    args = argparse.Namespace(max_duration=100, min_duration=100, target_queries_nb=5)
    assert growing_requests.continue_condition(current_timestamp, start_queries_timestamp, nb_queries_launched,
                                                    n, len_queries_dataset, args)

    # Nothing reached
    len_queries_dataset = 1000
    args = argparse.Namespace(max_duration=100, min_duration=100, target_queries_nb=100)
    assert growing_requests.continue_condition(current_timestamp, start_queries_timestamp, nb_queries_launched,
                                                    n, len_queries_dataset, args)
