import pytest
import numpy as np
from benchmark_llm_serving import io_classes


def test_QueryOutput_init():
    # Default init
    query_output = io_classes.QueryOutput()
    default_attributes = {"prompt": "", "starting_timestamp": 0.0, "ending_timestamp": 0.0, "prompt_length": 0,
            "response_length": 0, "generated_text": "", "total_query_time": 0.0, "timestamp_of_first_token": 0.0, 
            "time_to_first_token": 0.0, "timestamp_of_tokens_arrival": [], "delta_time_of_tokens_arrival": [],
            "completion_time_from_first_token": 0.0, "median_time_between_tokens": 0.0, "total_waiting_time": 0.0,
            "speed_from_beginning": 0.0, "speed_from_first_token": 0.0, "success": False, "error": "", "timeout": False,
            "speed_without_waiting_time": 0.0}
    for attribute, default_value in default_attributes.items():
        assert getattr(query_output, attribute) == default_value
    
    # Init with value
    init_attributes = {"prompt": "Hey", "starting_timestamp": 3.5, "ending_timestamp": 7.9, "prompt_length": 10,
            "response_length": 20, "generated_text": "It's me", "total_query_time": 5.0, "timestamp_of_first_token": 4.5, 
            "time_to_first_token": 23, "timestamp_of_tokens_arrival": [2.3, 5.3], "delta_time_of_tokens_arrival": [2.1],
            "completion_time_from_first_token": 12143.0, "median_time_between_tokens": 0.34, "total_waiting_time": 12.4,
            "speed_from_beginning": 234.3, "speed_from_first_token": 12.2, "success": True, "error": "ERROR", "timeout": True,
            "speed_without_waiting_time": 334.0}
    query_output = io_classes.QueryOutput(**init_attributes)
    for attribute, value in init_attributes.items():
        assert getattr(query_output, attribute) == value

def test_calculate_derived_stats():
    init_attributes = {"prompt": "Hey", "starting_timestamp": 3.5, "ending_timestamp": 6.95, "prompt_length": 10,
           "generated_text": "It's me",  "timestamp_of_tokens_arrival": [3.7, 3.75, 3.85, 3.95, 4.10, 6.3, 6.4, 6.45, 6.55, 6.65, 6.75, 6.85, 6.95],
           "success": True}
    query_output = io_classes.QueryOutput(**init_attributes)
    query_output.calculate_derived_stats()
    assert query_output.response_length == len(init_attributes["timestamp_of_tokens_arrival"])
    assert query_output.total_query_time == init_attributes["ending_timestamp"] - init_attributes["starting_timestamp"]
    target_delta_time_of_tokens_arrival = [init_attributes["timestamp_of_tokens_arrival"][i+1] - init_attributes["timestamp_of_tokens_arrival"][i] 
                                                for i in range(len(init_attributes["timestamp_of_tokens_arrival"]) - 1)]
    assert query_output.delta_time_of_tokens_arrival == target_delta_time_of_tokens_arrival
    assert query_output.timestamp_of_first_token == init_attributes["timestamp_of_tokens_arrival"][0]
    assert query_output.completion_time_from_first_token == init_attributes["ending_timestamp"] - init_attributes["timestamp_of_tokens_arrival"][0]
    assert query_output.time_to_first_token == init_attributes["timestamp_of_tokens_arrival"][0] - init_attributes["starting_timestamp"]
    assert query_output.median_time_between_tokens == np.median(target_delta_time_of_tokens_arrival)
    assert query_output.total_waiting_time == query_output.calculate_total_waiting_time()
    assert query_output.speed_from_beginning == query_output.response_length / query_output.total_query_time
    assert query_output.speed_from_first_token == query_output.response_length / (query_output.total_query_time - query_output.time_to_first_token)
    assert query_output.speed_without_waiting_time == query_output.response_length / (query_output.total_query_time - query_output.total_waiting_time)


def test_calculate_total_waiting_time():
    init_attributes = {"prompt": "Coucou", "starting_timestamp": 3.5, "ending_timestamp": 6.95, "prompt_length": 10,
           "generated_text": "C'est moi",  "timestamp_of_tokens_arrival": [3.7, 3.75, 3.85, 3.95, 4.10, 6.3, 6.4, 6.45, 6.55, 6.65, 6.75, 6.85, 6.95],
           "success": True}
    query_output = io_classes.QueryOutput(**init_attributes)
    query_output.calculate_derived_stats()
    time_first_token = init_attributes["timestamp_of_tokens_arrival"][0] - init_attributes["starting_timestamp"]
    delta_time_of_tokens_arrival = [init_attributes["timestamp_of_tokens_arrival"][i+1] - init_attributes["timestamp_of_tokens_arrival"][i] 
                                                for i in range(len(init_attributes["timestamp_of_tokens_arrival"]) - 1)]
    median_delta = np.median(delta_time_of_tokens_arrival)
    assert query_output.median_time_between_tokens == pytest.approx(median_delta)
    std_delta = np.std(delta_time_of_tokens_arrival, ddof=1)
    typical_generation_time = median_delta + 3 * std_delta
    big_delta_time_of_tokens_arrival = [delta-typical_generation_time for delta in delta_time_of_tokens_arrival if delta > typical_generation_time]
    additional_waiting_time = sum(big_delta_time_of_tokens_arrival)
    
    assert  query_output.total_waiting_time == pytest.approx(time_first_token + additional_waiting_time)


def test_to_dict():
    default_attributes = {"prompt": "", "starting_timestamp": 0.0, "ending_timestamp": 0.0, "prompt_length": 0,
            "response_length": 0, "generated_text": "", "total_query_time": 0.0, "timestamp_of_first_token": 0.0, 
            "time_to_first_token": 0.0, "timestamp_of_tokens_arrival": [], "delta_time_of_tokens_arrival": [],
            "completion_time_from_first_token": 0.0, "median_time_between_tokens": 0.0, "total_waiting_time": 0.0,
            "speed_from_beginning": 0.0, "speed_from_first_token": 0.0, "success": False, "error": "", "timeout": False,
            "speed_without_waiting_time": 0.0}
    query_output = io_classes.QueryOutput()
    assert query_output.to_dict() == default_attributes
    
    init_attributes = {"prompt": "Hey", "starting_timestamp": 3.5, "ending_timestamp": 7.9, "prompt_length": 10,
            "response_length": 20, "generated_text": "It's me", "total_query_time": 5.0, "timestamp_of_first_token": 4.5, 
            "time_to_first_token": 23, "timestamp_of_tokens_arrival": [2.3, 5.3], "delta_time_of_tokens_arrival": [2.1],
            "completion_time_from_first_token": 12143.0, "median_time_between_tokens": 0.34, "total_waiting_time": 12.4,
            "speed_from_beginning": 234.3, "speed_from_first_token": 12.2, "success": True, "error": "ERROR", "timeout": True,
            "speed_without_waiting_time": 334.0}
    query_output = io_classes.QueryOutput(**init_attributes)
    assert query_output.to_dict() == init_attributes


def test_QueryInput_init():
    # Default init
    query_input = io_classes.QueryInput(prompt="Hey", internal_id=42)
    default_attributes = {"prompt": "Hey", "internal_id": 42, "scheduled_delta": 0, "scheduled_timestamp": 0}
    for attribute, default_value in default_attributes.items():
        assert getattr(query_input, attribute) == default_value


    init_attributes = {"prompt": "Hey. How are you?", "internal_id": 33, "scheduled_delta": 0.5, "scheduled_timestamp": 34423.2}
    query_input = io_classes.QueryInput(**init_attributes)
    for attribute, value in init_attributes.items():
        assert getattr(query_input, attribute) == value
    

def test_add_starting_timestamp():
    init_attributes = {"prompt": "Hey. How are you?", "internal_id": 33, "scheduled_delta": 0.5, "scheduled_timestamp": 0}
    query_input = io_classes.QueryInput(**init_attributes)
    for starting_timestamp in [0.0, 3553.234, 214353.2]:
        query_input.add_starting_timestamp(starting_timestamp)
        assert query_input.scheduled_timestamp == pytest.approx(starting_timestamp + query_input.scheduled_delta)