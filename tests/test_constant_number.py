import types
import pytest
import argparse

from benchmark_llm_serving.query_profiles import constant_number


@pytest.mark.asyncio
async def test_get_async_generator():
    my_list = ['Hey', 'How are you ?', "Fine, thank you."]
    my_generator = constant_number.get_async_generator(my_list)
    print(type(my_generator))
    assert isinstance(my_generator, types.AsyncGeneratorType)
    list_from_generator = []
    async for element in my_generator:
        list_from_generator.append(element)
    assert my_list == list_from_generator


def test_continue_condition():
    current_timestamp = 10
    start_queries_timestamp = 0
    count_query = 10
    
    # Max duration reached
    args = argparse.Namespace(max_duration=10, min_duration=3, target_queries_nb=100)
    assert not constant_number.continue_condition(current_timestamp, start_queries_timestamp, args, count_query)

    # Min duration reached and target_queries_nb reached
    args = argparse.Namespace(max_duration=100, min_duration=3, target_queries_nb=5)
    assert not constant_number.continue_condition(current_timestamp, start_queries_timestamp, args, count_query)

    # Min duration reached but target_queries_nb not reached
    args = argparse.Namespace(max_duration=100, min_duration=3, target_queries_nb=100)
    assert constant_number.continue_condition(current_timestamp, start_queries_timestamp, args, count_query)

    # Min duration not reached but target_queries_nb reached
    args = argparse.Namespace(max_duration=100, min_duration=11, target_queries_nb=5)
    assert constant_number.continue_condition(current_timestamp, start_queries_timestamp, args, count_query)

    # Nothing reached
    args = argparse.Namespace(max_duration=100, min_duration=100, target_queries_nb=100)
    assert constant_number.continue_condition(current_timestamp, start_queries_timestamp, args, count_query)
