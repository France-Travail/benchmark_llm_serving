import pytest
import random
import argparse
import numpy as np
import requests_mock

from benchmark_llm_serving import benchmark
from benchmark_llm_serving.io_classes import QueryOutput, QueryInput


def test_augment_dataset():
    dataset = ["Hey", "It's me"]
    
    new_dataset = benchmark.augment_dataset(dataset, 1)
    assert new_dataset == ["Hey"]

    new_dataset = benchmark.augment_dataset(dataset, 2)
    assert new_dataset == dataset

    new_dataset = benchmark.augment_dataset(dataset, 3)
    assert new_dataset == ["Hey", "It's me", "0 Hey"]

    new_dataset = benchmark.augment_dataset(dataset, 4)
    assert new_dataset == ["Hey", "It's me", "0 Hey", "0 It's me"]

    new_dataset = benchmark.augment_dataset(dataset, 5)
    assert new_dataset == ["Hey", "It's me", "0 Hey", "0 It's me", "1 Hey"]


def test_get_model_name_from_info_endpoint():
    mock_data = {
                "application": "happy_vllm",
                "version": "1.1.2",
                "vllm_version": "0.4.2",
                "model_name": "Meta-Llama-3-8B-Instruct",
                "truncation_side": "right",
                "max_length": 8192
}

    args = argparse.Namespace(base_url="http://my_url", info_endpoint='/info_endpoint')
    with requests_mock.Mocker() as mocked:
        mocked.get('http://my_url/info_endpoint', json=mock_data)
        assert benchmark.get_model_name_from_info_endpoint(args) == "Meta-Llama-3-8B-Instruct"


def test_add_application_parameters():
    mock_info_data = {
                "application": "happy_vllm",
                "version": "1.1.2",
                "vllm_version": "0.4.2",
                "model_name": "Meta-Llama-3-8B-Instruct",
                "truncation_side": "right",
                "max_length": 8192
                    }
    mock_launch_arguments_data = {
            "host": "0.0.0.0",
            "port": 8501,
            "model_name": "Meta-Llama-3-8B-Instruct",
            "app_name": "happy_vllm",
            "api_endpoint_prefix": "",
            "explicit_errors": False,
            "allow_credentials": False,
            "allowed_origins": [
                "*"
            ],
            "allowed_methods": [
                "*"
            ],
            "allowed_headers": [
                "*"
            ],
            "uvicorn_log_level": "info",
            "ssl_keyfile": None,
            "ssl_certfile": None,
            "ssl_ca_certs": None,
            "ssl_cert_reqs": 0,
            "root_path": None,
            "lora_modules": None,
            "chat_template": None,
            "response_role": "assistant",
            "with_launch_arguments": True,
            "model": "/home/data/models/Meta-Llama-3-8B-Instruct",
            "tokenizer": None,
            "skip_tokenizer_init": False,
            "revision": None,
            "code_revision": None,
            "tokenizer_revision": None,
            "tokenizer_mode": "auto",
            "trust_remote_code": False,
            "download_dir": None,
            "load_format": "auto",
            "dtype": "auto",
            "kv_cache_dtype": "auto",
            "quantization_param_path": None,
            "max_model_len": None,
            "guided_decoding_backend": "outlines",
            "worker_use_ray": False,
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 1,
            "max_parallel_loading_workers": None,
            "ray_workers_use_nsight": False,
            "block_size": 16,
            "enable_prefix_caching": False,
            "use_v2_block_manager": False,
            "num_lookahead_slots": 0,
            "seed": 0,
            "swap_space": 4,
            "gpu_memory_utilization": 0.9,
            "num_gpu_blocks_override": None,
            "max_num_batched_tokens": None,
            "max_num_seqs": 256,
            "max_logprobs": 5,
            "disable_log_stats": False,
            "quantization": None,
            "enforce_eager": False,
            "max_context_len_to_capture": None,
            "max_seq_len_to_capture": 8192,
            "disable_custom_all_reduce": False,
            "tokenizer_pool_size": 0,
            "tokenizer_pool_type": "ray",
            "tokenizer_pool_extra_config": None,
            "enable_lora": False,
            "max_loras": 1,
            "max_lora_rank": 16,
            "lora_extra_vocab_size": 256,
            "lora_dtype": "auto",
            "max_cpu_loras": None,
            "fully_sharded_loras": False,
            "device": "auto",
            "image_input_type": None,
            "image_token_id": None,
            "image_input_shape": None,
            "image_feature_size": None,
            "scheduler_delay_factor": 0.0,
            "enable_chunked_prefill": False,
            "speculative_model": None,
            "num_speculative_tokens": None,
            "speculative_max_model_len": None,
            "ngram_prompt_lookup_max": None,
            "ngram_prompt_lookup_min": None,
            "model_loader_extra_config": None,
            "served_model_name": None,
            "engine_use_ray": False,
            "disable_log_requests": False,
            "max_log_len": None
            }

    args = argparse.Namespace(base_url="http://my_url", info_endpoint='/info_endpoint', launch_arguments_endpoint="/launch_arguments_endpoint")
    # With model_name: None
    parameters = {"first_parameter": "my_first_value", "model": "my_model", "model_name": None}
    with requests_mock.Mocker() as mocked:
        mocked.get('http://my_url/info_endpoint', json=mock_info_data)
        mocked.get('http://my_url/launch_arguments_endpoint', json=mock_launch_arguments_data)
        
        parameters = benchmark.add_application_parameters(parameters, args)
    assert parameters["first_parameter"] == "my_first_value"
    assert parameters["model"] == "my_model"
    assert parameters['happy_vllm_version'] == mock_info_data['version']
    assert parameters["vllm_version"] == mock_info_data["vllm_version"]
    assert parameters["model_name"] == "Meta-Llama-3-8B-Instruct"
    for key, value in mock_launch_arguments_data.items():
        if key != "model"and key != "model_name":
            assert parameters[key] == value
    assert set(parameters) == {"first_parameter", "model", "happy_vllm_version", "vllm_version"}.union(set(mock_launch_arguments_data))

    # With model_name: not None
    parameters = {"first_parameter": "my_first_value", "model": "my_model", "model_name": "Hey"}
    with requests_mock.Mocker() as mocked:
        mocked.get('http://my_url/info_endpoint', json=mock_info_data)
        mocked.get('http://my_url/launch_arguments_endpoint', json=mock_launch_arguments_data)
        
        parameters = benchmark.add_application_parameters(parameters, args)
    assert parameters["first_parameter"] == "my_first_value"
    assert parameters["model"] == "my_model"
    assert parameters['happy_vllm_version'] == mock_info_data['version']
    assert parameters["vllm_version"] == mock_info_data["vllm_version"]
    assert parameters["model_name"] == "Hey"
    for key, value in mock_launch_arguments_data.items():
        if key != "model" and key != "model_name":
            assert parameters[key] == value
    assert set(parameters) == {"first_parameter", "model", "happy_vllm_version", "vllm_version"}.union(set(mock_launch_arguments_data))


def test_get_general_metrics():
    benchmark_results = [QueryOutput(starting_timestamp=9000, ending_timestamp=12000, prompt_length=20,
                                    timestamp_of_tokens_arrival=[9100 + i*150 + random.randrange(-10, 10) for i in range(20)],
                                    success=True),
                        QueryOutput(starting_timestamp=10500, ending_timestamp=13600, prompt_length=50,
                                    timestamp_of_tokens_arrival=[11000 + i*150 + random.randrange(-20, 20) for i in range(20)],
                                    success=True),
                        QueryOutput(starting_timestamp=10500, ending_timestamp=13600, prompt_length=50,
                                    timestamp_of_tokens_arrival=[11000 + i*150 + random.randrange(-20, 20) for i in range(20)],
                                    success=False, timeout=False),
                        QueryOutput(starting_timestamp=10500, ending_timestamp=13600, prompt_length=50,
                                    timestamp_of_tokens_arrival=[11000 + i*150 + random.randrange(-20, 20) for i in range(20)],
                                    success=False, timeout=True)]
    for result in benchmark_results:
        result.calculate_derived_stats()
    
    all_live_metrics = [{"num_requests_running": 1.0, "num_requests_waiting": 2.0, "gpu_cache_usage_perc": 30.0, "timestamp": 10000},
                        {"num_requests_running": 3.0, "num_requests_waiting": 0.0, "gpu_cache_usage_perc": 93.2, "timestamp": 11000}]
    benchmark_results_valid = [query_output for query_output in benchmark_results if query_output.success or query_output.timeout]
    total_waiting_time = sum([result.total_waiting_time for result in benchmark_results_valid])
    

    # happy_vllm backend
    args = argparse.Namespace(backend="happy_vllm")
    general_metrics = benchmark.get_general_metrics(benchmark_results, all_live_metrics, args)
    target_general_metrics = {"max_kv_cache": 93.2, "max_requests_running": 3.0, "max_requests_waiting": 2.0,
                              "total_number_of_queries": 4.0, "total_time": 9200, "total_time_from_first_token": 1100,
                              "total_number_of_ingested_tokens": 120, "total_number_of_generated_tokens": 60,
                              "nb_timeout_queries": 1, "nb_errored_queries": 2, "total_time_without_waiting_time": 9200-total_waiting_time}
    # Remove errored queries
    target_general_metrics["total_time_from_first_token"] = sum([result.completion_time_from_first_token for result in benchmark_results_valid])                
    target_general_metrics["speed"] = target_general_metrics['total_number_of_generated_tokens'] / target_general_metrics['total_time']
    target_general_metrics["speed_from_first_token"] = target_general_metrics['total_number_of_generated_tokens'] / target_general_metrics['total_time_from_first_token']
    target_general_metrics["speed_without_waiting_time"] = target_general_metrics['total_number_of_generated_tokens'] / target_general_metrics['total_time_without_waiting_time']
    
    assert set(general_metrics) == set(target_general_metrics)
    for key, value in target_general_metrics.items():
        assert general_metrics[key] == pytest.approx(value)

    # mistral backend
    args = argparse.Namespace(backend="mistral")
    general_metrics = benchmark.get_general_metrics(benchmark_results, all_live_metrics, args)
    target_general_metrics = {"max_kv_cache": -1, "max_requests_running": -1, "max_requests_waiting": -1,
                              "total_number_of_queries": 4.0, "total_time": 9200, "total_time_from_first_token": 1100,
                              "total_number_of_ingested_tokens": 120, "total_number_of_generated_tokens": 60,
                              "nb_timeout_queries": 1, "nb_errored_queries": 2, "total_time_without_waiting_time": 9200-total_waiting_time}
    # Remove errored queries
    target_general_metrics["total_time_from_first_token"] = sum([result.completion_time_from_first_token for result in benchmark_results_valid])                
    target_general_metrics["speed"] = target_general_metrics['total_number_of_generated_tokens'] / target_general_metrics['total_time']
    target_general_metrics["speed_from_first_token"] = target_general_metrics['total_number_of_generated_tokens'] / target_general_metrics['total_time_from_first_token']
    target_general_metrics["speed_without_waiting_time"] = target_general_metrics['total_number_of_generated_tokens'] / target_general_metrics['total_time_without_waiting_time']
    
    assert set(general_metrics) == set(target_general_metrics)
    for key, value in target_general_metrics.items():
        assert general_metrics[key] == pytest.approx(value)


def test_get_aggregated_metrics():
    benchmark_results = []
    # successful queries
    for n in range(50):
        starting_timestamp = 9000 + random.randrange(-1000, 1000)
        ending_timestamp = 12000 + random.randrange(-1000, 1000)
        benchmark_results.append(QueryOutput(starting_timestamp=starting_timestamp,
                                             ending_timestamp=ending_timestamp,
                                             prompt_length=20 + random.randrange(-10, 10),
                                             timestamp_of_tokens_arrival=[starting_timestamp - 200 + i*150 + random.randrange(-10, 10) for i in range(20)],
                                             success=True))
    # timeout queries
    for n in range(50):
        starting_timestamp = 9000 + random.randrange(-1000, 1000)
        ending_timestamp = 12000 + random.randrange(-1000, 1000)
        benchmark_results.append(QueryOutput(starting_timestamp=starting_timestamp,
                                             prompt_length=20 + random.randrange(-10, 10),
                                             timestamp_of_tokens_arrival=[starting_timestamp - 200 + i*150 + random.randrange(-10, 10) for i in range(20)],
                                             success=False, timeout=True))
    # errored queries
    for n in range(50):
        starting_timestamp = 9000 + random.randrange(-1000, 1000)
        ending_timestamp = 12000 + random.randrange(-1000, 1000)
        benchmark_results.append(QueryOutput(starting_timestamp=starting_timestamp,
                                             ending_timestamp=ending_timestamp,
                                             prompt_length=20 + random.randrange(-10, 10),
                                             timestamp_of_tokens_arrival=[starting_timestamp - 200 + i*150 + random.randrange(-10, 10) for i in range(20)],
                                             success=False, timeout=False))
    for result in benchmark_results:
        result.calculate_derived_stats()
    benchmark_results_no_errors = [result for result in benchmark_results if result.success or result.timeout]
    stat_list = ['time_to_first_token', "prompt_length", "response_length", "total_query_time", "completion_time_from_first_token",
                "median_time_between_tokens", "total_waiting_time", "speed_from_beginning", "speed_from_first_token",
                "speed_without_waiting_time"]
    target_aggregated_metrics = {}
    for stat in stat_list:
        all_stats = [getattr(result, stat) for result in benchmark_results_no_errors]
        if len(all_stats):
            target_aggregated_metrics[stat] = {
                "mean": np.mean(all_stats),
                "median": np.median(all_stats)
                }
            for percentile in [1, 5, 10, 25, 75, 90, 95, 99]:
                target_aggregated_metrics[stat][f"percentile_{percentile}"] = np.percentile(all_stats, percentile)
    aggregated_metrics = benchmark.get_aggregated_metrics(benchmark_results)
    assert set(aggregated_metrics) == set(stat_list)
    for stat in stat_list:
        assert set(aggregated_metrics[stat]) == set(target_aggregated_metrics[stat])
        for key, value in aggregated_metrics[stat].items():
            assert value == pytest.approx(target_aggregated_metrics[stat][key])