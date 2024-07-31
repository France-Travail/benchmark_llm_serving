import os
import pytest
import argparse
from pathlib import Path

from benchmark_llm_serving import backends
from benchmark_llm_serving import utils_metrics
from benchmark_llm_serving.io_classes import QueryOutput, QueryInput


def get_metrics_response():
    current_directory = Path(os.path.dirname(os.path.realpath(__file__)))
    metrics_response_file = current_directory / "data" / "metrics_response.txt"
    with open(metrics_response_file, 'r') as txt_file:
        metrics_response = txt_file.read()
    return metrics_response


def test_get_backend():
    # happy_vllm backend
    backend = backends.get_backend("happy_vllm")
    assert isinstance(backend, backends.BackendHappyVllm)
    assert backend.backend_name == "happy_vllm"
    assert backend.chunk_prefix == "data: "
    assert backend.last_chunk == "[DONE]"
    # mistral backend
    backend = backends.get_backend("mistral")
    assert isinstance(backend, backends.BackEndMistral)
    assert backend.backend_name == "mistral"
    assert backend.chunk_prefix == "data: "
    assert backend.last_chunk == "[DONE]"
    # ValueError
    with pytest.raises(ValueError):
        backends.get_backend("backend_not_implemented")


def test_backend_happy_vllm_get_payload():
    backend = backends.get_backend("happy_vllm")
    prompts_list = ["Hey. How are you?", "Fine, you ?"]
    model_list = ["My_awesome_model", "yet_another_model"]
    output_length_list = [100, 1000]
    for prompt in prompts_list:
        for model in model_list:
            for output_length in output_length_list:
                query_input = QueryInput(prompt=prompt, internal_id=0)
                args = argparse.Namespace(model=model, output_length=output_length)
                payload = backend.get_payload(query_input, args)
                target_payload = {"prompt": prompt,
                                    "model": model,
                                    "max_tokens": output_length,
                                    "min_tokens": output_length,
                                    "temperature": 0,
                                    "repetition_penalty": 1.2,
                                    "stream": True,
                                    "stream_options": {"include_usage": True}
                                            }
                assert payload == target_payload


def test_backend_happy_vllm_test_chunk_validity():
    backend = backends.get_backend("happy_vllm")
    for chunk in ['first chunk', "second chunk", "tok-i 1", "tok-o 1"]:
        chunk_validity = backend.test_chunk_validity(chunk)
        assert chunk_validity


def test_backend_happy_vllm_get_completions_headers():
    backend = backends.get_backend("happy_vllm")
    assert backend.get_completions_headers() == {}


def test_backend_happy_vllm_remove_response_prefix():
    backend = backends.get_backend("happy_vllm")
    for chunk_str in ["Hey", "How are you ?", "Fine thanks!"]:
        chunk = "data: " + chunk_str
        chunk = backend.remove_response_prefix(chunk)
        assert chunk == chunk_str


def test_backend_happy_vllm_check_end_of_stream():
    backend = backends.get_backend("happy_vllm")
    for chunk_not_done in ["not done", " [DONE]", "[DONE] "]:
        assert not(backend.check_end_of_stream(chunk_not_done))
    assert backend.check_end_of_stream("[DONE]")


def test_backend_happy_vllm_get_newly_generated_text():
    generated_text = "Hey ! How are you ?"
    json_chunk_completions = {"id": "cmpl-d85f82039b864ceb8d95be931b200745", "object": "chat.completion.chunk", 
                            "created": 1716468615, "model": "CodeLlama-34B-AWQ", 
                            "choices": [{"index": 0,"text": f"{generated_text}", 
                            "stop_reason": None, "logprobs": None, "finish_reason": None}]}
    
    backend = backends.get_backend("happy_vllm")
    newly_generated_text = backend.get_newly_generated_text(json_chunk_completions)
    assert newly_generated_text == generated_text


def test_backend_happy_vllm_add_prompt_length():
    backend = backends.get_backend("happy_vllm")
    prompt_tokens = 1234
    chunk_with_usage = {"id": "cmpl-d5edc7c2c3264f189b3c941630751d8e", 
                                  "object": "text_completion",
                                  "created": 1722328225,
                                  "model": "Vigostral-7B-Chat-AWQ", 
                                  "choices":[], 
                                  "usage": {"prompt_tokens": prompt_tokens,"total_tokens": 108,"completion_tokens": 86}}

    chunk_without_usage = {"id": "cmpl-d5edc7c2c3264f189b3c941630751d8e", 
                                  "object": "text_completion",
                                  "created": 1722328225,
                                  "model": "Vigostral-7B-Chat-AWQ", 
                                  "choices":[]}
    # with usage key
    output = QueryOutput()
    assert output.prompt_length == 0
    backend.add_prompt_length(chunk_with_usage, output)
    assert output.prompt_length == prompt_tokens

    output = QueryOutput()
    assert output.prompt_length == 0
    backend.add_prompt_length(chunk_without_usage, output)
    assert output.prompt_length == 0


def test_backend_happy_vllm_get_metrics_from_metrics_dict():
    backend = backends.get_backend("happy_vllm")
    metrics_response = get_metrics_response()
    parsed_metrics = utils_metrics.parse_metrics_response(metrics_response)
    live_metrics = backend.get_metrics_from_metrics_dict(parsed_metrics)
    assert isinstance(live_metrics, dict)
    assert set(live_metrics) == {"num_requests_running", "num_requests_waiting", "gpu_cache_usage_perc"}
    assert live_metrics['num_requests_running'] == pytest.approx(2.0)
    assert live_metrics['num_requests_waiting'] == pytest.approx(1.0)
    assert live_metrics['gpu_cache_usage_perc'] == pytest.approx(4.2)


def test_backend_mistral_get_payload():
    backend = backends.get_backend("mistral")
    prompts_list = ["Hey. How are you?", "Fine, you ?"]
    model_list = ["My_awesome_model", "yet_another_model"]
    output_length_list = [100, 1000]
    for prompt in prompts_list:
        for model in model_list:
            for output_length in output_length_list:
                query_input = QueryInput(prompt=prompt, internal_id=0)
                args = argparse.Namespace(model=model, output_length=output_length)
                payload = backend.get_payload(query_input, args)
                target_payload = {"messages": [{"role": "user", "content": prompt}],
                        "model": model,
                        "max_tokens": output_length,
                        "min_tokens": output_length,
                        "temperature": 0,
                        "stream": True
                                }
                assert payload == target_payload


def test_backend_mistral_test_chunk_validity():
    backend = backends.get_backend("mistral")
    # True
    for chunk in ['first chunk', "second chunk"]:
        chunk_validity = backend.test_chunk_validity(chunk)
        assert chunk_validity
    # False
    for chunk in ["tok-i 1", "tok-o 1"]:
        chunk_validity = backend.test_chunk_validity(chunk)
        assert not(chunk_validity)


def test_backend_mistral_get_completions_headers():
    backend = backends.get_backend("mistral")
    target_headers = {"Accept": "application/json",
                      "Content-Type": "application/json"}
    assert backend.get_completions_headers() == target_headers


def test_backend_mistral_remove_response_prefix():
    backend = backends.get_backend("mistral")
    for chunk_str in ["Hey", "How are you ?", "Fine thanks!"]:
        chunk = "data: " + chunk_str
        chunk = backend.remove_response_prefix(chunk)
        assert chunk == chunk_str


def test_backend_mistral_check_end_of_stream():
    backend = backends.get_backend("mistral")
    for chunk_not_done in ["not done", " [DONE]", "[DONE] "]:
        assert not(backend.check_end_of_stream(chunk_not_done))
    assert backend.check_end_of_stream("[DONE]")


def test_backend_mistral_get_newly_generated_text():
    generated_text = "Hey ! How are you ?"
    json_chunk_chat_completions = {"id":"cbaa5c28166d4b98b5256f1becc0364d", 
                                   "object":"chat.completion.chunk", 
                                   "created":1722322855, 
                                   "model":"mistral", 
                                   "choices":[{"index":0,"delta": {"content": f"{generated_text}"},"finish_reason": None,"logprobs": None}]}

    backend = backends.get_backend("mistral")
    newly_generated_text = backend.get_newly_generated_text(json_chunk_chat_completions)
    assert newly_generated_text == generated_text


def test_backend_mistral_add_prompt_length():
    prompt_tokens = 1234
    backend = backends.get_backend("mistral")
    json_chunk_with_usage = {"id":"cbaa5c28166d4b98b5256f1becc0364d",
                          "object": "chat.completion.chunk",
                          "created":1722322855, 
                          "model":"mistral", 
                          "choices":[{"index": 0,"delta": {"content":""}, "finish_reason": "stop", "logprobs": None}],
                          "usage": {"prompt_tokens": prompt_tokens,"total_tokens": 58,"completion_tokens": 46}}
    json_chunk_without_usage = {"id":"cbaa5c28166d4b98b5256f1becc0364d",
                          "object": "chat.completion.chunk",
                          "created":1722322855, 
                          "model":"mistral", 
                          "choices":[{"index": 0,"delta": {"content":""}, "finish_reason": "stop", "logprobs": None}]}

    #with usage key
    output = QueryOutput()
    assert output.prompt_length == 0
    backend.add_prompt_length(json_chunk_with_usage, output)
    assert output.prompt_length == prompt_tokens

    #without usage key
    output = QueryOutput()
    assert output.prompt_length == 0
    backend.add_prompt_length(json_chunk_without_usage, output)
    assert output.prompt_length == 0