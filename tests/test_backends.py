import pytest
import argparse

from benchmark_llm_serving import backends
from benchmark_llm_serving.io_classes import QueryOutput, QueryInput

def test_get_payload():
    # happy_vllm backend
    prompts_list = ["Hey. How are you?", "Fine, you ?"]
    model_list = ["My_awesome_model", "yet_another_model"]
    output_length_list = [100, 1000]
    for prompt in prompts_list:
        for model in model_list:
            for output_length in output_length_list:
                query_input = QueryInput(prompt=prompt, internal_id=0)
                args = argparse.Namespace(backend="happy_vllm", model=model, output_length=output_length)
                payload = backends.get_payload(query_input, args)
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
    
    # mistral backend
    prompts_list = ["Hey. How are you?", "Fine, you ?"]
    model_list = ["My_awesome_model", "yet_another_model"]
    output_length_list = [100, 1000]
    for prompt in prompts_list:
        for model in model_list:
            for output_length in output_length_list:
                query_input = QueryInput(prompt=prompt, internal_id=0)
                args = argparse.Namespace(backend="mistral", model=model, output_length=output_length)
                payload = backends.get_payload(query_input, args)
                target_payload = {"messages": [{"role": "user", "content": prompt}],
                        "model": model,
                        "max_tokens": output_length,
                        "min_tokens": output_length,
                        "temperature": 0,
                        "stream": True
                                }
                assert payload == target_payload

    # ValueError
    prompts_list = ["Hey. How are you?", "Fine, you ?"]
    model_list = ["My_awesome_model", "yet_another_model"]
    output_length_list = [100, 1000]
    for prompt in prompts_list:
        for model in model_list:
            for output_length in output_length_list:
                query_input = QueryInput(prompt=prompt, internal_id=0)
                args = argparse.Namespace(backend="not_implemented_backend", model=model, output_length=output_length)
                with pytest.raises(ValueError):
                    backends.get_payload(query_input, args)


def test_test_chunk_validity():
    # happy_vllm backend
    args = argparse.Namespace(backend="happy_vllm")
    for chunk in ['first chunk', "second chunk", "tok-i 1", "tok-o 1"]:
        chunk_validity = backends.test_chunk_validity(chunk, args)
        assert chunk_validity

    # mistral backend
    args = argparse.Namespace(backend="mistral")
    # True
    for chunk in ['first chunk', "second chunk"]:
        chunk_validity = backends.test_chunk_validity(chunk, args)
        assert chunk_validity
    # False
    for chunk in ["tok-i 1", "tok-o 1"]:
        chunk_validity = backends.test_chunk_validity(chunk, args)
        assert not(chunk_validity)

    # ValueError
    args = argparse.Namespace(backend="not_implemented_backend")
    for chunk in ['first chunk', "second chunk", "tok-i 1", "tok-o 1"]:
        with pytest.raises(ValueError):
            backends.test_chunk_validity(chunk, args)


def test_get_completions_headers():
    # happy_vllm backend
    args = argparse.Namespace(backend="happy_vllm")
    assert backends.get_completions_headers(args) == {}

    # mistral backend
    args = argparse.Namespace(backend="mistral")
    assert backends.get_completions_headers(args) == {"Accept": "application/json",
                "Content-Type": "application/json"}

    # ValueError
    args = argparse.Namespace(backend="not_implemented_backend")
    with pytest.raises(ValueError):
        backends.get_completions_headers(args)


def test_decode_remove_response_prefix():
    # happy_vllm backend
    args = argparse.Namespace(backend="happy_vllm")
    for chunk_str in ["Hey", "How are you ?", "Fine thanks!"]:
        chunk_bytes = bytes("data: " + chunk_str, "utf-8")
        chunk = backends.decode_remove_response_prefix(chunk_bytes, args)
        assert chunk == chunk_str

    # mistral backend
    args = argparse.Namespace(backend="mistral")
    for chunk_str in ["Hey", "How are you ?", "Fine thanks!"]:
        chunk_bytes = bytes("data: " + chunk_str, "utf-8")
        chunk = backends.decode_remove_response_prefix(chunk_bytes, args)
        assert chunk == chunk_str

    # ValueError
    args = argparse.Namespace(backend="not_implemented_backend")
    for chunk_str in ["Hey", "How are you ?", "Fine thanks!"]:
        chunk_bytes = bytes("data: " + chunk_str, "utf-8")
        with pytest.raises(ValueError):
            backends.decode_remove_response_prefix(chunk_bytes, args)


def test_check_end_of_stream():
    # happy_vllm backend
    args = argparse.Namespace(backend="happy_vllm")
    for chunk_not_done in ["not done", " [DONE]", "[DONE] "]:
        assert not(backends.check_end_of_stream(chunk_not_done, args))
    assert backends.check_end_of_stream("[DONE]", args)

    # mistral backend
    args = argparse.Namespace(backend="mistral")
    for chunk_not_done in ["not done", " [DONE]", "[DONE] "]:
        assert not(backends.check_end_of_stream(chunk_not_done, args))
    assert backends.check_end_of_stream("[DONE]", args)
    # ValueError
    args = argparse.Namespace(backend="not_implemented_backend")
    for chunk in ["not done", " [DONE]", "[DONE] ", "[DONE]"]:
        with pytest.raises(ValueError):
            backends.check_end_of_stream(chunk, args)


def test_get_newly_generated_text():
    generated_text = "Hey ! How are you ?"
    json_chunk_completions = {"id": "cmpl-d85f82039b864ceb8d95be931b200745", "object": "chat.completion.chunk", 
                            "created": 1716468615, "model": "CodeLlama-34B-AWQ", 
                            "choices": [{"index": 0,"text": f"{generated_text}", 
                            "stop_reason": None, "logprobs": None, "finish_reason": None}]}
    json_chunk_chat_completions = {"id":"cbaa5c28166d4b98b5256f1becc0364d", 
                                   "object":"chat.completion.chunk", 
                                   "created":1722322855, 
                                   "model":"mistral", 
                                   "choices":[{"index":0,"delta": {"content": f"{generated_text}"},"finish_reason": None,"logprobs": None}]}
    
    # happy_vllm backend
    args = argparse.Namespace(backend="happy_vllm")
    newly_generated_text = backends.get_newly_generated_text(json_chunk_completions, args)
    assert newly_generated_text == generated_text

    # mistral backend
    args = argparse.Namespace(backend="mistral")
    newly_generated_text = backends.get_newly_generated_text(json_chunk_chat_completions, args)
    assert newly_generated_text == generated_text

    # ValueError
    args = argparse.Namespace(backend="not_implemented_backend")
    with pytest.raises(ValueError):
        backends.get_newly_generated_text(json_chunk_chat_completions, args)


def test_add_prompt_length():
    prompt_tokens = 1234
    happy_vllm_chunk_with_usage = {"id": "cmpl-d5edc7c2c3264f189b3c941630751d8e", 
                                  "object": "text_completion",
                                  "created": 1722328225,
                                  "model": "Vigostral-7B-Chat-AWQ", 
                                  "choices":[], 
                                  "usage": {"prompt_tokens": prompt_tokens,"total_tokens": 108,"completion_tokens": 86}}

    happy_vllm_chunk_without_usage = {"id": "cmpl-d5edc7c2c3264f189b3c941630751d8e", 
                                  "object": "text_completion",
                                  "created": 1722328225,
                                  "model": "Vigostral-7B-Chat-AWQ", 
                                  "choices":[]}

    mistral_json_chunk_with_usage = {"id":"cbaa5c28166d4b98b5256f1becc0364d",
                          "object": "chat.completion.chunk",
                          "created":1722322855, 
                          "model":"mistral", 
                          "choices":[{"index": 0,"delta": {"content":""}, "finish_reason": "stop", "logprobs": None}],
                          "usage": {"prompt_tokens": prompt_tokens,"total_tokens": 58,"completion_tokens": 46}}
    mistral_json_chunk_without_usage = {"id":"cbaa5c28166d4b98b5256f1becc0364d",
                          "object": "chat.completion.chunk",
                          "created":1722322855, 
                          "model":"mistral", 
                          "choices":[{"index": 0,"delta": {"content":""}, "finish_reason": "stop", "logprobs": None}]}

    # happy_vllm backend with usage key
    args = argparse.Namespace(backend="happy_vllm")
    output = QueryOutput()
    assert output.prompt_length == 0
    backends.add_prompt_length(happy_vllm_chunk_with_usage, output, args)
    assert output.prompt_length == prompt_tokens

    # happy_vllm backend without usage key
    args = argparse.Namespace(backend="happy_vllm")
    output = QueryOutput()
    assert output.prompt_length == 0
    backends.add_prompt_length(happy_vllm_chunk_without_usage, output, args)
    assert output.prompt_length == 0

    # mistral backend with usage key
    args = argparse.Namespace(backend="mistral")
    output = QueryOutput()
    assert output.prompt_length == 0
    backends.add_prompt_length(mistral_json_chunk_with_usage, output, args)
    assert output.prompt_length == prompt_tokens

    # mistral backend without usage key
    args = argparse.Namespace(backend="mistral")
    output = QueryOutput()
    assert output.prompt_length == 0
    backends.add_prompt_length(mistral_json_chunk_without_usage, output, args)
    assert output.prompt_length == 0

    # ValueError
    args = argparse.Namespace(backend="not_implemented_backend")
    output = QueryOutput()
    with pytest.raises(ValueError):
        backends.add_prompt_length(mistral_json_chunk_with_usage, output, args)