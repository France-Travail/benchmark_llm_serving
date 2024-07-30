import argparse

from benchmark_llm_serving.io_classes import QueryOutput, QueryInput


IMPLEMENTED_BACKENDS = "'happy_vllm', 'mistral'"


def get_payload(query_input: QueryInput, args: argparse.Namespace) -> dict:
    """Gets the payload to give to the model

    Args:
        query_input (QueryInput) : The query input to use
        args (argparse.Namespace) : The cli args

    Returns:
        dict : The payload
    """
    temperature = 0
    repetition_penalty = 1.2
    if args.backend == "happy_vllm":
        return {"prompt": query_input.prompt,
                        "model": args.model,
                        "max_tokens": args.output_length,
                        "min_tokens": args.output_length,
                        "temperature": temperature,
                        "repetition_penalty": repetition_penalty,
                        "stream": True,
                        "stream_options": {"include_usage": True}
                                }
    elif args.backend == "mistral":
        return {"messages": [{"role": "user", "content": query_input.prompt}],
                        "model": args.model,
                        "max_tokens": args.output_length,
                        "min_tokens": args.output_length,
                        "temperature": temperature,
                        "stream": True
                                }
    else:
        raise ValueError(f"The specified backend {args.backend} is not implemented. Please use one of the following : {IMPLEMENTED_BACKENDS}")


def test_chunk_validity(chunk: str, args: argparse.Namespace) -> bool:
    """Tests if the chunk is valid or should not be considered.

    Args:
        chunk (str) : The chunk to consider
        args (argparse.Namespace) : The cli args

    Returns:
        bool : Whether the chunk is valid or not
    """
    if args.backend in ["happy_vllm"]:
        return True
    elif args.backend in ["mistral"]:
        if chunk[:4] == "tok-":
            return False
        else:
            return True
    else:
        raise ValueError(f"The specified backend {args.backend} is not implemented. Please use one of the following : {IMPLEMENTED_BACKENDS}")


def get_completions_headers(args: argparse.Namespace) -> dict:
    """Gets the headers (depending on the backend) to use for the request

    Args:
        args (argparse.Namespace) : The cli args
    
    Returns:
        dict: The headers

    """
    if args.backend in ["happy_vllm"]:
        return {}
    elif args.backend == "mistral":
        return {"Accept": "application/json",
                "Content-Type": "application/json"}
    else:
        raise ValueError(f"The specified backend {args.backend} is not implemented. Please use one of the following : {IMPLEMENTED_BACKENDS}")


def decode_remove_response_prefix(chunk_bytes: bytes, args: argparse.Namespace) -> str:
    """Removes the prefix in the response of a model and converts the bytes in str

    Args:
        chunk_bytes (bytes) : The chunk received
        args (argparse.Namespace) : The cli args

    Returns:
        str : The decoded string without the prefix
    """
    chunk = chunk_bytes.decode("utf-8")
    if args.backend in ["happy_vllm", "mistral"]:
        return chunk.removeprefix("data: ")
    else:
        raise ValueError(f"The specified backend {args.backend} is not implemented. Please use one of the following : {IMPLEMENTED_BACKENDS}")


def check_end_of_stream(chunk: str, args: argparse.Namespace) -> bool:
    """Checks if this is the last chunk of the stream

    Args:
        chunk (str) : The chunk to test
        args (argparse.Namespace) : The cli args

    Returns:
        bool : Whether it is the last chunk of the stream
    """
    if args.backend in ["happy_vllm", "mistral"]:
        return chunk == "[DONE]"
    else:
        raise ValueError(f"The specified backend {args.backend} is not implemented. Please use one of the following : {IMPLEMENTED_BACKENDS}")


def get_newly_generated_text(json_chunk: dict, args: argparse.Namespace) -> str:
    """Gets the newly generated text

    Args:
        json_chunk (dict) : The chunk containing the generated text
        args (argparse.Namespace) : The cli args

    Returns:
        str : The newly generated text
    """
    if args.backend == "happy_vllm":
        if len(json_chunk['choices']):
            data = json_chunk['choices'][0]['text']
            return data
    elif args.backend == "mistral":
        if len(json_chunk['choices']):
            data = json_chunk['choices'][0]['delta']["content"]
            return data
    else:
        raise ValueError(f"The specified backend {args.backend} is not implemented. Please use one of the following : {IMPLEMENTED_BACKENDS}")
    return ""


def add_prompt_length(json_chunk: dict, output: QueryOutput, args: argparse.Namespace) -> None:
    """Add the prompt length to the QueryOutput

    Args:
        json_chunk (dict) : The chunk containing the prompt length
        args (argparse.Namespace) : The cli args
    """
    if args.backend in ["happy_vllm", 'mistral']:
        if "usage" in json_chunk:
            if json_chunk['usage'] is not None:
                output.prompt_length = json_chunk['usage']['prompt_tokens']
    else:
        raise ValueError(f"The specified backend {args.backend} is not implemented. Please use one of the following : {IMPLEMENTED_BACKENDS}")