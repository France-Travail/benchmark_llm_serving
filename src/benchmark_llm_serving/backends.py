import argparse

from benchmark_llm_serving.io_classes import QueryOutput, QueryInput


class BackEnd():

    TEMPERATURE = 0
    REPETITION_PENALTY = 1.2

    def __init__(self, backend_name: str, chunk_prefix: str = "data: ", last_chunk: str = "[DONE]", metrics_endpoint_exists: bool = True):
        self.backend_name = backend_name
        self.chunk_prefix = chunk_prefix
        self.last_chunk = last_chunk
        self.metrics_endpoint_exists = metrics_endpoint_exists

    def get_payload(self, query_input: QueryInput, args: argparse.Namespace) -> dict:
        """Gets the payload to give to the model

        Args:
            query_input (QueryInput) : The query input to use
            args (argparse.Namespace) : The cli args

        Returns:
            dict : The payload
        """
        raise NotImplemented("The subclass should implement this method") # type: ignore

    def get_newly_generated_text(self, json_chunk: dict) -> str:
        """Gets the newly generated text

        Args:
            json_chunk (dict) : The chunk containing the generated text

        Returns:
            str : The newly generated text
        """
        raise NotImplemented("The subclass should implement this method") # type: ignore

    def get_metrics_from_metrics_dict(self, metrics_dict: dict) -> dict:
        """Gets the useful metrics from the parsed output of the /metrics endpoint

        Args:
            metrics_dict (dict) : The parsed output of the /metrics endpoint

        Returns:
            dict : The useful metrics
        """
        raise NotImplemented("The subclass should implement this method if metrics_endpoint_exists") # type: ignore
    
    def test_chunk_validity(self, chunk: str) -> bool:
        """Tests if the chunk is valid or should not be considered.

        Args:
            chunk (str) : The chunk to consider

        Returns:
            bool : Whether the chunk is valid or not
        """
        return True

    def get_completions_headers(self) -> dict:
        """Gets the headers (depending on the backend) to use for the request

        Returns:
            dict: The headers

        """
        return {}

    def remove_response_prefix(self, chunk: str) -> str:
        """Removes the prefix in the response of a model

        Args:
            chunk (str) : The chunk received

        Returns:
            str : The string without the prefix
        """
        return chunk.removeprefix(self.chunk_prefix)

    def check_end_of_stream(self, chunk: str) -> bool:
        """Checks whether this is the last chunk of the stream

        Args:
            chunk (str) : The chunk to test

        Returns:
            bool : Whether it is the last chunk of the stream
        """
        return chunk == self.last_chunk

    def add_prompt_length(self, json_chunk: dict, output: QueryOutput) -> None:
        """Add the prompt length to the QueryOutput if the key "usage" is in the chunk

        Args:
            json_chunk (dict) : The chunk containing the prompt length
            output (QueryOutput) : The output
        """
        if "usage" in json_chunk:
            if json_chunk['usage'] is not None:
                output.prompt_length = json_chunk['usage']['prompt_tokens']


class BackendHappyVllm(BackEnd):

    def get_payload(self, query_input: QueryInput, args: argparse.Namespace) -> dict:
        """Gets the payload to give to the model

        Args:
            query_input (QueryInput) : The query input to use
            args (argparse.Namespace) : The cli args

        Returns:
            dict : The payload
        """
        return {"prompt": query_input.prompt,
                        "model": args.model,
                        "max_tokens": args.output_length,
                        "min_tokens": args.output_length,
                        "temperature": self.TEMPERATURE,
                        "repetition_penalty": self.REPETITION_PENALTY,
                        "stream": True,
                        "stream_options": {"include_usage": True}
                                }

    def get_newly_generated_text(self, json_chunk: dict) -> str:
        """Gets the newly generated text

        Args:
            json_chunk (dict) : The chunk containing the generated text

        Returns:
            str : The newly generated text
        """
        if len(json_chunk['choices']):
            data = json_chunk['choices'][0]['text']
            return data
        else:
            return ""
    
    def get_metrics_from_metrics_dict(self, metrics_dict: dict) -> dict:
        """Gets the useful metrics from the parsed output of the /metrics endpoint

        Args:
            metrics_dict (dict) : The parsed output of the /metrics endpoint

        Returns:
            dict : The useful metrics
        """
        metrics = {}
        metrics['num_requests_running'] = metrics_dict['vllm:num_requests_running'][0]['value']
        metrics['num_requests_waiting'] = metrics_dict['vllm:num_requests_waiting'][0]['value']
        metrics['gpu_cache_usage_perc'] = metrics_dict['vllm:gpu_cache_usage_perc'][0]['value']
        return metrics


class BackEndMistral(BackEnd):

    def get_payload(self, query_input: QueryInput, args: argparse.Namespace) -> dict:
        """Gets the payload to give to the model

        Args:
            query_input (QueryInput) : The query input to use
            args (argparse.Namespace) : The cli args

        Returns:
            dict : The payload
        """
        return {"messages": [{"role": "user", "content": query_input.prompt}],
                        "model": args.model,
                        "max_tokens": args.output_length,
                        "min_tokens": args.output_length,
                        "temperature": self.TEMPERATURE,
                        "stream": True
                                }
        
    def test_chunk_validity(self, chunk: str) -> bool:
        """Tests if the chunk is valid or should not be considered.

        Args:
            chunk (str) : The chunk to consider

        Returns:
            bool : Whether the chunk is valid or not
        """
        if chunk[:4] == "tok-":
            return False
        else:
            return True

    def get_completions_headers(self) -> dict:
        """Gets the headers (depending on the backend) to use for the request

        Returns:
            dict: The headers

        """
        return {"Accept": "application/json",
                "Content-Type": "application/json"}

    def get_newly_generated_text(self, json_chunk: dict) -> str:
        """Gets the newly generated text

        Args:
            json_chunk (dict) : The chunk containing the generated text

        Returns:
            str : The newly generated text
        """
        if len(json_chunk['choices']):
            data = json_chunk['choices'][0]['delta']["content"]
            return data
        else:
            return ""


def get_backend(backend_name: str) -> BackEnd:
    implemented_backends = ["mistral", "happy_vllm"]
    if backend_name not in implemented_backends:
        raise ValueError(f"The specified backend {backend_name} is not implemented. Please use one of the following : {implemented_backends}")
    if backend_name == "happy_vllm":
        return BackendHappyVllm(backend_name, chunk_prefix="data: ", last_chunk="[DONE]", metrics_endpoint_exists=True)
    if backend_name == "mistral":
        return BackEndMistral(backend_name, chunk_prefix="data: ", last_chunk="[DONE]", metrics_endpoint_exists=False)
    return BackEnd("not_implemented")