import sys
import json
import argparse
from typing import Union


def get_parser_base_arguments() -> argparse.ArgumentParser:
    """Gets the parser with only base arguments
    
    Returns:
        ArgumentParser : The argparse parser
    """
    parser = argparse.ArgumentParser(description="Benchmark script")

    parser.add_argument("--dataset-folder",
                        type=str,
                        help="The path to the dataset folder")
    parser.add_argument("--base-url",
                        type=str,
                        help="Server or API base url if not using http host and port.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--step-live-metrics", type=float, default=0.01, 
                        help="The delta between two queries of the /metrics endpoints. Should be small since it blocks the completions queries")
    parser.add_argument("--max-queries", type=int, help="The maximum number of queries sent to the completions API. By default, it is the size of the dataset")
    parser.add_argument("--completions-endpoint", type=str, default="/v1/completions", help="The endpoint to the completions")
    parser.add_argument("--metrics-endpoint", type=str, default="/metrics/", help="The endpoint to the metrics")
    parser.add_argument("--info-endpoint", type=str, default="/v1/info", help="The endpoint to the metrics useless if the model name is provided "
                                                                               "and if the requested API is not happy_vLLM")
    parser.add_argument("--launch-arguments-endpoint", type=str, default="/v1/launch_arguments", help="The endpoint to get the launch arguments of happy_vllm")
    parser.add_argument("--backend", type=str, default="happy_vllm", help="The backend of the API we query")
    parser.add_argument("--model", type=str, help="The name of the model needed to query the completions API")
    parser.add_argument("--model-name", type=str, help="The name of the model to be displayed in the graphs")
    parser.add_argument("--max-duration", type=int, default=900, help="The maximal duration (in s) between the beginning of the queries and the end of the queries")
    parser.add_argument("--min-duration", type=int, help="The minimal duration during which the benchmark should run if there are still some prompts available")
    parser.add_argument("--target-queries-nb", type=int, help="If min-duration is reached and this number is reached, stop the benchmark")
    return parser


def add_arguments_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds several arguments specific to a scenario to the parser

    Args:
        parser (argparse.ArgumentParser) : The parser to add the arguments to

    Returns:
        argparse.ArgumentParser : The parser with the new arguments added
    """
    parser.add_argument("--prompt-length",
                        type=str,
                        default="0",
                        choices=["0", "32", "128", "256", "512", "1024", "2048", "4096"],
                        help="Which dataset to consider. 0 means the dataset with varying prompts length")
    parser.add_argument("--n-workers", type=int, default=1, help="The number of constant queries when query_profile == constant_number_of_queries")
    parser.add_argument("--query-profile", type=str, default="constant_number_of_queries", 
                        choices=["constant_number_of_queries", "request_rate", "growing_requests"],
                        help="The chosen profile of queries")
    parser.add_argument("--request-rate", type=float, default=float("inf"), help="Number of requests per second. If this is inf, "
                                                                                    "then all the requests are sent at time 0. "
                                                                                    "Otherwise, we use Poisson process to synthesize "
                                                                                    "the request arrival times.")
    parser.add_argument("--json-parameters", type=str, help="Path to a json containing parameters")
    parser.add_argument("--output-length", type=int, help="The number of tokens of the output")
    parser.add_argument("--query-metrics", action="store_true", help="Whether we should also get the results for each query in the output file")
    parser.add_argument("--output-file", type=str, help="Path to the output file")
    parser.add_argument("--with-kv-cache-profile", action="store_true", help="Whether we should also get the results for each query in the output file")
    return parser


def parse_args() -> argparse.Namespace:
    """Parses the args. We want this priority : args from cli > args from json > default values.

    Returns:
        NameSpace
    """
    # Gets the parser
    # The default value of the application variables are properly set
    # Those of the model variables are not
    parser = get_parser_base_arguments()
    parser = add_arguments_to_parser(parser)
    args = parser.parse_args()

    if args.json_parameters is not None:
        with open(args.json_parameters, 'r') as json_file:
            json_parameters = json.load(json_file)
        # Sets the default values of the model variables in the parser
        parser.set_defaults(**json_parameters)
    # Gets the args
    args = parser.parse_args()

    # Explicitly check for help flag for the providing help message to the entrypoint
    if '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        parser.print_help()
        sys.exit()
    return args