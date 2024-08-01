import argparse
from benchmark_llm_serving import utils_args


def test_get_parser_base_arguments():
    parser = utils_args.get_parser_base_arguments()
    base_arguments = {"--dataset-folder", "--base-url", "--host", "--port", "--step-live-metrics", "--max-queries",
                       "--completions-endpoint", "--metrics-endpoint", "--info-endpoint", "--launch-arguments-endpoint",
                       "--backend", "--model", "--max-duration", "--min-duration", "--target-queries-nb", "--help", "-h", "--model-name"}
    assert set(parser.__dict__["_option_string_actions"]) == base_arguments


def test_add_arguments_to_parser():
    parser = argparse.ArgumentParser(description="")
    parser = utils_args.add_arguments_to_parser(parser)
    base_arguments = {"--prompt-length", "--n-workers", "--query-profile", "--request-rate", "--json-parameters",
                       "--output-length", "--query-metrics", "--output-file", "--with-kv-cache-profile", "--help", "-h"}
    assert set(parser.__dict__["_option_string_actions"]) == base_arguments