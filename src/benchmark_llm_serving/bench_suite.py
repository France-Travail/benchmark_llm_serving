import os
import random
import string
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, List
from pydantic_settings import BaseSettings, SettingsConfigDict

from benchmark_llm_serving import utils
from benchmark_llm_serving.io_classes import QueryInput
from benchmark_llm_serving.make_readmes import make_readme
from benchmark_llm_serving.backends import get_backend, BackEnd
from benchmark_llm_serving.make_graphs import draw_and_save_graphs
from benchmark_llm_serving.benchmark import launch_benchmark, augment_dataset
from benchmark_llm_serving.utils_args import get_parser_base_arguments, add_arguments_to_parser


logger = logging.getLogger("Benchmark suite")
logging.basicConfig(level=logging.INFO)


def get_random_string(length: int = 4) -> str:
    """Generates a random string of letters 

    Args:
        length (int) : The length of the output string

    Returns:
        str : The random string

    """
    prefix = ""
    for i in range(length):
        prefix += random.choice(string.ascii_letters)
    return prefix


def add_prefixes_to_dataset(dataset: List[str], length_prefix: int) -> List[str]:
    """Add a random prefix for each prompt of the dataset

    Args:
        dataset (list) : The list of prompt
        length_prefix (int) : The length of the prefix

    Returns:
        list : The list of new prompts
    """
    return [get_random_string(length_prefix) + prompt for prompt in dataset]


class BenchmarkSettings(BaseSettings):
    """Application settings

    This class is used for settings management purpose, have a look at the pydantic
    documentation for more details : https://pydantic-docs.helpmanual.io/usage/settings/

    By default, it looks for environment variables (case insensitive) to set the settings
    if a variable is not found, it looks for a file name .env in your working directory
    where you can declare the values of the variables and finally it sets the values
    to the default ones
    """
    model: Optional[str] = None
    base_url: str = ""
    host: str = "localhost"
    port: int = 8000
    model_name: Optional[str] = None

    gpu_name: Optional[str] = None

    dataset_folder: str = "src/benchmark_llm_serving/datasets"
    output_folder: str = "results"

    speed_threshold: float = 20.0

    step_live_metrics: float = 0.01
    max_queries: int = 1000
    max_duration_prompt_ingestion: int = 900
    max_duration_kv_cache_profile: int = 900
    max_duration_speed_generation: int = 900
    min_duration_speed_generation: int = 60
    target_queries_nb_speed_generation: int = 100

    min_number_of_valid_queries: int = 50

    backend: str = "happy_vllm"
    completions_endpoint: str = "/v1/completions"
    metrics_endpoint: str = "/metrics/"
    info_endpoint: str = "/v1/info"
    launch_arguments_endpoint: str = "/v1/launch_arguments"

    model_config = SettingsConfigDict(env_file=".env", extra='ignore', protected_namespaces=('settings', ))


def main():
    # Define arguments for the bench_suite
    bench_settings = BenchmarkSettings()
    parser = get_parser_base_arguments()
    parser.add_argument("--output-folder", type=str, help="Path to the output folder")
    parser.add_argument("--max-duration-prompt-ingestion", type=int, help="The max duration for the prompt ingestion")
    parser.add_argument("--max-duration-kv-cache-profile", type=int, help="The max duration for the KV cache profile")
    parser.add_argument("--max-duration-speed-generation", type=int, help="The max duration for the speed generation")
    parser.add_argument("--min-duration-speed-generation", type=int, help="The min duration for the speed generation")
    parser.add_argument("--target-queries-nb-speed_generation", type=int, help="The target_queries for the speed generation")
    parser.add_argument("--speed-threshold", type=float, help="Accepted threshold for generation speed")
    parser.add_argument("--min-number-of-valid-queries", type=int, help="The minimal number of queries needed to consider a file for drawing the graphs")
    parser.set_defaults(**bench_settings.model_dump())

    parser = add_arguments_to_parser(parser)
    
    args = parser.parse_args()
    args.request_rate = 0

    suite_id = utils.get_now()

    # Create the output folder if it doesn't exist
    if os.path.isabs(args.output_folder):
        output_folder = args.output_folder
    else:
        current_directory = Path(os.path.dirname(os.path.realpath(__file__)))
        grand_parent_directory = current_directory.parent.parent.absolute()
        output_folder = os.path.join(grand_parent_directory, args.output_folder)
    raw_results_folder = os.path.join(output_folder, "raw_results")
    if not os.path.isdir(raw_results_folder):
        os.makedirs(raw_results_folder)

    input_lengths = ["32", "1024", "4096"]
    output_lengths = [16, 128, 1024]

    if args.base_url is None:
        args.base_url = f"http://{args.host}:{args.port}"

    # We load the datasets here in order not to do it in each call to the benchmarks and for insuring
    # that they have different prefixes
    datasets = {}
    for input_length in ["0"] + input_lengths:
        dataset = utils.load_dataset(args.dataset_folder, input_length)
        datasets[input_length] = dataset.copy()

    # Define the input/output length couple
    input_output_lengths = []
    for input_length in input_lengths:
        for output_length in output_lengths:
            input_output_lengths.append((input_length, output_length))

    backend = get_backend(args.backend)
    
    # Launch the benchmark for prompt ingestion speed
    now = utils.get_now()
    logger.info(f"{now} Beginning the benchmark for the prompt ingestion speed")
    args.prompt_length = "0"
    args.n_workers = 1
    args.query_profile = "constant_number_of_queries"
    args.output_length = 1
    args.query_metrics = True
    args.max_duration = args.max_duration_prompt_ingestion
    args.min_duration = None
    args.target_queries_nb = None
    args.with_kv_cache_profile = False
    # We launch the script several time in order to take the mean and have more robust results
    for i in range(4):
        now = utils.get_now()
        logger.info(f"{now} Benchmark for the prompt ingestion speed : instance {i} ")
        args.output_file = os.path.join(raw_results_folder, f"prompt_ingestion_{i}.json")
        dataset = add_prefixes_to_dataset(datasets[args.prompt_length], 4)
        launch_benchmark(args, dataset, suite_id, backend=backend)
        now = utils.get_now()
        logger.info(f"{now} Benchmark for the prompt ingestion speed : instance {i} : DONE")
    now = utils.get_now()
    logger.info(f"{now} Benchmark for the prompt ingestion speed : DONE")

    if backend.backend_name == "happy_vllm":
        # Launch the benchmark for the KV cache profile
        now = utils.get_now()
        logger.info(f"{now} Beginning the benchmarks for the KV cache profile")
        args.query_profile = "growing_requests"
        args.query_metrics = False
        args.with_kv_cache_profile = True
        args.max_duration = args.max_duration_kv_cache_profile
        args.min_duration = None
        args.target_queries_nb = None
        for input_length, output_length in input_output_lengths:
            args.prompt_length = input_length
            args.output_length = output_length
            args.output_file = os.path.join(raw_results_folder, f"kv_cache_profile_input_{input_length}_output_{output_length}.json")
            now = utils.get_now()
            dataset = add_prefixes_to_dataset(datasets[args.prompt_length], 4)
            logger.info(f"{now} Beginning the benchmark for the KV cache profile, input length : {input_length}, output_length : {output_length}")
            launch_benchmark(args, dataset, suite_id, backend=backend)
            now = utils.get_now()
            logger.info(f"{now} Benchmark for the KV cache profile, input length : {input_length}, output_length : {output_length} : DONE")
        now = utils.get_now()
        logger.info(f"{now} Benchmarks for the KV cache profile : DONE")

    # Launch the benchmark for generation_speed
    now = utils.get_now()
    logger.info(f"{now} Beginning the benchmarks for the generation speed")
    args.query_profile = "constant_number_of_queries"
    args.query_metrics = False
    args.with_kv_cache_profile = False
    args.max_duration = args.max_duration_speed_generation
    args.min_duration = args.min_duration_speed_generation
    args.target_queries_nb = args.target_queries_nb_speed_generation
    for input_length, output_length in input_output_lengths:
        now = utils.get_now()
        logger.info(f"{now} Beginning the benchmarks for the generation speed, input length : {input_length}, output_length : {output_length}")
        max_duration_reached = False
        timestamp_beginning = datetime.now().timestamp()
        for nb_constant_requests in [1, 2, 3, 4, 6, 8] + list(range(10, 41, 4)):
            if not max_duration_reached:
                args.prompt_length = input_length
                args.output_length = output_length
                args.n_workers = nb_constant_requests
                args.output_file = os.path.join(raw_results_folder, f"generation_speed_input_{input_length}_output_{output_length}_nb_requests_{nb_constant_requests}.json")
                now = utils.get_now()
                logger.info(f"{now} Benchmarks for the generation speed, input length : {input_length}, output_length : {output_length}, nb_requests : {nb_constant_requests}")
                dataset = add_prefixes_to_dataset(datasets[args.prompt_length], 4)
                launch_benchmark(args, dataset, suite_id, backend=backend)
                now = utils.get_now()
                logger.info(f"{now} Benchmarks for the generation speed, input length : {input_length}, output_length : {output_length}, nb_requests : {nb_constant_requests} : DONE")
            current_timestamp = datetime.now().timestamp()
            if current_timestamp - timestamp_beginning > args.max_duration:
                max_duration_reached = True
        now = utils.get_now()
        logger.info(f"{now} Benchmarks for the generation speed, input length : {input_length}, output_length : {output_length} : DONE")
    now = utils.get_now()
    logger.info(f"{now} Benchmarks for the generation speed : DONE")

    now = utils.get_now()
    logger.info(f"{now} Drawing graphs")
    draw_and_save_graphs(output_folder, speed_threshold=args.speed_threshold, gpu_name=args.gpu_name,
                        min_number_of_valid_queries=args.min_number_of_valid_queries, backend=backend)
    now = utils.get_now()
    logger.info(f"{now} Drawing graphs : DONE")

    now = utils.get_now()
    logger.info(f"{now} Making readme")
    make_readme(output_folder)
    now = utils.get_now()
    logger.info(f"{now} Making readme : DONE")

    now = utils.get_now()
    logger.info(f"{now} Zipping raw_results folder")
    shutil.make_archive(os.path.join(output_folder, "raw_results"), 'zip', raw_results_folder)
    shutil.rmtree(raw_results_folder)
    now = utils.get_now()
    logger.info(f"{now} Zipping raw_results folder : DONE")

    now = utils.get_now()
    logger.info(f"{now} Everything : DONE")

if __name__ == "__main__":
    main()