import os
import csv
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Any, Union
from matplotlib import pyplot as plt
from pydantic_settings import BaseSettings, SettingsConfigDict

from benchmark_llm_serving import utils
from benchmark_llm_serving.backends import get_backend, BackEnd

logger = logging.getLogger("Making the graphs")
logging.basicConfig(level=logging.INFO)


class GraphsSettings(BaseSettings):
    """Graphs settings

    This class is used for settings management purpose, have a look at the pydantic
    documentation for more details : https://pydantic-docs.helpmanual.io/usage/settings/

    By default, it looks for environment variables (case insensitive) to set the settings
    if a variable is not found, it looks for a file name .env in your working directory
    where you can declare the values of the variables and finally it sets the values
    to the default ones
    """
    output_folder: str = "results"
    speed_threshold: float = 20.0
    min_number_of_valid_queries: int = 50

    model_config = SettingsConfigDict(env_file=".env", extra='ignore', protected_namespaces=('settings', ))


def make_prompt_ingestion_graph(files: dict, report_folder: str) -> None:
    """Draws the prompt ingestion graph and saves the corresponding data

    Args:
        files (dict) : The files containing the results of the benchmarks
        report_folder (str) : The folder where the report should be written
    """
    prompt_ingestion_files = {key: value for key, value in files.items() if 'prompt_ingestion' in key}
    
    # First fit on all data
    prompt_lengths = []
    times_to_first_token = []
    for filename, results in prompt_ingestion_files.items():
        for query_output in results['individual_query']:
            prompt_lengths.append(query_output["prompt_length"])
            times_to_first_token.append(query_output["time_to_first_token"])
        model_name = results['parameters']['model_name']
    first_fit = np.poly1d(np.polyfit(prompt_lengths, times_to_first_token, 1))

    # Delete outliers
    cleaned_prompt_lengths = []
    cleaned_times_to_first_token = []
    for prompt_length, time_to_first_token in zip(prompt_lengths, times_to_first_token):
        if abs(first_fit(prompt_length) - time_to_first_token) / time_to_first_token < 0.05:
            cleaned_prompt_lengths.append(prompt_length)
            cleaned_times_to_first_token.append(time_to_first_token)

    # Second fit on data without outliers
    second_fit_coefficients = np.polyfit(cleaned_prompt_lengths, cleaned_times_to_first_token, 1)
    prompt_ingestion_coefficient = int(1 / second_fit_coefficients[0])
    second_fit = np.poly1d(second_fit_coefficients)

    # Figure definition
    fig, ax1 = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    ax1.set_xlabel('Prompt length', fontsize='14')
    ax1.set_ylabel('Time to first token', fontsize='14')

    # Scatter plot of the cleaned data
    ax1.scatter(cleaned_prompt_lengths, cleaned_times_to_first_token, s=12)
    unique_prompt_values = np.unique(cleaned_prompt_lengths)

    # Plot of the fit
    ax1.plot(unique_prompt_values, second_fit(unique_prompt_values), c="red", linestyle='--')

    plt.title(f"Model : {model_name} \n Prompt ingestion speed : {prompt_ingestion_coefficient} tokens per second", fontsize='16')
    # Save plot
    plt.savefig(os.path.join(report_folder, 'prompt_ingestion_graph.png'),bbox_inches='tight',dpi=75)

    # Save data
    graph_data = {"fit_coefficients": list(second_fit_coefficients),
                  "prompt_lengths": cleaned_prompt_lengths,
                  "times_to_first_token": times_to_first_token}
    with open(os.path.join(report_folder, "data", "prompt_ingestion_graph_data.json"), "w") as json_file:
        json.dump(graph_data, json_file, indent=4)


def make_speed_generation_graph_for_one_input_output(input_length: int, output_length: int,
                                                     speed_generation_files: dict, report_folder: str,
                                                     speed_threshold: float, backend: BackEnd) -> dict:
    """Draws the speed generation graph and save the corresponding data for a couple of
    input length/ output length. Also returns the corresponding speed and kv cache
    thresholds

    Args:
        input_length (int) : The length of the inputs
        output_length (int) : The length of the outputs
        speed_generation_files (dict) : The file containing the results of the speed generation benchmarks
        report_folder (str) : The folder where the report should be written
        speed_threshold (float) : The accepted speed generation to fix the threshold
        backend (BackEnd) : The backend

    Returns:
        dict : The accepted thresholds 
    """
    input_output_files = {key: value for key, value in speed_generation_files.items()
                            if f'input_{input_length}_output_{output_length}' in key}
    data_summary = {}
    # Definition of the data used
    lower_percentile = 'percentile_25'
    upper_percentile = 'percentile_75'
    name_speed = "speed_generation"
    name_TTFT = 'time_to_first_token'

    # For each benchmark result
    for filename, results in input_output_files.items():
        # Take the useful data and summarize it in data_summary
        model_name = results['parameters']['model_name']
        nb_parallel_requests = results['parameters']['n_workers']
        speed_from_beginning = results['aggregated_metrics']['speed_from_beginning']
        speed_from_beginning_median = speed_from_beginning['median']
        speed_generation = (speed_from_beginning_median - speed_from_beginning[lower_percentile],
                            speed_from_beginning_median,
                            speed_from_beginning[upper_percentile] - speed_from_beginning_median)
        time_to_first_token = results['aggregated_metrics']['time_to_first_token']
        time_to_first_token_median = time_to_first_token['median']
        time_to_first_token_generation = (time_to_first_token_median - time_to_first_token[lower_percentile],
                            time_to_first_token_median,
                            time_to_first_token[upper_percentile] - time_to_first_token_median)
        data_summary[nb_parallel_requests] = {name_speed: speed_generation,
                                              name_TTFT: time_to_first_token_generation,
                                              "max_kv_cache": results['general_metrics']['max_kv_cache'],
                                              "parallel_requests_nb": nb_parallel_requests}
    
    # Prepare lists to plot
    parallel_requests_nbs = sorted(list(data_summary))

    speed_generation_plot = []
    speed_generation_lower_percentiles = []
    speed_generation_upper_percentiles = []
    time_to_first_token_plot = []
    time_to_first_token_lower_percentiles = []
    time_to_first_token_upper_percentiles = []
    max_kv_cache = []

    for parallel_requests_nb in parallel_requests_nbs:
        speed_generation_plot.append(data_summary[parallel_requests_nb][name_speed][1])
        speed_generation_lower_percentiles.append(data_summary[parallel_requests_nb][name_speed][0])
        speed_generation_upper_percentiles.append(data_summary[parallel_requests_nb][name_speed][2])
        time_to_first_token_plot.append(data_summary[parallel_requests_nb][name_TTFT][1])
        time_to_first_token_lower_percentiles.append(data_summary[parallel_requests_nb][name_TTFT][0])
        time_to_first_token_upper_percentiles.append(data_summary[parallel_requests_nb][name_TTFT][2])
        max_kv_cache.append(data_summary[parallel_requests_nb]['max_kv_cache'])

    # Figure definition
    fig, ax1 = plt.subplots()
    fig.subplots_adjust(right=0.75)
    # Specify the type of ax2 and ax3 to avoid mypy errors
    ax3: Any = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.1))
    fig.set_size_inches(18.5, 10.5)
    ax1.set_xlabel('Number of parallel requests', fontsize='14')
    ax1.set_ylabel('Speed generation (tokens per second)', fontsize='14')
    
    ax3.set_ylabel('Time to first token (ms)', fontsize='14')

    if backend.backend_name == "happy_vllm":
        ax2: Any = ax1.twinx()
        ax2.set_ylabel('Max KV cache percentage', fontsize='14')
        ax2.set_ylim((0, 1.0))

    # Speed generation plot
    speed_generation_graph = ax1.errorbar(parallel_requests_nbs, speed_generation_plot, 
                yerr=[speed_generation_lower_percentiles, speed_generation_upper_percentiles],
                fmt='b-o',
                capsize=4, label="Speed generation")
    if backend.backend_name == "happy_vllm":
        # Max KV cache plot
        max_kv_cache_graph = ax2.plot(parallel_requests_nbs, max_kv_cache, color='green', linestyle="--", label="Max KV cache")
    # Time to first token generation plot                
    time_to_first_token_generation_graph = ax3.errorbar(parallel_requests_nbs, time_to_first_token_plot, 
                yerr=[time_to_first_token_lower_percentiles, time_to_first_token_upper_percentiles],
                fmt='r-o',
                capsize=4, label="Time to first token")
    if backend.backend_name == "happy_vllm":
        curves = [speed_generation_graph, max_kv_cache_graph[0], time_to_first_token_generation_graph]
    else:
        curves = [speed_generation_graph, time_to_first_token_generation_graph]
    # Legend
    ax1.legend(
        handles=curves,
        labels=[c.get_label() for c in curves]
    )

    plt.title(f"Model : {model_name} \n Speed generation | input length: {input_length} | output length : {output_length}", fontsize='16')
    # Save graph
    plt.savefig(os.path.join(report_folder, "speed_generation", f'speed_generation_graph_input_{input_length}_output_{output_length}.png'), bbox_inches='tight',dpi=75)

    # Save data
    with open(os.path.join(report_folder, "speed_generation", "data", f"speed_generation_graph_data_input_{input_length}_output_{output_length}.json"), 'w') as json_file:
        json.dump(data_summary, json_file, indent=4)

    # Calculates thresholds
    # Speed threshold
    speed_treshold_reached = [key for key, value in data_summary.items() if value[name_speed][1] >= speed_threshold]
    if len(speed_treshold_reached):
        max_requests_speed = max(speed_treshold_reached)
    else:
        max_requests_speed = 0
    if backend.backend_name == "happy_vllm":
        # KV cache threshold
        kv_treshold_not_reached = [key for key, value in data_summary.items() if value["max_kv_cache"] < 0.95]
        if len(kv_treshold_not_reached):
            max_kv_cache_requests = max(kv_treshold_not_reached)
        else:
            max_kv_cache_requests = 0
        return {"speed_threshold": max_requests_speed, "kv_cache_threshold": max_kv_cache_requests}
    else:
        return {"speed_threshold": max_requests_speed}


def make_speed_generation_graphs(files: dict, report_folder: str, speed_threshold: float, backend: BackEnd):
    """Draws the speed generation graphs and save the corresponding data. Also saves the thresholds,
    namely the accepted number of parallel requests with a KV cache inferior to 1.0 and 
    a speed generation above the speed_threshold

    Args:
        files (dict) : The files containing the results of the benchmarks
        report_folder (str) : The folder where the report should be written
        speed_threshold (float) : The accepted speed generation to fix the threshold
        backend (BackEnd) : The backend
    """
    speed_generation_files = {key: value for key, value in files.items() if 'generation_speed' in key}
    thresholds = []
    # Get input output couples
    input_output_couples = set()
    for filename in speed_generation_files:
        split_filename = filename.split('_')
        input_output_couples.add((int(split_filename[3]), int(split_filename[5])))
    # Make speed generation graph for each couple and save the corresponding thresholds
    for input_length, output_length in input_output_couples:
        thresholds_tmp = make_speed_generation_graph_for_one_input_output(input_length, output_length, speed_generation_files, report_folder, speed_threshold, backend)
        thresholds_tmp['input_length'] = input_length
        thresholds_tmp['output_length'] = output_length
        thresholds.append(thresholds_tmp.copy())

    # Write thresholds results in a csv
    with open(os.path.join(report_folder, "thresholds.csv"), "w", encoding="utf8", newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=thresholds[0].keys())
        writer.writeheader()
        writer.writerows(thresholds)


def make_kv_cache_profile_graph_for_one_input_output(input_length: int, output_length: int,
                                                     kv_cache_files: dict, report_folder: str) -> None:
    """Draws and saves the kv cache profile for a couple input length/ output length. We don't save
    the corresponding data since it is already in the raw results almost without parsing. 

    Args:
        input_length (int) : The length of the inputs
        output_length (int) : The length of the outputs
        kv_cache_files (dict) : The files containing the results of the benchmarks for the kv cache profile
        report_folder (str) : The folder where the report should be written
    """
    filename = f"kv_cache_profile_input_{input_length}_output_{output_length}.json"
    results = kv_cache_files[filename]
    model_name = results['parameters']['model']
    kv_cache_profile = sorted(results['kv_cache_profile'], key=lambda x:x["timestamp"])
    # Get the first timestamp to be able to plot durations instead of absolute timestamps
    first_timestamp = kv_cache_profile[0]['timestamp']

    # Prepare lists for plot
    duration = [data_point['timestamp'] - first_timestamp for data_point in kv_cache_profile]
    kv_cache = [data_point['gpu_cache_usage_perc'] for data_point in kv_cache_profile]
    running = [data_point['num_requests_running'] for data_point in kv_cache_profile]
    waiting = [data_point['num_requests_waiting'] for data_point in kv_cache_profile]

    # Figure definition
    fig, ax1 = plt.subplots()
    ax2: Any = ax1.twinx()
    fig.set_size_inches(18.5, 10.5)
    ax1.set_xlabel('Time (in second)', fontsize='14')
    ax1.set_ylabel('Percentage KV cache', fontsize='14')
    ax2.set_ylabel('Requests number', fontsize='14')
    ax1.set_ylim((0, 1.0))
    ax2.set_ylim(bottom=0, top=max(running+waiting)+1)

    # Plots the kv cache usage and running and waiting requests
    ax1.plot(duration, kv_cache, label="KV cache percentage", color='green')
    ax2.plot(duration, running, label="Running requests", color='blue')
    ax2.plot(duration, waiting, label="Waiting requests", color='red')

    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2)

    plt.title(f"Model : {model_name} \n KV cache profile | input length: {input_length} | output length : {output_length}", fontsize='16')
    # Save graph
    plt.savefig(os.path.join(report_folder, "kv_cache_profile", f'graph_kv_cache_profile_input_{input_length}_output_{output_length}.png'), bbox_inches='tight',dpi=75)


def make_kv_cache_profile_graphs(files: dict, report_folder: str) -> None:
    """Draws and saves the kv cache profile graphs.

    Args:
        files (dict) : The files containing the results of the benchmarks
        report_folder (str) : The folder where the report should be written
    """
    kv_cache_files = {key: value for key, value in files.items() if 'kv_cache_profile' in key}
    # Get input output couples
    input_output_couples = set()
    for filename in kv_cache_files:
        split_filename = filename.replace('.json', '').split('_')
        input_output_couples.add((int(split_filename[4]), int(split_filename[6])))
    # Make kv cache profile graph for each couple
    for input_length, output_length in input_output_couples:
        make_kv_cache_profile_graph_for_one_input_output(input_length, output_length, kv_cache_files, report_folder)


def make_total_speed_generation_graph(files: dict, report_folder: str) -> None:
    """Draws and saves the total speed generation graph. Also saves the corresponding data.

    Args:
        files (dict) : The files containing the results of the benchmarks
        report_folder (str) : The folder where the report should be written
    """
    speed_generation_files = {key: value for key, value in files.items() if 'generation_speed' in key}
    
    # First we get the data
    data_summary: dict[tuple, Any] = {}
    for filename, results in speed_generation_files.items():
        nb_parallel_requests = results['parameters']['n_workers']
        total_number_of_generated_tokens = results['general_metrics']['total_number_of_generated_tokens']
        total_time = results['general_metrics']['total_time']
        actual_total_time = results['general_metrics']['actual_total_time']
        input_length = int(results['parameters']['prompt_length'])
        output_length = results['parameters']['output_length']
        if (input_length, output_length) in data_summary:
            data_summary[(input_length, output_length)].append({"nb_parallel_requests": nb_parallel_requests,
                                                                "total_time": total_time,
                                                                "actual_total_time": actual_total_time,
                                                                "total_number_of_generated_tokens": total_number_of_generated_tokens})
        else:
            data_summary[(input_length, output_length)] = [{"nb_parallel_requests": nb_parallel_requests,
                                                                "total_time": total_time,
                                                                "actual_total_time": actual_total_time,
                                                                "total_number_of_generated_tokens": total_number_of_generated_tokens}]
    
    # Figure definition
    fig, ax1 = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    ax1.set_xlabel('Number of parallel requests', fontsize='14')
    ax1.set_ylabel('Total speed generation (tokens per second)', fontsize='14')

    # For each input length/output length couple
    for input_length, output_length in data_summary:
        # We sort the data with growing number of parallel requests
        data_summary[(input_length, output_length)] = sorted(data_summary[(input_length, output_length)], key=lambda x: x['nb_parallel_requests'])
        # We calculate the difference between the actual total time and the total time for one requests
        # The logic being that this difference is roughly the "lost" time used for setting the queries and 
        # should be subtracted of all actual total time
        one_request = data_summary[(input_length, output_length)][0]
        one_request_nb_parallel = one_request["nb_parallel_requests"]
        if one_request_nb_parallel != 1:
            raise ValueError(f"The minimal number of request should be 1 for the couple ({input_length}, {output_length}). But it is {one_request_nb_parallel}")
        time_difference = one_request['actual_total_time'] - one_request['total_time']
        # Calculate speed
        for value_nb_request in data_summary[(input_length, output_length)]:
            tokens_nb = value_nb_request["total_number_of_generated_tokens"]
            time_elapsed = value_nb_request['actual_total_time'] - time_difference
            value_nb_request["speed"] = tokens_nb / time_elapsed
        # Plot the graph
        nb_parallel_requests = [value["nb_parallel_requests"] for value in data_summary[(input_length, output_length)]]
        speed = [value["speed"] for value in data_summary[(input_length, output_length)]]
        ax1.plot(nb_parallel_requests, speed, label=f"input length :{input_length}, output length : {output_length}")
    
    plt.legend()

    plt.savefig(os.path.join(report_folder, f'total_speed_generation_graph.png'), bbox_inches='tight',dpi=75)


    # Save data
    data_to_save = {}
    for i, (input_length, output_length) in enumerate(data_summary):
        data_to_save[i] = {'data': data_summary[(input_length, output_length)].copy()}
        data_to_save[i]['input_length'] = input_length
        data_to_save[i]['output_length'] = output_length
    with open(os.path.join(report_folder, "data", "total_speed_generation_graph_data.json"), 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)


def save_common_parameters(files: dict, report_folder: str, gpu_name: str):
    """Saves the common parameters of all the benchmarks.

    Args:
        files (dict) : The files containing the results of the benchmarks
        report_folder (str) : The folder where the report should be written
        gpu_name (str) : The name of the GPU
    """
    common_parameters: dict[str, Any] = {}
    for results in files.values():
        # Initialization
        if len(common_parameters) == 0:
            common_parameters = results['parameters'].copy()
        else:
            # We pop the parameters which don't have the same values in all benchmarks
            # since they are not "common" between all benchmarks
            for key, value in results['parameters'].items():
                if key in common_parameters:
                    if value != common_parameters[key]:
                        common_parameters.pop(key)
    common_parameters["gpu_name"] = gpu_name

    with open(os.path.join(report_folder, 'parameters.json'), 'w') as json_file:
        json.dump(common_parameters, json_file, indent=4)


def draw_and_save_graphs(output_folder: str, speed_threshold: float = 20.0, gpu_name: Union[str, None] = None,
                         min_number_of_valid_queries: int = 50, backend: Union[BackEnd, str] = "happy_vllm"):
    """Draws and saves all the graphs and corresponding data for benchmark results 
    obtained via bench_suite.py

    Args:
        output_folder (str) : The folder where the results of the benchmarks are
        speed_threshold (float) : The accepted speed generation to fix the threshold
        gpu_name (str) : The name of the gpu
        backend (str) : The backend
    """
    if isinstance(backend, str):
        backend = get_backend(backend)

    # Manage output path
    if not os.path.isabs(output_folder):
        current_directory = Path(os.path.dirname(os.path.realpath(__file__)))
        grand_parent_directory = current_directory.parent.parent.absolute()
        output_folder = os.path.join(grand_parent_directory, output_folder)
    raw_result_folder = os.path.join(output_folder, "raw_results")

    # Make report folder and subfolders
    report_folder = os.path.join(output_folder, "report")
    subfolders = [os.path.join(report_folder, "data"),
                 os.path.join(report_folder, "kv_cache_profile"),
                 os.path.join(report_folder, "speed_generation"),
                 os.path.join(report_folder, "speed_generation", "data")]
    for folder in [report_folder] + subfolders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    now = utils.get_now()
    logger.info(f"{now} Loading the results")
    # Get the data
    filenames = list(os.walk(raw_result_folder))[0][2]
    files = {}
    for filename in filenames:
        if ".json" in filename and "graph" not in filename:
            with open(os.path.join(raw_result_folder, filename), 'r') as json_file:
                files[filename] = json.load(json_file)
    files = {key: value for key, value in files.items() 
             if value['general_metrics']['total_number_of_queries'] - value['general_metrics']['nb_errored_queries'] + value['general_metrics']['nb_timeout_queries'] >= min_number_of_valid_queries}
    
    now = utils.get_now()
    logger.info(f"{now} Making prompt ingestion graph")
    make_prompt_ingestion_graph(files, report_folder)
    now = utils.get_now()
    logger.info(f"{now} Making speed generation graphs")
    make_speed_generation_graphs(files, report_folder, speed_threshold, backend)
    if backend.backend_name == "happy_vllm":
        now = utils.get_now()
        logger.info(f"{now} Making kv cache profile graphs")
        make_kv_cache_profile_graphs(files, report_folder)
    now = utils.get_now()
    logger.info(f"{now} Making total speed generation graph")
    make_total_speed_generation_graph(files, report_folder)
    save_common_parameters(files, report_folder, gpu_name)
    now = utils.get_now()
    logger.info(f"{now} Graphs done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graphs script")
    parser.add_argument("--output-folder", type=str, help="Path to the output folder")
    parser.add_argument("--speed-threshold", type=float, default=20.0, help="Accepted threshold for generation speed")
    parser.add_argument("--gpu-name", help="The name of the GPU")
    parser.add_argument("--min-number-of-valid-queries", type=int, help="The minimal number of queries needed to consider a file for drawing the graphs")
    parser.add_argument("--backend", type=str, default='happy_vllm', help="The backend")
    graph_settings = GraphsSettings()
    parser.set_defaults(**graph_settings.model_dump())
    args = parser.parse_args()
    draw_and_save_graphs(output_folder=args.output_folder, speed_threshold=args.speed_threshold, gpu_name=args.gpu_name,
                        min_number_of_valid_queries=args.min_number_of_valid_queries, backend=args.backend)