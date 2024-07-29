import os
import json
import argparse
import numpy as np
from mdutils import Html
from mdutils.mdutils import MdUtils


def get_max_tokens_in_kv_cache(output_folder: str) -> int:
    """Calculates the maximum number of tokens the KV cache can contain. 

    Args:
        output_folder (str) : The folder which contain all the results

    Returns:
        int : The number of tokens the KV cache can contain 
    """
    raw_result_folder = os.path.join(output_folder, "raw_results")
    all_raw_filenames = list(os.walk(raw_result_folder))[0][2]
    kv_cache_filenames = [filename for filename in all_raw_filenames if "kv_cache_profile" in filename]
    if len(kv_cache_filenames) == 0:
        return -1
    max_tokens_in_kv_cache_list = []
    for filename in kv_cache_filenames:
        path = os.path.join(raw_result_folder, filename)
        with open(path, 'r') as json_file:
            kv_cache_profile = json.load(json_file)
        kv_cache_profile_general = kv_cache_profile["general_metrics"]
        # Total number of tokens (prompts+generated)
        total_tokens = kv_cache_profile_general["total_number_of_ingested_tokens"] + kv_cache_profile_general["total_number_of_generated_tokens"]
        mean_tokens_per_query = total_tokens / kv_cache_profile_general["total_number_of_queries"]
        
        kv_cache_profile_list = kv_cache_profile['kv_cache_profile']
        kv_cache_profile_list = sorted(kv_cache_profile_list, key=lambda x: x["gpu_cache_usage_perc"]) 
        max_kv_cache_result = kv_cache_profile_list[-1]
        max_cache_usage_perc = max_kv_cache_result["gpu_cache_usage_perc"]
        nb_requests = max_kv_cache_result["num_requests_running"]
        # Extrapolated max number of tokens in the KV cache
        max_tokens_in_kv =  int(mean_tokens_per_query * nb_requests / max_cache_usage_perc)
        max_tokens_in_kv_cache_list.append(max_tokens_in_kv)
    return int(np.median(max_tokens_in_kv_cache_list))


def add_summary_section(mdfile: MdUtils, output_folder: str, report_folder: str) -> MdUtils:
    """Adds the summary section to the readme.

    Args:
        mdfile (MdUtils) : The future readme
        output_folder (str) : The folder which contain all the results
        report_folder (str) : The folder containing the report

    Returns:
        MdUtils : The future readme
    """
    prompt_ingestion_file = os.path.join(report_folder, "data", "prompt_ingestion_graph_data.json")
    with open(prompt_ingestion_file, 'r') as json_file:
        prompt_ingestion = json.load(json_file)
    

    # Summary table
    mdfile.new_header(level=1, title='Main metrics')
    mdfile.new_paragraph("The main metrics are summarized in the following table : ")

    summary_data = ["Metric", "Value"]
    
    # Prompt ingestion speed
    prompt_ingestion_speed = int(1 / prompt_ingestion["fit_coefficients"][0])
    # Precision 250 tokens per second
    prompt_ingestion_speed = int(250 * (prompt_ingestion_speed//250))
    summary_data.extend(["Prompt ingestion speed", f"~{prompt_ingestion_speed} t/s "])

    # Speed generation
    speed_generation_filename = "speed_generation_graph_data_input_1024_output_128.json"
    speed_generation_file = os.path.join(report_folder, "speed_generation", "data", speed_generation_filename)
    with open(speed_generation_file, 'r') as json_file:
        speed_generation = json.load(json_file)
    speed_generation_value = int(speed_generation["10"]["speed_generation"][1])
    summary_data.extend(["Mean generation speed for 10 parallel requests with a prompt of 1024 tokens and 128 tokens generated",
                        f"~{speed_generation_value} t/s for each request"])

    # Max tokens in KV cache
    max_tokens_in_kv_cache = get_max_tokens_in_kv_cache(output_folder)
    if max_tokens_in_kv_cache == -1:
        summary_data.extend(["Estimate of the max nb of tokens in KV cache", "NA"])
    else:
        # Precision 25k tokens
        approximate_max_tokens_in_kv_cache = int(25 * (max_tokens_in_kv_cache // 25000))
        summary_data.extend(["Estimate of the max nb of tokens in KV cache", f"~{approximate_max_tokens_in_kv_cache}k tokens"])
    

    mdfile.new_line()
    mdfile.new_table(columns=2, rows=len(summary_data)//2, text=summary_data, text_align='center')
    return mdfile


def add_total_generation_speed_section(mdfile: MdUtils) -> MdUtils:
    """Adds the total generation speed section to the readme

    Args:
        mdfile (MdUtils) : The future readme

    Returns:
        MdUtils : The future readme
    """
    mdfile.new_header(level=1, title='Total generation speed')
    mdfile.new_paragraph("The following graph shows the total throughput with respect to "
                        "the number of parallel queries for different scenarii :")
    graph_file = "./total_speed_generation_graph.png"
    mdfile.new_paragraph(Html.image(path=graph_file, size='800'))
    mdfile.new_line()
    return mdfile


def add_generation_speed_section(mdfile: MdUtils) -> MdUtils:
    """Adds the generation speed section to the readme

    Args:
        mdfile (MdUtils) : The future readme

    Returns:
        MdUtils : The future readme
    """
    mdfile.new_header(level=1, title='Generation speed and latency')
    mdfile.new_paragraph("The following graphs shows the throughput for each request "
                        "with respect to the number of parallel queries. It also shows "
                        "the time to first token and the corresponding gpu usage. "
                        "The lines show the median whereas the bars show the "
                        "25th and 75th percentile")
    for input_size, output_size in [(4096, 128), (32, 16)]:
        mdfile.new_paragraph(f"This graph is for prompt size of roughly {input_size} tokens and "
                             f"{output_size} generated tokens :")
        filename = f"speed_generation_graph_input_{input_size}_output_{output_size}.png"
        graph_file = f"./speed_generation/{filename}"
        mdfile.new_paragraph(Html.image(path=graph_file, size='800'))
        mdfile.new_paragraph()
    return mdfile


def add_parameters_section(mdfile: MdUtils, parameters: dict) -> MdUtils:
    """Adds the parameters section to the readme

    Args:
        mdfile (MdUtils) : The future readme
        parameters (dict) : The launch parameters

    Returns:
        MdUtils : The future readme
    """
    # Parameters table
    mdfile.new_header(level=1, title='Parameters')
    mdfile.new_paragraph("The api from happy_vllm was launched using the following arguments : ")
    rows_nb = len(parameters) + 1

    table_data = ["Parameter", "Value"]
    for key, value in parameters.items():
        table_data.extend([key, value])

    mdfile.new_line()
    mdfile.new_table(columns=2, rows=rows_nb, text=table_data, text_align='center')
    return mdfile


def make_readme(output_folder: str) -> None:
    """Writes the readme

    Args:
        output_folder (str) : The folder which contain all the results
    """
    report_folder = os.path.join(output_folder, "report")
    output_file = os.path.join(report_folder, "README.md")
    parameters_file = os.path.join(report_folder, "parameters.json")
    with open(parameters_file, 'r') as json_file:
        parameters = json.load(json_file)
    
    model_name = parameters['model']
    gpu_name = parameters['gpu_name']


    mdfile = MdUtils(file_name=output_file,title=f"Model card for {model_name} on {gpu_name}")
    mdfile = add_summary_section(mdfile, output_folder, report_folder)
    mdfile = add_total_generation_speed_section(mdfile)
    mdfile = add_generation_speed_section(mdfile)
    mdfile = add_parameters_section(mdfile, parameters)
    mdfile.create_md_file()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Readmes script")
    parser.add_argument("--output-folder", type=str, help="Path to the output folder")
    args = parser.parse_args()
    make_readme(output_folder=args.output_folder)