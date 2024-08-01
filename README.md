# Benchmark LLM serving
[![pypi badge](https://img.shields.io/pypi/v/benchmark_llm_serving.svg)](https://pypi.python.org/pypi/benchmark_llm_serving)
[![Generic badge](https://img.shields.io/badge/python-3.10|3.11-blue.svg)](https://shields.io/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

![Build & Tests](https://github.com/France-Travail/benchmark_llm_serving/actions/workflows/build_and_tests.yaml/badge.svg)
![Wheel setup](https://github.com/France-Travail/benchmark_llm_serving/actions/workflows/wheel.yaml/badge.svg)

benchmark_llm_serving is a script aimed at benchmarking the serving API of LLMs. For now, two backends are implemented : [mistral](https://docs.mistral.ai/api/) and [vLLM](https://github.com/vllm-project/vllm) (via [happy-vllm](https://github.com/France-Travail/happy_vllm) which is an API layer on vLLM adding new endpoints and permitting a configuration via environment variables).

## Installation

It is advised to clone the repository in order to get the datasets used for the benchmarks (you can find them in `src/benchmark_llm_serving/datasets`) and build it from source:

```bash
git clone https://github.com/France-Travail/benchmark_llm_serving.git
cd benchmark_llm_serving
pip install -e .
```

You can also install benchmark_llm_serving using pip:

```bash
pip install benchmark_llm_serving
```
and download the datasets directly from the repository

## Quickstart

Launch the script bench_suite.py via the entrypoint `bench-suite` if you want a complete benchmarking of your API deployed via happy_vllm. This will launch several individual benchmarks which will be aggregated to draw graphs used to compare the models. All the results will be saved in the output folder (by default `results`). 

You can specify the launch arguments either via the CLI or a .env (see the `.env.example` for an example). If you cloned this repo and are benchmarking an API deployed via happy_vllm, you only need to specify the arguments `base-url` or the couple `host`/`port`. For example you can write :

`bench-suite --host 127.0.0.1 --port 5000`

**Be careful**, with the default arguments (those written in `.env.example`) the whole bench suite can be quite long (around 15 hours).

## Results

After the bench suite ends, you obtain a folder containing :

 - The results of all the benchmarks (in the zip file `raw_results.zip`)
 - A folder `report` containing the aggregation of all the individual benchmarks. More specifically:
   - `parameters.json` containing all the parameters for the bench, in particular, the arguments used to launch the `happy_vllm` API
   - `prompt_ingestion_graph.png` containing the graph of the speed of prompt ingestion by the model. It is the time taken to produce the first token vs the length of the prompt. The speed is the slope of this line and is indicated in the title of the graph. The data used for this graph is contained in the `data` folder.
   - `thresholds.csv` is a .csv containing, for each couple of input length/output length, the number of parallel requests such that : the kv cache usage is inferior to 100% and the speed generation is above a specified threshold (by default, 20 tokens per second)
   - `total_speed_generation_graph.png` is a graph containing, for each couple of input length/output length, the total speed generation vs the number of parallel requests. So, for example, if the model can answer to 10 parallel requests each with a speed of 20 tokens per second, the value on the graph will be 200 tokens per second (20 x 10). The data used for this graph is contained in the `data` folder.
   - If the backend is `happy_vllm` : a folder `kv_cache_profile` containing, for each couple of input length/output length, a graph showing the response of the LLMs to n requests launched at the same time. On the y-axis, you have the kv cache usage, the number of requests running and the number of requests waiting. On the x-axis, you have the time. The graph is obtained by sending one request, watching the response of the LLM then two requests, then three requests, ...
   - A folder `speed_generation` containing, for each couple of input length/output length, a graph showing the speed generation (per request) in token per second vs the number of parallel requests. The graph also shows the time to the first token generated in milliseconds. If the backend is `happy_vllm` it also shows the max kv cache usage for this number of parallel requests. The corresponding data is in the `data` folder

Note that the various input lengths are "32", "1024" and "4096" to simulate small, medium and long prompt. These length are to be understood as roughly this size (and generally speaking a bit above this size). The various output lengths are 16, 128 and 1024. Contrary to the input lengths, these are exact : the model produced exactly this number of tokens.


## Launch arguments


Here is a list of the arguments:
 - `model` : The name of the model you need to query the model. If you are using happy_vllm, you don't need to give it since it will automatically fetch it
 - `base-url` : The base url for the API you want to benchmark
 - `host` : The host of the API (if you specify a base-url, you don't need to specify a host)
 - `port` : The port of the API (if you specify a base-url, you don't need to specify a port)
 - `dataset-folder` : The folder where the datasets for querying the API are (by default, it is in `src/benchmark_llm_serving/datasets`)
 - `output-folder` : The folder where the results will be written (by default in the `results` folder)
 - `gpu-name`: The name of the GPU on which the model is (default `None`)
 - `step-live-metrics` : The time, in second, between two querying of the `/metrics/` endpoint of happy_vllm (default `0.01`)
 - `max-queries` : The maximal number of query for each bench (default `1000`)
 - `max-duration-prompt-ingestion` : The max duration (in seconds) for the execution of an individual script benchmarking the prompt ingestion ( default `900`)
 - `max-duration-kv-cache-profile` : The max duration (in seconds) for the execution of an individual script benchmarking the KV cache usage ( default `900`)
 - `max-duration-speed-generation` : The max duration (in seconds) for the execution of an individual script benchmarking the speed generation ( default `900`). It is also the max duration permitted for the launch of all the scripts benchmarking the speed generation for a given couple of input length/output length.
 - `min-duration-speed-generation` : For each individual script benchmarking the speed generation, if this min duration (in seconds) is reached and the target-queries-nb is also reached, the script will end (default `60`)
 - `target-queries-nb-speed-generation` : For each individual script benchmarking the speed generation, if this target-queries-nb is reached and the min-duration is also reached, the script will end (default `100`)
 - `min-number-of-valid-queries`: The minimal number of valid queries that should be present in a file to be considered for graph drawing (default `50`)
 - `backend` : Only `happy_vllm`and `mistral` are supported. 
 - `completions-endpoint` : The endpoint for completions (default `/v1/completions`)
 - `metrics-endpoint` : The endpoint for the metrics (default `/metrics/`)
 - `info-endpoint` : The info endpoint  (default `/v1/info`)
 - `launch-arguments-endpoint` : The endpoint for getting the launch arguments of the API (default `/v1/launch_arguments`)
 - `speed-threshold` : The speed generation above which the model is considered ok (default value `20`). It is only useful when writing `thresholds.csv` 
 - `model-name`: The name that should be displayed on the graph (default value : `None`). If it is `None`, the name displayed will be the one of the argument `model`
