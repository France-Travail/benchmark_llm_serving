

benchmark_llm_serving is a script aimed at benchmarking the serving API of LLMs. For now, it is focused on LLMs served via [vllm](https://github.com/vllm-project/vllm) and more specifically via [happy-vllm](https://github.com/OSS-Pole-Emploi/happy_vllm) which is an API layer on vLLM adding new endpoints and permitting a configuration via environment variables. 

## Installation

You can install benchmark_llm_serving using pip:

```bash
pip install benchmark_llm_serving
```

Or build it from source:

```bash
git clone https://github.com/OSS-Pole-Emploi/benchmark_llm_serving.git
cd benchmark_llm_serving
pip install -e .
```

## Quickstart

Launch the script bench_suite if you want a complete benchmarking of your API deployed via happy_vllm. This will launch several individual benchmarks which will be aggregated to draw graphs used to compare the models. All the results will be saved in the output folder

You can specify the launch arguments either via the CLI or a .env (see the .env.example for an example). If you cloned this repo and are benchmarking an API deployed via happy_vllm, you only need to specify the arguments base-url or the couple host/port. For example you can do :

`python src/benchmark_llm_serving/bench_suite.py --host 127.0.0.1 --port 5000`

## Launch arguments


Here is a list of the arguments:
 - `model` : The name of the model you need to query the model. If you are using happy_vllm, you don't need to give it since it will automatically fetch it
 - `base-url` : The base url for the API you want to benchmark
 - `host` : The host of the API (if you specify a base-url, you don't need to specify a host)
 - `port` : The port of the API (if you specify a base-url, you don't need to specify a port)
 - `dataset-folder` : The folder where the datasets for querying the API are (by default, it is in `datasets`)
 - `output-folder` : The folder where the results will be written (by default in the `results` folder)
 - `step-live-metrics` : The time, in second, between two querying of the `/metrics/` endpoint of happy_vllm (default `0.01`)
 - `max-queries` : The maximal number of query for each bench (default `1000`)
 - `max-duration-prompt-ingestion` : The max duration (in seconds) for the execution of an individual script benchmarking the prompt ingestion ( default `900`)
 - `max-duration-kv-cache-profile` : The max duration (in seconds) for the execution of an individual script benchmarking the KV cache usage ( default `900`)
 - `max-duration-speed-generation` : The max duration (in seconds) for the execution of an individual script benchmarking the speed generation ( default `900`). It is also the max duration permitted for the launch of all the scripts benchmarking the speed generation for a given couple of input length/output length.
 - `min-duration-speed-generation` : For each individual script benchmarking the speed generation, if this min duration (in seconds) is reached and the target-queries-nb is also reached, the script will end (default `60`)
 - `target-queries-nb-speed-generation` : For each individual script benchmarking the speed generation, if this target-queries-nb is reached and the min-duration is also reached, the script will end (default `100`)
 - `backend` : For now, only happy_vllm is supported. 
 - `completions-endpoint` : The endpoint for completions (default `/v1/completions`)
 - `metrics-endpoint` : The endpoint for the metrics (default `/metrics/`)
 - `info-endpoint` : The info endpoint  (default `/v1/info`)
 - `launch-arguments-endpoint` : The endpoint for getting the launch arguments of the API (default `/v1/launch_arguments`)