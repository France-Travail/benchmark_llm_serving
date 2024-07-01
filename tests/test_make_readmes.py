import os
from pathlib import Path

from benchmark_llm_serving import make_readmes


def get_output_folder():
    current_directory = Path(os.path.dirname(os.path.realpath(__file__)))
    output_folder = current_directory / "data"
    return output_folder


def test_get_max_tokens_in_kv_cache():
    output_folder = get_output_folder()
    max_tokens_in_kv_cache = make_readmes.get_max_tokens_in_kv_cache(output_folder)
    assert max_tokens_in_kv_cache == 2500