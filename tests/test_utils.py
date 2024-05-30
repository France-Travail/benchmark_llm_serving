import importlib

from benchmark_llm_serving import utils

def test_get_package_version():
    # Nominal case
    version = utils.get_package_version()
    assert version == importlib.metadata.version("benchmark_llm_serving")