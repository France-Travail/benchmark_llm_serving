import importlib.metadata


def get_package_version() -> str:
    '''Returns the current version of the package

    Returns:
        str: version of the package
    '''
    version = importlib_metadata.version("benchmark_llm_serving")
    return version