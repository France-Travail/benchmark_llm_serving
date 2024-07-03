import os
import pytest
import asyncio
import importlib.metadata
from pathlib import Path

from benchmark_llm_serving import utils


def test_get_package_version():
    # Nominal case
    version = utils.get_package_version()
    assert version == importlib.metadata.version("benchmark_llm_serving")


async def waiting_task(time_to_wait):
    await asyncio.sleep(time_to_wait)
    return "Done"


def test_get_now():
    now = utils.get_now()
    assert isinstance(now, str)
    assert now[0] == '2'
    assert now[4] == '-'
    assert now[7] == '-'
    assert now[10:12] == ', '
    assert now[14] == ':'
    assert now[17] == ':'


def test_load_dataset():
    current_directory = Path(os.path.dirname(os.path.realpath(__file__)))
    dataset_folder = current_directory / "data"
    
    dataset = utils.load_dataset(dataset_folder=dataset_folder,
                                prompt_length="0")
    assert len(dataset) == 3
    assert isinstance(dataset, list)

    dataset = utils.load_dataset(dataset_folder=dataset_folder,
                                prompt_length="42")
    assert len(dataset) == 4
    assert isinstance(dataset, list)


@pytest.mark.asyncio
async def test_tasks_are_done():
    tasks = []
    time_step = 0.01
    for i in range(1, 4):
        tasks += [asyncio.create_task(waiting_task(i*time_step + time_step))]
        assert not(utils.tasks_are_done(tasks))
        await asyncio.sleep(time_step)
        assert not(utils.tasks_are_done(tasks))
        await asyncio.sleep(i*time_step + time_step)
        assert utils.tasks_are_done(tasks)

