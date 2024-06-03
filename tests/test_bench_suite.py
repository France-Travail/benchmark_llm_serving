import string
from benchmark_llm_serving import bench_suite


def test_get_random_string():
    for i in range(10):
        random_string = bench_suite.get_random_string(length=i)
        assert len(random_string) == i
        for character in random_string:
            assert character in string.ascii_letters


def test_add_prefixes_to_dataset():

    initial_dataset = ["Hey", "How are you ?"]
    for i in range(10):
        dataset = initial_dataset.copy()
        new_dataset = bench_suite.add_prefixes_to_dataset(dataset, i)
        for prompt_nb, new_prompt in enumerate(new_dataset):
            initial_prompt = initial_dataset[prompt_nb]
            assert initial_prompt == new_prompt[i:]