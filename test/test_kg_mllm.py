from kg_mllm.main import hello_torch


def test_hello_torch() -> None:
    assert hello_torch() == 'Hello PyTorch'
