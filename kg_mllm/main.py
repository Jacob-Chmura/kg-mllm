def hello_torch() -> str:
    import torch

    _ = torch.zeros(1337)
    return 'Hello PyTorch'


def main() -> None:
    hello_torch()


if __name__ == '__main__':
    main()
