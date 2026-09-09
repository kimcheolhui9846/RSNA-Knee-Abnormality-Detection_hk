import torch

def test_pytorch_installation():
    # 파이토치가 정상적으로 임포트되고 버전이 잡히는지 테스트
    assert torch.__version__ is not None
    print(f"PyTorch Version: {torch.__version__}")