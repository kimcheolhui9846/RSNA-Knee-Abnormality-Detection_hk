import importlib
import subprocess
import sys


def test_src_packages_importable() -> None:
    for name in ("src", "src.data", "src.models"):
        importlib.import_module(name)


def test_src_models_importable_without_torch() -> None:
    # 학습 의존성(torch) 없는 CI 환경에서도 패키지 import는 되어야 한다
    code = "import sys; sys.modules['torch'] = None; import src.models; print('ok')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
