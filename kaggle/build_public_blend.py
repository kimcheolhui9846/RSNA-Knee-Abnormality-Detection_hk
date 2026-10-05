"""공개 파이프라인 노트북 + 우리 앙상블 → 순위 혼합 제출 노트북을 만든다 (업로드는 하지 않는다).

공개 노트북(예: `yamadan96/rsna-knee-d4-public0946`, 공개 LB 0.943)의 셀을 **그대로** 두고
끝에 셀 두 개를 붙인다.
1. 공개 결과(`submission.csv`)를 `public_submission.csv`로 보관하고, 우리 제출 스크립트
   (`rsna-knee-code` 데이터셋의 `run_submission.py`)를 **별도 프로세스**로 돌려
   `ours_submission.csv`를 만든다.
   (공개 파이프라인과 패키지·메모리 상태를 섞지 않는다)
2. 라벨별 순위 혼합 `(1 - w) * rank(공개) + w * rank(우리)` → `submission.csv`.
   우리 단계가 실패하면 공개 결과를 그대로 제출한다.

실행: `python kaggle/build_public_blend.py --public <pulled 폴더> --user <kaggle 사용자명>
       --weight 0.1`
업로드(사용자 승인 후): `kaggle kernels push -p <out>`
"""

import argparse
import json
import shutil
from pathlib import Path

KERNEL_SLUG = "rsna-knee-public-blend"
CODE_SLUG, WEIGHTS_SLUG = "rsna-knee-code", "rsna-knee-weights"

ATTRIBUTION = """# 공개 파이프라인 + 우리 모델 순위 혼합

이 노트북의 앞부분(마지막 두 셀 전까지)은 공개 노트북 `{public_id}`의 코드를 **수정 없이** 쓴다.
그 코드와 가중치의 저작자: mattiaangeli, antoinegg1, prvsiyan, pilkwang, marwanmath,
dreaddevelopment, sofiaanjenje, jiweiliu, ryokucha 등 (각 데이터셋·노트북 참조).

마지막 두 셀만 우리 것이다:
우리 모델 앙상블(`{user}/{code}`, `{user}/{weights}`)을 별도 프로세스로 돌려
라벨별 순위로 {w:.0%} 섞는다.
"""

RUN_OURS = """# --- 우리 앙상블 (별도 프로세스) -------------------------------------------------
import os, shutil, subprocess, sys, time
from pathlib import Path

_t0 = time.time()
OURS_DEADLINE_SEC = 8.3 * 3600  # 노트북 시작부터 이 시각까지 우리 단계를 끝낸다
try:  # 공개 파이프라인이 잡아 둔 GPU 메모리를 돌려준다
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
except Exception:  # noqa: BLE001
    pass
_work = Path("/kaggle/working")
shutil.copy(_work / "submission.csv", _work / "public_submission.csv")
_ours = _work / "ours_submission.csv"
_runner = None
for _root, _dirs, _files in os.walk("/kaggle/input"):
    _dirs[:] = [d for d in _dirs if d not in ("train_series", "test_series")]
    if "run_submission.py" in _files and "rsna_knee_code.marker" in _files:
        _runner = Path(_root) / "run_submission.py"
        break
print("our runner:", _runner)
# 9시간 제한: 노트북이 시작된 뒤 지난 시간을 빼고 남은 만큼만 우리 단계에 준다.
# 넘으면 우리 단계를 끊고 공개 결과만 제출한다 (제출 전체가 시간 초과로 실패하지 않게).
try:
    import psutil

    _elapsed = time.time() - psutil.Process().create_time()
except Exception:  # noqa: BLE001
    _elapsed = None
_limit = None if _elapsed is None else max(0.0, OURS_DEADLINE_SEC - _elapsed)
print(f"notebook elapsed {_elapsed}, ours time limit {_limit}")
if _runner is not None and (_limit is None or _limit > 300):
    try:
        _r = subprocess.run(
            [sys.executable, str(_runner)],
            env={**os.environ, "KAGGLE_SUBMISSION_PATH": str(_ours)},
            capture_output=True, text=True, timeout=_limit,
        )
        print(_r.stdout[-4000:])
        print(_r.stderr[-4000:])
    except subprocess.TimeoutExpired:
        print("ours timed out — public only")
        _ours.unlink(missing_ok=True)
print(f"ours done in {(time.time() - _t0) / 60:.1f} min, exists={_ours.exists()}")
"""

BLEND = """# --- 순위 혼합 → submission.csv (우리 단계 실패 시 공개 결과 그대로) ---------------
import numpy as np
import pandas as pd

OUR_WEIGHT = {w}
_pub = pd.read_csv(_work / "public_submission.csv", dtype={{"StudyInstanceUID": str}})
_labels = [c for c in _pub.columns if c != "StudyInstanceUID"]
_final = _pub.copy()
try:
    _our = pd.read_csv(_ours, dtype={{"StudyInstanceUID": str}}).set_index("StudyInstanceUID")
    _our = _our.loc[_pub["StudyInstanceUID"], _labels]
    if not np.isfinite(_our.to_numpy(float)).all():
        raise ValueError("ours has non-finite values")
    for _c in _labels:
        _final[_c] = (1 - OUR_WEIGHT) * _pub[_c].rank(pct=True) + OUR_WEIGHT * _our[_c].rank(
            pct=True
        ).to_numpy()
    print(f"blended: public {{1 - OUR_WEIGHT:.2f}} + ours {{OUR_WEIGHT:.2f}}")
except Exception as _e:  # noqa: BLE001
    print("ours unavailable, submitting public only:", repr(_e))
assert len(_final) == len(_pub) and np.isfinite(_final[_labels].to_numpy(float)).all()
_final.to_csv(_work / "submission.csv", index=False)
print("submission.csv written", _final.shape)
"""


def _cell(kind: str, source: str, cid: str) -> dict:
    cell = {
        "cell_type": kind,
        "id": cid,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def build(public_dir: Path, user: str, weight: float, out: Path) -> Path:
    public_dir = Path(public_dir)
    meta = json.loads((public_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    nb = json.loads((public_dir / meta["code_file"]).read_text(encoding="utf-8"))
    for c in nb["cells"]:  # 공개 노트북의 저장된 출력은 버린다
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    head = ATTRIBUTION.format(
        public_id=meta["id"], user=user, code=CODE_SLUG, weights=WEIGHTS_SLUG, w=weight
    )
    nb["cells"] = [
        _cell("markdown", head, "ours-attribution"),
        *nb["cells"],
        _cell("code", RUN_OURS, "ours-run"),
        _cell("code", BLEND.format(w=weight), "ours-blend"),
    ]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / f"{KERNEL_SLUG}.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    new_meta = {
        "id": f"{user}/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": f"{KERNEL_SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [
            *meta.get("dataset_sources", []),
            f"{user}/{CODE_SLUG}",
            f"{user}/{WEIGHTS_SLUG}",
        ],
        "competition_sources": meta.get("competition_sources", []),
        "kernel_sources": meta.get("kernel_sources", []),
        "model_sources": meta.get("model_sources", []),
    }
    for key in ("docker_image", "machine_shape"):  # 공개 노트북과 같은 실행 환경
        if meta.get(key):
            new_meta[key] = meta[key]
    (out / "kernel-metadata.json").write_text(json.dumps(new_meta, indent=2), encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--public", type=Path, required=True, help="kaggle kernels pull -m 로 받은 폴더")
    p.add_argument("--user", required=True)
    p.add_argument("--weight", type=float, default=0.1, help="우리 앙상블 순위의 비율")
    p.add_argument("--out", type=Path, default=Path("outputs/kaggle_public_blend"))
    a = p.parse_args()
    print(build(a.public, a.user, a.weight, a.out))


if __name__ == "__main__":
    main()
