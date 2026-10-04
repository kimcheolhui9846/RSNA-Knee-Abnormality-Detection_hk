#!/usr/bin/env python3
"""
train.py — 학습 → safetensors 저장 → HF 비공개 레포 업로드 → Pod 자동 종료

이 파일은 '학습 레포(GitHub Private)'의 루트에 복사해서 쓴다.
학습 로직은 train_model(cfg)만 채우면 되고, 나머지(업로드/예외/종료)는 그대로 둔다.

로컬(RTX 3070 Ti 등) 파이프라인 검증:
    python train.py --smoke-test --no-upload --no-shutdown

종료 보장 구조:
    try:     train_model() → 가중치 업로드
    except:  traceback 기록 + 중간 체크포인트 업로드 시도
    finally: (1) status/로그 업로드 시도  (2) 무조건 Pod 종료 API 호출 — 각각 별도 try
※ SyntaxError(이 파일 자체의 문법 오류), import 단계 실패, OOM-kill(SIGKILL)은
  여기의 try/finally로 잡을 수 없다. start_up.sh의 EXIT trap과 워치독이 최종 안전망이다.

종료 코드: 0=성공, 1=일반 예외, 2=CUDA OOM, 3=업로드(네트워크) 실패,
           130=KeyboardInterrupt, 143=SIGTERM(stop/terminate/워치독)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import signal
import sys
import time
import traceback
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

# hf_transfer가 설치돼 있으면 대용량 업로드를 빠르게(옵션). huggingface_hub import 전에 정해야 한다.
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") is None:
    try:
        import hf_transfer  # noqa: F401

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    except ImportError:
        pass

log = logging.getLogger("train")


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
@dataclass
class Config:
    run_id: str
    data_dir: Path
    output_dir: Path
    checkpoint_dir: Path
    hf_repo_id: str | None
    git_commit: str
    smoke_test: bool = False
    # 하이퍼파라미터는 전부 YAML config에 둔다 (configs/expXXX.yaml)
    config_path: str = "configs/exp001_baseline.yaml"
    extra: dict = field(default_factory=dict)


class Terminated(BaseException):
    """SIGTERM을 받았을 때 던진다. BaseException이라 `except Exception`에 안 잡힌다."""


def _on_sigterm(signum, _frame):
    raise Terminated(f"signal {signum}")


_last_heartbeat = 0.0


def heartbeat() -> None:
    """학습이 진행 중임을 start_up.sh의 정지 감시(STALL_SEC)에 알린다.

    로그를 자주 찍지 않는 학습 루프에서는 step마다 불러 둔다(30초에 한 번만 실제로 기록한다).
    로그가 STALL_SEC(기본 10분) 넘게 안 늘고 이 함수도 안 불리면 멈춘 것으로 보고 Pod를 끝낸다.
    """
    global _last_heartbeat
    state_dir = os.environ.get("HARNESS_STATE_DIR")
    now = time.time()
    if state_dir and now - _last_heartbeat >= 30:
        _last_heartbeat = now
        try:
            # 내용을 바꿔 써야 감시가 알아챈다 (mtime은 볼륨 서버 시계라 믿지 않는다)
            Path(state_dir, "heartbeat").write_text(f"{now:.0f}\n", encoding="utf-8")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 학습 (이 함수만 채우면 된다)
# ---------------------------------------------------------------------------
def train_model(cfg: Config) -> Path:
    """학습 후 최종 .safetensors 경로를 반환한다.

    중간 체크포인트는 cfg.checkpoint_dir/*.safetensors 로 저장해 두면
    실패 시 가장 최근 파일을 자동으로 HF에 올린다.
    """
    import yaml

    from src.train import make_synthetic_data, run

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(cfg.config_path).read_text(encoding="utf-8"))
    data_dir = cfg.data_dir

    if cfg.smoke_test:
        # 실데이터 없이 같은 학습 코드를 아주 작게 끝까지 돌린다
        # (CPU로 돌릴 것: CUDA_VISIBLE_DEVICES=, 사전학습 가중치 다운로드 없음)
        data_dir = cfg.output_dir / "synthetic_data"
        # config가 고르는 라벨 파일 이름·캐시 위치(cache_dir)를 합성 데이터에도 그대로 맞춘다
        make_synthetic_data(
            data_dir,
            n_studies=8,
            depth=4,
            size=32,
            n_folds=2,
            n_pseudo=4,
            cache_subdir=config.get("cache_dir", "cache"),
        )
        labels_file = config.get("labels_file", "labels.csv")
        if labels_file != "labels.csv":
            shutil.copyfile(data_dir / "labels.csv", data_dir / labels_file)
        config.update(
            n_folds=2, epochs=1, batch_size=2, depth=4, size=32, target_slices=4, num_workers=0
        )
        config["model"] = {
            "backbone": "resnet18",
            "pretrained": False,
            "embed_dim": 32,
            "dropout": 0.0,
        }

    result = run(
        config,
        data_dir=data_dir,
        out_dir=cfg.output_dir,
        heartbeat=heartbeat,
        checkpoint_dir=cfg.checkpoint_dir,  # fold마다 저장 → 실패·시간 초과 때 하네스가 업로드
    )
    cfg.extra["result"] = {k: result[k] for k in ("macro_auc", "fold_std", "elapsed_sec")}
    return cfg.output_dir / "model.safetensors"


# ---------------------------------------------------------------------------
# 업로드
# ---------------------------------------------------------------------------
def gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return os.environ.get("GPU_TYPE", "cpu")


def with_retry(fn, what: str, attempts: int = 5, base: float = 5.0):
    """지수 백오프 재시도 (5s, 10s, 20s, 40s + 지터)."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise
            delay = base * 2**i + random.uniform(0, 2)
            log.warning(
                "%s 실패(%s: %s) → %.0fs 후 재시도 %d/%d",
                what,
                type(e).__name__,
                e,
                delay,
                i + 1,
                attempts - 1,
            )
            time.sleep(delay)


def hf_upload(cfg: Config, files: dict[str, Path], note: str) -> None:
    """files = {runs/{RUN_ID}/ 아래 상대경로: 로컬 경로} 를 한 커밋으로 올린다."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()  # 토큰은 HF_TOKEN env 또는 로컬 로그인에서 읽힌다
    with_retry(
        lambda: api.create_repo(cfg.hf_repo_id, private=True, exist_ok=True, repo_type="model"),
        "create_repo",
    )
    ops = [
        CommitOperationAdd(f"runs/{cfg.run_id}/{rel}", str(p))
        for rel, p in files.items()
        if p.exists()
    ]
    msg = f"[{cfg.run_id}] {note} | git {cfg.git_commit[:12] or 'unknown'} | gpu {gpu_name()}"
    with_retry(
        lambda: api.create_commit(cfg.hf_repo_id, ops, commit_message=msg, repo_type="model"),
        "upload",
    )
    log.info("HF 업로드 완료: %s/runs/%s (%d개 파일)", cfg.hf_repo_id, cfg.run_id, len(ops))


def latest_checkpoint(cfg: Config) -> Path | None:
    cks = sorted(cfg.checkpoint_dir.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
    return cks[-1] if cks else None


# ---------------------------------------------------------------------------
# Pod 종료
# ---------------------------------------------------------------------------
def shutdown_pod() -> None:
    """RunPod GraphQL로 자기 자신을 terminate(기본) 또는 stop 한다.

    ON_FINISH=stop: GPU 과금은 멈추지만 컨테이너/볼륨 디스크 요금은 계속 나간다.
    또 stop된 Pod를 다시 켜면 Start Command가 다시 돈다(start_up.sh의 재시작 가드가 재학습은 막음).
    """
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not pod_id:
        log.info("RUNPOD_POD_ID 없음(로컬 실행) → Pod 종료 생략")
        return
    if not re.fullmatch(r"[A-Za-z0-9]+", pod_id) or not key or "{{" in key:
        raise RuntimeError(
            "RUNPOD_POD_ID/RUNPOD_API_KEY가 유효하지 않아 Python에서 종료 불가 "
            "→ start_up.sh trap에 맡김"
        )
    if os.environ.get("ON_FINISH", "terminate") == "stop":
        q = f'mutation {{ podStop(input: {{podId: "{pod_id}"}}) {{ id desiredStatus }} }}'
    else:
        q = f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}'
    url = os.environ.get("RUNPOD_GRAPHQL_URL", "https://api.runpod.io/graphql")
    req = urllib.request.Request(
        url,
        data=json.dumps({"query": q}).encode(),
        # urllib 기본 User-Agent(Python-urllib)는 Cloudflare가 403(error code 1010)으로 막는다
        # (2026-10-02 실측)
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "runpod-hf-harness/1.0",
        },
    )
    state_dir = os.environ.get("HARNESS_STATE_DIR")
    if state_dir:
        Path(state_dir, "shutdown_requested").touch()

    def call():
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read() or b"{}")
        if body.get("errors"):
            raise RuntimeError(body["errors"][0].get("message"))

    with_retry(call, "Pod 종료", attempts=3, base=3)
    log.info("Pod %s 요청 완료 (%s)", os.environ.get("ON_FINISH", "terminate"), pod_id)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="학습 → safetensors → HF 업로드 → Pod 자동 종료")
    p.add_argument("--smoke-test", action="store_true", help="작은 더미 텐서로 파이프라인만 검증")
    p.add_argument("--no-upload", action="store_true", help="HF 업로드를 끈다 (로컬 검증용)")
    p.add_argument(
        "--no-shutdown", action="store_true", help="Pod 종료 API 호출을 끈다 (디버깅용, 비용 주의)"
    )
    p.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "outputs"),
        help="결과 저장 위치 (기본: outputs)",
    )
    p.add_argument(
        "--config", default="configs/exp001_baseline.yaml", help="실험 YAML (하이퍼파라미터 전부)"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for stream in (sys.stdout, sys.stderr):  # Windows 콘솔(cp949)에서 한글 깨짐 방지
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    run_id = os.environ.get("RUN_ID") or time.strftime("local-%Y%m%d-%H%M%S")
    out = Path(args.output_dir) / run_id
    cfg = Config(
        run_id=run_id,
        data_dir=Path(os.environ.get("DATA_DIR", "/workspace/data")),
        output_dir=out,
        checkpoint_dir=out / "checkpoints",
        hf_repo_id=os.environ.get("HF_REPO_ID") or None,
        git_commit=os.environ.get("GIT_COMMIT_SHA", ""),
        smoke_test=args.smoke_test,
        config_path=args.config,
    )
    out.mkdir(parents=True, exist_ok=True)
    # 로그: 콘솔(=start_up.sh가 tee로 LOG_FILE에 기록) + 결과 폴더의 train.log
    logging.basicConfig(
        level=logging.INFO,
        format="[train %(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out / "train.log", encoding="utf-8"),
        ],
    )
    signal.signal(signal.SIGTERM, _on_sigterm)

    upload = not args.no_upload and bool(cfg.hf_repo_id)
    if not args.no_upload and not cfg.hf_repo_id:
        log.warning("HF_REPO_ID가 없어 업로드를 생략합니다")
    config_path = out / "config.json"
    config_path.write_text(
        json.dumps(asdict(cfg), default=str, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    status = {
        "run_id": run_id,
        "status": "running",
        "exit_code": None,
        "writer": "train.py",
        "git_commit": cfg.git_commit,
        "gpu": gpu_name(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rc = 1
    try:
        log.info(
            "학습 시작: run_id=%s gpu=%s commit=%s",
            run_id,
            status["gpu"],
            cfg.git_commit[:12] or "-",
        )
        weights = train_model(cfg)
        if upload:
            files = {"model.safetensors": weights, "config.json": config_path}
            for name in ("metrics.json", "oof.csv"):  # 교차검증 결과 (src/train.py가 저장)
                files[name] = cfg.output_dir / name
            hf_upload(cfg, files, "final weights")
        rc = 0
        status["status"] = "succeeded"
    except BaseException as e:  # noqa: BLE001 — 종료 경로를 하나로 모으기 위해 의도적으로 넓게 잡는다
        rc, status["status"] = classify(e)
        status["error"] = f"{type(e).__name__}: {e}"
        log.error("학습 실패 (%s)\n%s", status["status"], traceback.format_exc())
        ck = latest_checkpoint(cfg)
        if upload and ck is not None:
            try:
                hf_upload(
                    cfg,
                    {f"checkpoints/{ck.name}": ck, "config.json": config_path},
                    f"checkpoint after {status['status']}",
                )
            except BaseException:  # noqa: BLE001
                log.error("체크포인트 업로드 실패\n%s", traceback.format_exc())
    finally:
        status["exit_code"] = rc
        status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # (1) status + 로그 업로드 — 실패해도 (2)는 반드시 실행
        try:
            status_path = out / "status.json"
            status_path.write_text(
                json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            state_dir = os.environ.get("HARNESS_STATE_DIR")
            if state_dir:  # start_up.sh가 status를 덮어쓰지 않도록 같은 내용을 공유
                Path(state_dir, "status.json").write_text(
                    status_path.read_text(encoding="utf-8"), encoding="utf-8"
                )
            if upload:
                files = {"status.json": status_path, "logs/train.log": out / "train.log"}
                if os.environ.get("LOG_FILE"):
                    lf = Path(os.environ["LOG_FILE"])
                    files[f"logs/{lf.name}"] = lf
                hf_upload(cfg, files, f"status={status['status']}")
        except BaseException:  # noqa: BLE001
            log.error("status/로그 업로드 실패\n%s", traceback.format_exc())
            if rc == 0:
                rc = 3
        # (2) Pod 종료 — 무조건 시도
        try:
            if args.no_shutdown:
                log.info("--no-shutdown → Pod 종료 생략")
            else:
                for h in logging.getLogger().handlers:
                    h.flush()
                shutdown_pod()
        except BaseException:  # noqa: BLE001
            log.error("Pod 종료 실패 (start_up.sh trap이 다시 시도)\n%s", traceback.format_exc())
    return rc


def classify(e: BaseException) -> tuple[int, str]:
    """예외 → (종료 코드, 상태 문자열)."""
    if isinstance(e, Terminated):
        # start_up.sh가 왜 SIGTERM을 보냈는지 표시 파일로 남긴다 (시간 초과 / 진행 없음)
        state_dir = Path(os.environ.get("HARNESS_STATE_DIR") or ".")
        if (state_dir / "watchdog_fired").exists():
            return 143, "timeout"
        if (state_dir / "stall_fired").exists():
            return 143, "stalled"
        return 143, "terminated"
    if isinstance(e, KeyboardInterrupt):
        return 130, "interrupted"
    try:
        import torch

        if isinstance(e, torch.cuda.OutOfMemoryError):
            return 2, "oom"
    except ImportError:
        pass
    net = (ConnectionError, TimeoutError, OSError)
    try:
        import requests

        net = net + (requests.exceptions.RequestException,)
    except ImportError:
        pass
    if isinstance(e, net) and not isinstance(e, (FileNotFoundError, PermissionError)):
        return 3, "network_error"
    return 1, "failed"


if __name__ == "__main__":
    sys.exit(main())
