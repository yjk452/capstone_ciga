import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch


# ----------------------------
# Logging
# ----------------------------
def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ensemble_mlp")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(fmt)

    logger.handlers.clear()
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# ----------------------------
# Checkpoint discovery
# ----------------------------
def _parse_epoch_step(name: str) -> Tuple[int, int]:
    em = re.search(r"epoch=(\d+)", name)
    sm = re.search(r"step=(\d+)", name)
    epoch = int(em.group(1)) if em else -1
    step = int(sm.group(1)) if sm else -1
    return epoch, step


def find_mlp_ckpt(block_dir: Path) -> Path:
    """
    block_dir: blocks/block_1 같은 디렉토리
    반환: checkpoints_mlp/ 안에서 '*-mlp.pt' 중 (epoch, step) 가장 큰 파일
    """
    ckpt_dir = block_dir / "checkpoints_mlp"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"{ckpt_dir} 디렉토리가 없습니다.")

    candidates = list(ckpt_dir.glob("*-mlp.pt"))
    if not candidates:
        raise FileNotFoundError(f"{ckpt_dir} 에 '*-mlp.pt' 파일이 없습니다.")

    candidates.sort(key=lambda p: _parse_epoch_step(p.name))
    return candidates[-1]


def _to_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """
    torch.save가 state_dict만 저장한 경우: dict[str, Tensor]
    혹은 {'state_dict': {...}} 형태일 수도 있어서 방어.
    """
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint format: {type(obj)}")


def load_state_dicts(
    root_dir: Path,
    start_block: int = 0,
    end_block: int = 8,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict[str, torch.Tensor]], List[int], List[Path]]:
    """
    root_dir/blocks/block_{start_block} ~ block_{end_block} 까지의 MLP state_dict를 읽어옴
    """
    state_dicts: List[Dict[str, torch.Tensor]] = []
    used_blocks: List[int] = []
    used_paths: List[Path] = []

    for i in range(start_block, end_block + 1):
        block_dir = root_dir / "blocks" / f"block_{i}"
        if not block_dir.is_dir():
            msg = f"[WARN] {block_dir} 가 존재하지 않아 건너뜀."
            (logger.warning if logger else print)(msg)
            continue

        try:
            ckpt_path = find_mlp_ckpt(block_dir)
        except FileNotFoundError as e:
            (logger.warning if logger else print)(f"[WARN] {e} — block_{i} 건너뜀.")
            continue

        (logger.info if logger else print)(f"[INFO] block_{i} 에서 MLP 로드: {ckpt_path}")
        sd_raw = torch.load(ckpt_path, map_location="cpu")
        sd = _to_state_dict(sd_raw)

        state_dicts.append(sd)
        used_blocks.append(i)
        used_paths.append(ckpt_path)

    if not state_dicts:
        raise RuntimeError("어떤 block에서도 MLP state_dict를 찾지 못했습니다.")

    return state_dicts, used_blocks, used_paths


# ----------------------------
# Ensemble (average)
# ----------------------------
def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    여러 개의 state_dict를 같은 키 기준으로 평균냄.
    - 모든 state_dict는 동일한 key set과 동일한 shape을 가져야 함.
    """
    avg_state: Dict[str, torch.Tensor] = {}
    n = len(state_dicts)
    ref = state_dicts[0]

    # 키 일치 확인 (안전)
    ref_keys = set(ref.keys())
    for idx, sd in enumerate(state_dicts[1:], start=1):
        if set(sd.keys()) != ref_keys:
            missing = ref_keys - set(sd.keys())
            extra = set(sd.keys()) - ref_keys
            raise ValueError(
                f"state_dict keys mismatch at index={idx}. missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}"
            )

    for k, v in ref.items():
        if not torch.is_tensor(v):
            # 텐서가 아니면 그냥 첫 번째 값 유지
            avg_state[k] = v
            continue

        stacked = torch.stack([sd[k].float() for sd in state_dicts], dim=0)  # [N, ...]
        avg_state[k] = stacked.mean(dim=0).type_as(v)

    return avg_state


def save_ensemble(root_dir: Path, state_dict: Dict[str, torch.Tensor], filename: str = "ensemble-mlp.pt") -> Path:
    out_dir = root_dir / "checkpoints_mlp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    torch.save(state_dict, out_path)
    return out_path


# ----------------------------
# TXT dump (log_weight_stats style)
# ----------------------------
def dump_state_dict_stats_like_log_weight_stats(
    state_dict: Dict[str, Any],
    file_path: Path,
    step: int = -1,
    title: str = "State Dict (structure + params)",
    append: bool = False,
):
    """
    ensamble-mlp_dump.txt의 "State Dict (structure + params)" 부분을
    아래 형식으로 기록:
      step: {step}
      {name:20s} shape=(...) min=... max=... mean=... std=...
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    with file_path.open(mode, encoding="utf-8") as f:
        f.write(f"=== {title} ===\n")
        f.write(f"step: {step}\n")
        for name in sorted(state_dict.keys()):
            v = state_dict[name]
            if not torch.is_tensor(v):
                continue

            t = v.detach().cpu()
            if t.numel() == 0:
                f.write(f"{name:20s} shape={tuple(t.shape)} min=nan max=nan mean=nan std=nan\n")
                continue

            f.write(
                f"{name:20s} "
                f"shape={tuple(t.shape)} "
                f"min={t.min().item():.4e} "
                f"max={t.max().item():.4e} "
                f"mean={t.mean().item():.4e} "
                f"std={t.std(unbiased=False).item():.4e}\n"
            )
        f.write("\n\n")


# ----------------------------
# Pipeline
# ----------------------------
def build_and_save_ensemble(
    root_dir_str: str,
    start_block: int = 0,
    end_block: int = 8,
    step: int = -1,
):
    """
    root_dir 아래 blocks/block_{start..end}/checkpoints_mlp/*-mlp.pt 를 찾아
    앙상블(평균) 후:
      - root_dir/checkpoints_mlp/ensemble-mlp.pt 저장
      - root_dir/log/ensemble-mlp_dump.txt 에 stats 저장
      - root_dir/log/ensemble_mlp.log 에 실행 로그 저장
    """
    root_dir = Path(root_dir_str).expanduser().resolve()

    log_dir = root_dir / "log"
    logger = setup_logger(log_dir / "ensemble_mlp.log")
    logger.info(f"root_dir = {root_dir}")
    logger.info(f"block range = [{start_block}, {end_block}]")
    logger.info(f"dump step = {step}")

    # 1) 블록별 state_dict 로드
    state_dicts, used_blocks, used_paths = load_state_dicts(
        root_dir, start_block=start_block, end_block=end_block, logger=logger
    )
    logger.info(f"used_blocks = {used_blocks}")
    for b, p in zip(used_blocks, used_paths):
        logger.info(f"block_{b} ckpt = {p}")

    # 2) 평균 앙상블
    ensemble_sd = average_state_dicts(state_dicts)
    logger.info(f"ensemble keys = {len(ensemble_sd)}")

    # 3) pt 저장
    pt_path = save_ensemble(root_dir, ensemble_sd, filename="ensemble-mlp.pt")
    logger.info(f"Ensemble MLP saved: {pt_path}")

    # 4) txt 덤프는 outputs/$NAME/log 로 저장
    dump_txt_path = log_dir / "ensemble-mlp_dump.txt"
    dump_state_dict_stats_like_log_weight_stats(
        state_dict=ensemble_sd,
        file_path=dump_txt_path,
        step=step,
        title="State Dict (structure + params)",
        append=False,
    )
    logger.info(f"Ensemble dump txt saved: {dump_txt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="blocks/block_* 의 MLP를 앙상블(평균)해서 ensemble-mlp.pt 저장 + stats txt 로그 생성"
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="blocks/ 가 들어있는 상위 경로 (예: /home/.../outputs/03_V2)",
    )
    parser.add_argument(
        "--start_block",
        type=int,
        default=0,
        help="시작 block id (기본 0)",
    )
    parser.add_argument(
        "--end_block",
        type=int,
        default=8,
        help="끝 block id (기본 8)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=-1,
        help="덤프 txt에 기록할 step 값 (기본 -1)",
    )
    args = parser.parse_args()

    build_and_save_ensemble(
        args.root_dir,
        start_block=args.start_block,
        end_block=args.end_block,
        step=args.step,
    )
