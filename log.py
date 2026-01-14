# log.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, Iterable
import threading
import shutil


# -------------------------
# Global states (module-level)
# -------------------------
_LOG_DIR: Optional[Path] = None
_ENABLED: bool = True
_LOCK = threading.Lock()


def set_dir(dir_path: Any) -> Path:
    """
    Set a global logging directory.
    After calling this once, print_to/log_weight_stats/step_log will write under this directory.
    """
    global _LOG_DIR
    p = Path(dir_path).expanduser()
    try:
        p = p.resolve()
    except Exception:
        p = Path(str(p))

    with _LOCK:
        p.mkdir(parents=True, exist_ok=True)
        _LOG_DIR = p
    return p


def set_flag(enabled: bool) -> None:
    """
    Enable/disable logging globally.
    If disabled, print_to/log_weight_stats/step_log become no-ops.
    """
    global _ENABLED
    with _LOCK:
        _ENABLED = bool(enabled)


def _is_enabled() -> bool:
    with _LOCK:
        return _ENABLED


def _get_dir(dir_path: Optional[Any] = None) -> Optional[Path]:
    """
    Internal: get the directory to write logs.
    If dir_path is explicitly provided, it is used.
    Otherwise use the global directory set by set_dir().
    """
    global _LOG_DIR
    if dir_path is not None:
        p = Path(dir_path).expanduser()
        try:
            p = p.resolve()
        except Exception:
            p = Path(str(p))
        p.mkdir(parents=True, exist_ok=True)
        return p

    return _LOG_DIR


def print_to(*args, sep: str = " ", end: str = "\n") -> None:
    """
    Write logs to a file under the global log directory.

    Supported calls:
      1) print_to(file_name, *values)
      2) (legacy) print_to(dir_path, file_name, *values)

    If logging is disabled or directory is not set, it becomes a no-op.
    """
    if not _is_enabled():
        return
    if len(args) < 2:
        return

    dir_path = None
    file_name = None
    values = ()

    # legacy signature heuristic
    if len(args) >= 3:
        a0, a1 = args[0], args[1]
        if isinstance(a0, (str, Path)) and isinstance(a1, (str, Path)):
            dir_path = a0
            file_name = str(a1)
            values = args[2:]
        else:
            file_name = str(args[0])
            values = args[1:]
    else:
        file_name = str(args[0])
        values = args[1:]

    target_dir = _get_dir(dir_path)
    if target_dir is None:
        return

    file_path = target_dir / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("a", encoding="utf-8") as f:
        print(*values, sep=sep, end=end, file=f)


def log_weight_stats(module, *args) -> None:
    """
    Log parameter statistics of a torch.nn.Module to a file.

    Supported calls:
      1) log_weight_stats(module, file_name, step=...)
      2) (legacy) log_weight_stats(module, file_name, dir_path, step=...)

    If logging is disabled or directory is not set, it becomes a no-op.
    """
    if not _is_enabled():
        return
    if module is None:
        return
    if len(args) < 1:
        return

    file_name = str(args[0])
    dir_path = args[1] if len(args) >= 2 else None

    target_dir = _get_dir(dir_path)
    if target_dir is None:
        return

    file_path = target_dir / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("a", encoding="utf-8") as f:
        for name, p in module.named_parameters():
            t = p.detach()
            if t.numel() == 0:
                f.write(f"{name:30s} shape={tuple(t.shape)} <empty>\n")
                continue
            f.write(
                f"{name:30s} "
                f"shape={tuple(t.shape)} "
                f"min={t.min().item():.4e} "
                f"max={t.max().item():.4e} "
                f"mean={t.mean().item():.4e} "
                f"std={t.std(unbiased=False).item():.4e}\n"
            )
        f.write("\n")


def step_log(step: int, prefix: str = "step") -> None:
    """
    Append a single line like: "<prefix>: <step>" to ALL *.txt files under the global log directory.

    - Only works when logging is enabled and set_dir() has been called.
    - Intended to be called from gaussian_splatting.py each step (or at selected steps).
    """
    if not _is_enabled():
        return

    target_dir = _get_dir(None)
    if target_dir is None:
        return

    line = f"=================================================\n{prefix}: {int(step)}\n"

    # 폴더 내 모든 txt에 추가 (하위 폴더까지 포함하려면 rglob 사용)
    for fp in target_dir.rglob("*.txt"):
        # 디렉토리는 제외
        if not fp.is_file():
            continue
        try:
            with fp.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            # 로깅은 학습을 깨면 안 되므로, 실패 시 조용히 스킵
            continue

