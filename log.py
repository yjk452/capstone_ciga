from pathlib import Path
import shutil

def print_to(dir_path, file_name, *args, sep=" ", end="\n"):
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / file_name

    with file_path.open("a", encoding="utf-8") as f:
        print(*args, sep=sep, end=end, file=f)

def nuke_dir(recreate: bool = True):
    d = Path("/home/jinholee/Ciga/logs/")
    if d.exists():
        shutil.rmtree(d)
    if recreate:
        d.mkdir(parents=True, exist_ok=True)        


def log_weight_stats(module, file_name, dir_path, step=-1):
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / file_name

    with file_path.open("a", encoding="utf-8") as f:
        f.write(f"step: {step}\n")
        for name, p in module.named_parameters():
            t = p.detach()
            f.write(
                f"{name:20s} "
                f"shape={tuple(t.shape)} "
                f"min={t.min().item():.4e} "
                f"max={t.max().item():.4e} "
                f"mean={t.mean().item():.4e} "
                f"std={t.std(unbiased=False).item():.4e}\n"
            )
        f.write("\n\n")