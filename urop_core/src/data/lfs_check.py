"""Inspect repository battery dataset files and verify real hardware datasets (NASA + Wang + Mendeley Excel)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config import PipelineConfig

LFS_HEADER = "version https://git-lfs.github.com/spec/v1"


@dataclass
class DatasetInventory:
    li_ion_files: list[Path]
    na_ion_3_2ah_files: list[Path]
    na_ion_10ah_files: list[Path]
    li_ion_pulled: bool
    na_ion_pulled: bool
    total_files: int
    total_declared_bytes: int
    pulled_files: int
    pointer_files: int
    mendeley_xlsx_files: list[Path] = field(default_factory=list)
    mendeley_pulled: bool = False

    @property
    def all_real_available(self) -> bool:
        return self.li_ion_pulled and self.na_ion_pulled

    def to_dict(self) -> dict:
        return {
            "li_ion_file_count": len(self.li_ion_files),
            "na_ion_3_2ah_file_count": len(self.na_ion_3_2ah_files),
            "na_ion_10ah_file_count": len(self.na_ion_10ah_files),
            "mendeley_xlsx_file_count": len(self.mendeley_xlsx_files),
            "li_ion_pulled": self.li_ion_pulled,
            "na_ion_pulled": self.na_ion_pulled,
            "mendeley_pulled": self.mendeley_pulled,
            "all_real_available": self.all_real_available,
            "total_files": self.total_files,
            "pulled_files": self.pulled_files,
            "pointer_files": self.pointer_files,
            "total_declared_mb": round(self.total_declared_bytes / (1024 * 1024), 2),
        }


def is_lfs_pointer(path: Path) -> tuple[bool, int]:
    """Return (is_pointer, declared_size_bytes)."""
    if not path.exists():
        return True, 0
    actual_size = path.stat().st_size
    if actual_size > 1024:
        return False, actual_size
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return True, actual_size
    if text.startswith(LFS_HEADER):
        declared = actual_size
        for line in text.splitlines():
            if line.startswith("size "):
                try:
                    declared = int(line.split()[1])
                except (ValueError, IndexError):
                    pass
        return True, declared
    return False, actual_size


def inspect_datasets(cfg: PipelineConfig) -> DatasetInventory:
    """Scan all expected dataset directories (NASA .zip, Wang Arbin .CSV, Mendeley .xlsx) and report status."""
    li_files = sorted(cfg.nasa_li_dir.glob("*.zip")) if cfg.nasa_li_dir.exists() else []
    na_3_2_files = (
        sorted(p for p in cfg.na_3_2ah_dir.rglob("*") if p.suffix.lower() == ".csv")
        if cfg.na_3_2ah_dir.exists()
        else []
    )
    na_10_files = (
        sorted(p for p in cfg.na_10ah_dir.rglob("*") if p.suffix.lower() == ".csv")
        if cfg.na_10ah_dir.exists()
        else []
    )
    mendeley_files = (
        sorted(p for p in cfg.dataset_root.glob("Dataset*.xlsx"))
        if cfg.dataset_root.exists()
        else []
    )

    all_files = li_files + na_3_2_files + na_10_files + mendeley_files
    pulled = 0
    pointers = 0
    declared_bytes = 0
    li_pulled_count = 0
    na_pulled_count = 0
    mendeley_pulled_count = 0

    for f in all_files:
        ptr, sz = is_lfs_pointer(f)
        declared_bytes += sz
        if ptr:
            pointers += 1
        else:
            pulled += 1
            if f in li_files:
                li_pulled_count += 1
            elif f in mendeley_files:
                mendeley_pulled_count += 1
            else:
                na_pulled_count += 1

    return DatasetInventory(
        li_ion_files=li_files,
        na_ion_3_2ah_files=na_3_2_files,
        na_ion_10ah_files=na_10_files,
        mendeley_xlsx_files=mendeley_files,
        li_ion_pulled=len(li_files) > 0 and li_pulled_count == len(li_files),
        na_ion_pulled=(len(na_3_2_files) + len(na_10_files)) > 0
        and na_pulled_count == (len(na_3_2_files) + len(na_10_files)),
        mendeley_pulled=len(mendeley_files) > 0 and mendeley_pulled_count == len(mendeley_files),
        total_files=len(all_files),
        total_declared_bytes=declared_bytes,
        pulled_files=pulled,
        pointer_files=pointers,
    )
