"""Central configuration and electrochemical specifications for Li-ion and Na-ion datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ChemistrySpec:
    """Electrochemical operating envelope and nominal ratings for a battery chemistry."""

    name: str
    v_min: float  # Lower cut-off voltage (V)
    v_max: float  # Upper cut-off voltage (V)
    v_nom: float  # Nominal voltage (V)
    default_q_nom: float  # Default nominal capacity (Ah)
    r_int_bol: float  # Typical beginning-of-life internal resistance (Ohm)
    t_min_safe: float  # Min safe charging temp (C)
    t_max_safe: float  # Max safe charging temp (C)
    max_c_rate_charge: float  # Max continuous charge C-rate
    coulombic_eff: float  # Nominal Coulombic efficiency
    eol_soh: float = 0.80  # End-of-life SoH threshold


CHEMISTRY_SPECS: dict[str, ChemistrySpec] = {
    # NASA PCoE 18650 Li-ion Battery Aging Dataset (B0005 - B0056)
    "li_ion": ChemistrySpec(
        name="li_ion",
        v_min=2.50,
        v_max=4.20,
        v_nom=3.70,
        default_q_nom=2.00,
        r_int_bol=0.085,
        t_min_safe=0.0,
        t_max_safe=45.0,
        max_c_rate_charge=1.50,
        coulombic_eff=0.998,
    ),
    # Wang et al. (2024) Zenodo 13836819: Transimage 3.2 Ah & HiNa 10 Ah Sodium-ion cells
    "na_ion": ChemistrySpec(
        name="na_ion",
        v_min=1.50,
        v_max=3.90,
        v_nom=3.00,
        default_q_nom=3.20,
        r_int_bol=0.045,
        t_min_safe=-5.0,
        t_max_safe=45.0,
        max_c_rate_charge=1.20,
        coulombic_eff=0.992,
    ),
}


@dataclass
class PipelineConfig:
    """Paths, seeds, Big Data lakehouse settings, and experiment hyperparameters."""

    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    seed: int = 42

    # Raw dataset directories in repository
    nasa_li_dir_name: str = "5. Battery Data Set"
    na_3_2ah_dir_name: str = "3.2Ah na-ion"
    na_10ah_dir_name: str = "10Ah na-ion"

    # Big Data Lakehouse & Outputs
    lake_dir_name: str = "data_lake"
    output_dir_name: str = "outputs"

    # Downsampling stride for real high-frequency CSVs (1-10 Hz -> manageable Big Data batches)
    real_csv_downsample: int = 20
    max_rows_per_chemistry_soc: int = 18000

    # Sequence model settings (LSTM / GRU)
    seq_len: int = 12
    seq_epochs: int = 18
    seq_batch_size: int = 128
    seq_hidden_dim: int = 48
    seq_lr: float = 2e-3

    # Transfer learning budgets (fraction of target training samples)
    tl_fractions: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)

    # Fast mode flag for tests
    fast_mode: bool = False

    @property
    def dataset_root(self) -> Path:
        # Look for raw LFS dataset folders in repo_root first, then parent (when housed under urop_core/)
        if (self.repo_root / self.nasa_li_dir_name).exists():
            return self.repo_root
        if (self.repo_root.parent / self.nasa_li_dir_name).exists():
            return self.repo_root.parent
        return self.repo_root

    @property
    def nasa_li_dir(self) -> Path:
        return self.dataset_root / self.nasa_li_dir_name

    @property
    def na_3_2ah_dir(self) -> Path:
        return self.dataset_root / self.na_3_2ah_dir_name

    @property
    def na_10ah_dir(self) -> Path:
        return self.dataset_root / self.na_10ah_dir_name

    @property
    def lake_dir(self) -> Path:
        return self.repo_root / self.lake_dir_name

    @property
    def output_dir(self) -> Path:
        return self.repo_root / self.output_dir_name

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    def ensure_dirs(self) -> None:
        for d in (
            self.lake_dir / "bronze",
            self.lake_dir / "silver",
            self.lake_dir / "gold",
            self.output_dir,
            self.tables_dir,
            self.figures_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
