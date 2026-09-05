"""Domain specification — the declarative description of ONE classification task.

The eval logic in this package is entirely domain-agnostic. Everything that
differs between complexity, sensitivity, cost and a team's own custom taxonomy is
data, and lives in a YAML file. That separation is the point: the controls in
`controls.py` caught seven wrong conclusions on this project, and they should be
runnable against a taxonomy their author never saw.

A spec answers five questions:
  1. what are the labels, and are they ORDERED?      (gates and folds need this)
  2. which datasets evaluate it, and how were they built?
  3. which gating thresholds does the deployment use, if any?
  4. what candidate folds should be considered?
  5. what is this signal's measured noise floor?     (so "a gain" has a meaning)
"""
from __future__ import annotations
import json, pathlib, dataclasses, hashlib
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


@dataclasses.dataclass
class Dataset:
    id: str
    path: str
    role: str = "eval"                  # eval | contested | secondary
    provenance: str = "unspecified"     # real-traffic | synthetic | mixed
    separability_from_production: float | None = None   # see §77; None = unmeasured
    jury_models: int | None = None
    notes: str = ""


@dataclasses.dataclass
class DomainSpec:
    name: str
    labels: list[str]
    ordered: bool = False
    gates: list[str] = dataclasses.field(default_factory=list)
    datasets: list[Dataset] = dataclasses.field(default_factory=list)
    folds: dict[str, dict[str, str]] = dataclasses.field(default_factory=dict)
    noise_floor: float | None = None
    text_field: str = "text"
    label_field: str = "tier"
    votes_field: str = "votes"
    description: str = ""

    def __post_init__(self):
        bad = [g for g in self.gates if g not in self.labels]
        if bad:
            raise ValueError(f"{self.name}: gate(s) {bad} not in labels")
        if self.gates and not self.ordered:
            raise ValueError(
                f"{self.name}: gates require ordered=true — a gate is 'at or above "
                f"tier X', which is meaningless without an order")
        for fname, table in self.folds.items():
            missing = set(self.labels) - set(table)
            if missing:
                raise ValueError(f"{self.name}: fold '{fname}' does not map {sorted(missing)}")

    @property
    def rank(self) -> dict[str, int]:
        return {l: i for i, l in enumerate(self.labels)}

    def fold_id(self, name: str) -> str:
        """Content hash of a fold's label map.

        Two folds with different names and identical maps are the same experiment.
        This project shipped `route` and `cx2` as separate models before an
        ensemble diagnostic revealed Jaccard 1.000 error overlap between them.
        """
        table = self.folds[name]
        return hashlib.blake2b(
            json.dumps([[k, table[k]] for k in self.labels]).encode(), digest_size=6
        ).hexdigest()

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "DomainSpec":
        p = pathlib.Path(path)
        raw = p.read_text()
        if p.suffix in (".yaml", ".yml"):
            if yaml is None:
                raise RuntimeError("pyyaml not installed; use a .json spec instead")
            d: dict[str, Any] = yaml.safe_load(raw)
        else:
            d = json.loads(raw)
        ds = [Dataset(**x) for x in d.pop("datasets", [])]
        return cls(datasets=ds, **d)

    @classmethod
    def builtin(cls, name: str) -> "DomainSpec":
        here = pathlib.Path(__file__).parent / "domains"
        for ext in (".yaml", ".json"):
            p = here / f"{name}{ext}"
            if p.exists():
                return cls.load(p)
        avail = sorted(x.stem for x in here.glob("*.*") if not x.stem.startswith("_"))
        raise FileNotFoundError(f"no builtin domain '{name}'; available: {avail}")
