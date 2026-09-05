"""clfeval — a generic classifier evaluation harness for semantic routing.

Domain-agnostic. Built-in domains: complexity, sensitivity, cost. Any other
taxonomy is a YAML file (see domains/_template.yaml).

The design premise: accuracy is four lines of code, so computing it is not the
job. The job is making the specific ways an evaluation lies impossible to skip.
"""
from .spec import DomainSpec, Dataset
from .runner import evaluate, HFAdapter, load_rows
from .sink import Sink
from .shadow import ShadowRun, ShadowConfig
from . import metrics, controls
__all__ = ["DomainSpec","Dataset","evaluate","HFAdapter","load_rows","Sink",
           "ShadowRun","ShadowConfig","metrics","controls"]
