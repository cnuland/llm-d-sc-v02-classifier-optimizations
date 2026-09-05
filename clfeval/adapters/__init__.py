from .base import ClassifierAdapter
from .huggingface import HuggingFaceAdapter
from .llmdsc import LlmDScAdapter
from .callable_ import CallableAdapter, majority_class_baseline
__all__ = ["ClassifierAdapter","HuggingFaceAdapter","LlmDScAdapter",
           "CallableAdapter","majority_class_baseline"]
