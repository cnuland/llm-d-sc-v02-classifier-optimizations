"""llm-d-sc gRPC adapter — the RUNTIME plane, and the canonical path.

This is what users actually deploy: the eval harness talks to the llm-d-sc
Classify service, which owns tokenization, the forward pass, batching and
backpressure. Qualifying the checkpoint alone leaves all of that unmeasured.

STATUS: WRITTEN, NOT VALIDATED. The only reachable llm-d-sc deployment was in
CrashLoopBackOff (its modelcar init container could not write to /shared) for the
duration of this work, so this adapter has never exchanged a message with a live
service. It is shipped unvalidated and labelled as such rather than presented
beside the model-plane numbers, which are real.
"""
from __future__ import annotations
import time
import numpy as np
from .base import ClassifierAdapter


class LlmDScAdapter(ClassifierAdapter):
    plane = "runtime"

    def __init__(self, target: str, task, timeout_s: float = 5.0,
                 stub_factory=None):
        self.target, self.task, self.timeout_s = target, task, timeout_s
        self.revision = "unknown"           # populated from the service if exposed
        self.runtime_revision = f"llm-d-sc@{target}"
        self._stub = stub_factory() if stub_factory else self._connect()

    def _connect(self):
        try:
            import grpc  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "grpcio not installed. The llm-d-sc adapter needs the generated "
                "Classify stubs from the llm-d-sc protobufs; see project docs."
            ) from e
        raise NotImplementedError(
            "Generated Classify stubs are not vendored here. Pass `stub_factory` "
            "with a client exposing .Classify(text) -> {label: score}. This path "
            "is UNVALIDATED — no live llm-d-sc service was reachable during "
            "development.")

    def predict_proba(self, texts):
        rows = []
        for t in texts:
            scores = self._stub.Classify(t)
            rows.append([float(scores.get(l, 0.0)) for l in self.task.labels])
        P = np.asarray(rows, dtype=float)
        s = P.sum(1, keepdims=True)
        return np.divide(P, s, out=np.full_like(P, 1/P.shape[1]), where=s > 0)
