"""MLflow sink — telemetry from POC deployments and offline evals, same schema.

Design constraints that come from the research rather than from MLflow:

  CONTROLS ARE NESTED RUNS, not tags. A failed control has to be as visible as a
  metric in the UI, because the failures this catches all produced numbers that
  looked publishable. Burying them in a tag string means nobody reads them.

  THE PRIMARY METRIC IS lift_over_chance, NOT accuracy. Sorting a leaderboard by
  accuracy reordered this project's results wrongly three times and once ranked a
  model that releases 14.88% of live secrets above one that releases none.

  A RUN THAT EMITS ONLY ACCURACY IS REJECTED. The contract is enforced here, at
  the boundary, so it cannot be skipped by a caller in a hurry.

Degrades to JSONL when mlflow is absent, so the harness can be dropped into a POC
environment before anyone has stood up a tracking server.
"""
from __future__ import annotations
import json, os, pathlib, time
from typing import Any

REQUIRED = {"accuracy", "majority_baseline", "lift_over_chance", "minority_share", "n"}


class Sink:
    def __init__(self, experiment: str, tracking_uri: str | None = None,
                 fallback_dir: str = "clfeval-runs"):
        self.experiment = experiment
        self.fallback = pathlib.Path(fallback_dir)
        self._mlflow = None
        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        try:
            import mlflow
            if uri: mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
            self._mlflow = mlflow
        except Exception as e:
            self.fallback.mkdir(parents=True, exist_ok=True)
            print(f"[clfeval] mlflow unavailable ({type(e).__name__}); "
                  f"writing to {self.fallback}/")

    def log_eval(self, *, run_name: str, params: dict[str, Any],
                 metrics: dict[str, float], controls: dict,
                 artifacts: dict[str, Any] | None = None, tags: dict | None = None):
        missing = REQUIRED - set(metrics)
        if missing:
            raise ValueError(
                f"metric contract violated: {sorted(missing)} missing. Accuracy "
                f"without a baseline is not a result — see clfeval.metrics.core()")
        tags = dict(tags or {})
        tags["controls_passed"] = str(controls.get("controls_passed"))
        if controls.get("controls_failed"):
            tags["controls_failed"] = ",".join(controls["controls_failed"])
        tags["primary_metric"] = "lift_over_chance"
        self._write(run_name, params, metrics, controls, artifacts, tags, kind="eval")

    def log_shadow(self, *, run_name: str, params: dict, metrics: dict,
                   drift: dict, tags: dict | None = None):
        """Phase-1 POC telemetry: no labels, no accuracy, no contract.

        Deliberately exempt from the metric contract. Shadow mode's job is to
        answer 'do the published numbers transfer to this traffic?' -- and
        drift/separability answers it without a single label.
        """
        tags = dict(tags or {})
        tags["mode"] = "shadow"
        tags["drift_status"] = str(drift.get("drift/status", "unmeasured"))
        m = {**metrics, **{k: v for k, v in drift.items() if isinstance(v, (int, float))}}
        self._write(run_name, params, m, {}, None, tags, kind="shadow")

    def _write(self, run_name, params, metrics, controls, artifacts, tags, kind):
        if self._mlflow:
            mf = self._mlflow
            with mf.start_run(run_name=run_name):
                mf.set_tags({**tags, "clfeval_kind": kind})
                mf.log_params({k: str(v)[:500] for k, v in params.items()})
                mf.log_metrics({k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float)) and v == v})
                for check in controls.get("checks", []):
                    with mf.start_run(run_name=f"control:{check['control']}", nested=True):
                        mf.set_tags({"passed": str(check["passed"]),
                                     "detail": str(check["detail"])[:500]})
                        mf.log_metric("passed", 1.0 if check["passed"] else 0.0)
                for name, obj in (artifacts or {}).items():
                    p = pathlib.Path(f"/tmp/clfeval-{name}")
                    p.write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=1))
                    mf.log_artifact(str(p))
            return
        rec = {"ts": time.time(), "experiment": self.experiment, "run": run_name,
               "kind": kind, "params": params, "metrics": metrics,
               "controls": controls, "tags": tags}
        with open(self.fallback / f"{self.experiment.replace('/','_')}.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
