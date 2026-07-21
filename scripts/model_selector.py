"""
Week 8 — Model Selector
========================
Picks the best model for a given deployment constraint, from a registry
of model metadata (accuracy, size on disk, inference latency). This is
the piece that turns "we trained 5 models" into an actual deployment
decision: cloud servers care about accuracy, edge devices care about
footprint, real-time control loops care about latency.

Usage
-----
    from model_selector import ModelRegistry

    registry = ModelRegistry()
    registry.register("xgboost", rmse=13.99, size_kb=420, latency_ms=1.2, cmapss_score=310)
    registry.register("lstm",    rmse=12.80, size_kb=1850, latency_ms=8.4, cmapss_score=267)
    registry.register("cnn",     rmse=18.06, size_kb=980, latency_ms=4.1, cmapss_score=649)
    registry.register("edge_mlp",rmse=16.20, size_kb=45,  latency_ms=0.3, cmapss_score=410)

    best_for_cloud   = registry.select(constraint="cloud")     # accuracy-optimal
    best_for_edge    = registry.select(constraint="edge")      # size-optimal
    best_for_realtime= registry.select(constraint="realtime")  # latency-optimal
    print(registry.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ModelMeta:
    name: str
    rmse: float
    size_kb: float
    latency_ms: float
    cmapss_score: float | None = None
    extra: dict = field(default_factory=dict)


# Constraint -> which metric to optimize, and in which direction.
# All three metrics here are "lower is better", so direction is fixed,
# but kept explicit in case you add a metric where higher is better later.
CONSTRAINT_METRIC = {
    "cloud": "rmse",  # cloud servers: accuracy is king, size/latency are cheap
    "edge": "size_kb",  # constrained hardware: footprint matters most
    "realtime": "latency_ms",  # control-loop / streaming: speed matters most
}


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, ModelMeta] = {}

    def register(
        self,
        name: str,
        rmse: float,
        size_kb: float,
        latency_ms: float,
        cmapss_score: float | None = None,
        **extra,
    ) -> None:
        self._models[name] = ModelMeta(
            name=name,
            rmse=rmse,
            size_kb=size_kb,
            latency_ms=latency_ms,
            cmapss_score=cmapss_score,
            extra=extra,
        )

    def select(self, constraint: str) -> ModelMeta:
        """
        Returns the ModelMeta that is optimal for the given constraint.
        constraint: "cloud" | "edge" | "realtime"
        """
        if constraint not in CONSTRAINT_METRIC:
            raise ValueError(
                f"Unknown constraint '{constraint}'. Must be one of {list(CONSTRAINT_METRIC)}"
            )
        if not self._models:
            raise ValueError("No models registered yet — call .register() first.")

        metric = CONSTRAINT_METRIC[constraint]
        best = min(self._models.values(), key=lambda m: getattr(m, metric))
        return best

    def select_weighted(
        self, weights: dict[str, float], normalize: bool = True
    ) -> tuple[str, pd.DataFrame]:
        """
        Multi-objective selection: combine rmse, size_kb, latency_ms into
        one weighted score (each metric min-max normalized to [0, 1] first
        so they're comparable, then lower weighted sum wins).

        weights: e.g. {"rmse": 0.6, "size_kb": 0.2, "latency_ms": 0.2}
        Use this when a real deployment isn't purely edge/cloud/realtime
        but a mix — e.g. "mostly accuracy, but size matters somewhat."
        """
        df = self.summary()
        metrics = list(weights.keys())
        norm = df.copy()
        for m in metrics:
            lo, hi = df[m].min(), df[m].max()
            norm[m + "_norm"] = 0.0 if hi == lo else (df[m] - lo) / (hi - lo)

        norm["weighted_score"] = sum(
            norm[m + "_norm"] * w for m, w in weights.items()
        )
        norm = norm.sort_values("weighted_score")
        best_name = norm.iloc[0]["name"]
        return best_name, norm

    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "name": m.name,
                "rmse": m.rmse,
                "size_kb": m.size_kb,
                "latency_ms": m.latency_ms,
                "cmapss_score": m.cmapss_score,
            }
            for m in self._models.values()
        ]
        return pd.DataFrame(rows)


if __name__ == "__main__":
    registry = ModelRegistry()
    registry.register("xgboost", rmse=13.99, size_kb=420, latency_ms=1.2, cmapss_score=310)
    registry.register("lstm", rmse=12.80, size_kb=1850, latency_ms=8.4, cmapss_score=267)
    registry.register("cnn", rmse=18.06, size_kb=980, latency_ms=4.1, cmapss_score=649)
    registry.register("edge_mlp", rmse=16.20, size_kb=45, latency_ms=0.3, cmapss_score=410)

    print("Full registry:")
    print(registry.summary().to_string(index=False))

    for constraint in ["cloud", "edge", "realtime"]:
        best = registry.select(constraint)
        print(f"\nBest for '{constraint}': {best.name} "
              f"(rmse={best.rmse}, size_kb={best.size_kb}, latency_ms={best.latency_ms})")

    print("\nWeighted multi-objective (60% accuracy, 20% size, 20% latency):")
    best_name, ranked = registry.select_weighted(
        {"rmse": 0.6, "size_kb": 0.2, "latency_ms": 0.2}
    )
    print(f"Winner: {best_name}")
    print(ranked[["name", "weighted_score"]].to_string(index=False))
