"""SpatialGNN: when the neighborhood is the label.

This package tests a sharp hypothesis: a graph neural network beats strong
non-graph baselines (MLP, XGBoost) precisely when the prediction target is
*defined by spatial neighborhood structure* and is not recoverable from a
node's own features. Tissue-domain / niche identification in spatial omics is
such a task -- a cell's domain is a property of where it sits, not just what it
expresses. We provide a controlled synthetic benchmark that proves the
mechanism and a real spatial-omics path that tests it honestly, both evaluated
under leakage-safe spatial splits where absolute position cannot be memorized.
"""

__version__ = "0.1.0"
