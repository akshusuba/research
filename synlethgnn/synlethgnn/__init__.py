"""SynLethGNN: making the GNN earn its keep on synthetic lethality prediction.

The central thesis of this package is that synthetic lethality (SL) is a
*topological* property of the gene interaction network (parallel-pathway
redundancy) rather than a property of any single gene's intrinsic features.
A graph neural network that propagates over the interaction graph can
therefore represent the SL mechanism directly, while a feature-only MLP
cannot. This package provides the data, models, splits, and ablations needed
to test that claim rigorously and to show *where* and *why* a GNN beats
non-graph baselines.
"""

__version__ = "0.1.0"
