"""spatial_prognosis: does tumor spatial *arrangement* predict outcome beyond *composition*?

The research question is biological, not methodological. The tumor
microenvironment's prognostic value is widely attributed to how cells are
spatially organized (e.g., immune infiltration vs. exclusion), not merely which
cells are present. We test that claim head-on: a spatial graph neural network
(which reads cell arrangement) is compared against composition-only baselines
(XGBoost/MLP on cell-type proportions, which are blind to arrangement). If the
GNN wins, spatial structure carries prognostic signal beyond composition; a
graph-shuffle ablation confirms the signal is the arrangement itself.
"""

__version__ = "0.1.0"
