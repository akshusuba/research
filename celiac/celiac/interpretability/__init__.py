"""
Interpretability module for celiac gut-brain GNN.

Provides:
- Attention weight visualization
- Multi-hop path analysis
- Case studies for biological validation
"""

from .attention_viz import (
    extract_attention_weights,
    visualize_attention_heatmap,
    get_top_attention_edges,
)
from .path_analysis import (
    find_paths,
    score_paths,
    extract_top_paths,
    visualize_path,
)
from .case_studies import (
    CED_CASE_STUDIES,
    validate_case_study,
    run_all_case_studies,
    generate_case_study_report,
)

__all__ = [
    # Attention
    'extract_attention_weights',
    'visualize_attention_heatmap',
    'get_top_attention_edges',
    # Path analysis
    'find_paths',
    'score_paths',
    'extract_top_paths',
    'visualize_path',
    # Case studies
    'CED_CASE_STUDIES',
    'validate_case_study',
    'run_all_case_studies',
    'generate_case_study_report',
]
