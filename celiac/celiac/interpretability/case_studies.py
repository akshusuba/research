"""
Case studies for biological validation of predictions.

Validates model predictions against known biological mechanisms
in the celiac gut-brain axis.
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path

from .path_analysis import find_paths, score_paths, Path as GraphPath


@dataclass
class CaseStudy:
    """A biological case study for validation."""
    name: str
    description: str
    source_type: str
    source_name: str
    target_type: str
    target_name: str
    expected_intermediates: List[str]
    expected_path_length: int
    literature_support: List[str]
    mechanism: str


# Curated case studies for celiac gut-brain axis
CED_CASE_STUDIES = [
    CaseStudy(
        name="TGM6_Ataxia",
        description="Transglutaminase 6 and cerebellar ataxia in celiac disease",
        source_type="gene",
        source_name="TGM6",
        target_type="phenotype",
        target_name="ataxia",
        expected_intermediates=["anti-TG6 antibodies", "cerebellar damage"],
        expected_path_length=2,
        literature_support=[
            "Hadjivassiliou et al. 2008 - Lancet Neurology",
            "Hadjivassiliou et al. 2013 - Movement Disorders",
        ],
        mechanism="Anti-TG6 antibodies cross-react with cerebellar tissue, causing neuronal damage and ataxia",
    ),
    CaseStudy(
        name="Tryptophan_Depression",
        description="Tryptophan metabolism and depression in celiac disease",
        source_type="metabolite",
        source_name="tryptophan",
        target_type="phenotype",
        target_name="depression",
        expected_intermediates=["serotonin", "5-HT depletion"],
        expected_path_length=3,
        literature_support=[
            "Russo et al. 2012 - Neuroimmunomodulation",
            "Lionetti et al. 2010 - Alimentary Pharmacology & Therapeutics",
        ],
        mechanism="Malabsorption and inflammation reduce tryptophan availability, decreasing serotonin synthesis",
    ),
    CaseStudy(
        name="IL15_Enteropathy",
        description="IL-15 driven enteropathy and systemic effects",
        source_type="gene",
        source_name="IL15",
        target_type="phenotype",
        target_name="enteropathy",
        expected_intermediates=["IEL activation", "epithelial damage"],
        expected_path_length=2,
        literature_support=[
            "Jabri & Sollid 2017 - Nature Reviews Immunology",
            "Meresse et al. 2004 - Immunity",
        ],
        mechanism="IL-15 drives cytotoxic T cell activation leading to villous atrophy",
    ),
    CaseStudy(
        name="HLA_DQ2_CeD",
        description="HLA-DQ2/DQ8 and celiac disease susceptibility",
        source_type="gene",
        source_name="HLA-DQA1",
        target_type="phenotype",
        target_name="celiac disease",
        expected_intermediates=["antigen presentation", "T cell response"],
        expected_path_length=2,
        literature_support=[
            "Sollid et al. 1989 - Journal of Experimental Medicine",
            "Lundin et al. 1993 - Journal of Experimental Medicine",
        ],
        mechanism="HLA-DQ2/DQ8 present deamidated gliadin peptides to CD4+ T cells",
    ),
    CaseStudy(
        name="Microbiome_Inflammation",
        description="Gut microbiome dysbiosis and neuroinflammation",
        source_type="microbe",
        source_name="Prevotella",
        target_type="phenotype",
        target_name="cognitive impairment",
        expected_intermediates=["SCFA depletion", "gut barrier", "neuroinflammation"],
        expected_path_length=4,
        literature_support=[
            "Caminero et al. 2019 - Gastroenterology",
            "Dinan & Cryan 2017 - Journal of Physiology",
        ],
        mechanism="Reduced Prevotella leads to SCFA depletion, increased gut permeability, and neuroinflammation",
    ),
    CaseStudy(
        name="Gluten_Neuropathy",
        description="Gluten exposure and peripheral neuropathy",
        source_type="gene",
        source_name="TGM2",
        target_type="phenotype",
        target_name="peripheral neuropathy",
        expected_intermediates=["anti-TG2", "cross-reactivity", "nerve damage"],
        expected_path_length=3,
        literature_support=[
            "Hadjivassiliou et al. 2006 - Neurology",
            "Briani et al. 2008 - Journal of Neurology",
        ],
        mechanism="Anti-TG2 antibodies may cross-react with neuronal tissue or affect nerve blood supply",
    ),
]


def find_node_by_name(
    data: HeteroData,
    node_type: str,
    name_pattern: str,
) -> Optional[int]:
    """
    Find node index by name pattern.

    Args:
        data: HeteroData object
        node_type: Type of node to search
        name_pattern: Name or pattern to match

    Returns:
        Node index or None if not found
    """
    if not hasattr(data[node_type], 'node_names'):
        return None

    names = data[node_type].node_names
    name_pattern_lower = name_pattern.lower()

    # Try exact match first
    for i, name in enumerate(names):
        if name.lower() == name_pattern_lower:
            return i

    # Try partial match
    for i, name in enumerate(names):
        if name_pattern_lower in name.lower():
            return i

    return None


def validate_case_study(
    case_study: CaseStudy,
    data: HeteroData,
    model: Optional[nn.Module] = None,
    max_hops: int = 4,
) -> Dict[str, Any]:
    """
    Validate a case study against the knowledge graph.

    Args:
        case_study: CaseStudy to validate
        data: HeteroData object
        model: Optional trained model for scoring
        max_hops: Maximum path length to search

    Returns:
        Validation results dict
    """
    results = {
        'case_study': case_study.name,
        'source_found': False,
        'target_found': False,
        'paths_found': 0,
        'shortest_path_length': None,
        'paths': [],
        'prediction_score': None,
        'validation_status': 'unknown',
    }

    # Find source node
    source_idx = find_node_by_name(data, case_study.source_type, case_study.source_name)
    if source_idx is not None:
        results['source_found'] = True
        results['source_idx'] = source_idx

    # Find target node
    target_idx = find_node_by_name(data, case_study.target_type, case_study.target_name)
    if target_idx is not None:
        results['target_found'] = True
        results['target_idx'] = target_idx

    # If both found, search for paths
    if results['source_found'] and results['target_found']:
        source = (case_study.source_type, source_idx)
        target = (case_study.target_type, target_idx)

        paths = find_paths(data, source, target, max_length=max_hops, max_paths=50)

        if paths:
            results['paths_found'] = len(paths)
            results['shortest_path_length'] = min(p.length for p in paths)

            # Score paths if model provided
            if model is not None:
                paths = score_paths(paths, model, data)

            # Store top paths
            results['paths'] = [p.to_dict() for p in paths[:5]]

            # Get model prediction score for direct link
            if model is not None:
                try:
                    model.eval()
                    edge_type = (case_study.source_type, 'associated_with', case_study.target_type)
                    if edge_type in data.edge_types:
                        edge_index = torch.tensor([[source_idx], [target_idx]])
                        with torch.no_grad():
                            z_dict = model(data)
                            score = model.decode(z_dict, edge_type, edge_index)
                            results['prediction_score'] = torch.sigmoid(score).item()
                except Exception as e:
                    results['prediction_error'] = str(e)

            # Determine validation status
            if results['shortest_path_length'] is not None:
                if results['shortest_path_length'] <= case_study.expected_path_length:
                    results['validation_status'] = 'validated'
                else:
                    results['validation_status'] = 'partial'
        else:
            results['validation_status'] = 'no_path'
    else:
        results['validation_status'] = 'nodes_not_found'

    return results


def run_all_case_studies(
    data: HeteroData,
    model: Optional[nn.Module] = None,
    case_studies: Optional[List[CaseStudy]] = None,
    max_hops: int = 4,
) -> List[Dict[str, Any]]:
    """
    Run all case studies and return results.

    Args:
        data: HeteroData object
        model: Optional trained model
        case_studies: List of case studies (default: CED_CASE_STUDIES)
        max_hops: Maximum path length

    Returns:
        List of validation results
    """
    if case_studies is None:
        case_studies = CED_CASE_STUDIES

    results = []
    for case_study in case_studies:
        print(f"\nValidating: {case_study.name}")
        print(f"  {case_study.source_type}:{case_study.source_name} → {case_study.target_type}:{case_study.target_name}")

        result = validate_case_study(case_study, data, model, max_hops)
        results.append(result)

        print(f"  Status: {result['validation_status']}")
        if result['paths_found'] > 0:
            print(f"  Paths found: {result['paths_found']}, shortest: {result['shortest_path_length']}")

    return results


def generate_case_study_report(
    results: List[Dict[str, Any]],
    case_studies: Optional[List[CaseStudy]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a report of case study validation results.

    Args:
        results: Results from run_all_case_studies
        case_studies: Original case studies for metadata
        output_path: Optional path to save report

    Returns:
        Report as markdown string
    """
    if case_studies is None:
        case_studies = CED_CASE_STUDIES

    case_study_dict = {cs.name: cs for cs in case_studies}

    lines = [
        "# Case Study Validation Report",
        "",
        "## Summary",
        "",
        f"Total case studies: {len(results)}",
        f"Validated: {sum(1 for r in results if r['validation_status'] == 'validated')}",
        f"Partial: {sum(1 for r in results if r['validation_status'] == 'partial')}",
        f"Not found: {sum(1 for r in results if r['validation_status'] == 'nodes_not_found')}",
        f"No path: {sum(1 for r in results if r['validation_status'] == 'no_path')}",
        "",
        "## Detailed Results",
        "",
    ]

    for result in results:
        cs = case_study_dict.get(result['case_study'])
        if cs is None:
            continue

        status_emoji = {
            'validated': '✓',
            'partial': '~',
            'no_path': '✗',
            'nodes_not_found': '?',
            'unknown': '?',
        }.get(result['validation_status'], '?')

        lines.extend([
            f"### {status_emoji} {cs.name}",
            "",
            f"**Description**: {cs.description}",
            "",
            f"**Path**: {cs.source_name} ({cs.source_type}) → {cs.target_name} ({cs.target_type})",
            "",
            f"**Mechanism**: {cs.mechanism}",
            "",
            f"**Literature**:",
        ])

        for ref in cs.literature_support:
            lines.append(f"- {ref}")

        lines.extend([
            "",
            f"**Validation Status**: {result['validation_status']}",
            "",
        ])

        if result['paths_found'] > 0:
            lines.extend([
                f"**Paths Found**: {result['paths_found']}",
                f"**Shortest Path Length**: {result['shortest_path_length']} (expected: {cs.expected_path_length})",
                "",
            ])

            if result.get('prediction_score') is not None:
                lines.append(f"**Model Prediction Score**: {result['prediction_score']:.4f}")
                lines.append("")

            if result.get('paths'):
                lines.append("**Top Paths**:")
                for i, path in enumerate(result['paths'][:3], 1):
                    lines.append(f"{i}. {path['path_string']} (score: {path['score']:.4f})")
                lines.append("")

        lines.append("---")
        lines.append("")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_path}")

    return report


def export_case_studies_to_json(
    results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Export case study results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results exported to {output_path}")
