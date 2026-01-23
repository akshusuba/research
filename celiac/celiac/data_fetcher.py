"""
Fetch data from external sources: GEO, Monarch, etc.
Uses standard library only for compatibility.
"""

import urllib.request
import urllib.parse
import json
import gzip
import ssl
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv

# SSL context for compatibility
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE


def fetch_url(url: str, timeout: int = 60) -> Optional[str]:
    """Fetch URL content with error handling."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Celiac-KG-Research)"}
        )
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as response:
            content = response.read()
            # Try to decode, handle gzip
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
                return gzip.decompress(content).decode('utf-8')
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return None


def fetch_json(url: str, timeout: int = 60) -> Optional[Dict]:
    """Fetch JSON from URL."""
    content = fetch_url(url, timeout)
    if content:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
    return None


# ============================================================================
# GEO Data Fetching
# ============================================================================

def fetch_geo_series_matrix(geo_id: str, output_dir: Path) -> Optional[Path]:
    """
    Download GEO series matrix file.
    Returns path to downloaded file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{geo_id}_series_matrix.txt"

    if output_file.exists():
        print(f"  Series matrix already exists: {output_file}")
        return output_file

    # Try to download series matrix
    base_url = "https://ftp.ncbi.nlm.nih.gov/geo/series"
    series_prefix = geo_id[:6] + "nnn"  # e.g., GSE164nnn
    matrix_url = f"{base_url}/{series_prefix}/{geo_id}/matrix/{geo_id}_series_matrix.txt.gz"

    print(f"  Downloading series matrix from: {matrix_url}")

    try:
        req = urllib.request.Request(matrix_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120, context=SSL_CONTEXT) as response:
            compressed = response.read()
            content = gzip.decompress(compressed).decode('utf-8')

        with open(output_file, 'w') as f:
            f.write(content)

        print(f"  Saved to: {output_file}")
        return output_file

    except Exception as e:
        print(f"  Error downloading series matrix: {e}")
        return None


def parse_geo_series_matrix(matrix_file: Path) -> Tuple[Dict, List[Dict], List[List[float]]]:
    """
    Parse GEO series matrix file.
    Returns: (series_info, samples, expression_data)
    """
    series_info = {}
    samples = []
    expression_data = []
    gene_ids = []

    in_data = False
    sample_ids = []

    with open(matrix_file, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith("!Series_"):
                key = line.split("\t")[0].replace("!Series_", "")
                value = "\t".join(line.split("\t")[1:]).strip('"')
                series_info[key] = value

            elif line.startswith("!Sample_"):
                key = line.split("\t")[0].replace("!Sample_", "")
                values = [v.strip('"') for v in line.split("\t")[1:]]

                # Initialize samples if needed
                if not samples:
                    samples = [{} for _ in values]

                for i, v in enumerate(values):
                    if key not in samples[i]:
                        samples[i][key] = v
                    else:
                        # Append if key exists (e.g., characteristics)
                        samples[i][key] = samples[i][key] + "; " + v

            elif line.startswith("\"ID_REF\""):
                in_data = True
                sample_ids = [s.strip('"') for s in line.split("\t")[1:]]
                continue

            elif in_data and line and not line.startswith("!"):
                parts = line.split("\t")
                if len(parts) > 1:
                    gene_id = parts[0].strip('"')
                    try:
                        values = [float(v) if v else 0.0 for v in parts[1:]]
                        gene_ids.append(gene_id)
                        expression_data.append(values)
                    except ValueError:
                        continue

    return series_info, samples, {"gene_ids": gene_ids, "expression": expression_data, "sample_ids": sample_ids}


# ============================================================================
# Monarch Initiative API
# ============================================================================

def fetch_monarch_gene_phenotypes(gene_symbol: str) -> Tuple[Optional[str], List[Dict]]:
    """
    Query Monarch for gene-phenotype associations.
    Returns: (gene_id, list of phenotypes)
    """
    # Search for gene
    encoded_gene = urllib.parse.quote(gene_symbol)
    search_url = f"https://api.monarchinitiative.org/v3/api/search?q={encoded_gene}&category=biolink:Gene&limit=5"

    data = fetch_json(search_url)
    if not data:
        return None, []

    # Find human gene
    gene_id = None
    for item in data.get("items", []):
        item_id = item.get("id", "")
        if "HGNC" in item_id or "NCBIGene" in item_id:
            # Prefer HGNC
            if "HGNC" in item_id:
                gene_id = item_id
                break
            elif gene_id is None:
                gene_id = item_id

    if not gene_id:
        return None, []

    # Get associations using correct endpoint
    encoded_id = urllib.parse.quote(gene_id)
    assoc_url = f"https://api.monarchinitiative.org/v3/api/association/all?subject={encoded_id}&category=biolink:GeneToPhenotypicFeatureAssociation&limit=200"

    assoc_data = fetch_json(assoc_url)
    if not assoc_data:
        return gene_id, []

    phenotypes = []
    seen = set()
    for assoc in assoc_data.get("items", []):
        obj_id = assoc.get("object", "")
        if obj_id.startswith("HP:") and obj_id not in seen:
            seen.add(obj_id)
            phenotypes.append({
                "id": obj_id,
                "label": assoc.get("object_label", ""),
                "source": assoc.get("primary_knowledge_source", ""),
            })

    return gene_id, phenotypes


def fetch_monarch_phenotype_genes(hpo_id: str) -> List[Dict]:
    """
    Get genes associated with an HPO phenotype.
    """
    encoded_id = urllib.parse.quote(hpo_id)
    assoc_url = f"https://api.monarchinitiative.org/v3/api/association/all?object={encoded_id}&category=biolink:GeneToPhenotypicFeatureAssociation&limit=500"

    data = fetch_json(assoc_url)
    if not data:
        return []

    genes = []
    seen = set()
    for assoc in data.get("items", []):
        subj_id = assoc.get("subject", "")
        if subj_id not in seen:
            seen.add(subj_id)
            genes.append({
                "id": subj_id,
                "label": assoc.get("subject_label", ""),
                "source": assoc.get("primary_knowledge_source", ""),
            })

    return genes


def fetch_all_gene_phenotype_edges(gene_symbols: List[str], output_file: Path) -> List[Dict]:
    """
    Fetch gene-phenotype edges for all given genes.
    Saves to CSV and returns list of edges.
    """
    edges = []

    print(f"  Fetching gene-phenotype edges for {len(gene_symbols)} genes...")

    for i, gene in enumerate(gene_symbols):
        gene_id, phenotypes = fetch_monarch_gene_phenotypes(gene)

        if gene_id and phenotypes:
            for pheno in phenotypes:
                edges.append({
                    "source": gene,
                    "source_id": gene_id,
                    "relation": "associated_with",
                    "target": pheno["id"],
                    "target_label": pheno["label"],
                    "evidence": "monarch",
                })
            print(f"    [{i+1}/{len(gene_symbols)}] {gene}: {len(phenotypes)} phenotypes")
        else:
            print(f"    [{i+1}/{len(gene_symbols)}] {gene}: no phenotypes found")

        time.sleep(0.3)  # Rate limiting

    # Save to CSV
    if edges:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=edges[0].keys())
            writer.writeheader()
            writer.writerows(edges)
        print(f"  Saved {len(edges)} edges to {output_file}")

    return edges


# ============================================================================
# Microbe-Metabolite edges (curated from literature)
# ============================================================================

# Curated microbe-metabolite relationships from literature
# Sources: gutMGene, VMH, primary literature
MICROBE_METABOLITE_EDGES = [
    # Tryptophan metabolism
    {"microbe": "Bifidobacterium", "metabolite": "tryptophan", "relation": "metabolizes", "evidence": "high", "source": "Gao 2018"},
    {"microbe": "Bifidobacterium", "metabolite": "indole", "relation": "produces", "evidence": "medium", "source": "Roager 2018"},
    {"microbe": "Lactobacillus", "metabolite": "tryptophan", "relation": "metabolizes", "evidence": "high", "source": "Gao 2018"},
    {"microbe": "Lactobacillus", "metabolite": "serotonin", "relation": "modulates", "evidence": "medium", "source": "Yano 2015"},
    {"microbe": "Escherichia", "metabolite": "indole", "relation": "produces", "evidence": "high", "source": "Lee 2015"},
    {"microbe": "Escherichia", "metabolite": "tryptophan", "relation": "metabolizes", "evidence": "high", "source": "Agus 2018"},
    {"microbe": "Bacteroides", "metabolite": "indole-3-propionic_acid", "relation": "produces", "evidence": "medium", "source": "Dodd 2017"},

    # Kynurenine pathway (inflammation-driven)
    {"microbe": "Escherichia", "metabolite": "kynurenine", "relation": "promotes", "evidence": "medium", "source": "Kennedy 2017"},
    {"microbe": "Klebsiella", "metabolite": "kynurenine", "relation": "promotes", "evidence": "low", "source": "inferred"},

    # SCFA production
    {"microbe": "Faecalibacterium", "metabolite": "butyrate", "relation": "produces", "evidence": "high", "source": "Louis 2014"},
    {"microbe": "Blautia", "metabolite": "acetate", "relation": "produces", "evidence": "high", "source": "Louis 2014"},
    {"microbe": "Blautia", "metabolite": "butyrate", "relation": "produces", "evidence": "medium", "source": "Louis 2014"},
    {"microbe": "Dorea", "metabolite": "acetate", "relation": "produces", "evidence": "medium", "source": "Louis 2014"},
    {"microbe": "Bacteroides", "metabolite": "propionate", "relation": "produces", "evidence": "high", "source": "Louis 2014"},
    {"microbe": "Bacteroides", "metabolite": "acetate", "relation": "produces", "evidence": "high", "source": "Louis 2014"},
    {"microbe": "Prevotella", "metabolite": "propionate", "relation": "produces", "evidence": "medium", "source": "Chen 2017"},

    # GABA production
    {"microbe": "Lactobacillus", "metabolite": "GABA", "relation": "produces", "evidence": "high", "source": "Barrett 2012"},
    {"microbe": "Bifidobacterium", "metabolite": "GABA", "relation": "produces", "evidence": "high", "source": "Barrett 2012"},

    # LPS (pro-inflammatory)
    {"microbe": "Escherichia", "metabolite": "LPS", "relation": "produces", "evidence": "high", "source": "textbook"},
    {"microbe": "Klebsiella", "metabolite": "LPS", "relation": "produces", "evidence": "high", "source": "textbook"},
    {"microbe": "Pseudomonas", "metabolite": "LPS", "relation": "produces", "evidence": "high", "source": "textbook"},
]


# Curated metabolite-gene relationships
METABOLITE_GENE_EDGES = [
    # Tryptophan/Serotonin pathway
    {"metabolite": "tryptophan", "gene": "TPH1", "relation": "substrate_of", "evidence": "high", "source": "KEGG"},
    {"metabolite": "tryptophan", "gene": "TPH2", "relation": "substrate_of", "evidence": "high", "source": "KEGG"},
    {"metabolite": "tryptophan", "gene": "IDO1", "relation": "substrate_of", "evidence": "high", "source": "KEGG"},
    {"metabolite": "serotonin", "gene": "SLC6A4", "relation": "transported_by", "evidence": "high", "source": "KEGG"},
    {"metabolite": "serotonin", "gene": "HTR1A", "relation": "activates", "evidence": "high", "source": "KEGG"},
    {"metabolite": "serotonin", "gene": "HTR2A", "relation": "activates", "evidence": "high", "source": "KEGG"},

    # Kynurenine pathway
    {"metabolite": "kynurenine", "gene": "IDO1", "relation": "product_of", "evidence": "high", "source": "KEGG"},
    {"metabolite": "kynurenic_acid", "gene": "IDO1", "relation": "downstream_of", "evidence": "medium", "source": "KEGG"},
    {"metabolite": "quinolinic_acid", "gene": "IDO1", "relation": "downstream_of", "evidence": "medium", "source": "KEGG"},

    # SCFA receptors
    {"metabolite": "butyrate", "gene": "FFAR2", "relation": "activates", "evidence": "high", "source": "Brown 2003"},
    {"metabolite": "butyrate", "gene": "FFAR3", "relation": "activates", "evidence": "high", "source": "Brown 2003"},
    {"metabolite": "propionate", "gene": "FFAR2", "relation": "activates", "evidence": "high", "source": "Brown 2003"},
    {"metabolite": "propionate", "gene": "FFAR3", "relation": "activates", "evidence": "high", "source": "Brown 2003"},
    {"metabolite": "acetate", "gene": "FFAR2", "relation": "activates", "evidence": "high", "source": "Brown 2003"},

    # Inflammatory signaling
    {"metabolite": "LPS", "gene": "TLR4", "relation": "activates", "evidence": "high", "source": "textbook"},
    {"metabolite": "LPS", "gene": "NFKB1", "relation": "activates", "evidence": "high", "source": "textbook"},
    {"metabolite": "LPS", "gene": "TNF", "relation": "induces", "evidence": "high", "source": "textbook"},
    {"metabolite": "LPS", "gene": "IL6", "relation": "induces", "evidence": "high", "source": "textbook"},

    # GABA signaling (simplified)
    {"metabolite": "GABA", "gene": "GABRA1", "relation": "activates", "evidence": "high", "source": "textbook"},

    # Indole derivatives
    {"metabolite": "indole", "gene": "AHR", "relation": "activates", "evidence": "high", "source": "Zelante 2013"},
    {"metabolite": "indole-3-propionic_acid", "gene": "AHR", "relation": "activates", "evidence": "medium", "source": "Zelante 2013"},
]


def get_curated_edges(output_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Get curated microbe-metabolite and metabolite-gene edges.
    Saves to CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save microbe-metabolite edges
    mm_file = output_dir / "microbe_metabolite_edges.csv"
    with open(mm_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["microbe", "metabolite", "relation", "evidence", "source"])
        writer.writeheader()
        writer.writerows(MICROBE_METABOLITE_EDGES)
    print(f"  Saved {len(MICROBE_METABOLITE_EDGES)} microbe-metabolite edges to {mm_file}")

    # Save metabolite-gene edges
    mg_file = output_dir / "metabolite_gene_edges.csv"
    with open(mg_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["metabolite", "gene", "relation", "evidence", "source"])
        writer.writeheader()
        writer.writerows(METABOLITE_GENE_EDGES)
    print(f"  Saved {len(METABOLITE_GENE_EDGES)} metabolite-gene edges to {mg_file}")

    return MICROBE_METABOLITE_EDGES, METABOLITE_GENE_EDGES


if __name__ == "__main__":
    # Test the functions
    from celiac.config import RAW_DIR, PROCESSED_DIR, KEY_GENES

    print("\n=== Testing Data Fetching ===\n")

    # Test GEO download
    print("1. Testing GEO download...")
    matrix_file = fetch_geo_series_matrix("GSE164883", RAW_DIR)
    if matrix_file:
        print("  Success!")

    # Test Monarch API
    print("\n2. Testing Monarch API for TGM6...")
    gene_id, phenotypes = fetch_monarch_gene_phenotypes("TGM6")
    print(f"  Gene ID: {gene_id}")
    print(f"  Phenotypes: {len(phenotypes)}")
    for p in phenotypes[:5]:
        print(f"    - {p['id']}: {p['label']}")

    # Test curated edges
    print("\n3. Getting curated edges...")
    mm_edges, mg_edges = get_curated_edges(PROCESSED_DIR)
    print(f"  Microbe-Metabolite: {len(mm_edges)}")
    print(f"  Metabolite-Gene: {len(mg_edges)}")
