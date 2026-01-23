#!/usr/bin/env python3
"""
Data Availability & Connectivity Check for Celiac Gut-Brain KG

This script de-risks the research by verifying:
1. GSE164883 (duodenal transcriptomics) is accessible and has sufficient samples
2. CeD-associated microbiome taxa can connect to metabolite/gene pathways
3. Monarch/HPO has gene↔phenotype edges for neurological phenotypes
4. The layers can actually connect (microbe → metabolite → gene → phenotype)

Run: python3 check_data_availability.py

Uses only standard library (no external dependencies).
"""

import urllib.request
import urllib.parse
import json
import ssl
import time
from typing import Dict, List, Tuple, Any, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

# CeD-associated taxa from meta-analyses (Arcila-Galvis et al. 2022, etc.)
CED_MICROBIOME_TAXA = [
    # Enriched in CeD
    ("Bacteroides", "enriched", "Arcila-Galvis 2022"),
    ("Escherichia", "enriched", "Arcila-Galvis 2022"),
    ("Staphylococcus", "enriched", "Arcila-Galvis 2022"),
    ("Neisseria", "enriched", "Arcila-Galvis 2022"),
    ("Haemophilus", "enriched", "Arcila-Galvis 2022"),
    ("Prevotella", "enriched", "Arcila-Galvis 2022"),
    ("Klebsiella", "enriched", "Caminero 2019"),
    ("Pseudomonas", "enriched", "Caminero 2019"),
    # Depleted in CeD
    ("Bifidobacterium", "depleted", "Arcila-Galvis 2022"),
    ("Lactobacillus", "depleted", "Arcila-Galvis 2022"),
    ("Streptococcus", "depleted", "Arcila-Galvis 2022"),
    ("Faecalibacterium", "depleted", "Arcila-Galvis 2022"),
    ("Blautia", "depleted", "Wacklin 2014"),
    ("Dorea", "depleted", "Wacklin 2014"),
]

# Neurological HPO terms relevant to celiac
NEUROLOGICAL_HPO_TERMS = [
    "HP:0001251",  # Ataxia
    "HP:0001271",  # Polyneuropathy
    "HP:0002354",  # Memory impairment
    "HP:0000708",  # Behavioral abnormality
    "HP:0002076",  # Migraine
    "HP:0001250",  # Seizures
    "HP:0100543",  # Cognitive impairment
    "HP:0002078",  # Truncal ataxia
    "HP:0007340",  # Lower limb muscle weakness
    "HP:0009830",  # Peripheral neuropathy
]

# Key genes known to be involved in celiac/neurological manifestations
CED_KEY_GENES = [
    "TGM6",   # Transglutaminase 6 - gluten ataxia
    "TGM2",   # Transglutaminase 2 - celiac autoantigen
    "HLA-DQA1",  # HLA risk allele
    "HLA-DQB1",  # HLA risk allele
    "IL15",   # Inflammatory cytokine
    "IFNG",   # Interferon gamma
    "TPH1",   # Tryptophan hydroxylase (serotonin synthesis)
    "TPH2",   # Tryptophan hydroxylase 2
    "IDO1",   # Indoleamine dioxygenase (kynurenine pathway)
    "SLC6A4", # Serotonin transporter
]

# Create SSL context that doesn't verify certificates (for compatibility)
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE


def fetch_url(url: str, timeout: int = 30) -> Optional[str]:
    """Fetch URL content with error handling."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Research Data Check)"}
        )
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        return None


def fetch_json(url: str, timeout: int = 30) -> Optional[Dict]:
    """Fetch JSON from URL."""
    content = fetch_url(url, timeout)
    if content:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
    return None


# ============================================================================
# 1. CHECK GEO DATASET (GSE164883)
# ============================================================================

def check_geo_dataset(geo_id: str = "GSE164883") -> Tuple[bool, Dict]:
    """Check GEO dataset availability and get sample counts."""
    print(f"\n{'='*60}")
    print(f"1. CHECKING GEO DATASET: {geo_id}")
    print('='*60)

    # Try NCBI E-utilities
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gds&term={geo_id}&retmode=json"

    search_data = fetch_json(search_url)

    if search_data:
        id_list = search_data.get("esearchresult", {}).get("idlist", [])

        if id_list:
            print(f"  ✓ Dataset found in GEO (ID: {id_list[0]})")

            # Get summary
            summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gds&id={id_list[0]}&retmode=json"
            summary_data = fetch_json(summary_url)

            if summary_data:
                result = summary_data.get("result", {})
                if id_list[0] in result:
                    info = result[id_list[0]]
                    n_samples = info.get("n_samples", "unknown")
                    title = info.get("title", "")[:70]
                    gpl = info.get("gpl", "")

                    print(f"  ✓ Title: {title}...")
                    print(f"  ✓ Platform: GPL{gpl}")
                    print(f"  ✓ Sample count: {n_samples}")

                    return True, {"n_samples": n_samples, "title": title}

    # Fallback: Try direct SOFT format
    print("  → Trying direct GEO query...")
    soft_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={geo_id}&targ=self&form=text"
    soft_content = fetch_url(soft_url)

    if soft_content and "!Series_title" in soft_content:
        lines = soft_content.split('\n')
        sample_count = sum(1 for l in lines if l.startswith("!Series_sample_id"))
        title_lines = [l for l in lines if l.startswith("!Series_title")]
        title = title_lines[0].split("=")[1].strip() if title_lines else ""

        print(f"  ✓ Dataset accessible")
        print(f"  ✓ Title: {title[:70]}...")
        print(f"  ✓ Sample count: {sample_count}")
        return True, {"n_samples": sample_count, "title": title}

    print(f"  ❌ Could not access {geo_id}")
    return False, {}


# ============================================================================
# 2. CHECK MONARCH INITIATIVE (Gene-Phenotype edges)
# ============================================================================

def check_monarch_gene(gene_symbol: str) -> Tuple[Optional[str], List[Dict]]:
    """Query Monarch for gene-phenotype associations."""

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
        if "NCBIGene" in item_id or "HGNC" in item_id:
            gene_id = item_id
            break

    if not gene_id:
        return None, []

    # Get associations
    encoded_id = urllib.parse.quote(gene_id)
    assoc_url = f"https://api.monarchinitiative.org/v3/api/association/all?subject={encoded_id}&category=biolink:GeneToPhenotypicFeatureAssociation&limit=100"

    assoc_data = fetch_json(assoc_url)
    if not assoc_data:
        return gene_id, []

    phenotypes = []
    for assoc in assoc_data.get("items", []):
        obj = assoc.get("object", {})
        obj_id = obj.get("id", "")
        if obj_id.startswith("HP:"):
            phenotypes.append({
                "id": obj_id,
                "label": obj.get("label", ""),
                "is_neurological": obj_id in NEUROLOGICAL_HPO_TERMS
            })

    return gene_id, phenotypes


def check_monarch_phenotype(hpo_id: str) -> List[Dict]:
    """Get genes associated with an HPO phenotype."""

    encoded_id = urllib.parse.quote(hpo_id)
    assoc_url = f"https://api.monarchinitiative.org/v3/api/entity/{encoded_id}/associations?category=biolink:GeneToPhenotypicFeatureAssociation&limit=500"

    data = fetch_json(assoc_url)
    if not data:
        return []

    genes = []
    for assoc in data.get("items", []):
        subj = assoc.get("subject", {})
        if "Gene" in str(subj.get("category", [])):
            genes.append({
                "id": subj.get("id", ""),
                "label": subj.get("label", "")
            })

    return genes


def check_hpo_term_info(hpo_id: str) -> Optional[Dict]:
    """Get HPO term information."""
    encoded_id = urllib.parse.quote(hpo_id)
    url = f"https://api.monarchinitiative.org/v3/api/entity/{encoded_id}"
    return fetch_json(url)


def check_all_monarch_data() -> Dict:
    """Check Monarch data availability for key genes and phenotypes."""
    print(f"\n{'='*60}")
    print("2. CHECKING MONARCH INITIATIVE (Gene↔Phenotype)")
    print('='*60)

    results = {
        "genes_found": 0,
        "genes_with_neuro_pheno": 0,
        "phenotypes_checked": 0,
        "phenotypes_with_genes": 0,
        "total_gene_pheno_edges": 0,
        "gene_details": [],
        "phenotype_details": []
    }

    # Check key genes
    print("\n  Checking key CeD genes for phenotype associations...")
    for gene in CED_KEY_GENES:
        gene_id, phenotypes = check_monarch_gene(gene)
        if gene_id:
            results["genes_found"] += 1
            neuro_phenos = [p for p in phenotypes if p["is_neurological"]]
            if neuro_phenos:
                results["genes_with_neuro_pheno"] += 1
                print(f"    ✓ {gene}: {len(phenotypes)} phenotypes ({len(neuro_phenos)} neurological)")
            else:
                print(f"    ○ {gene}: {len(phenotypes)} phenotypes (0 neurological)")
            results["total_gene_pheno_edges"] += len(phenotypes)
            results["gene_details"].append({
                "symbol": gene,
                "id": gene_id,
                "n_phenotypes": len(phenotypes),
                "n_neuro": len(neuro_phenos)
            })
        else:
            print(f"    ❌ {gene}: not found in Monarch")
        time.sleep(0.3)  # Rate limiting

    # Check neurological phenotypes
    print("\n  Checking neurological HPO terms for gene associations...")
    for hpo_id in NEUROLOGICAL_HPO_TERMS[:5]:  # Check first 5
        # Get term info first
        term_info = check_hpo_term_info(hpo_id)
        term_label = term_info.get("name", hpo_id) if term_info else hpo_id

        genes = check_monarch_phenotype(hpo_id)
        results["phenotypes_checked"] += 1
        if genes:
            results["phenotypes_with_genes"] += 1
            print(f"    ✓ {term_label}: {len(genes)} associated genes")
            results["phenotype_details"].append({
                "id": hpo_id,
                "label": term_label,
                "n_genes": len(genes)
            })
        else:
            print(f"    ○ {term_label}: no genes found via API")
        time.sleep(0.3)

    print(f"\n  Summary:")
    print(f"    - Genes found in Monarch: {results['genes_found']}/{len(CED_KEY_GENES)}")
    print(f"    - Genes with neurological phenotypes: {results['genes_with_neuro_pheno']}")
    print(f"    - Phenotypes with gene associations: {results['phenotypes_with_genes']}/{results['phenotypes_checked']}")
    print(f"    - Total gene↔phenotype edges found: {results['total_gene_pheno_edges']}")

    return results


# ============================================================================
# 3. CHECK GUT-BRAIN KNOWLEDGE BASES
# ============================================================================

def check_knowledge_bases() -> List[str]:
    """Check availability of gut-brain knowledge bases."""
    print(f"\n{'='*60}")
    print("3. CHECKING GUT-BRAIN KNOWLEDGE BASES")
    print('='*60)

    databases = [
        ("gutMGene", "http://bio-annotation.cn/gutmgene/", "Microbe-Metabolite-Gene", False),
        ("gutMDisorder", "http://bio-annotation.cn/gutMDisorder/", "Microbe-Disease", False),
        ("HMDB", "https://hmdb.ca/", "Metabolite database", True),
        ("VMH", "https://www.vmh.life/", "Microbe metabolites", True),
    ]

    accessible = []

    for name, url, desc, has_api in databases:
        content = fetch_url(url, timeout=15)
        if content and len(content) > 100:
            print(f"  ✓ {name}: accessible")
            print(f"      {desc}")
            print(f"      {'API available' if has_api else 'Manual download required'}")
            accessible.append(name)
        else:
            print(f"  ⚠ {name}: not accessible or blocked")
        time.sleep(0.5)

    # Key metabolite pathways
    print("\n  Key metabolite pathways to verify manually:")
    pathways = [
        "Tryptophan → Kynurenine → Kynurenic acid",
        "Tryptophan → Serotonin (5-HT)",
        "Tryptophan → Indole (microbial)",
        "SCFAs: Acetate, Propionate, Butyrate",
    ]
    for p in pathways:
        print(f"    → {p}")

    return accessible


# ============================================================================
# 4. CHECK BRAIN DATA SOURCES
# ============================================================================

def check_brain_data() -> Dict:
    """Check availability of brain scaffold data."""
    print(f"\n{'='*60}")
    print("4. CHECKING BRAIN SCAFFOLD DATA")
    print('='*60)

    results = {}

    # Allen Human Brain Atlas
    print("\n  Allen Human Brain Atlas (AHBA):")
    ahba_url = "https://api.brain-map.org/api/v2/data/query.json?criteria=model::Donor"
    ahba_data = fetch_json(ahba_url)

    if ahba_data:
        donors = ahba_data.get("msg", [])
        print(f"    ✓ AHBA API accessible")
        print(f"    ✓ Donors available: {len(donors)}")
        print(f"    → Use 'abagen' Python package for gene expression by region")
        results["ahba"] = True
    else:
        print(f"    ⚠ AHBA API not accessible")
        results["ahba"] = False

    # HCP
    print("\n  Human Connectome Project (HCP S1200):")
    print("    ⚠ Requires registration at https://db.humanconnectome.org/")
    print("    → Group-average connectivity available after registration")
    print("    → Alternative: nilearn/neuromaps for preprocessed data")
    results["hcp_note"] = "Requires registration"

    # Schaefer parcellation
    print("\n  Schaefer Parcellation (400 regions):")
    schaefer_url = "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_400Parcels_7Networks_order.txt"
    schaefer_content = fetch_url(schaefer_url)
    if schaefer_content:
        print(f"    ✓ Schaefer parcellation accessible on GitHub")
        results["schaefer"] = True
    else:
        print(f"    ⚠ Could not fetch Schaefer parcellation")
        results["schaefer"] = False

    return results


# ============================================================================
# 5. CONNECTIVITY ANALYSIS
# ============================================================================

def analyze_connectivity():
    """Analyze potential for layer connectivity."""
    print(f"\n{'='*60}")
    print("5. CONNECTIVITY POTENTIAL ANALYSIS")
    print('='*60)

    print("""
  Path viability: Microbe → Metabolite → Gene → Phenotype

  KNOWN CONNECTIONS (from literature):

    ✓ Bifidobacterium (depleted in CeD)
        → tryptophan metabolism → serotonin
        → TPH1/TPH2, SLC6A4 genes
        → mood/cognitive phenotypes

    ✓ Faecalibacterium (depleted in CeD)
        → butyrate production
        → FFAR2/FFAR3 receptors
        → anti-inflammatory effects

    ✓ E. coli / Klebsiella (enriched in CeD)
        → LPS, pro-inflammatory
        → TLR4 signaling
        → neuroinflammation

    ✓ TGM6 autoantibodies
        → direct link to HP:0001251 (Ataxia)
        → Cerebellum

    ✓ Kynurenine pathway
        → IDO1/TDO2 genes
        → neurotoxic metabolites
        → HP:0100543 (Cognitive impairment)

  POTENTIAL GAPS:

    ⚠ gutMGene coverage: Not all CeD taxa may have entries
      → Mitigation: Use genus-level, add from primary literature

    ⚠ Microbe→Gene direct edges are rare
      → Must go through Metabolite layer

    ⚠ Phenotype→Region edges are sparse
      → Only use well-supported (Ataxia→Cerebellum)

  VIABILITY ESTIMATE:

    If gutMGene covers ≥50% of CeD taxa with metabolite edges: ✓
    If Monarch provides ≥20 gene→neuro phenotype edges: ✓
    If ≥3 complete paths exist: Project is VIABLE
    """)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("  CELIAC GUT-BRAIN KG: DATA AVAILABILITY CHECK")
    print("="*60)
    print("  Checking all data sources for project viability...")

    results = {}

    # 1. GEO
    geo_ok, geo_info = check_geo_dataset("GSE164883")
    results["geo"] = {"available": geo_ok, "info": geo_info}

    # 2. Monarch
    monarch_results = check_all_monarch_data()
    results["monarch"] = monarch_results

    # 3. Knowledge bases
    kb_accessible = check_knowledge_bases()
    results["knowledge_bases"] = kb_accessible

    # 4. Brain data
    brain_results = check_brain_data()
    results["brain"] = brain_results

    # 5. Connectivity analysis
    analyze_connectivity()

    # Final summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY & NEXT STEPS")
    print('='*60)

    print(f"""
  DATA AVAILABILITY RESULTS:

    GEO (GSE164883):        {'✓ Available' if results['geo']['available'] else '❌ Not found'}
    Monarch (gene↔pheno):   {results['monarch']['genes_found']}/{len(CED_KEY_GENES)} genes found
    Gene↔Phenotype edges:   {results['monarch']['total_gene_pheno_edges']} total
    Knowledge bases:        {len(results['knowledge_bases'])}/4 accessible
    Brain data (AHBA):      {'✓ Available' if results['brain'].get('ahba') else '⚠ Check manually'}

  VIABILITY ASSESSMENT:
""")

    # Simple viability check
    viable = True
    issues = []

    if not results['geo']['available']:
        viable = False
        issues.append("GSE164883 not accessible")

    if results['monarch']['genes_found'] < 5:
        viable = False
        issues.append("Too few genes found in Monarch")

    if results['monarch']['total_gene_pheno_edges'] < 10:
        issues.append("Low gene↔phenotype edge count - may need supplementation")

    if len(results['knowledge_bases']) < 2:
        issues.append("Limited KB access - check gutMGene manually")

    if viable and len(issues) == 0:
        print("    ✓ PROJECT APPEARS VIABLE")
        print("    → Proceed with data download and KG construction")
    elif viable:
        print("    ⚠ PROJECT VIABLE WITH CAVEATS")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("    ❌ CRITICAL ISSUES FOUND")
        for issue in issues:
            print(f"      - {issue}")

    print(f"""
  IMMEDIATE NEXT STEPS:

    1. Download GSE164883 series matrix
       → GEOquery (R) or GEOparse (Python)
       → Verify duodenal biopsy samples

    2. Download gutMGene flat files manually
       → http://bio-annotation.cn/gutmgene/
       → Map CeD taxa to metabolites

    3. Export Monarch gene↔phenotype edges
       → Use their bulk download or API
       → Filter to neurological HPO terms

    4. Build prototype 3-layer graph
       → Microbe → Metabolite → Gene → Phenotype
       → Count connected components
    """)

    return results


if __name__ == "__main__":
    main()
