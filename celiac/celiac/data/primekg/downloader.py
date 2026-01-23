"""
PrimeKG downloader.

Downloads PrimeKG from Harvard Dataverse.
PrimeKG: A Knowledge Graph for Precision Medicine
- 100K+ nodes, 4M+ edges
- 29 relation types
- Integrates 20+ biomedical resources
"""

import os
import urllib.request
import gzip
import shutil
from pathlib import Path
from typing import Optional
import ssl


# PrimeKG download URL from Harvard Dataverse
PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"

# Alternative mirror URLs
PRIMEKG_MIRRORS = [
    "https://dataverse.harvard.edu/api/access/datafile/6180620",
]

# Expected file info
PRIMEKG_FILENAME = "kg.csv"
PRIMEKG_COMPRESSED = "kg.csv.gz"


def download_file(
    url: str,
    dest_path: str,
    chunk_size: int = 8192,
    show_progress: bool = True,
) -> bool:
    """
    Download a file from URL.

    Args:
        url: URL to download from
        dest_path: Destination path
        chunk_size: Download chunk size
        show_progress: Whether to show progress

    Returns:
        True if successful
    """
    try:
        # Create SSL context that doesn't verify (for some corporate networks)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Open URL
        with urllib.request.urlopen(url, context=ssl_context) as response:
            total_size = response.headers.get('content-length')
            if total_size:
                total_size = int(total_size)

            # Download
            downloaded = 0
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if show_progress and total_size:
                        pct = downloaded / total_size * 100
                        print(f"\rDownloading: {pct:.1f}% ({downloaded / 1e6:.1f} MB)", end='')

            if show_progress:
                print()  # New line after progress

        return True

    except Exception as e:
        print(f"Download failed: {e}")
        return False


def download_primekg(
    output_dir: str = 'data/primekg',
    force: bool = False,
    show_progress: bool = True,
) -> str:
    """
    Download PrimeKG dataset.

    Args:
        output_dir: Directory to save the dataset
        force: Force re-download even if file exists
        show_progress: Show download progress

    Returns:
        Path to the downloaded/extracted CSV file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / PRIMEKG_FILENAME
    compressed_path = output_dir / PRIMEKG_COMPRESSED

    # Check if already downloaded
    if csv_path.exists() and not force:
        print(f"PrimeKG already exists at {csv_path}")
        return str(csv_path)

    print("Downloading PrimeKG from Harvard Dataverse...")
    print("This may take a few minutes (~500MB compressed)")

    # Try each mirror
    success = False
    for url in PRIMEKG_MIRRORS:
        print(f"Trying: {url}")
        if download_file(url, str(compressed_path), show_progress=show_progress):
            success = True
            break

    if not success:
        raise RuntimeError("Failed to download PrimeKG from all mirrors")

    # Check if file is gzipped
    try:
        with gzip.open(compressed_path, 'rb') as f_in:
            # Try to read first few bytes to verify it's gzipped
            f_in.read(100)

        # Decompress
        print("Decompressing...")
        with gzip.open(compressed_path, 'rb') as f_in:
            with open(csv_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove compressed file
        os.remove(compressed_path)

    except gzip.BadGzipFile:
        # File is not gzipped, rename it
        print("File is not compressed, renaming...")
        shutil.move(compressed_path, csv_path)

    print(f"PrimeKG downloaded to {csv_path}")
    return str(csv_path)


def verify_primekg(csv_path: str) -> dict:
    """
    Verify PrimeKG file and return basic statistics.

    Args:
        csv_path: Path to PrimeKG CSV

    Returns:
        Dict with statistics
    """
    import csv

    stats = {
        'num_edges': 0,
        'node_types': set(),
        'relation_types': set(),
        'sample_rows': [],
    }

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            stats['num_edges'] += 1

            if 'x_type' in row:
                stats['node_types'].add(row['x_type'])
            if 'y_type' in row:
                stats['node_types'].add(row['y_type'])
            if 'relation' in row:
                stats['relation_types'].add(row['relation'])

            # Store sample rows
            if i < 5:
                stats['sample_rows'].append(row)

            # Progress every 500k rows
            if (i + 1) % 500000 == 0:
                print(f"Processed {i + 1:,} edges...")

    stats['node_types'] = list(stats['node_types'])
    stats['relation_types'] = list(stats['relation_types'])

    return stats


def get_primekg_info() -> dict:
    """Get information about PrimeKG structure."""
    return {
        'description': 'Precision Medicine Knowledge Graph',
        'source': 'Harvard Dataverse',
        'paper': 'Chandak et al. 2023',
        'expected_nodes': '~129,000',
        'expected_edges': '~8,000,000',
        'node_types': [
            'gene/protein',
            'drug',
            'effect/phenotype',
            'disease',
            'biological_process',
            'molecular_function',
            'cellular_component',
            'exposure',
            'pathway',
            'anatomy',
        ],
        'relation_types': [
            'drug-drug interaction',
            'drug-protein (carrier)',
            'drug-protein (enzyme)',
            'drug-protein (target)',
            'drug-protein (transporter)',
            'drug-effect (indication)',
            'drug-effect (off-label use)',
            'drug-effect (side effect)',
            'drug-disease (indication)',
            'drug-disease (off-label use)',
            'drug-disease (contraindication)',
            'protein-protein interaction',
            'protein-GO (biological process)',
            'protein-GO (molecular function)',
            'protein-GO (cellular component)',
            'protein-pathway',
            'protein-disease',
            'protein-effect (phenotype)',
            'disease-disease',
            'disease-phenotype',
            'disease-GO (biological process)',
            'exposure-disease',
            'exposure-protein',
            'exposure-GO (biological process)',
            'anatomy-protein (expression)',
            'anatomy-GO (biological process)',
        ],
    }
