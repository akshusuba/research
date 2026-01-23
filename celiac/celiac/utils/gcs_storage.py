"""
Google Cloud Storage utilities for model backup.

Provides:
- Automatic backup of models to GCS
- Sync between local and GCS storage
- Model versioning and checkpointing
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import shutil


# Default GCS bucket for celiac models
DEFAULT_BUCKET = "celiac-gut-brain-models"


def get_gcs_bucket() -> str:
    """Get GCS bucket name from environment or default."""
    return os.environ.get("CELIAC_GCS_BUCKET", DEFAULT_BUCKET)


def gcs_path(local_path: str, bucket: Optional[str] = None) -> str:
    """
    Convert local path to GCS path.

    Args:
        local_path: Local file path
        bucket: GCS bucket name (uses default if not specified)

    Returns:
        GCS URI (gs://bucket/path)
    """
    if bucket is None:
        bucket = get_gcs_bucket()

    # Get relative path from project root
    local_path = Path(local_path)
    try:
        # Try to make path relative to celiac project
        rel_path = local_path.relative_to(Path("/home/elrarun/research/celiac"))
    except ValueError:
        rel_path = local_path.name

    return f"gs://{bucket}/celiac/{rel_path}"


def upload_to_gcs(
    local_path: str,
    gcs_uri: Optional[str] = None,
    bucket: Optional[str] = None,
    verbose: bool = True,
) -> bool:
    """
    Upload a file to GCS.

    Args:
        local_path: Local file path
        gcs_uri: Full GCS URI (if None, auto-generated from local path)
        bucket: GCS bucket name
        verbose: Print progress

    Returns:
        True if successful
    """
    local_path = Path(local_path)
    if not local_path.exists():
        print(f"Error: {local_path} does not exist")
        return False

    if gcs_uri is None:
        gcs_uri = gcs_path(str(local_path), bucket)

    cmd = ["gcloud", "storage", "cp", str(local_path), gcs_uri]

    if verbose:
        print(f"Uploading {local_path} -> {gcs_uri}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if verbose:
            print(f"  Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e.stderr}")
        return False


def download_from_gcs(
    gcs_uri: str,
    local_path: str,
    verbose: bool = True,
) -> bool:
    """
    Download a file from GCS.

    Args:
        gcs_uri: GCS URI (gs://bucket/path)
        local_path: Local destination path
        verbose: Print progress

    Returns:
        True if successful
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["gcloud", "storage", "cp", gcs_uri, str(local_path)]

    if verbose:
        print(f"Downloading {gcs_uri} -> {local_path}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if verbose:
            print(f"  Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e.stderr}")
        return False


def sync_to_gcs(
    local_dir: str,
    bucket: Optional[str] = None,
    verbose: bool = True,
) -> bool:
    """
    Sync a local directory to GCS.

    Args:
        local_dir: Local directory path
        bucket: GCS bucket name
        verbose: Print progress

    Returns:
        True if successful
    """
    local_dir = Path(local_dir)
    if not local_dir.exists():
        print(f"Error: {local_dir} does not exist")
        return False

    gcs_uri = gcs_path(str(local_dir), bucket)

    cmd = ["gcloud", "storage", "rsync", "-r", str(local_dir), gcs_uri]

    if verbose:
        print(f"Syncing {local_dir} -> {gcs_uri}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if verbose:
            print(f"  Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e.stderr}")
        return False


def sync_from_gcs(
    local_dir: str,
    bucket: Optional[str] = None,
    verbose: bool = True,
) -> bool:
    """
    Sync from GCS to local directory.

    Args:
        local_dir: Local directory path
        bucket: GCS bucket name
        verbose: Print progress

    Returns:
        True if successful
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    gcs_uri = gcs_path(str(local_dir), bucket)

    cmd = ["gcloud", "storage", "rsync", "-r", gcs_uri, str(local_dir)]

    if verbose:
        print(f"Syncing {gcs_uri} -> {local_dir}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if verbose:
            print(f"  Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e.stderr}")
        return False


def list_gcs_files(
    gcs_uri: str,
    pattern: Optional[str] = None,
) -> List[str]:
    """
    List files in GCS location.

    Args:
        gcs_uri: GCS URI prefix
        pattern: Optional glob pattern

    Returns:
        List of file URIs
    """
    cmd = ["gcloud", "storage", "ls", gcs_uri]
    if pattern:
        cmd.append(f"**/{pattern}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    except subprocess.CalledProcessError:
        return []


class ModelCheckpointer:
    """
    Model checkpointing with automatic GCS backup.

    Usage:
        checkpointer = ModelCheckpointer("my_experiment")
        checkpointer.save(model, metrics, epoch=10)
        model = checkpointer.load_best()
    """

    def __init__(
        self,
        experiment_name: str,
        local_dir: str = "models",
        bucket: Optional[str] = None,
        sync_to_gcs: bool = True,
    ):
        """
        Args:
            experiment_name: Name for this experiment
            local_dir: Local directory for checkpoints
            bucket: GCS bucket name
            sync_to_gcs: Whether to automatically sync to GCS
        """
        self.experiment_name = experiment_name
        self.local_dir = Path(local_dir) / experiment_name
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.bucket = bucket
        self.sync_enabled = sync_to_gcs
        self.metadata_file = self.local_dir / "checkpoints.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"checkpoints": [], "best_checkpoint": None}

    def _save_metadata(self):
        """Save checkpoint metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def save(
        self,
        model,
        metrics: Dict[str, float],
        epoch: int,
        is_best: bool = False,
    ) -> str:
        """
        Save a model checkpoint.

        Args:
            model: PyTorch model
            metrics: Dict of metrics
            epoch: Current epoch
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        import torch

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch{epoch}_{timestamp}.pt"
        checkpoint_path = self.local_dir / checkpoint_name

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "timestamp": timestamp,
        }
        torch.save(checkpoint, checkpoint_path)

        # Update metadata
        checkpoint_info = {
            "name": checkpoint_name,
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": timestamp,
            "is_best": is_best,
        }
        self.metadata["checkpoints"].append(checkpoint_info)

        if is_best:
            self.metadata["best_checkpoint"] = checkpoint_name
            # Also save as best.pt
            best_path = self.local_dir / "best.pt"
            shutil.copy(checkpoint_path, best_path)

        self._save_metadata()

        # Sync to GCS
        if self.sync_enabled:
            upload_to_gcs(str(checkpoint_path), bucket=self.bucket, verbose=False)
            upload_to_gcs(str(self.metadata_file), bucket=self.bucket, verbose=False)
            if is_best:
                upload_to_gcs(str(best_path), bucket=self.bucket, verbose=False)
            print(f"Checkpoint saved and synced to GCS: {checkpoint_name}")
        else:
            print(f"Checkpoint saved locally: {checkpoint_name}")

        return str(checkpoint_path)

    def load(
        self,
        checkpoint_name: Optional[str] = None,
        model=None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_name: Name of checkpoint (None = latest)
            model: Optional model to load state dict into

        Returns:
            Checkpoint dict
        """
        import torch

        if checkpoint_name is None:
            if self.metadata["checkpoints"]:
                checkpoint_name = self.metadata["checkpoints"][-1]["name"]
            else:
                raise ValueError("No checkpoints found")

        checkpoint_path = self.local_dir / checkpoint_name

        # Try to download from GCS if not found locally
        if not checkpoint_path.exists() and self.sync_enabled:
            gcs_uri = gcs_path(str(checkpoint_path), self.bucket)
            download_from_gcs(gcs_uri, str(checkpoint_path))

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if model is not None:
            model.load_state_dict(checkpoint["model_state_dict"])

        return checkpoint

    def load_best(self, model=None) -> Dict[str, Any]:
        """Load the best checkpoint."""
        best_name = self.metadata.get("best_checkpoint")
        if best_name is None:
            raise ValueError("No best checkpoint found")
        return self.load(best_name, model)

    def sync_all_to_gcs(self):
        """Sync all local checkpoints to GCS."""
        if not self.sync_enabled:
            print("GCS sync is disabled")
            return

        sync_to_gcs(str(self.local_dir), self.bucket)


def backup_models_to_gcs(
    models_dir: str = "/home/elrarun/research/celiac/models",
    bucket: Optional[str] = None,
):
    """
    Backup all models to GCS.

    Args:
        models_dir: Local models directory
        bucket: GCS bucket name
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        print(f"Models directory {models_dir} does not exist")
        return

    print(f"Backing up models from {models_dir} to GCS...")

    # Sync entire models directory
    sync_to_gcs(str(models_dir), bucket)

    print("Backup complete!")


def restore_models_from_gcs(
    models_dir: str = "/home/elrarun/research/celiac/models",
    bucket: Optional[str] = None,
):
    """
    Restore models from GCS.

    Args:
        models_dir: Local models directory
        bucket: GCS bucket name
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Restoring models from GCS to {models_dir}...")

    # Sync from GCS
    sync_from_gcs(str(models_dir), bucket)

    print("Restore complete!")


if __name__ == "__main__":
    # Test GCS connectivity
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "backup":
            backup_models_to_gcs()
        elif command == "restore":
            restore_models_from_gcs()
        elif command == "test":
            print("Testing GCS connectivity...")
            files = list_gcs_files(f"gs://{get_gcs_bucket()}/")
            if files:
                print(f"Found {len(files)} files in bucket")
            else:
                print("Bucket is empty or not accessible")
        else:
            print(f"Unknown command: {command}")
            print("Usage: python gcs_storage.py [backup|restore|test]")
    else:
        print("GCS Storage Utility")
        print("Usage: python gcs_storage.py [backup|restore|test]")
