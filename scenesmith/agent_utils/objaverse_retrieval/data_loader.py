"""Data loading utilities for Objaverse (ObjectThor) preprocessed indices and embeddings."""

import json
import logging

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

console_logger = logging.getLogger(__name__)


@dataclass
class ObjaverseMeshMetadata:
    """Metadata for a single Objaverse mesh."""

    uid: str
    """Objaverse/ObjectThor unique identifier."""

    name: str
    """Human-readable object name (from description)."""

    category: str
    """scenesmith category (large_objects, small_objects, etc.)."""

    bounding_box: tuple[float, float, float]
    """Bounding box dimensions (x, y, z) in meters from GLB."""

    description: str | None = None
    """Object description text (optional)."""


@dataclass
class ObjaversePreprocessedData:
    """Container for all preprocessed Objaverse data."""

    metadata_by_category: dict[str, list[ObjaverseMeshMetadata]]
    """Maps category names to mesh metadata lists."""

    clip_embeddings: np.ndarray
    """CLIP embeddings array (N, 768)."""

    embedding_index: list[str]
    """Maps array index to mesh UID."""

    object_categories: dict[str, list[str]]
    """Maps object types to UID lists."""

    _metadata_by_uid: dict[str, ObjaverseMeshMetadata] = field(
        init=False, default_factory=dict, repr=False
    )
    """Private O(1) lookup index from UID to metadata."""

    def __post_init__(self):
        """Build metadata lookup index after initialization."""
        self._metadata_by_uid = {
            m.uid: m for meshes in self.metadata_by_category.values() for m in meshes
        }

    def get_metadata(self, uid: str) -> ObjaverseMeshMetadata | None:
        """Get metadata for a specific mesh UID (O(1) lookup).

        Args:
            uid: Objaverse mesh UID to look up.

        Returns:
            Mesh metadata if found, None otherwise.
        """
        return self._metadata_by_uid.get(uid)

    def get_embedding_index(self, uid: str) -> int | None:
        """Get the embedding array index for a mesh UID.

        Args:
            uid: Objaverse mesh UID to look up.

        Returns:
            Array index if found, None otherwise.
        """
        try:
            return self.embedding_index.index(uid)
        except ValueError:
            return None


def load_preprocessed_data(preprocessed_path: Path) -> ObjaversePreprocessedData:
    """Load all preprocessed Objaverse data.

    Args:
        preprocessed_path: Path to directory containing preprocessed files.

    Returns:
        Loaded preprocessed data.

    Raises:
        FileNotFoundError: If required files are missing.
        ValueError: If data format is invalid.
    """
    console_logger.info(f"Loading Objaverse preprocessed data from {preprocessed_path}")

    metadata_path = preprocessed_path / "metadata_index.json"
    embeddings_path = preprocessed_path / "clip_embeddings.npy"
    embedding_index_path = preprocessed_path / "embedding_index.yaml"
    categories_path = preprocessed_path / "object_categories.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not embedding_index_path.exists():
        raise FileNotFoundError(f"Embedding index not found: {embedding_index_path}")
    if not categories_path.exists():
        raise FileNotFoundError(f"Categories file not found: {categories_path}")

    with open(metadata_path, "r") as f:
        metadata_data = json.load(f)

    # Build metadata_by_category from flat metadata list.
    metadata_by_category: dict[str, list[ObjaverseMeshMetadata]] = {}
    total_entries = 0

    for uid, entry in metadata_data.items():
        total_entries += 1

        # Extract bounding box from metadata.
        bbox = entry.get("bounding_box", [1.0, 1.0, 1.0])
        if isinstance(bbox, dict):
            bbox = [bbox.get("x", 1.0), bbox.get("y", 1.0), bbox.get("z", 1.0)]

        metadata = ObjaverseMeshMetadata(
            uid=uid,
            name=entry.get("name", entry.get("description", uid)),
            category=entry.get("category", "small_objects"),
            bounding_box=tuple(bbox),
            description=entry.get("description", ""),
        )

        category = metadata.category
        if category not in metadata_by_category:
            metadata_by_category[category] = []
        metadata_by_category[category].append(metadata)

    console_logger.info(f"Loaded {total_entries} Objaverse entries")

    # Load CLIP embeddings.
    clip_embeddings = np.load(embeddings_path)
    console_logger.info(
        f"Loaded CLIP embeddings: shape={clip_embeddings.shape}, "
        f"dtype={clip_embeddings.dtype}"
    )

    # Load embedding index (UID list).
    with open(embedding_index_path, "r") as f:
        embedding_index_data = yaml.safe_load(f)
        # Handle both dict format {"uids": [...]} and plain list format.
        if isinstance(embedding_index_data, dict):
            embedding_index = embedding_index_data.get("uids", [])
        else:
            embedding_index = embedding_index_data

    # Load object categories.
    with open(categories_path, "r") as f:
        categories_data = json.load(f)

    object_categories = {
        category: uid_list for category, uid_list in categories_data.items()
    }

    console_logger.info(
        f"Loaded {len(metadata_by_category)} categories, "
        f"{len(embedding_index)} mesh embeddings, "
        f"{len(object_categories)} object categories"
    )

    return ObjaversePreprocessedData(
        metadata_by_category=metadata_by_category,
        clip_embeddings=clip_embeddings,
        embedding_index=embedding_index,
        object_categories=object_categories,
    )


def construct_objaverse_mesh_path(data_path: Path, uid: str) -> Path:
    """Construct the file path for an Objaverse mesh.

    ObjectThor stores assets in {data_path}/assets/{uid}/{uid}.glb structure.

    Args:
        data_path: Root directory of Objaverse data (containing assets/ subdir).
        uid: Objaverse mesh UID.

    Returns:
        Path to the GLB mesh file.

    Raises:
        FileNotFoundError: If the constructed path does not exist.
    """
    # ObjectThor stores assets in assets/ subdirectory.
    mesh_path = data_path / "assets" / uid / f"{uid}.glb"

    if not mesh_path.exists():
        pkl_path = data_path / "assets" / uid / f"{uid}.pkl.gz"
        if pkl_path.exists():
            return pkl_path
        raise FileNotFoundError(f"Objaverse mesh not found: {mesh_path}")

    return mesh_path


def load_objaverse_mesh_from_pkl(data_path: Path, uid: str) -> "trimesh.Trimesh":
    """Load an Objaverse mesh from .pkl.gz file directly as a trimesh object.
    Fallback for when .glb files are not available.
    """
    import gzip
    import pickle

    import numpy as np
    import trimesh

    pkl_path = data_path / "assets" / uid / f"{uid}.pkl.gz"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Objaverse pkl not found: {pkl_path}")
    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # vertices is a list of {"x":..,"y":..,"z":..} dicts (Unity/Thor format)
    verts_raw = data["vertices"]
    vertices = np.array([[v["x"], v["y"], v["z"]] for v in verts_raw], dtype=np.float64)
    # triangles is a flat list of ints; reshape into (N,3) faces
    tris_raw = data["triangles"]
    triangles = np.array(tris_raw, dtype=np.int64).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    return mesh
