"""Hypha service for correcting TIFF instance-segmentation masks."""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
from pathlib import Path

import numpy as np
from imjoy_rpc.hypha import connect_to_server
from skimage.draw import polygon
from skimage.measure import find_contours
from tifffile import imread, imwrite

LOGGER = logging.getLogger(__name__)

DEFAULT_PLUGIN_URL = (
    "https://raw.githubusercontent.com/m-albert/hypha-correction-tool/"
    "main/correction_tool.imjoy.html"
)


def discover_image_pairs(
    root: Path,
    mask_suffix: str = "_masks.tif",
    corrected_suffix: str = "_masks_corrected.tif",
) -> list[dict[str, Path | str]]:
    """Find image/mask pairs below *root* without treating outputs as inputs."""
    root = root.resolve()
    pairs: list[dict[str, Path | str]] = []

    for mask_path in sorted(root.rglob(f"*{mask_suffix}")):
        if mask_path.name.endswith(corrected_suffix):
            continue
        relative_mask = mask_path.relative_to(root)
        relative_basename = str(relative_mask)[: -len(mask_suffix)]
        image_path = root / f"{relative_basename}.tif"
        if not image_path.is_file():
            raise FileNotFoundError(
                f"No image found for mask {mask_path}; expected {image_path}"
            )
        pairs.append(
            {
                "basename": relative_basename,
                "image": image_path,
                "mask": mask_path,
                "corrected": root / f"{relative_basename}{corrected_suffix}",
            }
        )

    if not pairs:
        raise FileNotFoundError(
            f"No masks ending in {mask_suffix!r} were found below {root}"
        )
    return pairs


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Scale an image to uint8 for transfer to the browser."""
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected a 2-D or RGB image, got shape {image.shape}")
    image = image.astype(np.float32, copy=False)
    low = float(np.nanmin(image))
    high = float(np.nanmax(image))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Image contains only non-finite values")
    if high == low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def mask_to_paths(mask: np.ndarray) -> list[list[list[float]]]:
    """Convert each non-zero instance to one editable outer contour.

    Padding is important: more than half of the masks in the target dataset have
    instances touching an image edge.  ``find_contours`` otherwise leaves those
    contours open.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}")

    paths: list[list[list[float]]] = []
    for label_id in np.unique(mask):
        if label_id == 0:
            continue
        contours = find_contours(np.pad(mask == label_id, 1), 0.5)
        if not contours:
            continue
        # A label can contain holes or tiny detached fragments.  The largest
        # contour is the editable outer boundary; unchanged labels are copied
        # pixel-for-pixel during saving, preserving their holes/fragments.
        contour = max(contours, key=len) - 1
        paths.append(np.column_stack((contour[:, 1], contour[:, 0])).tolist())
    return paths


def _feature_coordinates(feature: dict) -> np.ndarray | None:
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates")
    if coordinates is None:
        return None
    if geometry.get("type") == "Polygon":
        if not coordinates:
            return None
        coordinates = coordinates[0]
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[0] < 3 or coordinates.shape[1] != 2:
        return None
    return coordinates


def _rasterize_path(coordinates: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    region = np.zeros(shape, dtype=bool)
    rows, columns = polygon(coordinates[:, 1], coordinates[:, 0], shape=shape)
    region[rows, columns] = True
    return region


def features_to_mask(features: dict, original_mask: np.ndarray) -> np.ndarray:
    """Apply edited contours while preserving untouched labels exactly."""
    if not isinstance(features, dict):
        raise TypeError("Expected a GeoJSON FeatureCollection")

    returned_paths = [
        coordinates
        for feature in features.get("features", [])
        if (coordinates := _feature_coordinates(feature)) is not None
    ]
    original_label_ids = [int(value) for value in np.unique(original_mask) if value]
    original_paths = {
        label_id: np.asarray(path, dtype=float)
        for label_id, path in zip(original_label_ids, mask_to_paths(original_mask))
    }

    output = np.zeros_like(original_mask)
    assigned_labels: set[int] = set()
    pending: list[tuple[np.ndarray, np.ndarray]] = []

    # Exact paths survive the browser's two-decimal GeoJSON serialization and
    # allow a no-op save to remain pixel-identical to the source mask.
    for coordinates in returned_paths:
        exact_label = next(
            (
                label_id
                for label_id, original in original_paths.items()
                if label_id not in assigned_labels
                and coordinates.shape == original.shape
                and np.allclose(coordinates, original, atol=0.01)
            ),
            None,
        )
        if exact_label is not None:
            output[original_mask == exact_label] = exact_label
            assigned_labels.add(exact_label)
        else:
            pending.append(
                (coordinates, _rasterize_path(coordinates, original_mask.shape))
            )

    next_label = max(original_label_ids, default=0) + 1
    for _coordinates, region in pending:
        if not np.any(region):
            continue

        candidate_pixels = original_mask[region]
        candidate_pixels = candidate_pixels[candidate_pixels != 0]
        matches = [
            (int(label_id), int(count))
            for label_id, count in zip(*np.unique(candidate_pixels, return_counts=True))
            if int(label_id) not in assigned_labels
        ]
        matched_label = None
        if matches:
            label_id, overlap = max(matches, key=lambda item: item[1])
            original_area = int(np.count_nonzero(original_mask == label_id))
            if overlap / min(int(np.count_nonzero(region)), original_area) >= 0.2:
                matched_label = label_id

        if matched_label is None:
            matched_label = next_label
            next_label += 1
        else:
            assigned_labels.add(matched_label)
        output[region] = matched_label

    return output


def build_tree(basenames: list[str]) -> list[dict]:
    """Build the directory tree expected by the Kaibu tree widget."""
    root_nodes: list[dict] = []
    for basename in basenames:
        nodes = root_nodes
        parts = Path(basename).parts
        for index, part in enumerate(parts):
            leaf = index == len(parts) - 1
            existing = next((node for node in nodes if node["title"] == part), None)
            if existing is None:
                existing = {"title": part}
                if leaf:
                    existing.update(
                        {
                            "isLeaf": True,
                            "isDraggable": False,
                            "data": {"image_basename": basename},
                        }
                    )
                else:
                    existing.update({"children": [], "isExpanded": index == 0})
                nodes.append(existing)
            if not leaf:
                nodes = existing["children"]
    return root_nodes


async def start_server(
    server_url: str,
    path_to_images: str | os.PathLike[str],
    *,
    mask_suffix: str = "_masks.tif",
    corrected_suffix: str = "_masks_corrected.tif",
    plugin_url: str = DEFAULT_PLUGIN_URL,
):
    """Register the correction service and return ``(server, annotator_url)``."""
    root = Path(path_to_images).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    pairs = discover_image_pairs(root, mask_suffix, corrected_suffix)
    by_basename = {str(pair["basename"]): pair for pair in pairs}
    basenames = list(by_basename)
    LOGGER.info("Found %d image/mask pairs below %s", len(pairs), root)

    def get_next_image_basename(image_basename: str) -> str:
        index = basenames.index(image_basename)
        return basenames[(index + 1) % len(basenames)]

    def get_data_by_basename(image_basename: str | None = None):
        image_basename = image_basename or basenames[0]
        pair = by_basename[image_basename]
        corrected_path = Path(pair["corrected"])
        loaded_saved = corrected_path.is_file()
        mask_path = corrected_path if loaded_saved else Path(pair["mask"])

        image = normalize_image(imread(pair["image"]))
        mask = imread(mask_path)
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch for {image_basename}: "
                f"{image.shape[:2]} versus {mask.shape}"
            )
        LOGGER.info("Loaded %s (%s)", image_basename, mask_path.name)
        return image, mask_to_paths(mask), image_basename, loaded_saved

    def save_correction(
        annotation_features: dict,
        image_basename: str,
    ):
        pair = by_basename[image_basename]
        corrected_path = Path(pair["corrected"])
        source_path = corrected_path if corrected_path.is_file() else Path(pair["mask"])
        original_mask = imread(source_path)
        corrected_mask = features_to_mask(annotation_features, original_mask)

        temporary_path = corrected_path.with_name(f".{corrected_path.stem}.tmp.tif")
        imwrite(temporary_path, corrected_mask, photometric="minisblack")
        os.replace(temporary_path, corrected_path)
        LOGGER.info("Saved %s", corrected_path)
        return str(corrected_path)

    server = await connect_to_server({"server_url": server_url})
    token = await server.generate_token()
    service = await server.register_service(
        {
            "name": "Segmentation Mask Correction Tool",
            "id": "correction-tool",
            "config": {"visibility": "public"},
            "get_data_by_basename": get_data_by_basename,
            "save_correction": save_correction,
            "get_widget_node_list_of_basenames": lambda: build_tree(basenames),
            "get_next_image_basename": get_next_image_basename,
        }
    )

    config = {
        "server_url": server_url,
        "workspace": server.config.workspace,
        "annotation_service_id": service["id"],
        "token": token,
    }
    encoded_plugin_url = urllib.parse.quote(plugin_url, safe="/:")
    encoded_config = urllib.parse.quote(
        json.dumps(config, separators=(",", ":")), safe="/"
    )
    annotator_url = (
        f"https://imjoy.io/lite?plugin={encoded_plugin_url}&config={encoded_config}"
    )
    return server, annotator_url
