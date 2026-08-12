import tempfile
import unittest
from pathlib import Path

import numpy as np
from tifffile import imwrite

from correction_server import (
    discover_image_pairs,
    features_to_mask,
    mask_to_paths,
)


def feature_collection(paths, geometry_type="LineString"):
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": geometry_type,
                    "coordinates": [path] if geometry_type == "Polygon" else path,
                },
                "properties": {},
            }
            for path in paths
        ],
    }


class MaskConversionTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((12, 14), dtype=np.uint16)
        self.mask[0:7, 0:6] = 1  # touches two image edges
        self.mask[2:4, 2:4] = 0  # hole that is not exposed as an editable path
        self.mask[6:11, 8:13] = 2

    def test_noop_round_trip_is_pixel_exact(self):
        result = features_to_mask(
            feature_collection(mask_to_paths(self.mask)), self.mask
        )
        np.testing.assert_array_equal(result, self.mask)

    def test_polygon_round_trip_is_pixel_exact(self):
        features = feature_collection(mask_to_paths(self.mask), "Polygon")
        result = features_to_mask(features, self.mask)
        np.testing.assert_array_equal(result, self.mask)

    def test_missing_feature_deletes_instance(self):
        paths = mask_to_paths(self.mask)
        result = features_to_mask(feature_collection(paths[:1]), self.mask)
        self.assertTrue(np.any(result == 1))
        self.assertFalse(np.any(result == 2))

    def test_new_feature_gets_a_new_label(self):
        empty = np.zeros((12, 14), dtype=np.uint16)
        square = [[2, 2], [5, 2], [5, 5], [2, 5], [2, 2]]
        result = features_to_mask(feature_collection([square]), empty)
        self.assertEqual(int(result.max()), 1)
        self.assertGreater(np.count_nonzero(result), 0)


class DiscoveryTests(unittest.TestCase):
    def test_discovers_pairs_and_ignores_corrected_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = np.zeros((4, 4), dtype=np.uint8)
            imwrite(root / "sample.tif", image)
            imwrite(root / "sample_masks.tif", image)
            imwrite(root / "sample_masks_corrected.tif", image)

            pairs = discover_image_pairs(root)

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["basename"], "sample")
            self.assertEqual(
                Path(pairs[0]["corrected"]).name, "sample_masks_corrected.tif"
            )


if __name__ == "__main__":
    unittest.main()
