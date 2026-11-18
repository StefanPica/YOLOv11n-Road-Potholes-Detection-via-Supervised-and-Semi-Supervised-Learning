import unittest

import numpy as np
from PIL import Image

from evaluate_corruption_robustness import (
    apply_corruption,
    select_affected_indices,
    validate_test_scope,
)


class CorruptionRobustnessTests(unittest.TestCase):
    def test_coverage_subsets_are_deterministic_and_nested(self) -> None:
        coverages = (0.1, 0.25, 0.5, 1.0)
        first = select_affected_indices(394, coverages, seed=42)
        second = select_affected_indices(394, coverages, seed=42)

        self.assertEqual(first, second)
        self.assertEqual([len(first[value]) for value in coverages], [40, 99, 197, 394])
        self.assertTrue(first[0.1] < first[0.25] < first[0.5] < first[1.0])

    def test_corruptions_preserve_dimensions_and_leave_source_unchanged(self) -> None:
        y_coordinates, x_coordinates = np.indices((32, 32))
        pixels = (((x_coordinates // 4 + y_coordinates // 4) % 2) * 255).astype(np.uint8)
        image = Image.fromarray(np.stack([pixels, pixels, pixels], axis=-1))
        original = image.copy()
        cases = (
            ("low_light", 0.6),
            ("blur", 1.0),
            ("jpeg", 40),
            ("gaussian_noise", 15),
        )

        for corruption, parameter in cases:
            with self.subTest(corruption=corruption):
                result = apply_corruption(image, corruption, parameter, seed=42)
                self.assertEqual(result.size, image.size)
                self.assertFalse(np.array_equal(np.asarray(result), np.asarray(original)))
                self.assertTrue(np.array_equal(np.asarray(image), np.asarray(original)))

    def test_gaussian_noise_is_repeatable_for_the_same_seed(self) -> None:
        image = Image.new("RGB", (32, 32), color=(128, 128, 128))
        first = apply_corruption(image, "gaussian_noise", 15, seed=123)
        second = apply_corruption(image, "gaussian_noise", 15, seed=123)

        self.assertTrue(np.array_equal(np.asarray(first), np.asarray(second)))

    def test_test_split_allows_only_one_frozen_condition(self) -> None:
        with self.assertRaises(SystemExit):
            validate_test_scope("test", ["blur", "jpeg"], ["mild"], [0.25])

        validate_test_scope("test", ["blur"], ["moderate"], [0.25])
        validate_test_scope("valid", ["blur", "jpeg"], ["mild", "strong"], [0.1, 1.0])


if __name__ == "__main__":
    unittest.main()
