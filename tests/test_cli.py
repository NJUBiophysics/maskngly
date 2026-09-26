import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from maskngly.cli import analyze


class AnalysisTests(unittest.TestCase):
    def test_csv_and_optional_exports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / "proteins.fasta"
            fasta.write_text(">example Sites: 2,7\nANVTAANVTAA\n")
            matrix = np.zeros((11, 11), dtype=np.float32)
            matrix[3, 1] = 10
            matrix[8, 1] = 10
            with patch("maskngly.cli.InfluenceModel") as model:
                model.return_value.matrix.return_value = matrix
                analyze(fasta, root / "scores.csv")
                self.assertFalse((root / "scores").exists())
                with (root / "scores.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([row["score"] for row in rows], ["1.0", "0.0"])
                analyze(fasta, root / "scores.csv", save_npy=True, save_png=True)
                self.assertEqual(model.return_value.matrix.call_count, 2)
            np.testing.assert_array_equal(
                np.load(root / "scores/matrices/example.npy"), matrix
            )
            for position in (2, 7):
                self.assertTrue(
                    (root / f"scores/profiles/example_{position}.png").exists()
                )

    def test_long_sequence_positions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / "proteins.fasta"
            fasta.write_text(">long Sites: 1090\n" + "A" * 1100 + "\n")
            matrix = np.zeros((1022, 1022), dtype=np.float32)
            # Window begins at protein position 79; N maps to local position 1012.
            matrix[1013, 1011] = 10
            with patch("maskngly.cli.InfluenceModel") as model:
                model.return_value.matrix.return_value = matrix
                analyze(fasta, root / "scores.csv", save_npy=True)
            with (root / "scores.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                (row["position"], row["window_start"], row["score"]),
                ("1090", "79", "1.0"),
            )
            self.assertTrue((root / "scores/matrices/long_79.npy").exists())


if __name__ == "__main__":
    unittest.main()
