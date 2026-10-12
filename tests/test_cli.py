import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import call, patch

import numpy as np

from maskngly.cli import analyze, main


class AnalysisTests(unittest.TestCase):
    def test_unsupported_sequences_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / "proteins.fasta"
            fasta.write_text(
                ">unknown_first\nANxTAA\n"
                ">valid_first\nANVTAA\n"
                ">ambiguous\nANBZAA\n"
                ">valid_last\nnataaa\n"
                ">unknown_last\nANXTAA\n"
            )
            stderr = io.StringIO()
            with patch("maskngly.cli.InfluenceModel") as model, redirect_stderr(stderr):
                model.return_value.matrix.return_value = np.zeros(
                    (6, 6), dtype=np.float32
                )
                analyze(fasta, root / "scores.csv")
                self.assertEqual(
                    model.return_value.matrix.call_args_list,
                    [call("ANVTAA"), call("NATAAA")],
                )
            with (root / "scores.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["protein_id"] for row in rows], ["valid_first", "valid_last"]
            )
            warnings = stderr.getvalue()
            for identifier in ("unknown_first", "ambiguous", "unknown_last"):
                self.assertIn(f"warning: skipping protein '{identifier}'", warnings)
            self.assertIn("unsupported residues ['X']", warnings)
            self.assertIn("unsupported residues ['B', 'Z']", warnings)

    def test_all_unsupported_sequences_exit_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / "proteins.fasta"
            fasta.write_text(">unknown\nANxTAA\n>ambiguous\nANBTAA\n")
            output = root / "scores.csv"
            stderr = io.StringIO()
            with (
                patch("maskngly.cli.InfluenceModel") as model,
                patch(
                    "sys.argv",
                    ["maskngly", "--input", str(fasta), "--output", str(output)],
                ),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as exit_result,
            ):
                main()
            self.assertEqual(exit_result.exception.code, 1)
            model.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(stderr.getvalue().count("warning: skipping protein"), 2)
            self.assertIn("no valid protein sequences", stderr.getvalue())

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
                self.assertTrue(
                    all(row["png_path"] == row["npy_path"] == "" for row in rows)
                )
                analyze(fasta, root / "scores.csv", save_npy=True, save_png=True)
                self.assertEqual(model.return_value.matrix.call_count, 2)
            np.testing.assert_array_equal(
                np.load(root / "scores/matrices/000000001.npy"), matrix
            )
            for number in (1, 2):
                self.assertTrue(
                    (root / f"scores/profiles/{number:09d}.png").exists()
                )
            with (root / "scores.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["npy_path"] for row in rows],
                ["scores/matrices/000000001.npy"] * 2,
            )
            self.assertEqual(
                [row["png_path"] for row in rows],
                [f"scores/profiles/{number:09d}.png" for number in (1, 2)],
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
            self.assertEqual(row["npy_path"], "scores/matrices/000000001.npy")
            self.assertTrue((root / row["npy_path"]).exists())

    def test_punctuated_identifiers_and_numeric_exports(self) -> None:
        identifiers = (
            "PHI_I1CCV9_PHI:4843_Fob2",
            "PHI:11037_NRPS14_(VM1G_03342)",
            "../../PHI:123/[a+b]<c>",
            "PHI|123",
            "PHI_123",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / "proteins.fasta"
            fasta.write_text("".join(f">{name}\nANVTAA\n" for name in identifiers))
            output = root / "results" / "my-analysis.csv"
            matrix = np.arange(36, dtype=np.float32).reshape(6, 6)
            with patch("maskngly.cli.InfluenceModel") as model:
                model.return_value.matrix.return_value = matrix
                analyze(fasta, output, save_npy=True, save_png=True)
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["protein_id"] for row in rows], list(identifiers))
            for number, row in enumerate(rows, 1):
                self.assertEqual(
                    row["npy_path"], f"my-analysis/matrices/{number:09d}.npy"
                )
                self.assertEqual(
                    row["png_path"], f"my-analysis/profiles/{number:09d}.png"
                )
                np.testing.assert_array_equal(
                    np.load(output.parent / row["npy_path"]), matrix
                )
                self.assertTrue((output.parent / row["png_path"]).exists())

    def test_empty_identifier_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / "proteins.fasta"
            fasta.write_text(">\nANVTAA\n")
            with self.assertRaisesRegex(ValueError, "no protein identifier"):
                analyze(fasta, root / "scores.csv")


if __name__ == "__main__":
    unittest.main()
