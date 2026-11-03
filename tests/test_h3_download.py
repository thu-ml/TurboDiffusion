import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from turbodiffusion.inference import download_h3


class H3DownloadTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = (
            "adaln/adaln.safetensors", "adaln/adaln.json", "transformer/model.safetensors",
            "README.md", "LICENSE", "NOTICE",
        )
        self.records = {}
        for name in self.names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = name.encode()
            path.write_bytes(data)
            self.records[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.write_manifest()

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps({"files": self.records}))

    def test_full_download_pins_revisions_and_selects_every_component(self):
        with patch.object(download_h3, "snapshot_download") as snapshot:
            download_h3.download(self.root, components_dir=self.root / "components")
        first, second = snapshot.call_args_list
        self.assertEqual(first.kwargs["revision"], download_h3.DMD_REVISION)
        self.assertEqual(first.kwargs["allow_patterns"], list(download_h3.DMD_PATTERNS))
        self.assertEqual(second.kwargs["revision"], download_h3.COMPONENT_REVISION)
        self.assertEqual(second.kwargs["allow_patterns"], list(download_h3.COMPONENT_PATTERNS))

    def test_refresh_downloads_both_adaln_files_and_checks_their_hashes(self):
        (self.root / "transformer/model.safetensors").unlink()
        with patch.object(download_h3, "snapshot_download") as snapshot:
            download_h3.download(self.root, adaln_only=True)
        self.assertEqual(snapshot.call_args.kwargs["allow_patterns"], list(download_h3.ADALN_PATTERNS))
        self.assertEqual(download_h3.verify_manifest(self.root, download_h3.ADALN_PATTERNS), 2)
        table = self.root / "adaln/adaln.safetensors"
        table.write_bytes(b"x" * table.stat().st_size)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            download_h3.verify_manifest(self.root, download_h3.ADALN_PATTERNS)

    def test_missing_files_and_unmatched_patterns_are_rejected(self):
        (self.root / "adaln/adaln.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "missing or incorrect-size"):
            download_h3.verify_manifest(self.root, download_h3.ADALN_PATTERNS)
        with self.assertRaisesRegex(ValueError, "no files matching"):
            download_h3.verify_manifest(self.root, ("unknown/*",))

    def test_unsafe_manifest_paths_are_rejected(self):
        self.records["../outside"] = {"size": 1, "sha256": ""}
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "unsafe manifest path"):
            download_h3.verify_manifest(self.root, ("../*",))


if __name__ == "__main__":
    unittest.main()
