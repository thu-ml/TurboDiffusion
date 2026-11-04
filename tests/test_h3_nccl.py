"""NCCL checks query the actual mapped binary, not PyTorch's compile-time version."""
import ctypes
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from turbodiffusion.h3.nccl import runtime_nccl


class H3NCCLTest(unittest.TestCase):
    def test_actual_runtime_version_and_deduplicated_library_path(self):
        maps = "0000-1000 r--p 000000 00:00 1 /tmp/libnccl.so.2\n" * 2

        def get_version(pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 22809
            return 0

        library = types.SimpleNamespace(ncclGetVersion=Mock(side_effect=get_version))
        with patch.object(Path, "read_text", return_value=maps), patch(
            "turbodiffusion.h3.nccl.ctypes.CDLL", return_value=library
        ):
            version, path = runtime_nccl()
        self.assertEqual(version, 22809)
        self.assertEqual(path, Path("/tmp/libnccl.so.2"))

    def test_missing_or_multiple_runtime_libraries_are_rejected(self):
        for maps in ("", "0000 r--p 0 00:00 1 /one/libnccl.so.2\n"
                           "0000 r--p 0 00:00 1 /two/libnccl.so.2\n"):
            with self.subTest(maps=maps), patch.object(Path, "read_text", return_value=maps):
                with self.assertRaisesRegex(RuntimeError, "expected one"):
                    runtime_nccl()

    def test_nccl_api_errors_are_reported(self):
        maps = "0000 r--p 0 00:00 1 /tmp/libnccl.so.2\n"
        library = types.SimpleNamespace(ncclGetVersion=Mock(return_value=3))
        with patch.object(Path, "read_text", return_value=maps), patch(
            "turbodiffusion.h3.nccl.ctypes.CDLL", return_value=library
        ):
            with self.assertRaisesRegex(RuntimeError, "status 3"):
                runtime_nccl()


if __name__ == "__main__":
    unittest.main()
