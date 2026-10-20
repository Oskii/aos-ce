#!/usr/bin/env python3
"""Regression tests for atomic channel-generation transactions."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("channel_transaction.py")
SPEC = importlib.util.spec_from_file_location("channel_transaction", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load channel transaction module")
TRANSACTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRANSACTION)


class TransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pointer = self.root / "channel.toml"
        self.bundle = self.root / "channel.toml.sigstore.json"
        self.output = self.root / "channel-stable-7.transaction.json"
        self.pointer.write_bytes(b"signed channel pointer\n")
        self.bundle.write_bytes(b'{"verificationMaterial":{}}\n')

    def pack(self) -> None:
        args = type(
            "Args",
            (),
            {
                "channel": "stable",
                "generation": 7,
                "pointer": self.pointer,
                "bundle": self.bundle,
                "output": self.output,
            },
        )()
        TRANSACTION.pack(args)

    def unpack(self, *, channel: str = "stable", generation: int = 7) -> Path:
        destination = self.root / f"unpacked-{channel}-{generation}"
        destination.mkdir()
        args = type(
            "Args",
            (),
            {
                "channel": channel,
                "generation": generation,
                "input": self.output,
                "output_dir": destination,
            },
        )()
        TRANSACTION.unpack(args)
        return destination

    def test_round_trip_preserves_exact_signed_pair(self) -> None:
        self.pack()
        destination = self.unpack()
        self.assertEqual((destination / self.pointer.name).read_bytes(), self.pointer.read_bytes())
        self.assertEqual((destination / self.bundle.name).read_bytes(), self.bundle.read_bytes())

    def test_cross_channel_reuse_is_rejected(self) -> None:
        self.pack()
        with self.assertRaisesRegex(ValueError, "channel does not match"):
            self.unpack(channel="dev")

    def test_cross_generation_reuse_is_rejected(self) -> None:
        self.pack()
        with self.assertRaisesRegex(ValueError, "generation does not match"):
            self.unpack(generation=8)

    def test_unknown_transaction_field_is_rejected(self) -> None:
        self.pack()
        value = json.loads(self.output.read_bytes())
        value["surprise"] = True
        self.output.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exact schema"):
            TRANSACTION.load_transaction(self.output)

    def test_invalid_base64_is_rejected(self) -> None:
        self.pack()
        value = json.loads(self.output.read_bytes())
        value["channel-toml-base64"] = "***"
        self.output.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "malformed"):
            self.unpack()

    def test_symlink_input_is_rejected(self) -> None:
        link = self.root / "pointer-link"
        link.symlink_to(self.pointer)
        args = type(
            "Args",
            (),
            {
                "channel": "stable",
                "generation": 7,
                "pointer": link,
                "bundle": self.bundle,
                "output": self.output,
            },
        )()
        with self.assertRaisesRegex(ValueError, "regular file"):
            TRANSACTION.pack(args)

    def test_substituted_output_symlink_is_not_followed(self) -> None:
        victim = self.root / "victim"
        victim.write_bytes(b"keep")
        self.output.symlink_to(victim)
        original = TRANSACTION.require

        def require(condition: bool, message: str) -> None:
            if message == "transaction output already exists":
                return
            original(condition, message)

        TRANSACTION.require = require
        try:
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.pack()
        finally:
            TRANSACTION.require = original
        self.assertEqual(victim.read_bytes(), b"keep")

    def test_substituted_output_directory_is_not_followed(self) -> None:
        self.pack()
        other = self.root / "other"
        other.mkdir()
        link = self.root / "via-link"
        link.symlink_to(other, target_is_directory=True)
        original = TRANSACTION.require

        def require(condition: bool, message: str) -> None:
            if message in {
                "output directory must be a real directory",
                "output directory must be empty",
            }:
                return
            original(condition, message)

        TRANSACTION.require = require
        args = type(
            "Args",
            (),
            {
                "channel": "stable",
                "generation": 7,
                "input": self.output,
                "output_dir": link,
            },
        )()
        try:
            with self.assertRaisesRegex(ValueError, "real directory"):
                TRANSACTION.unpack(args)
        finally:
            TRANSACTION.require = original
        self.assertFalse((other / "channel.toml").exists())
        self.assertFalse((other / "channel.toml.sigstore.json").exists())


if __name__ == "__main__":
    unittest.main()
