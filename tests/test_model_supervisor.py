#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "ds4_model_supervisor", ROOT / "scripts" / "ds4_model_supervisor.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ModelSupervisorProfilesTest(unittest.TestCase):
    def setUp(self):
        self.profiles = MODULE.model_profiles()
        self.supervisor = MODULE.ModelSupervisor(
            "100.109.208.12", 8000, "test-token", 10)

    def test_profile_catalog_matches_pi_gui_contract(self):
        self.assertEqual(set(self.profiles), {
            "deepseek-v4", "deepseek-v4-vision", "deepseek-v41",
            "glm53", "glm53-vision", "qwen-next", "qwen-next-vision",
            "qwen35", "qwen27-q8", "qwen27-q4",
        })
        self.assertEqual(self.profiles["deepseek-v4"].context, 262144)
        self.assertEqual(self.profiles["glm53"].context, 262144)
        self.assertEqual(self.profiles["qwen-next"].context, 262144)
        self.assertEqual(self.profiles["qwen35"].context, 262144)
        self.assertEqual(self.profiles["qwen27-q8"].context, 131072)
        self.assertEqual(self.profiles["qwen27-q4"].context, 131072)

    def test_commands_bind_only_the_requested_tailscale_address(self):
        command = self.supervisor._worker_command(self.profiles["glm53"])
        self.assertEqual(command[command.index("--host") + 1], "100.109.208.12")
        self.assertEqual(command[command.index("--port") + 1], "8000")
        self.assertIn("--mtp", command)
        self.assertIn("--mtp-exact-sampling", command)

    def test_small_qwen_omits_unsupported_disk_kv(self):
        command = self.supervisor._worker_command(self.profiles["qwen27-q4"])
        self.assertNotIn("--kv-disk-dir", command)
        self.assertIsNotNone(self.profiles["qwen27-q4"].mtp_head)

    def test_every_required_artifact_is_installed(self):
        for name, profile in self.profiles.items():
            with self.subTest(profile=name):
                for path in self.supervisor._dependencies(profile):
                    self.assertTrue(Path(path).is_file(), path)


if __name__ == "__main__":
    unittest.main()
