"""
Simplified package installation and functionality tests for tensorrt-edgellm.
"""

import subprocess
from typing import Dict

import pytest


class PackageTester:
    """Simplified package tester"""

    def test_package_import(self) -> Dict[str, any]:
        """Test package import and basic functionality"""
        try:
            import tensorrt_edgellm

            # Check version
            version = getattr(tensorrt_edgellm, '__version__', None)
            if not version:
                return {"success": False, "error": "Package version not found"}

            # Check required functions
            required_functions = [
                "quantize_and_save_llm", "llm_export", "visual_export"
            ]
            for func_name in required_functions:
                if not hasattr(tensorrt_edgellm, func_name):
                    return {
                        "success": False,
                        "error": f"Missing function: {func_name}"
                    }

            return {
                "success": True,
                "message": f"Package imported successfully, version: {version}"
            }

        except ImportError as e:
            return {"success": False, "error": f"Import error: {str(e)}"}

    def test_command_line_tools(self) -> Dict[str, any]:
        """Test command-line tools availability"""
        tools = [
            "tensorrt-edgellm-quantize-llm", "tensorrt-edgellm-export-llm",
            "tensorrt-edgellm-export-visual"
        ]

        for tool in tools:
            try:
                result = subprocess.run([tool, "--help"],
                                        capture_output=True,
                                        text=True,
                                        timeout=10)
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Tool {tool} failed: {result.stderr}"
                    }
            except FileNotFoundError:
                return {
                    "success": False,
                    "error": f"Tool {tool} not found in PATH"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error testing {tool}: {str(e)}"
                }

        return {"success": True, "message": "All command-line tools available"}


class TestPackage:
    """Package installation and functionality tests"""

    @pytest.fixture(scope="class")
    def package_tester(self):
        return PackageTester()

    def test_package_import(self, package_tester):
        """Test package import"""
        result = package_tester.test_package_import()
        assert result[
            "success"], f"Package import failed: {result.get('error', 'Unknown error')}"

    def test_command_line_tools(self, package_tester):
        """Test command-line tools"""
        result = package_tester.test_command_line_tools()
        assert result[
            "success"], f"Command-line tools test failed: {result.get('error', 'Unknown error')}"
