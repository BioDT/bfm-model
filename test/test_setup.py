"""
Run from root folder: `python -m test.test_setup`
"""

import os
import unittest


class TestSetup(unittest.TestCase):
    """
    Tests the setup file (pyproject.toml) of this project.
    """

    def test_project_setup(self):
        import bfm_model
        from bfm_model.bfm import utils


if __name__ == "__main__":
    unittest.main()
