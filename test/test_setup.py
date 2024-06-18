import os
import unittest


class TestSetup(unittest.TestCase):
    """
    Tests the setup file (pyproject.toml) of this project.
    """

    def test_project_setup(self):
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        pyproject_path = os.path.join(current_script_dir, "..", "pyproject.toml")

        with open(pyproject_path, "r") as file:
            content = file.read()
            self.assertIn('name = "biodiversity_foundation_model"', content)
            self.assertIn("version = ", content)
            self.assertIn('authors = ["Sebastian Gribincea <sebastian.gribincea@tno.nl>"]', content)


if __name__ == "__main__":
    unittest.main()
