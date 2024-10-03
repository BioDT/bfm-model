.PHONY: setup_venv compile_dependencies install_dependencies test clean

# set the name of the venv
VENV = venv

# creates a venv, if it doesn't exist locally
setup_venv:
	@echo "Creating virtual environment."
	if [ -d $(VENV) ]; then \
		echo "Virtual environment already exists."; \
		exit 1; \
	fi

	python -m venv $(VENV)
	@echo "Activating virtual environment."
	. $(VENV)/bin/activate; pip install --upgrade pip

# create a requirements.txt file, from the dependencies in the pyproject.toml file
compile_dependencies: setup_venv
	@echo "Compiling dependencies in a requirements.txt ."
	pip install poetry poetry-plugin-export
	poetry config warnings.export false
	poetry export -f requirements.txt --output requirements.txt

# install dependencies from requirements.txt
install_dependencies:
	@echo "Installing dependencies."
	. $(VENV)/bin/activate; pip install -r requirements.txt

# running the available tests
test:
	@echo "Running tests."
	. $(VENV)/bin/activate; python -m unittest discover -s tests

# remove venv
clean:
	@echo "Cleaning up."
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
