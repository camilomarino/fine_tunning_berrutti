.PHONY: verify

verify:
	python -m unittest discover -s tests -p 'test_*.py'
	python -m compileall -q experiments
	ruff check experiments tests
	ruff format --check experiments tests
