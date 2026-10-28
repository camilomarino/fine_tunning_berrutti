.PHONY: verify

verify:
	python -m unittest discover -s tests -t . -p 'test_*.py'
	python -m compileall -q data archive_berrutti finetune evaluation reporting experiments
	ruff check data archive_berrutti finetune evaluation reporting experiments tests
	ruff format --check data archive_berrutti finetune evaluation reporting experiments tests
