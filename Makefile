.PHONY: pull_and_start

pull_and_start:
	git pull
	docker compose build
	docker compose run --rm -e POPULATE_DB=1 --entrypoint "python seed.py" app
	docker compose up
