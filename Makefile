.PHONY: pull_and_start clear_database

pull_and_start:
	git pull
	docker compose build
	docker compose run --rm -e POPULATE_DB=1 --entrypoint "python seed.py" app
	docker compose up

clear_database:
	docker compose down
	docker volume rm book-app_app-data || true
