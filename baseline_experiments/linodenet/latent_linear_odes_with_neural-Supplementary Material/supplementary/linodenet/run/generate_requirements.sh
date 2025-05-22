#!/usr/bin/env bash
poetry export --without-hashes --output requirements.txt
poetry export  --without-hashes --output requirements-dev.txt --with dev
