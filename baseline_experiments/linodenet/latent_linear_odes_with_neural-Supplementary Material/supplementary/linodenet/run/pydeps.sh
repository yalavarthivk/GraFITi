#!/usr/bin/env bash
# execute pydeps and create dependency graph
pydeps src/linodenet/ --cluster --rankdir BT --max-bacon=1
