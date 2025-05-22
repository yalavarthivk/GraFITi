#!/usr/bin/env bash
# execute pydeps and create dependency graph
pydeps src/tsdm/ --cluster --rankdir BT --max-bacon=1
