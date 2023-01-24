#!/usr/bin/env bash

# search for triple quotes that are not raw.
regex='(?<=[^r])"""(?=[^\n\s])'
echo "$regex"
