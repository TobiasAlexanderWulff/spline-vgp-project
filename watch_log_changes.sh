#!/usr/bin/env bash

inotifywait -m -r -e modify logs |
while read -r directory events filename; do
	if [[ "$filename" == *.log ]]; then
		dir_name=$(basename "$directory")
		tail -n 1 "$directory$filename" | sed "s/^/[$dir_name] /"
	fi
done
