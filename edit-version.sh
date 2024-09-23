#!/bin/bash

if [ -z "$1" ]; then
    echo "Must supply a version to set"
    exit 1
fi

sed -i "s/^version = .*/version = \"$1\"/" pyproject.toml || exit $?
sed -i "s/^base_version = .*/base_version = \"$1\"/" corrscope/version.py || exit $?

# https://blog.danslimmon.com/2019/07/15/do-nothing-scripting-the-key-to-gradual-automation/
echo "Update CHANGELOG.md?"
read -rp "- Press Enter to continue: "
