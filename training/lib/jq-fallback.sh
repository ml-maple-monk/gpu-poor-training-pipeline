#!/bin/bash
# training/lib/jq-fallback.sh — portable JSON extraction helper
#
# Usage: source training/lib/jq-fallback.sh
#
# Defines function `jqx` that accepts a jq-style path and reads JSON from stdin.
# Uses real `jq` if available; falls back to Python3.
#
# Supported path forms (subset of jq):
#   .key              -> object field
#   .[0]              -> array index
#   .[0].key          -> combined
#   .[]               -> array iteration (returns newline-separated values)
#
# Examples:
#   echo '{"login":"alice"}' | jqx '.login'          -> alice
#   echo '[{"run_name":"foo"}]' | jqx '.[0].run_name' -> foo
#   echo '{"a":{"b":"x"}}' | jqx '.a.b'             -> x

jqx() {
    local path="$1"
    if command -v jq &>/dev/null; then
        jq -r "$path"
    else
        python3 -c "
import json, sys, re

def resolve(data, path):
    # Strip leading dot
    path = path.lstrip('.')
    if not path:
        return data
    # Split on dots but keep bracket notation intact
    parts = re.split(r'\.(?![^\[]*\])', path)
    for part in parts:
        if not part:
            continue
        # Array index: [N]
        m = re.match(r'^\[(-?\d+)\]$', part)
        if m:
            data = data[int(m.group(1))]
        # Array iterate: []
        elif part == '[]':
            for item in data:
                print(item if not isinstance(item, (dict, list)) else json.dumps(item))
            return None
        # Object key with optional brackets: key or [\"key\"]
        else:
            key = re.sub(r'^\[\"(.*)\"\]$', r'\1', part)
            data = data[key]
    return data

data = json.load(sys.stdin)
result = resolve(data, '$path')
if result is None:
    pass
elif isinstance(result, (dict, list)):
    print(json.dumps(result))
elif isinstance(result, bool):
    print('true' if result else 'false')
else:
    print(result)
"
    fi
}

# jqx_file <file> <path> — read from file instead of stdin
jqx_file() {
    local file="$1"
    local path="$2"
    jqx "$path" < "$file"
}
