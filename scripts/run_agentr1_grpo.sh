#!/usr/bin/env bash
set -euo pipefail

exec python3 -m deep_research_rl training launch "$@"
