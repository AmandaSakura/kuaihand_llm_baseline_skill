#!/usr/bin/env bash
set -euo pipefail

REPO_SLUG="${REPO_SLUG:-AmandaSakura/kuaihand_llm_baseline_skill}"
REPO_URL="https://github.com/${REPO_SLUG}"

if ! command -v gh >/dev/null 2>&1; then
  echo "[star-repo] GitHub CLI 'gh' is not installed."
  echo "[star-repo] To star this repository manually, open: ${REPO_URL}"
  exit 0
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "[star-repo] GitHub CLI is installed but not authenticated."
  echo "[star-repo] Run 'gh auth login', or star this repository manually: ${REPO_URL}"
  exit 0
fi

if gh repo star "${REPO_SLUG}" >/dev/null 2>&1; then
  echo "[star-repo] Starred ${REPO_SLUG}."
else
  echo "[star-repo] Could not star ${REPO_SLUG}; you can do it manually: ${REPO_URL}"
fi
