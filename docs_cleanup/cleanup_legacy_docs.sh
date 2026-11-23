#!/usr/bin/env bash
# Remove only known, historical root-level Markdown reports.
# Defaults to a dry run. Run from the repository root:
#   bash cleanup_legacy_docs.sh
#   bash cleanup_legacy_docs.sh --apply
set -euo pipefail
if [[ ! -d .git ]]; then
  echo 'Run this script from the Git repository root.' >&2
  exit 2
fi
if [[ "${1:-}" != '' && "${1:-}" != '--apply' ]]; then
  echo 'Usage: bash cleanup_legacy_docs.sh [--apply]' >&2
  exit 2
fi
legacy=(
  DEVELOPMENT_STATUS.md
  LIPF6_REVERSIBLE_LANGEVIN_NOTES.md
  PAPER_ENERGY_ACCOUNTING_PHASE4.md
  PAPER_INTEGRATION_AUDIT_PHASE2.md
  PAPER_REPRODUCTION_STATUS.md
  PAPER_SOURCE_CODE_AUDIT_PHASE5.md
  PAPER_SPECIFICATION_PHASE3.md
  PHASE6_PAIRED_NVT_PILOT.md
  PHASE7_FIXED_CONFIGURATION_ENERGY_AUDIT.md
  REFACTOR_NOTES.md
)
found=()
for file in "${legacy[@]}"; do
  if [[ -f "$file" ]] && git ls-files --error-unmatch -- "$file" >/dev/null 2>&1; then
    found+=("$file")
  fi
done
if (( ${#found[@]} == 0 )); then
  echo 'No tracked root-level legacy reports found.'
  exit 0
fi
printf 'Historical root-level Markdown files identified:\n'
printf '  %s\n' "${found[@]}"
if [[ "${1:-}" == '--apply' ]]; then
  git rm -- "${found[@]}"
  echo 'Staged removals. Review with: git diff --cached --stat'
else
  echo 'DRY RUN: no files changed. To stage these removals, rerun with --apply.'
fi
# Validation records in validation/*.md are intentionally preserved for review.

