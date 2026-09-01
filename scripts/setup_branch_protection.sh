#!/usr/bin/env bash
set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: GitHub CLI (gh) is required to configure branch protection." >&2
  exit 1
fi

repo="${1:-$(gh repo view --json nameWithOwner --jq '.nameWithOwner' 2>/dev/null || true)}"

if [[ -z "${repo}" ]]; then
  echo "Usage: $0 [owner/repo]" >&2
  echo "Example: $0 owner/repo" >&2
  exit 1
fi

# Keep review requirements at 0 for now; this can be raised later if the contributor base grows.
branch_protection_payload=$(cat <<JSON
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["CI"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": false,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 0
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": false,
  "lock_branch": false,
  "allow_fork_syncing": true
}
JSON
)

gh api --method PUT "repos/${repo}/branches/main/protection" --input - <<<"${branch_protection_payload}" >/dev/null

echo "Updated branch protection for ${repo}:main (PR required, CI required, force pushes and deletions disabled)."
