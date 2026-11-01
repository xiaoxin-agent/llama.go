# OpenClaw Team

This repository is being maintained locally by an OpenClaw-led multi-agent workflow.

## Roles

- Lead Maintainer: owns the roadmap, state, integration, and user-facing updates.
- Architecture Explorer: reviews module boundaries and inference-runtime design.
- Quality/CI Explorer: reviews tests, CI, docs, reproducibility, and validation gates.
- Bug/Risk Explorer: looks for defects, misleading interfaces, and near-term risks.
- Worker Agents: implement narrow file-scoped tasks with disjoint ownership.
- Reviewer Agent: reviews diffs and regression risk before work is reported done.

## Operating Loop

1. Check the active OpenClaw goal and this repository's git state.
2. Fetch upstream state when network and permissions allow.
3. Pick one small sprint objective.
4. Use subagents only for bounded independent subtasks.
5. Integrate locally and run available validation.
6. Record any missing tools or unverified assumptions.
7. Ask before pushing branches, opening PRs, or making other external changes.

## Initial Sprint

Sprint 0: make project status trustworthy.

- Document the branch that contains real code.
- Fix stale project status documentation.
- Align quantization support docs with implementation.
- Verify row-stride support against dequantization support.
- Produce file-scoped next tasks for inference-core work.

## Environment Notes

- `git` is available in the initial OpenClaw container.
- `go` was not available during the first setup check, so Go tests could not be run locally yet.
- `rg` was not available during the first setup check; use `find` or `grep` when needed.
