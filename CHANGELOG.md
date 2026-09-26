# Changelog

This file records some changes to the SALT artifact package. Version 1.1 is the first public Zenodo release. The v1.0.x-ae versions were review snapshots used during PACT 2026 artifact evaluation and were not published as Zenodo releases.

## Unreleased

- Expanded the Chameleon/Trovi workflow with site-aware reservation, recovery, and cleanup guidance.
- Documented PMC event semantics and added temporary sibling-CPU isolation with automatic restoration for pinned collections.
- Recorded a future Cachegrind Runner refactor from its SQLite connection pool to a dedicated database-writer thread.

## v1.1

- Prepared the post-evaluation archival release.
- Added publication-ready PDF figures with embedded TrueType fonts.
- Reported Figure 4's mean absolute error as MAPE while retaining the equivalent MARE value in the documentation.
- Expanded release, reproduction, and generated-output documentation.

## Artifact-evaluation review snapshots

### v1.0.2-ae — 2026-08-23

- Improved documentation and release packaging.
- Made timing provenance work when Git metadata is absent from an archive.

### v1.0.1-ae — 2026-08-13

- Polished the PACT 2026 artifact packaging and Docker configuration.
