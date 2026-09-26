# SALT on Chameleon Cloud

`chameleon-salt.ipynb` is the evaluator-facing Trovi notebook. It performs the
complete resource lifecycle:

1. Checks every `compute_skylake` calendar at the explicitly selected site
   (`CHI@TACC` by default, or `CHI@UC`), selects the specific node with the
   earliest full interval, and reserves it with one floating IP.
2. Launches `CC-Ubuntu24.04`, waits for SSH, and records the allocated CPU and
   system configuration.
3. Runs `chameleon/setup-node.sh` to install compiler/build dependencies,
   Linux performance-counter tools, and Docker Engine.
4. Packages the required source files already present in the Trovi workspace,
   verifies the upload by SHA-256, builds the Docker image, and launches its
   test and experiment entry points. If the optional v1.1 release archive is
   present, the notebook verifies and uses that instead.
5. Downloads compressed result directories back into the Trovi/Jupyter
   workspace before deleting the server and lease.

The reservation profiles are deliberately separate so a short rerun cannot
silently reconnect to an incompatible long lease:

- `smoke`: 4 hours for reduced end-to-end validation;
- `full`: 16 hours by default for the complete evaluation;
- `debug`: 48 hours for developer troubleshooting and either experiment path.

The 48-hour profile is not the reviewer default: a longer interval is harder to
schedule, and Chameleon charges the reserved duration until teardown. Resource
names include both the site and profile. Change only `SITE` and `RUN_PROFILE` in
Section 1 when a different choice is needed.

The notebook offers three experiment paths:

- `smoke`: reduced end-to-end validation of all three workflows;
- `reproduce`: the complete paper workflow (approximately 8--9 hours after the
  image is built);
- optional fresh PMCs: Linux generic L1D/read/miss collection on the allocated
  node.

The normal `smoke` and `reproduce` commands use the checked-in Intel Core
i7-7700 measurements for Figure 4. This is the portable reproduction path. A
Xeon Gold 6126 is a Skylake-SP processor and can run the generic Linux
L1D/read/miss event. On this validated platform, that selector maps to
`L1D.REPLACEMENT`, not the raw `MEM_LOAD_RETIRED.L1_MISS` event. Its fresh
replacement totals should not be presented as replacements for the paper's
i7-7700 measurements. The notebook stores fresh measurements under a separately
named result directory.

For the optional fresh-PMC path, the notebook temporarily offlines only the
sibling thread on the selected measurement core and restores it after the run;
it leaves SMT enabled on every other core.

## Trovi contents

The normal GitHub-to-Trovi import includes everything required. Keep the
repository layout, including at least:

```text
chameleon-salt.ipynb
chameleon/setup-node.sh
Dockerfile
Cargo.toml
Cargo.lock
analyzer/
artifact/
benchmarks/
cachegrind-runner/
denning/
raffine/
salt_vs_hw_misses_package/
scripts/
```

`salt-pact26-v1.1.tar.gz` is ignored by Git and is optional. When it is present,
its expected SHA-256 is
`53d8df9da4fc714b86f9fb884f4fcd8b1ed4c0cbec65a401409fd951f5e5b559`.
When it is absent, the notebook creates `salt-trovi-source.tar.gz` from the
versioned source tree and verifies that archive after uploading it to the node.

## Recovery and cleanup

Provisioning uses stable, user-, site-, and profile-specific names. Before
creating anything, the notebook checks the selected site for that exact managed
lease. When no node has a long enough interval immediately, it reserves the
earliest future interval and prints its UTC start time. At that time, rerun
Sections 1 and 2 and continue with Section 3. If the kernel restarts after
launch, rerun Section 1 with the same site/profile and use the recovery cell.

If Nova places a server in `ERROR`, Section 3 reports the Nova fault and stops
before floating-IP association. Rerunning Section 3 once deletes only that
failed server and retries within the same active lease. If `No valid host was
found` repeats, run teardown and try a different interval, site, or image; ask
Chameleon support to inspect the reservation if the failure persists.

The `neutronclient` deprecation text sometimes printed by `python-chi` is a
non-fatal library warning. Diagnose a failed launch from the server status and
Nova fault shown by Section 3, not from that warning.

Always download wanted results before running the final cleanup cell. Cleanup
uses the currently selected site and profile, checks the exact managed names,
deletes the server, waits for Nova to confirm its removal, and then deletes the
lease so the node and reserved floating IP are released immediately.
