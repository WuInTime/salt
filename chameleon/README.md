# SALT on Chameleon Cloud

`chameleon-salt.ipynb` is the evaluator-facing Trovi notebook. It performs the
complete resource lifecycle:

1. Selects `CHI@UC` and reserves one `compute_skylake` bare-metal node plus one
   floating IP for 16 hours.
2. Launches `CC-Ubuntu24.04`, waits for SSH, and records the allocated CPU and
   system configuration.
3. Runs `chameleon/setup-node.sh` to install compiler/build dependencies,
   Linux performance-counter tools, and Docker Engine.
4. Uploads the immutable `salt-pact26-v1.1.tar.gz` release, verifies its
   SHA-256 digest, builds the Docker image, and launches its test and experiment
   entry points.
5. Downloads compressed result directories back into the Trovi/Jupyter
   workspace before deleting the server and lease.

The notebook offers three experiment paths:

- `smoke`: reduced end-to-end validation of all three workflows;
- `reproduce`: the complete paper workflow (approximately 8--9 hours after the
  image is built);
- optional fresh PMCs: hardware L1D-miss collection on the allocated node.

The normal `smoke` and `reproduce` commands use the checked-in Intel Core
i7-7700 measurements for Figure 4. This is the portable reproduction path. A
Xeon Gold 6126 is a Skylake-SP processor and can run the generic Linux L1D-miss
event, but its fresh counter totals should not be presented as replacements for
the paper's i7-7700 measurements. The notebook stores fresh measurements under
a separately named result directory.

## Trovi contents

Include these files in the Trovi artifact and keep their relative paths:

```text
chameleon-salt.ipynb
chameleon/setup-node.sh
salt-pact26-v1.1.tar.gz
```

The expected SHA-256 for `salt-pact26-v1.1.tar.gz` is
`53d8df9da4fc714b86f9fb884f4fcd8b1ed4c0cbec65a401409fd951f5e5b559`.

The top-level repository files may also be included, but the notebook uploads
the release archive so that the evaluated source is fixed and compact.

## Recovery and cleanup

Provisioning uses stable, user-specific names and idempotent `python-chi`
submissions. If the notebook kernel restarts, run the configuration cell and
then the recovery cell to reconnect to the existing lease and server.

Always download wanted results before running the final cleanup cell. Cleanup
deletes the server, waits for Nova to confirm its removal, and then deletes the
lease so the node and reserved floating IP are released immediately.
