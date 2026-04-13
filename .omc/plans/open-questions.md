# Open Questions

## verda-dstack-integration — 2026-04-12 (R2)
- [ ] Does `minimind/trainer/train_pretrain.py` already accept `--seq_len` and `--out_dir` flags? — Determines whether Step 5 invokes it directly or Step 7's patch must add the args. Confirm on first execution before scheduling an H100 run.
- [ ] Does the operator's GitHub org enforce SSO for PATs targeting GHCR? — If yes, the README in Step 10 must document the "Configure SSO" authorize step, otherwise `docker login ghcr.io` will 403 even with a valid `write:packages` PAT.
- [ ] Is there a Verda quota cap beyond the 1-concurrent-H100 assumption in P2? — Affects whether `run.sh remote` should refuse to start when another run is active, or just queue.
- [ ] Should smoke runs publish under `ghcr.io/${GH_USER}/verda-minimind:smoke` instead of `:latest`? — Avoids a smoke image accidentally becoming the dev default; decide before first `scripts/smoke.sh remote` execution.
- [ ] Should `MLFLOW_ARTIFACT_UPLOAD=0` be enforced by the image (baked env default) or only by `remote-entrypoint.sh`? — Belt-and-suspenders; decide in Step 5 review.
- [ ] Is `cloudflared` allowed by the operator's network policy, or do we need a fallback (e.g., ngrok, Tailscale)? — If blocked, B1 is invalidated and B2 becomes the forced choice.

## repo-readme-extreme-detail — 2026-04-12
- [ ] CI wiring for `scripts/doc-anchor-check.sh` — GitHub Actions vs local pre-commit vs Makefile target? — Determines where the anchor invariant is enforced; deferred per M1.
- [ ] Should `DSTACK_SERVER_ADMIN_TOKEN` migrate from compose `environment:` to a secret file now? — Tightens security item #5; currently tagged `accepted-debt`.
- [ ] Named-tunnel migration plan for cloudflared (vs current Quick Tunnel)? — Addresses security items #6 and #7 simultaneously.
- [ ] Should `TROUBLESHOOTING.md` get its own anchor namespace distinct from subsystem anchors? — Affects collision risk as anchors grow past ~40.
- [ ] Are the 7 enumerated security exposures exhaustive for posture coverage? — Additions (e.g. HF token handling, artifact bucket ACLs) may be needed before first external review.
