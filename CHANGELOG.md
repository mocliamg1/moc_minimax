# Changelog

## Unreleased

- Added linear LoKr and full-difference comparison, including effective-dimension matching across adapter formats and ranks.
- Added weighted LoRA/LoKr merge with streamed safetensors export, exact shared-factor LoKr retention, optional SVD compression, and per-block storage/compression error reports.

- Added CPU LoRA loader and effective-weight comparison nodes with strict dimension/rank matching, per-block differences, inline text, and JSON text reports.
- Added file-change cache invalidation, comparison tests, and two-loader example workflows.

## 0.3.0 — 2026-08-21

- Added a separate Image to Video + References node that preserves the native Image to Video interface and adds autogrowing standard `IMAGE` sockets for non-temporal references.
- Kept temporal anchors in `minimax_keyframes` and reference media in `minimax_refs` so neither changes the other's role or numbering.
- Updated experimental reference-span weighting to skip temporal condition rows in hybrid packed layouts.

## 0.2.0 — 2026-08-13

- Added optional per-image source-isolation masks without changing native H3 or existing node outputs.
- Added white-keep/white-remove polarity, grow/shrink, feathering, neutral fill, blurred background, and padded crop presentation.
- Added mask coverage, bounds, dimensions, processing settings, warnings, guided prompt boundaries, and manifest provenance.
- Preserved the exact unmasked image tensor path and reference-record schema version 1.

## 0.1.0 — 2026-08-13

- Added named image, video, audio, and reference-set builders.
- Added stable aliases, guided semantic roles/priorities, copy boundaries, and audio relationships.
- Added exact native ordering, media trimming, soundtrack pairing, validation, inline reports, and manifests.
- Added native MiniMax H3 delegation with compatible `CONDITIONING` and `LATENT` outputs.
- Added opt-in experimental direct-reference value gating and attention priors.
- Added API/UI preflight examples, ComfyUI contract validation, CI, and unit coverage.
