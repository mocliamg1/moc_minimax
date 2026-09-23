# MOC MiniMax H3 References

A richer authoring layer for ComfyUI's native **MiniMax H3 Reference to Video** node.

The extension keeps ComfyUI's MiniMax H3 encoding and latent preparation as the source of truth, then adds:

- stable prompt aliases such as `@hero`, `@walk`, and `@voice`;
- identity, style, composition, motion, camera, performance, voice, music, ambience, and other reference roles;
- semantic `Primary`, `Supporting`, `Weak`, and `Disabled` priorities;
- per-reference `use_for` and `do_not_copy` scopes;
- deterministic video trimming and 24 fps preparation;
- optional paired video soundtracks with an explicit audio role and relationship;
- resolved mappings, validation, manifests, and inline diagnostic reports;
- optional per-image source-isolation masks with neutral, blurred, or cropped presentation;
- per-image detail selection and native canvas/token estimates;
- global visual/audio reference-fidelity controls; and
- opt-in, experimental model-level signal weighting.

The normal path delegates to the native node. Numeric per-reference weighting is separate and experimental because MiniMax H3 does not expose a trained, calibrated reference-strength parameter.

## Requirements and model setup

- ComfyUI **0.32.0 or newer** with native MiniMax H3 support
- Python 3.10 or newer
- the models used by ComfyUI's official H3 reference-to-video workflow

Start with the official [MiniMax H3 R2V workflow template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json). It documents the currently recommended files in [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):

```text
ComfyUI/
└── models/
    ├── diffusion_models/
    │   └── minimax_h3_ref2va_pruned_int8_convrot.safetensors
    ├── text_encoders/
    │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
    └── vae/
        ├── minimax_h3_video_vae_fp16.safetensors
        └── minimax_h3_audio_vae_fp32.safetensors
```

Direct official downloads: [Ref2VA diffusion model](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors), [Qwen3-VL text encoder](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors), [video VAE](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors), and [audio VAE](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors).

Reference-to-video requires the **Ref2VA** diffusion checkpoint. The `fl2va` model used by the official T2V/I2V templates is a different model and is not a drop-in replacement. Filenames may evolve; follow the official R2V template if its model list changes.

Update ComfyUI first using the [official update guide](https://docs.comfy.org/installation/update_comfyui) if the native `MiniMaxH3ReferenceToVideo` node or `minimax` CLIP type is missing.

No extra Python packages are required beyond ComfyUI's runtime.

## Install

Clone this repository into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mocliamg1/moc_minimax.git
```

Restart ComfyUI. The nodes appear under **MiniMax H3 → MOC References**.

## Quick start

1. Open the official H3 R2V template so its model loading, sampler, decode, and video-save chain is already correct.
2. Create references with any mix of:
   - `MOC • H3 Image Reference`
   - `MOC • H3 Video Reference`
   - `MOC • H3 Audio Reference`
3. Connect the builders to `MOC • H3 Reference Set`.
4. Write the target scene using stable aliases such as `@hero` and `@walk`.
5. Use `MOC • H3 Compile Reference Prompt` to inspect mapping and diagnostics without loading H3 models.
6. Replace only the official workflow's native authoring node with `MOC • H3 Reference to Video+`. Connect the same H3 CLIP, video VAE, and audio VAE.
7. Connect its first two outputs where the native node's `positive` and `latent` outputs were connected.

For the stock Image to Video controls with extra image sockets, use **MOC • H3 Image to Video (Simple)**. Connect a `Load Image` output directly to its reference image socket; another socket appears when it is connected, up to nine images. No reference builders or extra settings are needed.

For a reference-sizing control, use `MOC • H3 Image to Video + References`.

The first two outputs deliberately match the native node's order:

1. `positive` — `CONDITIONING`
2. `latent` — `LATENT`

The remaining outputs expose the compiled prompt, report, and JSON manifest.

## How reference conditioning works

The native H3 node has two complementary conditioning paths:

- the text encoder receives the prompt plus H3's semantic representation of visual references; and
- image, video, and audio VAEs create direct reference latents that are packed with the generated audio/video sequence.

When an Image Reference mask is connected, MOC first creates one deterministic RGB reference image. That same processed RGB tensor is passed to the native node, so both Qwen's semantic image path and the visual VAE see the isolated source. The mask is not passed as a separate model control because H3 has no native per-reference mask input.

This is **source isolation only**. It helps prevent a reference background, pose, or nearby object from becoming reference evidence. It does not define where the retained subject must appear in generated frames, and it is not an inpainting or regional-generation mask.

Native tags depend on packing order: pictures first, then videos, then standalone audio. A soundtrack paired with a video receives an audio tag immediately before that video. MOC resolves stable aliases to those positional tags, for example:

```text
@hero       -> <Picture 1>
@walk_audio -> <Audio 1>
@walk       -> <Video 1>
@voice      -> <Audio 2>
```

With a soundtrack connected to video alias `@walk`, MOC reserves the generated alias `@walk_audio`. Do not create another reference with that name.

In `guided` prompt mode, roles, priorities, and scopes become explicit natural-language instructions for H3. They help communicate intent and resolve ambiguity, but they are **prompt guidance**, not guaranteed gates, calibrated probabilities, or official MiniMax strength controls. Always inspect the result and refine the prompt for the specific references.

For example, use native signal weights for the ordinary workflow:

| Alias | Media | Role | Priority | Signal weight |
|---|---|---|---|---:|
| `@hero` | image | identity | primary | 1.0 |
| `@walk` | video | motion | supporting | 1.0 |
| `@look` | image | style | weak | 1.0 |

```text
A cinematic tracking shot of @hero walking through a rainy alley.
Use @walk for gait and physical timing.
Borrow a subtle teal-and-amber palette from @look.
```

Guided mode resolves the aliases and prepends scoped instructions resembling:

```text
<Picture 1> is the primary identity reference (fully_preserved).
Use it for identity-defining facial features, hair, body proportions, and recurring wardrobe details.
Do not copy pose, camera angle, background, and lighting unless the scene explicitly requests them.

<Picture 2> is the weak style reference (weak_reference).
Use it for rendering style, palette, texture language, and lighting character.
Do not copy the source person, object identity, pose, composition, and scene content.

<Video 1> is the supporting motion reference (partially_preserved).
Use it for action, gait, timing, physical rhythm, and motion trajectory.
Do not copy the source actor, face, wardrobe, environment, visual style, and camera unless explicitly requested.
```

## Nodes

### MOC • H3 Image Reference

Builds one named image reference.

- `role`: identity, object, style, composition, environment, lighting/color, or general
- `priority`: semantic prompt priority; `disabled` excludes the record from encoding and numbering
- `signal_weight`: metadata for the optional experimental MODEL patch; `1.0` is native
- `detail`: inherit the downstream default, match the output area, or retain up to a 2048px short edge
- `use_for` / `do_not_copy`: optional exact scope overrides
- optional `mask`: a ComfyUI MASK used to isolate valid source pixels; white is retained by default
- `mask_polarity`: choose whether white keeps or removes source pixels
- `mask_presentation`: replace excluded pixels with neutral gray, a blurred background, or crop around the retained bounds
- advanced mask shaping: signed expansion/erosion, Gaussian feathering, neutral level, blur radius, and crop padding

Only the first item in an IMAGE batch and the first item in a MASK batch are used. A mismatched mask is resized to the source image with bilinear interpolation. Reports and manifests include mask coverage, bounds, source/resolved dimensions, presentation mode, and processing warnings.

Mask processing order is polarity, grow/shrink, feather, then presentation. An empty retained region is an error; coverage below 1% or above 99% is reported as a warning. If no mask is connected, the image tensor follows the original unmasked path unchanged.

### MOC • H3 Video Reference

Builds a named video reference from an IMAGE batch.

- roles include motion, camera, performance, identity, style, environment, and full reference;
- `input_fps` is the frame rate represented by the incoming batch, not necessarily the source file's original rate;
- trim controls prepare a deterministic 24 fps batch;
- native H3 then caps it to the target length and trims it to its `17k+5` frame grid;
- `soundtrack_policy` can trim a connected soundtrack to the native video duration, keep it, or ignore it; and
- `soundtrack_role` and `soundtrack_relationship` describe how guided prompting should use that audio.

A connected soundtrack uses the generated alias `@name_audio`. `soundtrack_relationship=reference` transfers audible attributes without asking for signal reuse. `fully_copy`, `partially_copy`, and `weak_reference` are explicit generative instructions for authorized material; even copy modes are not byte-exact audio editing.

For best results, keep each video and audio reference within the official 2–15 second recommendation.

### MOC • H3 Audio Reference

Builds a voice, prosody/dialogue, music, ambience, sound-effect, timing, or general audio reference with trim controls.

The `relationship` choices match paired soundtracks: `reference`, `fully_copy`, `partially_copy`, and `weak_reference`. Copy relationships should only be used when reuse is authorized, and they still describe a generative request rather than deterministic signal copying.

### MOC • H3 Reference Set

Collects up to 15 builder records, validates record structure, aliases, counts, reserved-name collisions, and prepared source-modality duration totals, excludes disabled records, and resolves native ordering:

1. pictures;
2. videos, with each paired soundtrack immediately before its video; and
3. standalone audio.

It returns the runtime set, mapping, report, JSON manifest, and structural validity. Target-dependent checks—such as native video-grid trimming and total effective durations after target-length capping—run in Compile and Reference to Video+ because they require the intended output length.

### MOC • H3 Compile Reference Prompt

A non-encoding, non-blocking preflight. It produces the compiled prompt, mapping, diagnostics, manifest, and `valid` result for the intended resolution, duration, and default image-detail policy. Its report is displayed inline using ComfyUI's PreviewText UI.

Prompt modes:

- `guided`: resolve aliases and add role/priority/scope instructions;
- `aliases_only`: resolve aliases without adding guidance; and
- `manual`: preserve direct `<Picture N>`, `<Video N>`, and `<Audio N>` authoring while validating referenced tags.

Validation modes affect preflight readiness without hiding its report:

- `warn`: `valid=false` only for errors; warnings remain advisory; and
- `strict`: `valid=false` for errors or warnings.

Use the same `default_image_detail`, width, height, and length here as on the main node for representative native canvas, frame, duration, and direct-token estimates.

### MOC • H3 Reference to Video+

The main replacement node delegates image/video/audio encoding and latent creation to ComfyUI's native `MiniMaxH3ReferenceToVideo`, then attaches copied per-reference metadata to the returned conditioning.

### MOC • H3 Image to Video (Simple)

The stock H3 inputs (`clip`, `vae`, `prompt`, `width`, `height`, `length`, `first_frame`, `last_frame`) plus optional dynamic `IMAGE` inputs. One reference socket appears initially; connecting it reveals the next, up to nine references. The only outputs are `positive` and `latent`, so reconnect these to the same places as the stock node.

Leave the extra image sockets empty to run ComfyUI's stock `MiniMaxH3ImageToVideo` directly. With extra images connected, the node uses the existing hybrid reference pipeline with sizing fixed to `match`. First/last frames remain temporal anchors; extra images are independent references, numbered `<Picture 1>`, `<Picture 2>`, etc. from top to bottom, skipping empty sockets. For example: `The person from <Picture 1> walks through the room from <Picture 2>.`

Additional references use H3's reference conditioning, so use a reference-capable model and a ComfyUI build that supports references alongside first/last keyframes, as with the hybrid node below. The existing nodes remain available for saved workflows and advanced controls.

### MOC • H3 Image to Video + References

This separate hybrid node starts with the default MiniMax H3 Image to Video inputs—`clip`, `vae`, `prompt`, `width`, `height`, `length`, `first_frame`, and `last_frame`—then adds up to nine autogrowing standard `IMAGE` reference inputs. It does not require MOC reference builders, a Reference Set, or an audio VAE. The first image is fixed to frame 0 and the last image to the resolved final frame. Temporal images are stored in `minimax_keyframes`; the additional images remain non-temporal `minimax_refs`.

The node's two outputs match the default node: `positive` and `latent`. Its prompt can address the additional references as `<Picture 1>`, `<Picture 2>`, and so on. Use a current ComfyUI build whose MiniMax H3 packed layout accepts `minimax_keyframes` and `minimax_refs` together.

The main node enforces validation:

- `warn` continues through advisory warnings but always stops on errors; and
- `strict` stops on warnings as well as errors.

Reference Set, Compile, and Reference to Video+ display their reports inline with PreviewText. The report and manifest also remain available as string outputs.

Advanced global controls:

- `visual_reference_fidelity`: native default `0.999`; lowering it adds noise to all direct visual-reference latents; and
- `audio_reference_fidelity`: native default `1.0`; lowering it adds noise to all direct audio-reference latents.

These affect a whole modality, not individual references.

### MOC • H3 Apply Reference Weights [Experimental]

An optional `MODEL → MODEL` patch. Place it after model loading/model-sampling patches and feed its output to the guider/sampler. Non-unit builder weights do nothing unless this patched MODEL is actually used for sampling.

With every weight at `1.0`, or with patch strength `0`, it preserves the native attention path.

Two experimental methods are available:

- `value_gate` (default): gates direct reference value contributions for generated audio/video queries; and
- `attention_prior`: applies `log(weight)` as a reference-key prior for generated queries, forces PyTorch SDPA for that path, and may be slower or use more memory.

Important limitations:

- The patch affects the direct VAE-latent reference path only. It does not numerically weight Qwen's semantic vision tokens.
- Role/priority prompt guidance and numeric direct-signal weights are separate mechanisms.
- Weights are not calibrated probabilities, percentages, or official MiniMax parameters.
- Weight `0` under `value_gate` suppresses value content but is not mathematically identical to removing the key; `attention_prior` more directly blocks access.
- Both methods reject any existing `optimized_attention_override` when non-unit weights are active. Remove the conflicting attention patch or return all weights to `1.0`.
- Do not use `(<Picture 1>:1.5)` as a substitute. H3 does not interpret that syntax as reference strength.

## Reference limits

The extension checks the native/published reference limits:

- up to 9 images;
- up to 3 videos;
- up to 3 standalone audio clips;
- 2–15 seconds recommended per video/audio clip;
- 15 seconds recommended total per video or audio modality; and
- 12 mixed files in the hosted API, reported as an advisory because local ComfyUI may not enforce it.

Paired video soundtracks count as audio files for mixed-file and effective-audio-duration diagnostics, but not as standalone audio slots.

## Examples

- [`examples/minimax_h3_reference_plus_api.json`](examples/minimax_h3_reference_plus_api.json) is a clean API-format authoring/preflight prompt. It uses two generated placeholder images, needs no H3 checkpoints, and stops at Compile.
- [`example_workflows/minimax_h3_moc_authoring_preflight.json`](example_workflows/minimax_h3_moc_authoring_preflight.json) is the same idea in loadable ComfyUI UI format. Replace the placeholder `Empty Image` nodes with real loaders before adapting it to generation.
- For full generation, begin with the official [H3 R2V workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json) and replace its native reference-authoring node as described above.

The preflight examples keep all signal weights at `1.0`. Experimental weights belong in a separate generation workflow where the MODEL patch connection is visible.

To use source isolation with ComfyUI's `Load Image`, connect its `MASK` output to the corresponding Image Reference node. Inspect the node summary and downstream report for coverage and bounds before loading H3. The examples leave masks disconnected so they remain queueable without uploaded image assets.

## Design and compatibility

This extension does not shadow the native node ID and does not vendor MiniMax model code. Existing native workflows remain untouched. Delegation keeps ComfyUI authoritative for H3 VAE preparation, tag presentation, reference layout, and AV latent creation.

Runtime records use `schema_version: 1`; enum values and manifest fields are intended to remain stable across compatible releases.

## Development

Run pure-Python tests:

```bash
python -m unittest discover -s tests -v
```

Run import/schema checks with the same Python environment that starts ComfyUI:

```bash
/path/to/ComfyUI/python_embeded/python.exe scripts/validate_comfy.py /path/to/ComfyUI
```

For a manual/venv install, activate ComfyUI's environment first and run:

```bash
python scripts/validate_comfy.py /path/to/ComfyUI
```

## License

MIT. MiniMax H3 model weights have their own license; this repository does not redistribute them.

## Compare MiniMax H3 LoRAs

Under **MiniMax H3 → MOC LoRA**, connect two **MOC • H3 Load LoRA** nodes to **MOC • H3 Compare LoRAs**. Each loader selects a file from ComfyUI's configured LoRA folders and loads it safely on CPU. No base model or GPU is needed. Replacing a file invalidates the loader cache using its path, size, timestamps, and inode.

The comparison displays a readable report inline and returns `report` and `report_json` as STRING outputs. The JSON is serialized text, not a custom dictionary socket. Load [the example workflow](example_workflows/minimax_h3_lora_compare.json) and select your two actual LoRA files; an [API example](examples/minimax_h3_lora_compare_api.json) is also included.

Comparison supports standard two-dimensional LoRA up/down and A/B pairs (including `.default.weight`), direct or decomposed linear LoKr factors, and full weight-difference adapters (`.diff`). `matching=strict` (default) requires identical normalized target layers, adapter formats, ranks, and factor shapes. `matching=effective_dimensions` allows different formats, ranks, and Kronecker partitions, while requiring identical target layers and effective weight dimensions. Use this mode to compare a compressed merge with its full-difference reference. Incompatible files return diagnostics, `compatible: false`, and `score: null`. Recognized `base_model.model.`, `model.diffusion_model.`, and `diffusion_model.` wrappers are removed; architectural names are not guessed or remapped. DoRA, Tucker/convolutional adapters, unknown weight entries, malformed pairs, and nonfinite values are rejected explicitly.

Scores compare effective updates at strength 1. LoRA uses `ΔW = (alpha / rank) × B × A`, with missing alpha defaulting to rank. LoKr reconstructs `ΔW = scale × kron(W1, W2)` using ComfyUI's additive weight-patch scaling: alpha is divided by the rank of the last decomposed factor (W2 when both are decomposed); missing alpha means scale 1. When both factors are stored directly, alpha is ignored. These decisions are recorded per module. The magnitude-sensitive score is `100 × (1 − ||X−Y||² / (||X||+||Y||)²)`: identical updates score 100, opposite updates score 0, and doubling an update scores about 88.89. Two zero updates score 100; comparing zero with nonzero scores 0. Cosine similarity is also reported, or null for a zero norm. **These are weight-update similarities and do not predict visual similarity.**

Overall and block metrics aggregate squared norms and inner products before scoring. CPU float64 Gram calculations handle LoRA pairs; Kronecker inner-product identities handle LoKr pairs with matching partitions. Other format combinations compare bounded row chunks without retaining full model updates. A module differs when `||X−Y|| > 1e-8 + 1e-5 × max(||X||, ||Y||)`; a block differs if any of its modules does. Reports retain full block-family paths, list differing blocks first by lowest score, and include non-block modules separately. JSON schema version 1 includes sources, compatibility, metric definition, overall/block/module metrics, alpha defaults, tolerances, and diagnostics.


## Merge LoRA / LoKr checkpoints

Connect two to eight **MOC • H3 Load LoRA** outputs to **MOC • H3 Merge LoRA / LoKr**. Set each `strength_a` … `strength_h` to the value you use in your working stack. Strengths are summed without normalization; zero excludes an input. Different internal ranks and adapter formats are allowed. Shared target layers must have identical effective weight dimensions, and layers present in only some inputs are preserved. This handles additive linear LoRA/LoKr updates, not full base-model checkpoints or DoRA.

The node writes a numbered `.safetensors` file under `ComfyUI/output/loras/` by default. It returns a connected adapter object, readable report, JSON text report, and the saved path. Existing files are never overwritten, and failed merges do not publish a partial safetensors file. The saved file includes source names, input strengths, format, and per-module error information in its metadata.

Three export choices are available:

| `output_format` | Behavior |
|---|---|
| `full_diff` (default) | Sum effective updates and save native ComfyUI `.diff` tensors. Preserves the weighted sum within storage precision, but can produce a very large file. |
| `lokr_shared_or_diff` | Keep a compact LoKr representation when all contributors to a layer have an exactly identical reconstructed W1 or W2 with matching partitions. Fold strengths and alpha scales into the other factor. Fall back to `.diff` for other layers. No approximate factor averaging. |
| `lora_svd` | Approximate each merged layer with a standard LoRA using the selected `rank`, capped by its weight dimensions. CPU SVD can be slow and require substantial memory on large layers. |

`storage_dtype` defaults to float32; float16 and bfloat16 are optional. Error reports measure the exported tensors **after** storage conversion against the requested weighted sum, so they include rounding as well as SVD truncation. Overall and per-block relative errors are Frobenius difference norms divided by reference-update norms; non-block modules have their own group. Low weight error does not guarantee unchanged generations.

Merging and export process one layer at a time. Tensor bytes are spooled to disk and published as one file, so the exporter does not retain a full dense MiniMax delta in RAM. Full-difference export still needs space for both the spool and final staging file (roughly twice the output size during writing), plus memory for the largest layer and its working tensors. The returned adapter maps the saved safetensors file on CPU.

Move or copy the saved file into a configured ComfyUI LoRA folder to use it with a native LoRA loader, and **apply the merged adapter at strength 1.0**: the original strengths are already baked in. Keep the same base model and sampling settings when comparing the merge with your original stack; floating-point application order can cause differences. Architecture-specific layer names are preserved; this does not translate between different model architectures.

Load [the three-checkpoint merge workflow](example_workflows/minimax_h3_lokr_merge.json), or use [the API example](examples/minimax_h3_lokr_merge_api.json). Replace placeholder filenames and sample strengths with your actual settings. To evaluate compression, make one `full_diff` merge and one `lora_svd` merge with identical inputs/strengths, then connect their adapter outputs to Compare with `matching=effective_dimensions`.
