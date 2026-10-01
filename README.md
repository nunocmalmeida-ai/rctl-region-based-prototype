# RCTL region-based ultrasound fitting lab

A self-contained browser proof of concept for exploring RCTL-style model fitting to 3D ultrasound volumes. It runs on a deterministic synthetic phantom immediately, supports local 8-bit volume import, procedural implant-like objects, artifact simulation, orthogonal slices, and educational fitting comparisons. It does not use a server, patient data, or a clinical model.

## Setup and commands

Requires Node.js 20.19+ (or 22.12+) and npm.

```sh
npm install
npm run dev
```

Open the local URL printed by Vite. To create and preview the production bundle:

```sh
npm run build
npm run preview
```

Run the numerical/data-loader tests with `npm test`. The app has no account, external service, or runtime network requirement.

## Two-stage pipeline: coarse initialization + refinement

The fitting workflow is split into two explicit, separately selectable stages, mirroring how a real re-acquisition/tracking system must first (re-)find an object before an EKF-style per-frame measurement update can refine it:

- **Stage 1 · Coarse initialization / re-acquisition** (`src/core/initialization.js`) — a global search over the whole volume that proposes one or more ranked candidate initial poses, with no assumption that the previous pose is a good starting point.
- **Stage 2 · Local refinement** (`src/core/fitting.js`, unchanged) — the existing coordinate-descent optimizer, seeded from Stage 1's chosen candidate (or a naive/manual baseline for comparison) and behaving exactly as before.

The **Stage 1 · Initialization** panel lets you pick one of five methods, run it, and inspect the returned candidate pose(s), their score, and (in synthetic mode, where the true embedded pose is known) their translation/rotation error *before* Stage 2 ever runs. The **Seed Stage 2 refinement from** control then chooses whether **Fit model to volume** starts from the Stage 1 output or from the naive volume-center baseline, so the value added by Stage 1 can be judged directly from the **Pipeline metrics** card (initialization error vs. post-refinement error vs. whether refinement converged to a qualitatively correct solution).

A shared **random perturbation** control (used by the manual method) offsets the known ground-truth pose by a random translation/rotation before Stage 1 runs, so you can test how far off a naive/previous-frame seed can be and how well each automatic method recovers from it.

Implemented Stage 1 methods:

- **Global template matching** — a coarse-to-fine (multi-resolution) search that compares a binarized bright-voxel mask of the object model, placed at trial translations (and, if a rotation step/range is configured, trial rotations) on a downsampled grid, against a similarly downsampled bright-voxel mask of the volume, using SSD, NCC, or thresholded-overlap as the similarity metric. Each resolution level narrows the search window around the previous level's best candidates(s), which keeps the search tractable in-browser while remaining a faithful (if small-scale) instance of multi-resolution template matching.
- **Generalized Hough transform (simplified)** — builds an R-table of reference-to-surface vectors sampled from the object model, then has every candidate bright/edge voxel in the volume vote for the translations implied by each table vector, accumulating votes in a discretized 3-D translation accumulator and returning the peak(s). *Simplification:* the classical technique indexes the R-table by local gradient orientation so that only compatible surface points vote for a given voxel; this prototype's R-table has no orientation index (every table vector votes for every candidate voxel) and the accumulator is translation-only (rotation is held fixed from the current rotation/config), trading angular selectivity for a search that stays fast in JavaScript.
- **ICP-style point-set alignment** — extracts a candidate point cloud from bright voxels above a threshold (optionally narrowed to the largest connected component), samples a point cloud from the model surface, pre-aligns the two clouds by their centroids/bounding boxes (addressing ICP's well-known sensitivity to initial alignment), then iterates nearest-neighbor correspondence + rigid-transform solving via Kabsch/SVD (a minimal from-scratch 3×3 SVD built on Jacobi eigendecomposition, since no linear-algebra library is used) until the pose change drops below a tolerance or the iteration budget is spent. Outlier correspondences beyond a configurable distance percentile are rejected at each iteration.
- **Blob/connected-component centroid + PCA orientation** — thresholds bright voxels, extracts 6/18/26-connected components, keeps the most plausible blob by a configurable expected volume range, and returns its centroid and principal axes (via the eigenvectors of its voxel covariance matrix) as a fast, cheap initial pose. This is presented as the naive baseline the other three methods should improve upon.
- **Manual/user-specified pose** — the original behavior, kept as an explicit selectable method so automatic and manual seeding can be compared directly; combined with the shared perturbation control this also doubles as the "naive/previous-frame seed" stress test for the other methods.

## Pre-processing filters

The new **Pre-processing** panel (`src/core/preprocessing.js`) applies up to two chained filters to the volume before it reaches Stage 1, Stage 2, both, or neither — these are independent toggles ("Apply to Stage 1 initialization input" / "Apply to Stage 2 refinement input"), not a single global switch, so you can, for example, denoise only the input to initialization while fitting against the raw volume.

- **Gaussian smoothing** — a full-fidelity separable 3-D convolution (configurable sigma), applied as three successive 1-D passes along X, Y, and Z.
- **Median filtering** — a full-fidelity 3-D median filter over a configurable odd kernel size, useful for speckle suppression.
- **Anisotropic diffusion (Perona–Malik)** — the classic discrete edge-preserving diffusion update, iterated for a configurable number of steps with a configurable conductance (kappa) and a choice of exponential or quadratic edge-stopping function; a full-fidelity 3-D implementation of the textbook ultrasound despeckling filter.
- **Monogenic-signal-based phase congruency (simplified)** — *not* a true 3-D monogenic signal / multi-scale log-Gabor implementation. It is a documented, simplified per-slice (2-D) approximation: a Gaussian-smoothed "even" channel and central-difference "odd" channels are combined into a local-energy-like phase-congruency proxy, usable as an edge-like feature or confidence weight. Treat it as illustrative of the concept, not a faithful reproduction of the monogenic-signal literature.
- **Wavelet-based despeckling** — a genuine single-level 2-D Haar discrete wavelet transform applied per z-slice, with a Donoho-style universal soft/hard threshold (estimated from the HH sub-band's median absolute deviation, or a user-adjustable scale) applied to the detail sub-bands before inverse-transforming; a simplified but real instance of wavelet-shrinkage despeckling. Odd-sized slices have a minor known boundary-handling simplification (documented in code).
- **None** — passthrough; combined with a second independent filter slot, this also allows chaining exactly one filter, or none at all, in addition to up to two filters in sequence.

## Using the prototype

1. Start with the **Synthetic demo** (48³ voxels); a bright sphere is embedded in a locally generated, noisy tissue/blood-pool phantom.
2. Choose a sphere, box, cylinder, torus, or procedural implant-like frame. Adjust its size, 3-D center and Euler rotation, echogenicity, blend, and optional shadow.
3. Adjust Gaussian noise, multiplicative speckle, dropout, clipping, and the deterministic seed. **Embed / regenerate** explicitly creates a new derived volume; the base is kept unchanged.
4. Switch the display between original, embedded, difference, and ground-truth overlay. Axial, coronal, and sagittal slices are independently selectable.
5. Select one of the educational measurement approaches and a model representation, then run **Fit model to volume**. The optimizer history and (for the synthetic inserted object) translation, relative size, Dice, IoU, and score are reported.
6. A local volume can be selected in the Dataset section. For `.raw`, enter dimensions in voxels (X, Y, Z), then load. `.npy` files must be 3-D, C-order, `uint8` arrays in `(Z, Y, X)` order. For local volumes, the configured synthetic object is embedded over the loaded base; the known mask describes that inserted object, not an annotation of the input scan.

The upload stays in the browser. `.raw` means exactly one unsigned byte per voxel in C order with X varying fastest, then Y, then Z; there is no header or automatic normalization. `.npy` supports NumPy format v1/v2 and C-order `uint8`; floating-point, compressed, Fortran-order, NIfTI, DICOM, and MetaImage files are not supported. Convert/rescale those formats before loading. For example, with NumPy and an array named `volume_zyx`:

```python
import numpy as np
volume = np.asarray(volume_zyx, dtype=np.uint8, order="C")
assert volume.ndim == 3
volume.tofile("volume.raw")
print("Set X,Y,Z to:", volume.shape[2], volume.shape[1], volume.shape[0])
```

No data volume is bundled or downloaded automatically. One public source of real 3D echocardiograms is the [EchoNet/3d-echo dataset release](https://github.com/echonet/3d-echo/releases/tag/v1.0) (29 author-recorded 3D echo volumes; download details and the associated paper are on the [project page](https://github.com/echonet/3d-echo)). Reproducibly download and inspect the upstream archive with:

```sh
curl -fL https://github.com/echonet/3d-echo/releases/download/v1.0/dataset.zip -o /tmp/echonet-3d-echo-dataset.zip
unzip -l /tmp/echonet-3d-echo-dataset.zip
unzip /tmp/echonet-3d-echo-dataset.zip -d /tmp/echonet-3d-echo
```

Convert one compatible volume to the raw/NPY input format above; the archive is not a browser-ready fixture and this prototype does not pretend to include or automatically parse it. The upstream project requests citation of Vukadinovic et al., *Automated Interpretable 2D Video Extraction from 3D Echocardiography* (2025, [arXiv:2511.15946](https://arxiv.org/abs/2511.15946)). The release page/repository does not state a clear dataset license; verify the current terms with the authors before research redistribution, commercial use, or other use beyond viewing/downloading. Neither those volumes nor that study's annotations are bundled here.

## What is implemented (and simplified)

- A typed-array volume generator and reversible base/derived volume pair; four primitive models plus a frame; deterministic noise, dropouts, clipping, and an optional acoustic shadow.
- 2-D orthogonal slice visualization, thresholded difference display, and a red synthetic ground-truth mask overlay. There is no 3-D surface renderer.
- One shared voxel-membership model interface and sampled data score for edge/normal-profile-inspired, region mean/variance, Gaussian likelihood-ratio, thin-shell template, and hybrid modes.
- Translation and radius/scale coordinate descent with an iteration history. Robust Huber/L1/quadratic threshold residuals, search radius, sampling density, local window, regularization, and iteration count affect the score/optimizer. The optional temporal checkbox applies a simple exponential pose blend between fits, not a Kalman filter.
- Smooth mode is an ellipsoidal implicit primitive; rigid mode preserves the selected primitive; CAD-like mode uses a low-dimensional anisotropic ellipsoid and a global scale parameter. These are procedural voxel models, not Doo–Sabin subdivision surfaces or imported CAD meshes.
- Synthetic-only Dice/IoU and pose/size errors are computed against the inserted synthetic mask. A local input scan has no annotation: metrics against an app-inserted object are not validated segmentation measures.
- Stage 1 coarse initialization (template matching, generalized Hough voting, ICP, blob+PCA) is a from-scratch, dependency-free prototype: the Hough R-table has no gradient-orientation index and a translation-only accumulator, ICP's rigid solve uses a minimal hand-rolled 3×3 SVD (no linear-algebra library), and all searches operate on small/downsampled grids for browser performance. These are simplifications relative to production-grade global initialization, not full re-implementations.
- The monogenic-signal/phase-congruency filter is a simplified per-slice 2-D proxy (Gaussian "even" channel + central-difference "odd" channels), not a true Riesz-transform/multi-scale monogenic signal. The wavelet despeckling filter is a real but single-level 2-D Haar transform, not a multi-level or biorthogonal wavelet family.

This is not an implementation of the original RCTL library: there is no production Kalman filter, Doo–Sabin surface, trained tracker, DICOM/NIfTI pipeline, physical voxel spacing, temporal sequence, gradient-based mesh optimizer, or validated device detection. The likelihood, edge/profile, shell, and hybrid terms are deliberately small educational proxies. The optimizer is local and can converge to a poor fit; displayed confidence is a normalized objective proxy, not calibrated probability. Data are not clinically validated. **Research prototype only—not for diagnosis, treatment, or clinical use.**

## License / data provenance

The repository code is provided under the ISC License (see `LICENSE`). Synthetic data are generated in the browser and are not medical images. Any separately downloaded dataset remains under its own terms; see the upstream links above.
