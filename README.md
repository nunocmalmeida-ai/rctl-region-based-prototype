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

This is not an implementation of the original RCTL library: there is no production Kalman filter, Doo–Sabin surface, trained tracker, DICOM/NIfTI pipeline, physical voxel spacing, temporal sequence, gradient-based mesh optimizer, or validated device detection. The likelihood, edge/profile, shell, and hybrid terms are deliberately small educational proxies. The optimizer is local and can converge to a poor fit; displayed confidence is a normalized objective proxy, not calibrated probability. Data are not clinically validated. **Research prototype only—not for diagnosis, treatment, or clinical use.**

## License / data provenance

The repository code is provided under the ISC License (see `LICENSE`). Synthetic data are generated in the browser and are not medical images. Any separately downloaded dataset remains under its own terms; see the upstream links above.
