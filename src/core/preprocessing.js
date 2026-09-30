// Classical pre-processing filters applied to a volume before either the Stage 1
// coarse-initialization search or the Stage 2 refinement measurement step (see
// initialization.js and fitting.js). Every filter returns a new volume object
// (same `dims`/`mask`/`truth`/`label`, a fresh `data` typed array) so filters can
// be chained without mutating the input. Each function documents whether it is a
// full-fidelity implementation of the classical technique or a deliberately
// simplified/2-D approximation made for in-browser performance, consistent with
// the project's existing "what is implemented and simplified" documentation style.
import { index3d } from "./volume.js";

export const FILTER_TYPES = ["none", "gaussian", "median", "anisotropic", "phase-congruency", "wavelet"];

function cloneVolume(volume, data) {
  return { ...volume, data };
}

// --- (a) Gaussian smoothing -------------------------------------------------
// Full-fidelity: a real separable 3-D Gaussian convolution (three 1-D passes
// along x, y, z), which is mathematically identical to a single 3-D Gaussian
// convolution but far cheaper (O(n) taps per axis instead of O(n^3)).
function gaussianKernel1D(sigma) {
  const radius = Math.max(1, Math.ceil(sigma * 3));
  const kernel = new Float32Array(radius * 2 + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i += 1) {
    const value = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel[i + radius] = value;
    sum += value;
  }
  for (let i = 0; i < kernel.length; i += 1) kernel[i] /= sum;
  return { kernel, radius };
}

export function gaussianSmooth3D(volume, options = {}) {
  const sigma = Math.max(0.01, Number(options.sigma ?? 1));
  const { kernel, radius } = gaussianKernel1D(sigma);
  const [nx, ny, nz] = volume.dims;
  const clampIndex = (value, max) => Math.max(0, Math.min(max, value));
  function convolveAxis(src, axis) {
    const dst = new Float32Array(src.length);
    for (let z = 0; z < nz; z += 1) {
      for (let y = 0; y < ny; y += 1) {
        for (let x = 0; x < nx; x += 1) {
          let sum = 0;
          for (let k = -radius; k <= radius; k += 1) {
            let xx = x;
            let yy = y;
            let zz = z;
            if (axis === 0) xx = clampIndex(x + k, nx - 1);
            else if (axis === 1) yy = clampIndex(y + k, ny - 1);
            else zz = clampIndex(z + k, nz - 1);
            sum += src[index3d(volume.dims, xx, yy, zz)] * kernel[k + radius];
          }
          dst[index3d(volume.dims, x, y, z)] = sum;
        }
      }
    }
    return dst;
  }
  let data = volume.data;
  data = convolveAxis(data, 0);
  data = convolveAxis(data, 1);
  data = convolveAxis(data, 2);
  return cloneVolume(volume, data);
}

// --- (b) Median filtering ----------------------------------------------------
// Full-fidelity: a real 3-D sliding-window median (cubic kernel, edge-clamped),
// the classic speckle-suppression filter.
export function medianFilter3D(volume, options = {}) {
  const requestedSize = Math.max(1, Math.round(options.kernelSize ?? 3));
  const half = Math.floor(requestedSize / 2);
  const kernelSize = 2 * half + 1; // force an odd effective kernel size (even inputs round down)
  const [nx, ny, nz] = volume.dims;
  const data = new Float32Array(volume.data.length);
  const clamp = (value, max) => Math.max(0, Math.min(max, value));
  const window = new Float32Array(kernelSize ** 3);
  for (let z = 0; z < nz; z += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        let count = 0;
        for (let dz = -half; dz <= half; dz += 1) {
          const zz = clamp(z + dz, nz - 1);
          for (let dy = -half; dy <= half; dy += 1) {
            const yy = clamp(y + dy, ny - 1);
            for (let dx = -half; dx <= half; dx += 1) {
              const xx = clamp(x + dx, nx - 1);
              window[count] = volume.data[index3d(volume.dims, xx, yy, zz)];
              count += 1;
            }
          }
        }
        const sorted = Array.from(window.subarray(0, count)).sort((a, b) => a - b);
        data[index3d(volume.dims, x, y, z)] = sorted[Math.floor(count / 2)];
      }
    }
  }
  return cloneVolume(volume, data);
}

// --- (c) Anisotropic diffusion (Perona-Malik) -------------------------------
// Full-fidelity (for a 6-connected discrete lattice): the classic explicit
// update I(t+1) = I(t) + lambda * sum_neighbors c(|grad|) * grad, with the
// exponential or quadratic edge-stopping function g(.). This is the standard
// edge-preserving ultrasound despeckling filter from Perona & Malik (1990),
// applied here on the 6-neighbor 3-D lattice instead of the original 2-D image.
function edgeStop(gradient, kappa, kind) {
  if (kind === "quadratic") return 1 / (1 + (gradient / kappa) ** 2);
  return Math.exp(-((gradient / kappa) ** 2));
}

export function anisotropicDiffusion3D(volume, options = {}) {
  const iterations = Math.max(1, Math.round(options.iterations ?? 6));
  const kappa = Math.max(1e-3, Number(options.kappa ?? 20));
  const lambda = Math.max(0, Math.min(0.2, Number(options.lambda ?? 0.12)));
  const edgeFunction = options.edgeFunction === "quadratic" ? "quadratic" : "exponential";
  const [nx, ny, nz] = volume.dims;
  let data = new Float32Array(volume.data);
  const neighborOffsets = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  for (let iter = 0; iter < iterations; iter += 1) {
    const next = new Float32Array(data.length);
    for (let z = 0; z < nz; z += 1) {
      for (let y = 0; y < ny; y += 1) {
        for (let x = 0; x < nx; x += 1) {
          const idx = index3d(volume.dims, x, y, z);
          const center = data[idx];
          let update = 0;
          for (const [dx, dy, dz] of neighborOffsets) {
            const xx = x + dx;
            const yy = y + dy;
            const zz = z + dz;
            if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz) continue;
            const neighbor = data[index3d(volume.dims, xx, yy, zz)];
            const gradient = neighbor - center;
            update += edgeStop(Math.abs(gradient), kappa, edgeFunction) * gradient;
          }
          next[idx] = center + lambda * update;
        }
      }
    }
    data = next;
  }
  return cloneVolume(volume, data);
}

// --- (d) Monogenic-signal-inspired local phase congruency -------------------
// SIMPLIFIED 2-D APPROXIMATION. The real monogenic signal (Felsberg & Sommer,
// 2001) applies a Riesz transform (a rotation-invariant, vector-valued analogue
// of the Hilbert transform) to a bank of band-pass (log-Gabor) filtered images
// at multiple scales, then fuses the resulting local energies into a scale- and
// contrast-invariant phase congruency map (Kovesi, 1999). Implementing the true
// frequency-domain Riesz transform and multi-scale log-Gabor bank is out of
// scope for an in-browser prototype. Instead, per axial (z) slice, we build a
// single-scale quadrature-like triplet:
//   - an "even" (band-pass) channel = I - Gaussian_sigma(I)
//   - two "odd" channels approximated by central finite-difference derivatives
//     along x and y of the raw image (a crude stand-in for the two Riesz
//     components, which are true 90-degree-phase-shifted filters)
// and combine them into a single-scale local-energy-based phase congruency
// proxy: pc = ||(even, oddX, oddY)|| / (|I| + eps). This captures the same
// *intuition* (edges/features produce high local "energy" across the
// quadrature channels regardless of absolute contrast) but is NOT the
// literature technique; it omits multi-scale fusion, the true Riesz kernel,
// and noise-floor compensation used in real phase congruency formulations.
export function phaseCongruency2D(volume, options = {}) {
  const sigma = Math.max(0.5, Number(options.sigma ?? 1.6));
  const [nx, ny, nz] = volume.dims;
  const smoothed = gaussianSmooth3D(volume, { sigma }).data;
  const data = new Float32Array(volume.data.length);
  const eps = 1e-3;
  for (let z = 0; z < nz; z += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        const idx = index3d(volume.dims, x, y, z);
        const bandpass = volume.data[idx] - smoothed[idx];
        const xp = Math.min(nx - 1, x + 1);
        const xm = Math.max(0, x - 1);
        const yp = Math.min(ny - 1, y + 1);
        const ym = Math.max(0, y - 1);
        const oddX = (volume.data[index3d(volume.dims, xp, y, z)] - volume.data[index3d(volume.dims, xm, y, z)]) / 2;
        const oddY = (volume.data[index3d(volume.dims, x, yp, z)] - volume.data[index3d(volume.dims, x, ym, z)]) / 2;
        const localEnergy = Math.hypot(bandpass, oddX, oddY);
        const amplitude = Math.abs(volume.data[idx]) + eps;
        data[idx] = Math.max(0, Math.min(255, (localEnergy / amplitude) * 255));
      }
    }
  }
  return cloneVolume(volume, data);
}

// --- (e) Wavelet-based despeckling ------------------------------------------
// SIMPLIFIED: a real, invertible single-level 2-D Haar discrete wavelet
// transform (per axial z-slice) with soft/hard thresholding of the detail
// (LH/HL/HH) coefficients followed by the exact inverse transform. This is a
// genuine (if shallow) wavelet shrinkage despeckling pipeline in the spirit of
// Donoho's universal threshold, not a placeholder: the forward/inverse Haar
// steps are mathematically exact for even-sized slices. Simplifications versus
// the wider wavelet-despeckling literature: only one decomposition level (real
// systems typically use 3-5 levels of a smoother biorthogonal/Daubechies
// wavelet), and slices with an odd dimension leave one trailing row/column
// outside the transformed region (documented, minor edge effect).
function haarRowTransform(matrix, rows, cols) {
  const evenCols = Math.floor(cols / 2) * 2;
  const half = evenCols / 2;
  const output = matrix.map((row) => Float32Array.from(row));
  for (let r = 0; r < rows; r += 1) {
    const row = matrix[r];
    const newRow = output[r];
    for (let i = 0; i < half; i += 1) {
      const a = row[2 * i];
      const b = row[2 * i + 1];
      newRow[i] = (a + b) / Math.SQRT2;
      newRow[half + i] = (a - b) / Math.SQRT2;
    }
  }
  return output;
}

function haarRowInverse(matrix, rows, cols) {
  const evenCols = Math.floor(cols / 2) * 2;
  const half = evenCols / 2;
  const output = matrix.map((row) => Float32Array.from(row));
  for (let r = 0; r < rows; r += 1) {
    const row = matrix[r];
    const newRow = output[r];
    for (let i = 0; i < half; i += 1) {
      const a = row[i];
      const d = row[half + i];
      newRow[2 * i] = (a + d) / Math.SQRT2;
      newRow[2 * i + 1] = (a - d) / Math.SQRT2;
    }
  }
  return output;
}

function transposeMatrix(matrix, rows, cols) {
  const result = [];
  for (let c = 0; c < cols; c += 1) {
    const row = new Float32Array(rows);
    for (let r = 0; r < rows; r += 1) row[r] = matrix[r][c];
    result.push(row);
  }
  return result;
}

function softOrHardThreshold(value, t, mode) {
  if (Math.abs(value) <= t) return 0;
  if (mode === "hard") return value;
  return Math.sign(value) * (Math.abs(value) - t);
}

function estimateUniversalThreshold(full, rows, cols, halfRows, halfCols, multiplier) {
  const hh = [];
  for (let r = halfRows; r < rows; r += 1) {
    for (let c = halfCols; c < cols; c += 1) hh.push(full[r][c]);
  }
  if (!hh.length) return 0;
  hh.sort((a, b) => a - b);
  const median = hh[Math.floor(hh.length / 2)];
  const deviations = hh.map((v) => Math.abs(v - median)).sort((a, b) => a - b);
  const mad = deviations[Math.floor(deviations.length / 2)];
  const sigma = mad / 0.6745;
  const n = Math.max(2, rows * cols);
  return multiplier * sigma * Math.sqrt(2 * Math.log(n));
}

export function waveletDespeckle2D(volume, options = {}) {
  const mode = options.mode === "hard" ? "hard" : "soft";
  const manualThreshold = options.threshold;
  const thresholdScale = Number(options.thresholdScale ?? 1);
  const [nx, ny, nz] = volume.dims;
  const data = new Float32Array(volume.data.length);
  for (let z = 0; z < nz; z += 1) {
    const matrix = [];
    for (let y = 0; y < ny; y += 1) {
      const row = new Float32Array(nx);
      for (let x = 0; x < nx; x += 1) row[x] = volume.data[index3d(volume.dims, x, y, z)];
      matrix.push(row);
    }
    let full = haarRowTransform(matrix, ny, nx);
    full = transposeMatrix(full, ny, nx);
    full = haarRowTransform(full, nx, ny);
    full = transposeMatrix(full, nx, ny);
    const halfRows = Math.floor(ny / 2);
    const halfCols = Math.floor(nx / 2);
    const t = manualThreshold !== undefined && manualThreshold !== null
      ? Number(manualThreshold)
      : estimateUniversalThreshold(full, ny, nx, halfRows, halfCols, thresholdScale);
    for (let r = 0; r < ny; r += 1) {
      for (let c = 0; c < nx; c += 1) {
        if (r < halfRows && c < halfCols) continue;
        full[r][c] = softOrHardThreshold(full[r][c], t, mode);
      }
    }
    let inv = transposeMatrix(full, ny, nx);
    inv = haarRowInverse(inv, nx, ny);
    inv = transposeMatrix(inv, nx, ny);
    inv = haarRowInverse(inv, ny, nx);
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) data[index3d(volume.dims, x, y, z)] = inv[y][x];
    }
  }
  return cloneVolume(volume, data);
}

// --- (f) passthrough + chaining ---------------------------------------------
export function identityFilter(volume) {
  return cloneVolume(volume, new Float32Array(volume.data));
}

export function applyFilter(volume, type, params = {}) {
  if (type === "gaussian") return gaussianSmooth3D(volume, params);
  if (type === "median") return medianFilter3D(volume, params);
  if (type === "anisotropic") return anisotropicDiffusion3D(volume, params);
  if (type === "phase-congruency") return phaseCongruency2D(volume, params);
  if (type === "wavelet") return waveletDespeckle2D(volume, params);
  return identityFilter(volume);
}

// Applies an ordered list of {type, params} steps in sequence (chaining), e.g.
// [{type: "median", params: {kernelSize: 3}}, {type: "gaussian", params: {sigma: 1}}].
// "none" steps are skipped. Returns a new volume; does not mutate the input.
export function applyPreprocessingChain(volume, steps = []) {
  let current = volume;
  for (const step of steps) {
    if (!step || step.type === "none") continue;
    current = applyFilter(current, step.type, step.params ?? {});
  }
  return current === volume ? identityFilter(volume) : current;
}
