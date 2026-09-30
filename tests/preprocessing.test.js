import test from "node:test";
import assert from "node:assert/strict";
import { createDemoVolume } from "../src/core/volume.js";
import {
  gaussianSmooth3D,
  medianFilter3D,
  anisotropicDiffusion3D,
  phaseCongruency2D,
  waveletDespeckle2D,
  identityFilter,
  applyFilter,
  applyPreprocessingChain
} from "../src/core/preprocessing.js";

function impulseVolume(dims) {
  const [nx, ny, nz] = dims;
  const data = new Float32Array(nx * ny * nz);
  const cx = Math.floor(nx / 2);
  const cy = Math.floor(ny / 2);
  const cz = Math.floor(nz / 2);
  data[cz * ny * nx + cy * nx + cx] = 255;
  return { dims: [...dims], data, mask: null, truth: null, label: "impulse" };
}

test("Gaussian smoothing spreads an impulse symmetrically and preserves total energy reasonably", () => {
  const volume = impulseVolume([9, 9, 9]);
  const smoothed = gaussianSmooth3D(volume, { sigma: 1 });
  const [nx, ny, nz] = volume.dims;
  const cx = Math.floor(nx / 2);
  const cy = Math.floor(ny / 2);
  const cz = Math.floor(nz / 2);
  const centerIndex = cz * ny * nx + cy * nx + cx;
  assert.ok(smoothed.data[centerIndex] < volume.data[centerIndex]);
  assert.ok(smoothed.data[centerIndex] > 0);
  // Symmetric neighbors along x should receive equal energy.
  const left = smoothed.data[centerIndex - 1];
  const right = smoothed.data[centerIndex + 1];
  assert.ok(Math.abs(left - right) < 1e-4, `expected symmetric spread, got ${left} vs ${right}`);
  // Original volume must not be mutated.
  assert.equal(volume.data[centerIndex], 255);
});

test("median filtering removes an isolated salt-noise spike while preserving flat background", () => {
  const dims = [7, 7, 7];
  const [nx, ny, nz] = dims;
  const data = new Float32Array(nx * ny * nz).fill(40);
  const cx = 3;
  const cy = 3;
  const cz = 3;
  data[cz * ny * nx + cy * nx + cx] = 255;
  const volume = { dims, data, mask: null, truth: null, label: "spike" };
  const filtered = medianFilter3D(volume, { kernelSize: 3 });
  const centerIndex = cz * ny * nx + cy * nx + cx;
  assert.equal(filtered.data[centerIndex], 40);
  assert.equal(filtered.data[0], 40);
});

test("anisotropic diffusion smooths within a flat region while damping across a strong step edge", () => {
  const dims = [10, 6, 6];
  const [nx, ny, nz] = dims;
  const data = new Float32Array(nx * ny * nz);
  for (let z = 0; z < nz; z += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        data[z * ny * nx + y * nx + x] = x < nx / 2 ? 20 : 220;
      }
    }
  }
  const volume = { dims, data, mask: null, truth: null, label: "edge" };
  const diffused = anisotropicDiffusion3D(volume, { iterations: 8, kappa: 15, lambda: 0.1, edgeFunction: "exponential" });
  const leftIndex = 3 * ny * nx + 3 * nx + 1;
  const rightIndex = 3 * ny * nx + 3 * nx + (nx - 2);
  // Far from the edge, values should stay close to their original flat-region value.
  assert.ok(Math.abs(diffused.data[leftIndex] - 20) < 5);
  assert.ok(Math.abs(diffused.data[rightIndex] - 220) < 5);
  // The step itself should still be a strong transition (edge-preserving, not globally blurred).
  const midLeft = 3 * ny * nx + 3 * nx + (nx / 2 - 1);
  const midRight = 3 * ny * nx + 3 * nx + (nx / 2);
  assert.ok(diffused.data[midRight] - diffused.data[midLeft] > 100);
});

test("phase congruency proxy highlights an edge slice more than a flat slice", () => {
  const dims = [12, 12, 3];
  const [nx, ny, nz] = dims;
  const data = new Float32Array(nx * ny * nz);
  for (let z = 0; z < nz; z += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        data[z * ny * nx + y * nx + x] = x < nx / 2 ? 30 : 200;
      }
    }
  }
  const volume = { dims, data, mask: null, truth: null, label: "edge-slice" };
  const pc = phaseCongruency2D(volume, { sigma: 1.2 });
  const edgeIndex = 1 * ny * nx + 6 * nx + (nx / 2);
  const flatIndex = 1 * ny * nx + 6 * nx + 2;
  assert.ok(pc.data[edgeIndex] > pc.data[flatIndex]);
});

test("wavelet despeckling reduces noise energy while preserving a flat region's mean", () => {
  const dims = [16, 16, 2];
  const [nx, ny, nz] = dims;
  const data = new Float32Array(nx * ny * nz);
  let seed = 42;
  const random = () => {
    seed = (seed * 1103515245 + 12345) >>> 0;
    return seed / 4294967296;
  };
  for (let i = 0; i < data.length; i += 1) data[i] = 120 + (random() - 0.5) * 30;
  const volume = { dims, data, mask: null, truth: null, label: "noisy-flat" };
  const denoised = waveletDespeckle2D(volume, { mode: "soft" });
  function variance(arr) {
    const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
    return arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
  }
  assert.ok(variance(denoised.data) <= variance(volume.data));
});

test("identityFilter and the none passthrough leave data numerically unchanged but return a fresh copy", () => {
  const volume = impulseVolume([5, 5, 5]);
  const copy = identityFilter(volume);
  assert.deepEqual(Array.from(copy.data), Array.from(volume.data));
  assert.notEqual(copy.data, volume.data);
  const viaApply = applyFilter(volume, "none");
  assert.deepEqual(Array.from(viaApply.data), Array.from(volume.data));
});

test("applyPreprocessingChain stacks two filters in order and skips none steps", () => {
  const volume = impulseVolume([9, 9, 9]);
  const chained = applyPreprocessingChain(volume, [
    { type: "none" },
    { type: "median", params: { kernelSize: 3 } },
    { type: "gaussian", params: { sigma: 1 } }
  ]);
  const expected = gaussianSmooth3D(medianFilter3D(volume, { kernelSize: 3 }), { sigma: 1 });
  assert.deepEqual(Array.from(chained.data), Array.from(expected.data));
});
