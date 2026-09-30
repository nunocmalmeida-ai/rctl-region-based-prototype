import test from "node:test";
import assert from "node:assert/strict";
import { createDemoVolume, embedObject } from "../src/core/volume.js";
import {
  templateMatchInit,
  buildRTable,
  accumulateHoughVotes,
  findAccumulatorPeaks,
  houghInit,
  icpAlign,
  icpInit,
  connectedComponents,
  pcaOrientation,
  blobPcaInit,
  manualInit,
  perturbPose,
  poseError
} from "../src/core/initialization.js";

function sphereVolume(dims, center, radius, intensity = 230) {
  const base = createDemoVolume(dims);
  const object = { shape: "sphere", radius, center, intensity, blend: 1, shadow: false, representation: "rigid" };
  return embedObject(base, object, { gaussian: 0, speckle: 0, dropout: 0, seed: 3, clip: true });
}

test("template matching scores the true embedded pose higher than an off-center pose and selects it as best candidate", () => {
  const center = [14, 11, 13];
  const radius = 4;
  const volume = sphereVolume([24, 24, 24], center, radius);
  const config = { shape: "sphere", representation: "rigid", radius, rotation: [0, 0, 0] };
  const candidates = templateMatchInit(volume, config, { metric: "ncc", levels: 2, gridStep: 2, maxCandidates: 3, threshold: 150 });
  assert.ok(candidates.length > 0);
  const best = candidates[0];
  const distanceToTruth = Math.hypot(...best.center.map((v, i) => v - center[i]));
  assert.ok(distanceToTruth <= 3, `expected best candidate near ${center}, got ${best.center}`);
  assert.ok(best.score >= candidates[candidates.length - 1].score);
});

test("generalized Hough accumulator peaks locate a known bright blob", () => {
  const center = [15, 12, 10];
  const radius = 3;
  const volume = sphereVolume([22, 22, 20], center, radius);
  const config = { shape: "sphere", representation: "rigid", radius, rotation: [0, 0, 0] };
  const rTable = buildRTable(config, 32);
  assert.ok(rTable.length > 10);
  const acc = accumulateHoughVotes(volume, rTable, { threshold: 150, binSize: 1, voxelStride: 1 });
  const peaks = findAccumulatorPeaks(acc, 3, 2);
  assert.ok(peaks.length > 0);
  const best = peaks[0];
  const distanceToTruth = Math.hypot(...best.center.map((v, i) => v - center[i]));
  assert.ok(distanceToTruth <= 4, `expected top peak near ${center}, got ${best.center}`);
});

test("houghInit returns ranked candidates with normalized scores", () => {
  const center = [13, 13, 13];
  const radius = 3;
  const volume = sphereVolume([20, 20, 20], center, radius);
  const config = { shape: "sphere", representation: "rigid", radius, rotation: [0, 0, 0] };
  const candidates = houghInit(volume, config, { threshold: 150, binSize: 1, sampleCount: 24, maxCandidates: 2 });
  assert.ok(candidates.length > 0);
  assert.equal(candidates[0].method, "hough");
  assert.ok(candidates[0].score <= 1 && candidates[0].score > 0);
});

test("icpAlign recovers a known translation offset between two point sets", () => {
  const target = [];
  for (let i = 0; i < 40; i += 1) {
    const angle = (i / 40) * Math.PI * 2;
    target.push([Math.cos(angle) * 5, Math.sin(angle) * 5, (i % 5) - 2]);
  }
  const trueOffset = [3, -2, 1.5];
  const source = target.map((p) => [p[0] - trueOffset[0], p[1] - trueOffset[1], p[2] - trueOffset[2]]);
  const { t, history } = icpAlign(source, target, { maxIterations: 15, tolerance: 1e-4, outlierRejectionPercentile: 1 });
  assert.ok(history.length > 0);
  assert.ok(Math.hypot(...t.map((v, i) => v - trueOffset[i])) < 0.2, `expected recovered translation near ${trueOffset}, got ${t}`);
  assert.ok(history[history.length - 1].meanError < 0.2);
});

test("icpInit aligns a sampled model surface to a bright synthetic blob and recovers its center", () => {
  const center = [15, 14, 12];
  const radius = 5;
  const volume = sphereVolume([26, 26, 24], center, radius, 240);
  const config = { shape: "sphere", representation: "rigid", radius, rotation: [0, 0, 0] };
  const candidates = icpInit(volume, config, { threshold: 150, maxIterations: 25, maxVolumePoints: 120, maxModelPoints: 60 });
  assert.equal(candidates.length, 1);
  const distanceToTruth = Math.hypot(...candidates[0].center.map((v, i) => v - center[i]));
  assert.ok(distanceToTruth <= 2.5, `expected ICP center near ${center}, got ${candidates[0].center}`);
});

test("connected-component extraction isolates a single bright blob", () => {
  const center = [10, 9, 8];
  const radius = 3;
  const volume = sphereVolume([18, 18, 16], center, radius, 240);
  const components = connectedComponents(volume, 150, 6);
  assert.ok(components.length >= 1);
  const largest = components.slice().sort((a, b) => b.length - a.length)[0];
  assert.ok(largest.length > 5);
});

test("PCA orientation recovers principal axis of an elongated synthetic point cloud", () => {
  const points = [];
  for (let i = -10; i <= 10; i += 1) points.push([i, i * 0.01, 0]);
  const pca = pcaOrientation(points);
  const dominant = pca.axes[0];
  const alignment = Math.abs(dominant[0]);
  assert.ok(alignment > 0.99, `expected dominant axis aligned with x, got ${dominant}`);
});

test("blob+PCA heuristic returns a centroid close to the embedded object center", () => {
  const center = [12, 10, 9];
  const radius = 4;
  const volume = sphereVolume([20, 20, 18], center, radius, 240);
  const config = { shape: "sphere", radius };
  const candidates = blobPcaInit(volume, config, { threshold: 150, minVolume: 5 });
  assert.ok(candidates.length > 0);
  const distanceToTruth = Math.hypot(...candidates[0].center.map((v, i) => v - center[i]));
  assert.ok(distanceToTruth <= 1.5, `expected centroid near ${center}, got ${candidates[0].center}`);
});

test("manual initialization wraps the provided pose unchanged as a single candidate", () => {
  const pose = { center: [5, 6, 7], radius: 4, rotation: [10, 0, 0] };
  const candidates = manualInit(pose);
  assert.equal(candidates.length, 1);
  assert.deepEqual(candidates[0].center, pose.center);
  assert.equal(candidates[0].method, "manual");
});

test("perturbPose is deterministic for a fixed pseudo-random function and poseError measures the induced offset", () => {
  const truth = { center: [10, 10, 10], rotation: [0, 0, 0] };
  const sequence = [0.75, 0.25, 0.9, 0.1, 0.6, 0.4];
  let index = 0;
  const fakeRandom = () => sequence[index++ % sequence.length];
  const perturbed = perturbPose({ center: truth.center, radius: 4, rotation: truth.rotation }, { translation: 4, rotation: 20 }, fakeRandom);
  const error = poseError(perturbed, truth);
  assert.ok(error.translationError > 0);
  assert.ok(error.rotationError > 0);
});
