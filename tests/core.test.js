import test from "node:test";
import assert from "node:assert/strict";
import { compareMasks, fitVolume } from "../src/core/fitting.js";
import { createDemoVolume, embedObject, loadNpyUint8, loadUint8Volume } from "../src/core/volume.js";

test("demo volume and seeded embedding are reproducible without mutating the base", () => {
  const base = createDemoVolume([20, 20, 20]);
  const original = new Float32Array(base.data);
  const object = { shape: "sphere", radius: 3, center: [11, 10, 10], intensity: 250, blend: 1, shadow: false };
  const settings = { gaussian: 0, speckle: 0, dropout: 0, seed: 9, clip: true };
  const first = embedObject(base, object, settings);
  const second = embedObject(base, object, settings);
  assert.deepEqual(first.data, second.data);
  assert.deepEqual(base.data, original);
  assert.equal(first.mask.reduce((sum, value) => sum + value, 0) > 0, true);
  assert.equal(first.data.some((value, index) => value !== base.data[index]), true);
});

test("local raw volume checks dimensions and exact byte count", () => {
  const loaded = loadUint8Volume(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]).buffer, [2, 2, 2]);
  assert.deepEqual(loaded.dims, [2, 2, 2]);
  assert.deepEqual(Array.from(loaded.data), [1, 2, 3, 4, 5, 6, 7, 8]);
  assert.throws(() => loadUint8Volume(new ArrayBuffer(7), [2, 2, 2]), /Expected 8 bytes/);
});

test("C-order 3D uint8 NPY arrays load in x-y-z UI dimension order", () => {
  const header = "{'descr': '|u1', 'fortran_order': False, 'shape': (2, 2, 2), }";
  const padded = `${header}${" ".repeat((16 - ((10 + header.length + 1) % 16)) % 16)}\n`;
  const prefix = new Uint8Array(10 + padded.length);
  prefix.set([0x93, 78, 85, 77, 80, 89, 1, 0, padded.length, 0]);
  prefix.set(new TextEncoder().encode(padded), 10);
  const file = new Uint8Array(prefix.length + 8);
  file.set(prefix);
  file.set([1, 2, 3, 4, 5, 6, 7, 8], prefix.length);
  const loaded = loadNpyUint8(file.buffer);
  assert.deepEqual(loaded.dims, [2, 2, 2]);
  assert.deepEqual(Array.from(loaded.data), [1, 2, 3, 4, 5, 6, 7, 8]);
});

test("coordinate descent returns finite history and synthetic overlap metrics", () => {
  const base = createDemoVolume([24, 24, 24]);
  const object = { shape: "sphere", radius: 4, center: [14, 10, 12], rotation: [0, 0, 0], intensity: 255, blend: 1, shadow: false };
  const embedded = embedObject(base, object, { seed: 2, clip: true });
  const config = { approach: "region", representation: "rigid", shape: "sphere", rotation: [0, 0, 0], searchRadius: 4, samples: 900, window: 2, regularization: 0, threshold: 100, robustLoss: "huber", iterations: 3 };
  const result = fitVolume(embedded, { center: [12, 12, 12], radius: 4 }, config);
  const metrics = compareMasks(embedded.dims, embedded.mask, result.pose, { ...config, truthCenter: object.center, truthRadius: object.radius });
  assert.equal(result.history.length, 3);
  assert.ok(Number.isFinite(result.score));
  assert.ok(metrics.dice >= 0 && metrics.dice <= 1);
  assert.ok(metrics.translationError >= 0);
});
