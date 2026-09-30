// Stage 1: coarse / global initialization for the two-stage pipeline.
//
//   Stage 1 (this file): coarse re-acquisition — produce a candidate initial
//   pose (or several ranked candidates) from the whole volume with no
//   assumption about where the object is.
//   Stage 2 (fitting.js, unchanged): local iterative refinement (coordinate
//   descent) starting from a seed pose, analogous to a per-frame EKF
//   measurement update.
//
// This module reuses the existing procedural model representation from
// volume.js (`containsObject`) instead of duplicating object/mesh geometry, so
// every coarse method below "renders" or "samples" the very same implicit
// model used by fitting.js and embedObject.
//
// Five selectable coarse-initialization methods are implemented:
//   (a) templateMatchInit  — multi-resolution (coarse-to-fine) translation +
//       small-rotation-set template correlation (SSD / NCC / thresholded
//       overlap) against a voxelized rendering of the model.
//   (b) houghInit           — generalized-Hough-transform-style voting: an
//       R-table of vectors from the model reference point to its surface is
//       used to cast votes from bright/edge voxels into a discretized
//       translation accumulator; peaks are returned as candidates.
//   (c) icpInit             — bright-voxel point cloud (optionally restricted
//       to the largest connected blob) aligned to a sampled model surface
//       point cloud via centroid pre-alignment + iterative closest point
//       (nearest-neighbor correspondence + Kabsch/SVD rigid solve).
//   (d) blobPcaInit          — thresholded connected-component blob extraction
//       + centroid/PCA-orientation heuristic (the fast/naive baseline).
//   (e) manualInit           — wraps a user-specified pose as a "candidate" so
//       manual seeding can be directly compared against the automatic methods.
//
// Every method returns an array of candidate poses:
//   { method, center: [x,y,z], radius, scale, rotation: [rx,ry,rz], score, rank, ... }
// ranked best-first, so the UI can show the top candidate(s) and their scores
// before handing the winner off to Stage 2 (fitVolume in fitting.js).
import { containsObject, index3d } from "./volume.js";

// ---------------------------------------------------------------------------
// Small linear-algebra helpers (3-vectors and 3x3 matrices as plain arrays).
// ---------------------------------------------------------------------------
function sub3(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function add3(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function dist3(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
function mean3(points) {
  if (!points.length) return [0, 0, 0];
  const sum = points.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]], [0, 0, 0]);
  return sum.map((v) => v / points.length);
}
function mulVec3(matrix, vector) {
  return [
    matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
    matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
    matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2]
  ];
}
function mul3(a, b) {
  const result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 3; c += 1) {
      result[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
    }
  }
  return result;
}
function transpose3(matrix) {
  return [
    [matrix[0][0], matrix[1][0], matrix[2][0]],
    [matrix[0][1], matrix[1][1], matrix[2][1]],
    [matrix[0][2], matrix[1][2], matrix[2][2]]
  ];
}
function det3(matrix) {
  const [[a, b, c], [d, e, f], [g, h, i]] = matrix;
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

// Jacobi eigenvalue algorithm for a real symmetric 3x3 matrix. This is a
// simple, classical, and numerically robust way to diagonalize small
// symmetric matrices without a general SVD routine; used both directly (PCA
// orientation) and as a building block for a minimal 3x3 SVD (Kabsch below).
function jacobiEigenSymmetric3(matrixIn) {
  const a = matrixIn.map((row) => row.slice());
  let v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 60; sweep += 1) {
    const off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off < 1e-12) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(a[p][q]) < 1e-14) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const sign = theta >= 0 ? 1 : -1;
      const t = sign / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      const app = a[p][p];
      const aqq = a[q][q];
      const apq = a[p][q];
      a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
      a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
      a[p][q] = 0;
      a[q][p] = 0;
      for (let k = 0; k < 3; k += 1) {
        if (k !== p && k !== q) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[p][k] = a[k][p];
          a[k][q] = s * akp + c * akq;
          a[q][k] = a[k][q];
        }
      }
      for (let k = 0; k < 3; k += 1) {
        const vkp = v[k][p];
        const vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq;
        v[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const eigenvalues = [a[0][0], a[1][1], a[2][2]];
  const eigenvectors = [
    [v[0][0], v[1][0], v[2][0]],
    [v[0][1], v[1][1], v[2][1]],
    [v[0][2], v[1][2], v[2][2]]
  ];
  return { eigenvalues, eigenvectors };
}

// Minimal Kabsch rigid-alignment solve: given corresponding centered point
// sets, builds the 3x3 cross-covariance H, derives its SVD H = U*Sigma*V^T
// from the eigendecomposition of H^T*H (a standard, if minimal, way to obtain
// a 3x3 SVD without a general-purpose linear-algebra library), and returns the
// optimal rotation R = V*U^T (with a reflection fix if det(R) < 0) and
// translation t mapping `sourcePoints` onto `targetPoints` in a least-squares
// sense.
function kabsch(sourcePoints, targetPoints) {
  const centroidP = mean3(sourcePoints);
  const centroidQ = mean3(targetPoints);
  const p = sourcePoints.map((point) => sub3(point, centroidP));
  const q = targetPoints.map((point) => sub3(point, centroidQ));
  const h = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < p.length; i += 1) {
    for (let a = 0; a < 3; a += 1) {
      for (let b = 0; b < 3; b += 1) h[a][b] += p[i][a] * q[i][b];
    }
  }
  const hth = mul3(transpose3(h), h);
  const { eigenvalues, eigenvectors } = jacobiEigenSymmetric3(hth);
  const order = [0, 1, 2].sort((a, b) => eigenvalues[b] - eigenvalues[a]);
  const singular = order.map((i) => Math.sqrt(Math.max(0, eigenvalues[i])));
  const vMat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  order.forEach((eigIndex, col) => {
    const vector = eigenvectors[eigIndex];
    vMat[0][col] = vector[0];
    vMat[1][col] = vector[1];
    vMat[2][col] = vector[2];
  });
  const uMat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let col = 0; col < 3; col += 1) {
    const v = [vMat[0][col], vMat[1][col], vMat[2][col]];
    const hv = mulVec3(h, v);
    const sigma = singular[col];
    if (sigma > 1e-9) {
      uMat[0][col] = hv[0] / sigma;
      uMat[1][col] = hv[1] / sigma;
      uMat[2][col] = hv[2] / sigma;
    } else {
      uMat[0][col] = col === 0 ? 1 : 0;
      uMat[1][col] = col === 1 ? 1 : 0;
      uMat[2][col] = col === 2 ? 1 : 0;
    }
  }
  let r = mul3(uMat, transpose3(vMat));
  if (det3(r) < 0) {
    for (let row = 0; row < 3; row += 1) {
      vMat[row][2] *= -1;
      uMat[row][2] *= -1;
    }
    r = mul3(uMat, transpose3(vMat));
  }
  const t = sub3(centroidQ, mulVec3(r, centroidP));
  return { R: r, t };
}

// Approximate extraction of Tait-Bryan (X then Y then Z) Euler angles in
// degrees from a rotation matrix, matching the composition order used by
// `rotateToLocal` in volume.js. This is only used to report/seed an
// orientation estimate for the coarse stage; Stage 2's coordinate-descent
// refinement (fitting.js) treats rotation as a fixed external parameter, so
// exact Euler-angle round-tripping is not required for correctness, only for
// a reasonable display/seed value.
function matrixToEulerDegrees(matrix) {
  const sy = -matrix[2][0];
  let x;
  let y;
  let z;
  if (Math.abs(sy) < 0.999999) {
    y = Math.asin(Math.max(-1, Math.min(1, sy)));
    x = Math.atan2(matrix[2][1], matrix[2][2]);
    z = Math.atan2(matrix[1][0], matrix[0][0]);
  } else {
    y = Math.asin(Math.max(-1, Math.min(1, sy)));
    x = Math.atan2(-matrix[1][2], matrix[1][1]);
    z = 0;
  }
  return [x, y, z].map((angle) => (angle * 180) / Math.PI);
}

function ensureProperRotation(matrix) {
  if (det3(matrix) < 0) {
    return [
      [matrix[0][0], matrix[0][1], -matrix[0][2]],
      [matrix[1][0], matrix[1][1], -matrix[1][2]],
      [matrix[2][0], matrix[2][1], -matrix[2][2]]
    ];
  }
  return matrix;
}

// ---------------------------------------------------------------------------
// Shared model-surface sampling (reused by the Hough and ICP methods).
// ---------------------------------------------------------------------------
function fibonacciSphereDirections(count) {
  const points = [];
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const n = Math.max(2, count);
  for (let i = 0; i < n; i += 1) {
    const yCoord = 1 - (i / (n - 1)) * 2;
    const radius = Math.sqrt(Math.max(0, 1 - yCoord * yCoord));
    const theta = goldenAngle * i;
    points.push([Math.cos(theta) * radius, yCoord, Math.sin(theta) * radius]);
  }
  return points;
}

// Marches a ray outward from the model's reference point (its local origin)
// along `direction` and returns the first inside -> outside crossing found
// via bisection. Scanning for the first transition (rather than assuming the
// origin itself is inside) lets this work even for non-star-shaped-from-center
// models such as the "torus" primitive in volume.js, whose center voxel is the
// hole, not solid material.
function raySurfacePoint(direction, config) {
  const scale = config.scale ?? 1;
  const maxRadius = config.radius * 2.4 + 3;
  const steps = 48;
  let previousInside = false;
  for (let i = 0; i <= steps; i += 1) {
    const t = (i / steps) * maxRadius;
    const point = direction.map((d) => d * t);
    const inside = containsObject(point, config.shape, config.radius, config.representation, config.rotation ?? [0, 0, 0], scale);
    if (previousInside && !inside) {
      let lo = ((i - 1) / steps) * maxRadius;
      let hi = t;
      for (let bisect = 0; bisect < 16; bisect += 1) {
        const mid = (lo + hi) / 2;
        const midPoint = direction.map((d) => d * mid);
        const midInside = containsObject(midPoint, config.shape, config.radius, config.representation, config.rotation ?? [0, 0, 0], scale);
        if (midInside) lo = mid; else hi = mid;
      }
      return direction.map((d) => d * lo);
    }
    previousInside = inside;
  }
  return null;
}

export function sampleModelSurfacePoints(config, sampleCount = 64) {
  const directions = fibonacciSphereDirections(sampleCount);
  const points = [];
  for (const direction of directions) {
    const point = raySurfacePoint(direction, config);
    if (point) points.push(point);
  }
  return points;
}

// ---------------------------------------------------------------------------
// (a) Global exhaustive / multi-resolution template matching.
// ---------------------------------------------------------------------------
function buildModelMask(config, rotation, extent) {
  const size = extent * 2 + 1;
  const mask = new Uint8Array(size * size * size);
  let count = 0;
  const scale = config.scale ?? 1;
  for (let dz = -extent; dz <= extent; dz += 1) {
    for (let dy = -extent; dy <= extent; dy += 1) {
      for (let dx = -extent; dx <= extent; dx += 1) {
        const inside = containsObject([dx, dy, dz], config.shape, config.radius, config.representation, rotation, scale);
        if (inside) {
          mask[(dz + extent) * size * size + (dy + extent) * size + (dx + extent)] = 1;
          count += 1;
        }
      }
    }
  }
  return { mask, size, extent, count };
}

function scoreTemplateAtPose(volume, maskObj, center, metric, threshold) {
  const { mask, extent, size } = maskObj;
  const [nx, ny, nz] = volume.dims;
  let sumT = 0;
  let sumV = 0;
  let sumTT = 0;
  let sumVV = 0;
  let sumTV = 0;
  let sumSqDiff = 0;
  let n = 0;
  let insideCount = 0;
  let insideBright = 0;
  let outsideCount = 0;
  let outsideBright = 0;
  for (let dz = -extent; dz <= extent; dz += 1) {
    const z = center[2] + dz;
    if (z < 0 || z >= nz) continue;
    for (let dy = -extent; dy <= extent; dy += 1) {
      const y = center[1] + dy;
      if (y < 0 || y >= ny) continue;
      for (let dx = -extent; dx <= extent; dx += 1) {
        const x = center[0] + dx;
        if (x < 0 || x >= nx) continue;
        const templateValue = mask[(dz + extent) * size * size + (dy + extent) * size + (dx + extent)];
        const raw = volume.data[index3d(volume.dims, x, y, z)];
        const v = raw / 255;
        sumT += templateValue;
        sumV += v;
        sumTT += templateValue * templateValue;
        sumVV += v * v;
        sumTV += templateValue * v;
        sumSqDiff += (templateValue - v) ** 2;
        n += 1;
        if (templateValue) {
          insideCount += 1;
          if (raw >= threshold) insideBright += 1;
        } else {
          outsideCount += 1;
          if (raw >= threshold) outsideBright += 1;
        }
      }
    }
  }
  if (n === 0) return -Infinity;
  if (metric === "ssd") return -sumSqDiff / n;
  if (metric === "overlap") return insideBright / Math.max(1, insideCount) - 0.5 * (outsideBright / Math.max(1, outsideCount));
  const meanT = sumT / n;
  const meanV = sumV / n;
  const covariance = sumTV / n - meanT * meanV;
  const stdT = Math.sqrt(Math.max(0, sumTT / n - meanT * meanT));
  const stdV = Math.sqrt(Math.max(0, sumVV / n - meanV * meanV));
  return covariance / (stdT * stdV + 1e-6);
}

export function templateMatchInit(volume, config, options = {}) {
  const metric = options.metric ?? "ncc";
  const threshold = options.threshold ?? 160;
  const levels = Math.max(1, Math.round(options.levels ?? 3));
  const finestStep = Math.max(1, Math.round(options.gridStep ?? 2));
  const rotations = options.rotations && options.rotations.length ? options.rotations : [config.rotation ?? [0, 0, 0]];
  const extent = Math.max(2, Math.round(config.radius * 1.3));
  const maxCandidates = Math.max(1, Math.round(options.maxCandidates ?? 3));
  const [nx, ny, nz] = volume.dims;
  if (nx <= extent * 2 || ny <= extent * 2 || nz <= extent * 2) return [];

  const maskCache = new Map();
  const maskFor = (rotation) => {
    const key = rotation.join(",");
    if (!maskCache.has(key)) maskCache.set(key, buildModelMask(config, rotation, extent));
    return maskCache.get(key);
  };

  let candidatesAll = [];
  for (const rotation of rotations) {
    const maskObj = maskFor(rotation);
    let step = finestStep * 2 ** (levels - 1);
    let centers = [];
    for (let z = extent; z < nz - extent; z += step) {
      for (let y = extent; y < ny - extent; y += step) {
        for (let x = extent; x < nx - extent; x += step) centers.push([x, y, z]);
      }
    }
    let scored = centers.map((center) => ({ center, rotation, score: scoreTemplateAtPose(volume, maskObj, center, metric, threshold) }));
    scored.sort((a, b) => b.score - a.score);
    let top = scored.slice(0, maxCandidates * 3);
    for (let level = levels - 1; level > 0; level -= 1) {
      step = finestStep * 2 ** (level - 1);
      const refined = [];
      for (const candidate of top) {
        for (let dz = -1; dz <= 1; dz += 1) {
          for (let dy = -1; dy <= 1; dy += 1) {
            for (let dx = -1; dx <= 1; dx += 1) {
              const center = [candidate.center[0] + dx * step, candidate.center[1] + dy * step, candidate.center[2] + dz * step];
              if (center[0] < extent || center[0] >= nx - extent) continue;
              if (center[1] < extent || center[1] >= ny - extent) continue;
              if (center[2] < extent || center[2] >= nz - extent) continue;
              refined.push({ center, rotation, score: scoreTemplateAtPose(volume, maskObj, center, metric, threshold) });
            }
          }
        }
      }
      refined.sort((a, b) => b.score - a.score);
      top = refined.slice(0, maxCandidates * 3);
    }
    candidatesAll = candidatesAll.concat(top);
  }
  candidatesAll.sort((a, b) => b.score - a.score);
  return candidatesAll.slice(0, maxCandidates).map((candidate, i) => ({
    method: "template",
    center: candidate.center,
    radius: config.radius,
    scale: 1,
    rotation: candidate.rotation,
    score: candidate.score,
    rank: i + 1
  }));
}

// ---------------------------------------------------------------------------
// (b) Generalized-Hough-transform-style voting.
// ---------------------------------------------------------------------------
// SIMPLIFICATION: a classical generalized Hough transform indexes R-table
// entries by local boundary-gradient orientation, so each edge pixel/voxel
// only votes using the subset of table entries sharing its orientation, and
// the accumulator spans translation *and* rotation/scale. Here the R-table is
// a flat list of reference-to-surface vectors with no orientation indexing
// (every bright/edge voxel votes with every table vector), and the
// accumulator is translation-only (rotation is held fixed from `config`).
// This keeps the in-browser cost bounded while preserving the core idea:
// votes from genuine object-boundary voxels should coherently pile up near
// the true reference-point translation, while votes from noise/background
// scatter and do not form a strong peak.
export function buildRTable(config, sampleCount = 48) {
  return sampleModelSurfacePoints(config, sampleCount);
}

export function accumulateHoughVotes(volume, rTable, options = {}) {
  const threshold = options.threshold ?? 170;
  const binSize = Math.max(1, Math.round(options.binSize ?? 2));
  const stride = Math.max(1, Math.round(options.voxelStride ?? 1));
  const [nx, ny, nz] = volume.dims;
  const bx = Math.max(1, Math.ceil(nx / binSize));
  const by = Math.max(1, Math.ceil(ny / binSize));
  const bz = Math.max(1, Math.ceil(nz / binSize));
  const accumulator = new Float32Array(bx * by * bz);
  for (let z = 0; z < nz; z += stride) {
    for (let y = 0; y < ny; y += stride) {
      for (let x = 0; x < nx; x += stride) {
        const intensity = volume.data[index3d(volume.dims, x, y, z)];
        if (intensity < threshold) continue;
        for (const vector of rTable) {
          const cx = x - vector[0];
          const cy = y - vector[1];
          const cz = z - vector[2];
          const bxi = Math.floor(cx / binSize);
          const byi = Math.floor(cy / binSize);
          const bzi = Math.floor(cz / binSize);
          if (bxi < 0 || bxi >= bx || byi < 0 || byi >= by || bzi < 0 || bzi >= bz) continue;
          accumulator[bzi * bx * by + byi * bx + bxi] += 1;
        }
      }
    }
  }
  return { accumulator, dims: [bx, by, bz], binSize };
}

export function findAccumulatorPeaks(accumulatorObj, maxPeaks = 3, minDistanceBins = 2) {
  const { accumulator, dims, binSize } = accumulatorObj;
  const [bx, by] = dims;
  const entries = [];
  for (let i = 0; i < accumulator.length; i += 1) if (accumulator[i] > 0) entries.push(i);
  entries.sort((a, b) => accumulator[b] - accumulator[a]);
  const peaks = [];
  for (const index of entries) {
    const bxi = index % bx;
    const byi = Math.floor(index / bx) % by;
    const bzi = Math.floor(index / (bx * by));
    const tooClose = peaks.some((peak) => Math.hypot(peak.bin[0] - bxi, peak.bin[1] - byi, peak.bin[2] - bzi) < minDistanceBins);
    if (tooClose) continue;
    peaks.push({ bin: [bxi, byi, bzi], votes: accumulator[index], center: [(bxi + 0.5) * binSize, (byi + 0.5) * binSize, (bzi + 0.5) * binSize] });
    if (peaks.length >= maxPeaks) break;
  }
  return peaks;
}

export function houghInit(volume, config, options = {}) {
  const rTable = buildRTable(config, options.sampleCount ?? 48);
  if (!rTable.length) return [];
  const accumulatorObj = accumulateHoughVotes(volume, rTable, options);
  const peaks = findAccumulatorPeaks(accumulatorObj, options.maxCandidates ?? 3, options.minDistanceBins ?? 2);
  const maxVotes = Math.max(1, ...peaks.map((peak) => peak.votes));
  return peaks.map((peak, i) => ({
    method: "hough",
    center: peak.center,
    radius: config.radius,
    scale: 1,
    rotation: config.rotation ?? [0, 0, 0],
    score: peak.votes / maxVotes,
    votes: peak.votes,
    rank: i + 1
  }));
}

// ---------------------------------------------------------------------------
// (c) ICP-style point-set alignment.
// ---------------------------------------------------------------------------
export function extractBrightVoxels(volume, threshold) {
  const points = [];
  const [nx, ny, nz] = volume.dims;
  for (let z = 0; z < nz; z += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        if (volume.data[index3d(volume.dims, x, y, z)] >= threshold) points.push([x, y, z]);
      }
    }
  }
  return points;
}

const CONNECTIVITY_OFFSETS_6 = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
const CONNECTIVITY_OFFSETS_18 = [
  ...CONNECTIVITY_OFFSETS_6,
  [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
  [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
  [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
];
const CONNECTIVITY_OFFSETS_26 = [
  ...CONNECTIVITY_OFFSETS_18,
  [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
  [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
];

export function connectedComponents(volume, threshold, connectivity = 6) {
  const offsets = connectivity === 26 ? CONNECTIVITY_OFFSETS_26 : connectivity === 18 ? CONNECTIVITY_OFFSETS_18 : CONNECTIVITY_OFFSETS_6;
  const [nx, ny, nz] = volume.dims;
  const visited = new Uint8Array(nx * ny * nz);
  const components = [];
  for (let z = 0; z < nz; z += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        const idx = index3d(volume.dims, x, y, z);
        if (visited[idx] || volume.data[idx] < threshold) continue;
        const queue = [[x, y, z]];
        visited[idx] = 1;
        const points = [];
        while (queue.length) {
          const [cx, cy, cz] = queue.pop();
          points.push([cx, cy, cz]);
          for (const [dx, dy, dz] of offsets) {
            const nxp = cx + dx;
            const nyp = cy + dy;
            const nzp = cz + dz;
            if (nxp < 0 || nxp >= nx || nyp < 0 || nyp >= ny || nzp < 0 || nzp >= nz) continue;
            const nidx = index3d(volume.dims, nxp, nyp, nzp);
            if (visited[nidx] || volume.data[nidx] < threshold) continue;
            visited[nidx] = 1;
            queue.push([nxp, nyp, nzp]);
          }
        }
        components.push(points);
      }
    }
  }
  return components;
}

function nearestPoint(point, candidates) {
  let best = candidates[0];
  let bestDist = Infinity;
  for (const candidate of candidates) {
    const d = dist3(point, candidate);
    if (d < bestDist) {
      bestDist = d;
      best = candidate;
    }
  }
  return best;
}

function subsamplePoints(points, maxCount) {
  if (points.length <= maxCount) return points;
  const stride = points.length / maxCount;
  const result = [];
  for (let i = 0; i < maxCount; i += 1) result.push(points[Math.floor(i * stride)]);
  return result;
}

// Simplified ICP: (1) bounding-box/centroid pre-alignment to satisfy ICP's
// well-known requirement of a reasonable initial alignment, then (2) iterate
// nearest-neighbor correspondence + Kabsch/SVD rigid-transform solve, with a
// percentile-based distance cutoff for crude outlier rejection.
export function icpAlign(sourcePoints, targetPoints, options = {}) {
  const maxIterations = Math.max(1, Math.round(options.maxIterations ?? 20));
  const tolerance = Math.max(0, Number(options.tolerance ?? 1e-3));
  const outlierRejectionPercentile = Math.max(0.1, Math.min(1, Number(options.outlierRejectionPercentile ?? 0.9)));
  const sourceCentroid = mean3(sourcePoints);
  const targetCentroid = mean3(targetPoints);
  const initialTranslation = sub3(targetCentroid, sourceCentroid);
  let R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  let t = initialTranslation;
  let current = sourcePoints.map((point) => add3(point, initialTranslation));
  const history = [];
  let previousError = Infinity;
  for (let iter = 0; iter < maxIterations; iter += 1) {
    const correspondences = current.map((point) => nearestPoint(point, targetPoints));
    const distances = current.map((point, i) => dist3(point, correspondences[i]));
    const sortedDistances = [...distances].sort((a, b) => a - b);
    const cutoffIndex = Math.max(0, Math.min(sortedDistances.length - 1, Math.floor(sortedDistances.length * outlierRejectionPercentile) - 1));
    const cutoff = sortedDistances[cutoffIndex];
    const inlierSource = [];
    const inlierTarget = [];
    for (let i = 0; i < current.length; i += 1) {
      if (distances[i] <= cutoff) {
        inlierSource.push(current[i]);
        inlierTarget.push(correspondences[i]);
      }
    }
    if (inlierSource.length < 3) break;
    const { R: stepR, t: stepT } = kabsch(inlierSource, inlierTarget);
    current = current.map((point) => add3(mulVec3(stepR, point), stepT));
    R = mul3(stepR, R);
    t = add3(mulVec3(stepR, t), stepT);
    const meanError = distances.reduce((sum, d) => sum + d, 0) / Math.max(1, distances.length);
    history.push({ iteration: iter + 1, meanError });
    if (Math.abs(previousError - meanError) < tolerance) break;
    previousError = meanError;
  }
  return { R, t, history, alignedPoints: current };
}

export function icpInit(volume, config, options = {}) {
  const threshold = options.threshold ?? 170;
  const connectivity = options.connectivity ?? 6;
  const useLargestBlob = options.useLargestBlob ?? true;
  const maxVolumePoints = Math.max(4, Math.round(options.maxVolumePoints ?? 150));
  const maxModelPoints = Math.max(4, Math.round(options.maxModelPoints ?? 80));

  let volumePoints;
  if (useLargestBlob) {
    const components = connectedComponents(volume, threshold, connectivity);
    if (!components.length) return [];
    components.sort((a, b) => b.length - a.length);
    volumePoints = components[0];
  } else {
    volumePoints = extractBrightVoxels(volume, threshold);
  }
  if (volumePoints.length < 4) return [];
  volumePoints = subsamplePoints(volumePoints, maxVolumePoints);
  const modelPoints = sampleModelSurfacePoints(config, maxModelPoints);
  if (modelPoints.length < 4) return [];

  const { R, t, history } = icpAlign(modelPoints, volumePoints, options);
  const rotation = matrixToEulerDegrees(ensureProperRotation(R));
  const finalError = history.length ? history[history.length - 1].meanError : null;
  return [{
    method: "icp",
    center: t,
    radius: config.radius,
    scale: 1,
    rotation,
    score: finalError !== null ? 1 / (1 + finalError) : 0,
    meanResidual: finalError,
    iterations: history.length,
    rank: 1
  }];
}

// ---------------------------------------------------------------------------
// (d) Blob/connected-component centroid + PCA-orientation heuristic.
// ---------------------------------------------------------------------------
export function pcaOrientation(points) {
  const centroid = mean3(points);
  const covariance = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const point of points) {
    const d = sub3(point, centroid);
    for (let a = 0; a < 3; a += 1) {
      for (let b = 0; b < 3; b += 1) covariance[a][b] += d[a] * d[b];
    }
  }
  const n = Math.max(1, points.length);
  for (let a = 0; a < 3; a += 1) {
    for (let b = 0; b < 3; b += 1) covariance[a][b] /= n;
  }
  const { eigenvalues, eigenvectors } = jacobiEigenSymmetric3(covariance);
  const order = [0, 1, 2].sort((a, b) => eigenvalues[b] - eigenvalues[a]);
  return {
    centroid,
    eigenvalues: order.map((i) => eigenvalues[i]),
    axes: order.map((i) => eigenvectors[i])
  };
}

export function blobPcaInit(volume, config, options = {}) {
  const threshold = options.threshold ?? 170;
  const minVolume = options.minVolume ?? 5;
  const maxVolume = options.maxVolume ?? Infinity;
  const connectivity = options.connectivity ?? 6;
  const maxCandidates = Math.max(1, Math.round(options.maxCandidates ?? 1));
  const components = connectedComponents(volume, threshold, connectivity)
    .filter((points) => points.length >= minVolume && points.length <= maxVolume);
  if (!components.length) return [];
  components.sort((a, b) => b.length - a.length);
  return components.slice(0, maxCandidates).map((points, i) => {
    const pca = pcaOrientation(points);
    const rotationMatrix = ensureProperRotation([
      [pca.axes[0][0], pca.axes[1][0], pca.axes[2][0]],
      [pca.axes[0][1], pca.axes[1][1], pca.axes[2][1]],
      [pca.axes[0][2], pca.axes[1][2], pca.axes[2][2]]
    ]);
    const rotation = matrixToEulerDegrees(rotationMatrix);
    const estimatedRadius = Math.cbrt((3 * points.length) / (4 * Math.PI));
    return {
      method: "blob-pca",
      center: pca.centroid,
      radius: config.radius ?? estimatedRadius,
      scale: 1,
      rotation,
      score: points.length,
      voxelCount: points.length,
      rank: i + 1
    };
  });
}

// ---------------------------------------------------------------------------
// (e) Manual / user-specified initial pose.
// ---------------------------------------------------------------------------
export function manualInit(pose) {
  return [{
    method: "manual",
    center: [...pose.center],
    radius: pose.radius,
    scale: pose.scale ?? 1,
    rotation: pose.rotation ? [...pose.rotation] : [0, 0, 0],
    score: 1,
    rank: 1
  }];
}

// ---------------------------------------------------------------------------
// Shared helpers used by the UI (app.js).
// ---------------------------------------------------------------------------
export const INITIALIZATION_METHODS = ["template", "hough", "icp", "blob-pca", "manual"];

export function runInitialization(method, volume, config, options = {}) {
  if (method === "template") return templateMatchInit(volume, config, options);
  if (method === "hough") return houghInit(volume, config, options);
  if (method === "icp") return icpInit(volume, config, options);
  if (method === "blob-pca") return blobPcaInit(volume, config, options);
  if (method === "manual") return manualInit(options.manualPose ?? { center: volume.dims.map((d) => (d - 1) / 2), radius: config.radius });
  throw new Error(`Unknown initialization method: ${method}`);
}

// Synthetically offsets a (typically ground-truth) pose by a random
// translation/rotation so the user can test how far off a naive start is and
// how well each coarse method recovers from it.
export function perturbPose(pose, options = {}, randomFn = Math.random) {
  const translation = Number(options.translation ?? 0);
  const rotation = Number(options.rotation ?? 0);
  const signedRandom = () => randomFn() * 2 - 1;
  return {
    center: pose.center.map((value) => value + signedRandom() * translation),
    radius: pose.radius,
    scale: pose.scale ?? 1,
    rotation: (pose.rotation ?? [0, 0, 0]).map((value) => value + signedRandom() * rotation)
  };
}

function angleDifferenceDegrees(a, b) {
  return (((a - b) % 360) + 540) % 360 - 180;
}

// Translation and (if the pose carries an orientation) rotation error between
// a candidate pose and a known ground-truth pose — used to report the
// initialization error BEFORE handing off to Stage 2 refinement.
export function poseError(pose, truth) {
  const translationError = Math.hypot(...pose.center.map((value, i) => value - truth.center[i]));
  let rotationError = null;
  if (Array.isArray(pose.rotation) && Array.isArray(truth.rotation)) {
    const diffs = pose.rotation.map((value, i) => angleDifferenceDegrees(value, truth.rotation[i]));
    rotationError = Math.hypot(...diffs);
  }
  return { translationError, rotationError };
}
