import { containsObject, index3d } from "./volume.js";

function robust(value, kind) {
  if (kind === "l1") return Math.abs(value);
  if (kind === "quadratic") return value * value;
  const delta = 0.25;
  const absolute = Math.abs(value);
  return absolute <= delta ? 0.5 * value * value : delta * (absolute - 0.5 * delta);
}

function samplesForPose(volume, pose, config) {
  const [nx, ny, nz] = volume.dims;
  const [cx, cy, cz] = pose.center;
  const radius = pose.radius;
  const half = Math.ceil(radius * 1.55 + config.window);
  const stride = Math.max(1, Math.ceil(Math.cbrt((half * 2 + 1) ** 3 / config.samples)));
  const foreground = [];
  const background = [];
  const shell = [];
  let gradient = 0;
  let gradientCount = 0;
  const minX = Math.max(1, Math.floor(cx - half));
  const maxX = Math.min(nx - 2, Math.ceil(cx + half));
  const minY = Math.max(1, Math.floor(cy - half));
  const maxY = Math.min(ny - 2, Math.ceil(cy + half));
  const minZ = Math.max(1, Math.floor(cz - half));
  const maxZ = Math.min(nz - 2, Math.ceil(cz + half));
  for (let z = minZ; z <= maxZ; z += stride) {
    for (let y = minY; y <= maxY; y += stride) {
      for (let x = minX; x <= maxX; x += stride) {
        const index = index3d(volume.dims, x, y, z);
        const point = [x - cx, y - cy, z - cz];
        const inside = containsObject(point, config.shape, radius, config.representation, config.rotation, pose.scale);
        const intensity = volume.data[index] / 255;
        const signed = inside ? intensity - config.threshold / 255 : config.threshold / 255 - intensity;
        if (inside) foreground.push(intensity);
        const distance = Math.hypot(...point);
        if (distance >= radius * 1.05 && distance <= radius * 1.05 + config.window) background.push(intensity);
        if (distance >= radius * 0.78 && distance <= radius * 1.22) shell.push(intensity);
        if (Math.abs(signed) < 0.16) {
          const gx = volume.data[index + 1] - volume.data[index - 1];
          const gy = volume.data[index + nx] - volume.data[index - nx];
          const gz = volume.data[index + nx * ny] - volume.data[index - nx * ny];
          gradient += Math.hypot(gx, gy, gz) / (255 * 1.732);
          gradientCount += 1;
        }
      }
    }
  }
  return { foreground, background, shell, gradient: gradientCount ? gradient / gradientCount : 0 };
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function variance(values, average = mean(values)) {
  return values.length ? values.reduce((sum, value) => sum + (value - average) ** 2, 0) / values.length : 1;
}

export function scorePose(volume, pose, config) {
  const samples = samplesForPose(volume, pose, config);
  const inside = samples.foreground;
  const outside = samples.background;
  if (inside.length < 2 || outside.length < 2) return -1;
  const insideMean = mean(inside);
  const outsideMean = mean(outside);
  const contrast = Math.max(-1, Math.min(1, insideMean - outsideMean));
  const spread = Math.sqrt(variance(inside, insideMean));
  const thresholdPenalty = mean(inside.map((value) => robust(Math.max(0, config.threshold / 255 - value), config.robustLoss)));
  const region = contrast - 0.35 * spread - 0.5 * thresholdPenalty;
  const likelihood = Math.log((variance(outside, outsideMean) + 0.012) / (variance(inside, insideMean) + 0.012)) * 0.12 + contrast;
  const template = mean(samples.shell) - outsideMean;
  const edge = samples.gradient + contrast * 0.28;
  let score = config.approach === "edge" ? edge :
    config.approach === "likelihood" ? likelihood :
      config.approach === "template" ? template :
        config.approach === "hybrid" ? 0.55 * region + 0.45 * template : region;
  const scalePenalty = Math.abs(pose.scale - 1) * config.regularization * 0.003;
  return score - scalePenalty;
}

export function fitVolume(volume, initial, config) {
  let pose = { center: [...initial.center], radius: initial.radius, scale: 1 };
  const history = [];
  let best = scorePose(volume, pose, config);
  let step = Math.max(1, config.searchRadius / 2);
  for (let iteration = 0; iteration < config.iterations; iteration += 1) {
    let improved = false;
    for (const axis of [0, 1, 2, 3]) {
      for (const direction of [-1, 1]) {
        const candidate = { center: [...pose.center], radius: pose.radius, scale: pose.scale };
        if (axis < 3) candidate.center[axis] += direction * step;
        else if (config.representation === "cad") candidate.scale = Math.max(0.55, Math.min(1.6, candidate.scale + direction * step / Math.max(4, pose.radius)));
        else candidate.radius = Math.max(2, candidate.radius + direction * step);
        const score = scorePose(volume, candidate, config);
        if (score > best) {
          pose = candidate;
          best = score;
          improved = true;
        }
      }
    }
    history.push({ iteration: iteration + 1, score: Number(best.toFixed(4)), center: pose.center.map((v) => Number(v.toFixed(1))), scale: Number(pose.scale.toFixed(2)) });
    step = improved ? Math.max(0.5, step * 0.72) : Math.max(0.5, step * 0.5);
  }
  return { pose, score: best, confidence: Math.max(0, Math.min(1, (best + 0.15) / 0.85)), history };
}

export function compareMasks(dims, truthMask, pose, config) {
  if (!truthMask) return null;
  let intersection = 0;
  let predicted = 0;
  let truth = 0;
  const [nx, ny, nz] = dims;
  for (let z = 0; z < nz; z += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        const index = index3d(dims, x, y, z);
        const isPredicted = containsObject([x - pose.center[0], y - pose.center[1], z - pose.center[2]], config.shape, pose.radius, config.representation, config.rotation, pose.scale);
        if (isPredicted) predicted += 1;
        if (truthMask[index]) truth += 1;
        if (isPredicted && truthMask[index]) intersection += 1;
      }
    }
  }
  const dice = (2 * intersection) / Math.max(1, predicted + truth);
  const translationError = Math.hypot(...pose.center.map((value, axis) => value - config.truthCenter[axis]));
  const scaleError = Math.abs(pose.radius * pose.scale - config.truthRadius) / Math.max(1, config.truthRadius);
  return { dice, iou: intersection / Math.max(1, predicted + truth - intersection), translationError, scaleError };
}
