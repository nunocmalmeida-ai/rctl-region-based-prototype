export const DEFAULT_DIMS = [48, 48, 48];

export function seededRandom(seed) {
  let state = Number(seed) >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function createDemoVolume(dims = DEFAULT_DIMS) {
  const [nx, ny, nz] = dims;
  const data = new Float32Array(nx * ny * nz);
  const cx = (nx - 1) / 2;
  const cy = (ny - 1) / 2;
  const cz = (nz - 1) / 2;
  const truth = {
    shape: "sphere",
    center: [Math.round(nx * 0.56), Math.round(ny * 0.48), Math.round(nz * 0.52)],
    radius: Math.max(3, Math.round(Math.min(nx, ny, nz) * 0.145)),
    rotation: 25
  };
  const random = seededRandom(8173);
  for (let z = 0; z < nz; z += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        const dx = (x - cx) / (nx * 0.34);
        const dy = (y - cy) / (ny * 0.39);
        const dz = (z - cz) / (nz * 0.37);
        const radius = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const tissue = radius < 0.68 ? 19 : radius < 1.04 ? 92 : 34;
        const speckle = (random() - 0.5) * (radius < 0.68 ? 16 : 46);
        data[z * ny * nx + y * nx + x] = Math.max(0, Math.min(255, tissue + speckle));
      }
    }
  }
  return { dims: [...dims], data, truth, label: "Synthetic demo" };
}

export function index3d(dims, x, y, z) {
  return z * dims[0] * dims[1] + y * dims[0] + x;
}

export function rotateToLocal(point, rotation = [0, 0, 0]) {
  let [x, y, z] = point;
  const [rx, ry, rz] = rotation.map((angle) => (angle * Math.PI) / 180);
  let c = Math.cos(-rx);
  let s = Math.sin(-rx);
  [y, z] = [y * c - z * s, y * s + z * c];
  c = Math.cos(-ry);
  s = Math.sin(-ry);
  [x, z] = [x * c + z * s, -x * s + z * c];
  c = Math.cos(-rz);
  s = Math.sin(-rz);
  [x, y] = [x * c - y * s, x * s + y * c];
  return [x, y, z];
}

export function containsObject(point, shape, radius, representation = "rigid", rotation = [0, 0, 0], scale = 1) {
  const [x, y, z] = rotateToLocal(point, rotation);
  const r = Math.max(0.5, radius * scale);
  if (representation === "smooth") return (x * x + y * y + z * z) / (r * r) <= 1;
  if (representation === "cad") {
    const ax = r * 1.12;
    const ay = r * 0.88;
    const az = r;
    return (x * x) / (ax * ax) + (y * y) / (ay * ay) + (z * z) / (az * az) <= 1;
  }
  if (shape === "box") return Math.max(Math.abs(x), Math.abs(y), Math.abs(z)) <= r;
  if (shape === "cylinder") return x * x + y * y <= r * r && Math.abs(z) <= r;
  if (shape === "torus") return (Math.hypot(x, y) - r * 0.68) ** 2 + z * z <= (r * 0.26) ** 2;
  if (shape === "frame") {
    const bar = Math.max(1, r * 0.22);
    const insideU = Math.abs(x) <= r && Math.abs(y) <= r && Math.abs(z) <= bar &&
      (Math.abs(x) >= r - bar || y <= -r + bar);
    return insideU;
  }
  return (x * x + y * y + z * z) <= r * r;
}

export function embedObject(base, object, artifacts = {}) {
  const { dims } = base;
  const [nx, ny, nz] = dims;
  const data = new Float32Array(base.data);
  const radius = Number(object.radius);
  const center = object.center.map(Number);
  const rotation = object.rotation ?? [0, 0, 0];
  const blend = Math.max(0, Math.min(1, Number(object.blend ?? 0.9)));
  const extent = Math.ceil(radius * 1.8);
  let mask = new Uint8Array(data.length);
  for (let z = Math.max(0, center[2] - extent); z <= Math.min(nz - 1, center[2] + extent); z += 1) {
    for (let y = Math.max(0, center[1] - extent); y <= Math.min(ny - 1, center[1] + extent); y += 1) {
      for (let x = Math.max(0, center[0] - extent); x <= Math.min(nx - 1, center[0] + extent); x += 1) {
        const index = index3d(dims, x, y, z);
        if (!containsObject([x - center[0], y - center[1], z - center[2]], object.shape, radius, object.representation, rotation)) continue;
        mask[index] = 1;
        data[index] = data[index] * (1 - blend) + Number(object.intensity) * blend;
      }
    }
  }

  if (object.shadow) {
    for (let z = 0; z < nz; z += 1) {
      for (let y = 0; y < ny; y += 1) {
        for (let x = 0; x < nx; x += 1) {
          const index = index3d(dims, x, y, z);
          if (z <= center[2] + radius || mask[index]) continue;
          const lateral = Math.hypot(x - center[0], y - center[1]);
          if (lateral <= radius * 0.85) data[index] *= 0.48;
        }
      }
    }
  }

  const random = seededRandom(artifacts.seed ?? 1);
  const gaussian = Number(artifacts.gaussian ?? 0);
  const speckle = Number(artifacts.speckle ?? 0) / 100;
  const dropout = Number(artifacts.dropout ?? 0) / 100;
  const saturation = Boolean(artifacts.clip);
  let spare;
  for (let i = 0; i < data.length; i += 1) {
    if (dropout > 0 && random() < dropout) {
      data[i] = 0;
      continue;
    }
    let noise;
    if (spare !== undefined) {
      noise = spare;
      spare = undefined;
    } else {
      const u = Math.max(random(), 1e-12);
      const v = random();
      noise = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      spare = Math.sqrt(-2 * Math.log(u)) * Math.sin(2 * Math.PI * v);
    }
    data[i] += noise * gaussian + noise * data[i] * speckle;
    if (saturation) data[i] = Math.max(0, Math.min(255, data[i]));
  }
  return { dims: [...dims], data, mask, truth: { ...object, center: [...center] }, label: base.label };
}

export function loadUint8Volume(buffer, dims, label = "Local volume") {
  const [nx, ny, nz] = dims.map(Number);
  if (![nx, ny, nz].every((n) => Number.isInteger(n) && n >= 2 && n <= 512)) {
    throw new Error("Volume dimensions must be integers between 2 and 512.");
  }
  const expected = nx * ny * nz;
  if (buffer.byteLength !== expected) throw new Error(`Expected ${expected} bytes for ${nx}×${ny}×${nz}; received ${buffer.byteLength}.`);
  return { dims: [nx, ny, nz], data: new Float32Array(new Uint8Array(buffer)), mask: null, truth: null, label };
}

export function loadNpyUint8(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 10 || bytes[0] !== 0x93 || new TextDecoder().decode(bytes.slice(1, 6)) !== "NUMPY") {
    throw new Error("Not a supported NumPy .npy file.");
  }
  const major = bytes[6];
  if (major !== 1 && major !== 2) throw new Error(`NumPy format version ${major} is not supported.`);
  const headerLength = major === 1 ? bytes[8] | (bytes[9] << 8) : bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (bytes[11] << 24);
  const headerStart = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(bytes.slice(headerStart, headerStart + headerLength));
  if (!/['"]descr['"]\s*:\s*['"]\|u1['"]/.test(header) || /['"]fortran_order['"]\s*:\s*True/.test(header)) {
    throw new Error("Only C-order uint8 NumPy arrays are supported.");
  }
  const shapeMatch = /['"]shape['"]\s*:\s*\(([^)]*)\)/.exec(header);
  const shape = shapeMatch?.[1].split(",").map((part) => Number(part.trim())).filter(Number.isFinite);
  if (!shape || shape.length !== 3) throw new Error("Expected a three-dimensional NumPy array.");
  const dims = [shape[2], shape[1], shape[0]];
  const payloadStart = headerStart + headerLength;
  return loadUint8Volume(buffer.slice(payloadStart), dims, "Local NPY volume");
}
