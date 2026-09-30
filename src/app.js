import { compareMasks, fitVolume } from "./core/fitting.js";
import { createDemoVolume, embedObject, loadNpyUint8, loadUint8Volume, seededRandom } from "./core/volume.js";
import { applyPreprocessingChain } from "./core/preprocessing.js";
import { runInitialization, perturbPose, poseError } from "./core/initialization.js";
import "./style.css";

const $ = (id) => document.getElementById(id);
const state = {
  base: createDemoVolume(),
  embedded: null,
  result: null,
  fittedPose: null,
  file: null,
  stage1: { candidates: [], method: null, chosen: null }
};
function number(id) { return Number($(id).value); }
function currentObject() {
  return {
    shape: $("shape").value,
    radius: number("size"),
    center: [number("object-x"), number("object-y"), number("object-z")],
    rotation: [number("rotation-x"), number("rotation-y"), number("rotation-z")],
    intensity: number("intensity"),
    blend: number("blend") / 100,
    shadow: $("shadow").checked,
    representation: $("representation").value
  };
}
function currentArtifacts() {
  return { gaussian: number("gaussian"), speckle: number("speckle"), dropout: number("dropout"), clip: $("clip").checked, seed: number("seed") };
}
function fitConfig() {
  return {
    approach: $("approach").value,
    representation: $("representation").value,
    shape: $("shape").value,
    rotation: [number("rotation-x"), number("rotation-y"), number("rotation-z")],
    searchRadius: number("search-radius"),
    samples: number("samples"),
    window: number("window"),
    regularization: number("regularization") / 100,
    threshold: number("threshold"),
    robustLoss: $("robust-loss").value,
    iterations: number("iterations")
  };
}
function preprocessingStep(prefix) {
  const type = $(`${prefix}-type`).value;
  if (type === "gaussian") return { type, params: { sigma: Number($(`${prefix}-gaussian-sigma`).value) } };
  if (type === "median") return { type, params: { kernelSize: Number($(`${prefix}-median-kernel`).value) } };
  if (type === "anisotropic") {
    return {
      type,
      params: {
        iterations: Number($(`${prefix}-aniso-iterations`).value),
        kappa: Number($(`${prefix}-aniso-kappa`).value),
        edgeFunction: $(`${prefix}-aniso-edge`).value
      }
    };
  }
  if (type === "phase-congruency") return { type, params: { sigma: Number($(`${prefix}-pc-sigma`).value) } };
  if (type === "wavelet") return { type, params: { mode: $(`${prefix}-wavelet-mode`).value, thresholdScale: Number($(`${prefix}-wavelet-scale`).value) } };
  return { type: "none" };
}
function preprocessingSteps() {
  return [preprocessingStep("pre-1"), preprocessingStep("pre-2")];
}
function volumeWithPreprocessing(toggleId) {
  if (!state.embedded) return null;
  if (!$(toggleId).checked) return state.embedded;
  return applyPreprocessingChain(state.embedded, preprocessingSteps());
}
function modelPriorConfig() {
  return {
    shape: $("shape").value,
    representation: $("representation").value,
    radius: number("size"),
    rotation: [number("rotation-x"), number("rotation-y"), number("rotation-z")]
  };
}
function initOptionsForMethod(method, config) {
  if (method === "template") {
    const rotStep = number("init-template-rot-step");
    const rotRange = number("init-template-rot-range");
    const rotations = [];
    if (rotStep > 0 && rotRange > 0) {
      for (let delta = -rotRange; delta <= rotRange; delta += rotStep) {
        rotations.push([config.rotation[0], config.rotation[1], config.rotation[2] + delta]);
      }
    } else {
      rotations.push([...config.rotation]);
    }
    return {
      gridStep: number("init-template-grid"),
      levels: number("init-template-levels"),
      rotations,
      metric: $("init-template-metric").value,
      threshold: number("init-template-threshold"),
      maxCandidates: number("init-template-candidates")
    };
  }
  if (method === "hough") {
    return {
      binSize: number("init-hough-bin"),
      threshold: number("init-hough-threshold"),
      sampleCount: number("init-hough-samples"),
      maxCandidates: number("init-hough-candidates")
    };
  }
  if (method === "icp") {
    return {
      threshold: number("init-icp-threshold"),
      maxIterations: number("init-icp-iterations"),
      tolerance: number("init-icp-tolerance"),
      outlierRejectionPercentile: number("init-icp-outlier"),
      maxVolumePoints: number("init-icp-volume-points"),
      maxModelPoints: number("init-icp-model-points")
    };
  }
  if (method === "blob-pca") {
    return {
      threshold: number("init-blob-threshold"),
      minVolume: number("init-blob-min"),
      maxVolume: number("init-blob-max"),
      connectivity: Number($("init-blob-connectivity").value)
    };
  }
  if (method === "manual") {
    const truth = state.embedded?.truth;
    if ($("init-perturb-enable").checked && truth) {
      const random = seededRandom((number("seed") + 777) >>> 0);
      const perturbed = perturbPose(
        { center: [...truth.center], radius: truth.radius, rotation: truth.rotation ?? [0, 0, 0] },
        { translation: number("init-perturb-translation"), rotation: number("init-perturb-rotation") },
        random
      );
      return { manualPose: perturbed };
    }
    return {
      manualPose: {
        center: [number("init-manual-x"), number("init-manual-y"), number("init-manual-z")],
        radius: config.radius,
        rotation: [number("init-manual-rx"), number("init-manual-ry"), number("init-manual-rz")]
      }
    };
  }
  return {};
}
function formatCandidate(candidate, truth) {
  const center = candidate.center.map((v) => v.toFixed(1)).join(", ");
  const rotation = candidate.rotation.map((v) => v.toFixed(0)).join(", ");
  let errorText = "";
  if (truth) {
    const error = poseError(candidate, { center: truth.center, rotation: truth.rotation ?? [0, 0, 0] });
    errorText = ` · error ${error.translationError.toFixed(2)}vox` + (error.rotationError !== null ? ` / ${error.rotationError.toFixed(1)}°` : "");
  }
  return `<div class="candidate-row"><span class="rank">#${candidate.rank ?? 1}</span><span>center ${center} · rot ${rotation}${errorText}</span><span class="score">${candidate.score.toFixed(3)}</span></div>`;
}
function runStage1() {
  if (!state.embedded) regenerate();
  const method = $("init-method").value;
  const volume = volumeWithPreprocessing("pre-use-init");
  const config = modelPriorConfig();
  const options = initOptionsForMethod(method, config);
  const candidates = runInitialization(method, volume, config, options);
  state.stage1 = { candidates, method, chosen: candidates[0] ?? null };
  const truth = state.embedded.truth;
  if (!candidates.length) {
    $("init-status").textContent = `${method}: no candidates found with current parameters.`;
    $("init-candidates").innerHTML = "";
    $("pipeline-stage1-badge").textContent = "no candidates";
    return;
  }
  $("init-status").textContent = `${method}: ${candidates.length} candidate(s) returned.`;
  $("init-candidates").innerHTML = candidates.map((candidate) => formatCandidate(candidate, truth)).join("");
  $("pipeline-stage1-badge").textContent = `top score ${candidates[0].score.toFixed(3)}`;
}
function setDimensionControls(dims) {
  ["x", "y", "z"].forEach((axis, index) => {
    $(`dim-${axis}`).value = dims[index];
    $(`object-${axis}`).max = dims[index] - 1;
    $(`object-${axis}`).value = Math.min(number(`object-${axis}`), dims[index] - 1);
    $(`slice-${axis}`).max = dims[index] - 1;
    $(`slice-${axis}`).value = Math.min(Math.round((dims[index] - 1) / 2), dims[index] - 1);
    $(`init-manual-${axis}`).max = dims[index] - 1;
    $(`init-manual-${axis}`).value = Math.min(Math.round((dims[index] - 1) / 2), dims[index] - 1);
  });
}
function regenerate() {
  state.embedded = embedObject(state.base, currentObject(), currentArtifacts());
  state.result = null;
  state.fittedPose = null;
  state.lastSeedKey = null;
  state.stage1 = { candidates: [], method: null, chosen: null };
  $("fit-status").textContent = "Not fitted";
  $("metrics").textContent = "Generated volume. Fit the selected model to inspect its score.";
  $("history").replaceChildren();
  $("dataset-badge").textContent = state.base.label;
  $("init-status").textContent = "Not run yet.";
  $("init-candidates").innerHTML = "";
  $("pipeline-stage1-badge").textContent = "not run";
  $("pipeline-stage2-badge").textContent = "not run";
  $("pipeline-status").textContent = "Stage 1 not run";
  $("pipeline-metrics").textContent = "Run Stage 1 initialization, then Stage 2 fitting, to compare initialization error against post-refinement error.";
  drawSlices();
}

function makeSlice(canvas, axis, index, volume, overlay = null) {
  const [nx, ny, nz] = volume.dims;
  const width = axis === "sagittal" ? ny : nx;
  const height = axis === "axial" ? ny : nz;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  const image = context.createImageData(width, height);
  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const x = axis === "sagittal" ? index : column;
      const y = axis === "axial" ? ny - row - 1 : axis === "coronal" ? index : ny - column - 1;
      const z = axis === "axial" ? index : nz - row - 1;
      const offset = (row * width + column) * 4;
      const volumeIndex = z * nx * ny + y * nx + x;
      let intensity;
      if (overlay === "difference") intensity = Math.min(255, Math.abs(state.embedded.data[volumeIndex] - state.base.data[volumeIndex]) * 2.2);
      else intensity = volume.data[volumeIndex];
      let red = intensity;
      let green = intensity;
      let blue = intensity;
      if (overlay === "ground-truth" && state.embedded.mask?.[volumeIndex]) {
        red = Math.min(255, intensity * 0.35 + 170);
        green = intensity * 0.38;
        blue = intensity * 0.38;
      }
      image.data[offset] = red;
      image.data[offset + 1] = green;
      image.data[offset + 2] = blue;
      image.data[offset + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
}

function drawSlices() {
  if (!state.embedded) return;
  const dims = state.base.dims;
  const x = number("slice-x");
  const y = number("slice-y");
  const z = number("slice-z");
  const mode = $("view-mode").value;
  const selected = mode === "base" ? state.base : state.embedded;
  const overlay = mode === "difference" ? "difference" : mode === "overlay" && state.embedded.mask ? "ground-truth" : null;
  makeSlice($("axial"), "axial", z, selected, overlay);
  makeSlice($("coronal"), "coronal", y, selected, overlay);
  makeSlice($("sagittal"), "sagittal", x, selected, overlay);
  $("axial-label").textContent = `z=${z} / ${dims[2] - 1}`;
  $("coronal-label").textContent = `y=${y} / ${dims[1] - 1}`;
  $("sagittal-label").textContent = `x=${x} / ${dims[0] - 1}`;
}

function renderFit() {
  if (!state.embedded) regenerate();
  const object = state.embedded.truth;
  const seedSource = $("seed-source").value;
  const seedKey = seedSource === "coarse" && state.stage1.chosen
    ? `coarse:${state.stage1.chosen.center.join(",")}:${state.stage1.method}`
    : `naive:${seedSource}`;
  let initial;
  let initSourceLabel;
  if (state.fittedPose && state.lastSeedKey === seedKey) {
    // Continue refining from the previous fit result (unchanged existing behavior):
    // repeated "Fit" clicks keep iterating coordinate descent from where it left off.
    initial = state.fittedPose;
    initSourceLabel = state.lastInitSourceLabel;
  } else if (seedSource === "coarse" && state.stage1.chosen) {
    const chosen = state.stage1.chosen;
    initial = { center: [...chosen.center], radius: chosen.radius, rotation: [...chosen.rotation] };
    ["rotation-x", "rotation-y", "rotation-z"].forEach((id, axis) => { $(id).value = String(Math.round(chosen.rotation[axis])); });
    setRangeOutputs();
    initSourceLabel = `Stage 1 (${state.stage1.method})`;
  } else {
    initial = {
      center: state.embedded.dims.map((dimension) => (dimension - 1) / 2),
      radius: object ? object.radius * 0.85 : Math.min(...state.embedded.dims) * 0.14,
      rotation: [number("rotation-x"), number("rotation-y"), number("rotation-z")]
    };
    initSourceLabel = "Naive baseline (volume center)";
  }
  // Read the fit configuration only after any rotation-field updates above, so a
  // coarse-seeded candidate's rotation is reflected in the config used to fit.
  const config = fitConfig();
  state.lastSeedKey = seedKey;
  state.lastInitSourceLabel = initSourceLabel;
  const volume = volumeWithPreprocessing("pre-use-refine") ?? state.embedded;
  const fitted = fitVolume(volume, initial, config);
  if ($("temporal").checked && state.result) {
    fitted.pose.center = fitted.pose.center.map((position, axis) => position * 0.65 + state.result.pose.center[axis] * 0.35);
  }
  state.result = fitted;
  state.fittedPose = fitted.pose;
  $("fit-status").textContent = `${config.approach} · ${fitted.confidence.toFixed(2)} confidence`;
  $("pipeline-stage2-badge").textContent = `${fitted.confidence.toFixed(2)} confidence`;
  const truthMetrics = volume.mask && object
    ? compareMasks(volume.dims, volume.mask, fitted.pose, { ...config, truthCenter: object.center, truthRadius: object.radius })
    : null;
  $("metrics").innerHTML = truthMetrics
    ? `<div><b>Translation error</b><span>${truthMetrics.translationError.toFixed(2)} voxels</span></div>
       <div><b>Relative size error</b><span>${(truthMetrics.scaleError * 100).toFixed(1)}%</span></div>
       <div><b>Dice / IoU</b><span>${truthMetrics.dice.toFixed(3)} / ${truthMetrics.iou.toFixed(3)}</span></div>
       <div><b>Fit score</b><span>${fitted.score.toFixed(3)}</span></div>
       <div><b>Estimated center</b><span>${fitted.pose.center.map((value) => value.toFixed(1)).join(", ")}</span></div>`
    : `<div><b>Fit score</b><span>${fitted.score.toFixed(3)}</span></div>
       <div><b>Confidence proxy</b><span>${fitted.confidence.toFixed(2)}</span></div>
       <div><b>Estimated center (x,y,z)</b><span>${fitted.pose.center.map((value) => value.toFixed(1)).join(", ")}</span></div>
       <p class="muted">No ground-truth labels are available for this local volume; overlap/error metrics are omitted.</p>`;
  $("history").innerHTML = `<div class="history-heading">Coordinate-descent history</div>${fitted.history.map((row) =>
    `<div class="history-row"><span>iter ${row.iteration}</span><span>score ${row.score}</span><span>center ${row.center.join(", ")}</span></div>`
  ).join("")}`;

  const initError = object ? poseError(initial, { center: object.center, rotation: object.rotation ?? config.rotation }) : null;
  const convergedCorrectly = truthMetrics ? truthMetrics.translationError < object.radius * 0.75 : null;
  $("pipeline-status").textContent = `Seeded from: ${initSourceLabel}`;
  $("pipeline-metrics").innerHTML = object
    ? `<div><b>Seed source</b><span>${initSourceLabel}</span></div>
       <div><b>Initialization error (before Stage 2)</b><span>${initError.translationError.toFixed(2)} vox${initError.rotationError !== null ? ` / ${initError.rotationError.toFixed(1)}°` : ""}</span></div>
       <div><b>Post-refinement error</b><span>${truthMetrics.translationError.toFixed(2)} vox</span></div>
       <div><b>Converged to correct solution</b><span>${convergedCorrectly ? "yes" : "no"}</span></div>`
    : `<div><b>Seed source</b><span>${initSourceLabel}</span></div><p class="muted">No ground-truth labels are available for this local volume; initialization/refinement error comparison is omitted.</p>`;
}

function setRangeOutputs() {
  document.querySelectorAll('input[type="range"]').forEach((input) => {
    const output = $(`${input.id}-value`);
    if (output) output.value = input.value;
    if (output) output.textContent = input.id === "blend" || input.id === "dropout" || input.id === "speckle" || input.id === "regularization"
      ? `${input.value}%`
      : input.value;
  });
}

function randomizedSettings() {
  const random = seededRandom((number("seed") + 1) >>> 0);
  $("shape").value = ["sphere", "box", "cylinder", "torus", "frame"][Math.floor(random() * 5)];
  $("size").value = String(5 + Math.floor(random() * 6));
  ["x", "y", "z"].forEach((axis, index) => {
    const maximum = state.base.dims[index];
    $(`object-${axis}`).value = String(Math.round(maximum * (0.38 + random() * 0.24)));
  });
  ["rotation-x", "rotation-y", "rotation-z"].forEach((axis) => { $(axis).value = String(Math.floor(random() * 360) - 180); });
  $("seed").value = String((number("seed") + 1) >>> 0);
  setRangeOutputs();
  regenerate();
}

function updateMethodNote() {
  const notes = {
    region: "Region fitting maximizes robust foreground-vs-surrounding local mean contrast while penalizing within-region spread.",
    edge: "The RCTL-inspired baseline rewards sampled image-gradient magnitude near the model boundary plus foreground/background contrast.",
    likelihood: "The Gaussian likelihood-ratio proxy favors a bright, low-variance model interior over the local background shell.",
    template: "Thin-shell template matching compares intensities around the procedural model shell to the surrounding annulus.",
    hybrid: "Hybrid fitting combines region contrast and thin-shell template scores with fixed weights (0.55 / 0.45)."
  };
  $("method-note").textContent = notes[$("approach").value];
}

$("volume-file").addEventListener("change", (event) => { state.file = event.target.files?.[0] ?? null; });
$("load-volume").addEventListener("click", async () => {
  if (!state.file) { window.alert("Choose a local .raw or .npy volume first."); return; }
  try {
    const buffer = await state.file.arrayBuffer();
    state.base = state.file.name.toLowerCase().endsWith(".npy")
      ? loadNpyUint8(buffer)
      : loadUint8Volume(buffer, [number("dim-x"), number("dim-y"), number("dim-z")], `Local: ${state.file.name}`);
    state.base.label = `Local volume · ${state.file.name}`;
    state.fittedPose = null;
    setDimensionControls(state.base.dims);
    regenerate();
  } catch (error) {
    window.alert(error.message);
  }
});
$("reset-demo").addEventListener("click", () => {
  state.base = createDemoVolume();
  state.file = null;
  $("volume-file").value = "";
  setDimensionControls(state.base.dims);
  regenerate();
});
$("embed").addEventListener("click", regenerate);
$("randomize").addEventListener("click", randomizedSettings);
$("fit").addEventListener("click", renderFit);
$("run-init").addEventListener("click", runStage1);
$("seed-source").addEventListener("change", () => { if (state.result) renderFit(); });
$("approach").addEventListener("change", updateMethodNote);
["slice-x", "slice-y", "slice-z", "view-mode"].forEach((id) => $(id).addEventListener("input", drawSlices));
document.querySelectorAll('input[type="range"]').forEach((input) => input.addEventListener("input", setRangeOutputs));
["shape", "size", "object-x", "object-y", "object-z", "rotation-x", "rotation-y", "rotation-z", "intensity", "blend", "shadow", "gaussian", "speckle", "dropout", "clip", "seed", "representation"].forEach((id) => {
  $(id).addEventListener("change", () => {
    setRangeOutputs();
    if (state.embedded) regenerate();
  });
});
["search-radius", "samples", "window", "regularization", "threshold", "iterations", "robust-loss", "temporal"].forEach((id) => {
  $(id).addEventListener("change", () => { if (state.result) renderFit(); });
});

const PREPROCESSING_FILTER_TYPES = ["gaussian", "median", "anisotropic", "phase-congruency", "wavelet"];
function updatePreprocessingVisibility() {
  ["pre-1", "pre-2"].forEach((prefix) => {
    const type = $(`${prefix}-type`).value;
    PREPROCESSING_FILTER_TYPES.forEach((filterType) => {
      const panel = $(`${prefix}-params-${filterType}`);
      if (panel) panel.classList.toggle("hidden", type !== filterType);
    });
  });
}
const INIT_METHOD_TYPES = ["template", "hough", "icp", "blob-pca", "manual"];
function updateInitMethodVisibility() {
  const method = $("init-method").value;
  INIT_METHOD_TYPES.forEach((methodType) => {
    const panel = $(`init-params-${methodType}`);
    if (panel) panel.classList.toggle("hidden", method !== methodType);
  });
}
["pre-1-type", "pre-2-type"].forEach((id) => $(id).addEventListener("change", updatePreprocessingVisibility));
$("init-method").addEventListener("change", updateInitMethodVisibility);

function initialize() {
  const truth = state.base.truth;
  $("object-x").value = truth.center[0];
  $("object-y").value = truth.center[1];
  $("object-z").value = truth.center[2];
  $("slice-x").value = truth.center[0];
  $("slice-y").value = truth.center[1];
  $("slice-z").value = truth.center[2];
  setDimensionControls(state.base.dims);
  $("object-x").value = truth.center[0];
  $("object-y").value = truth.center[1];
  $("object-z").value = truth.center[2];
  $("init-manual-x").value = Math.round((state.base.dims[0] - 1) / 2);
  $("init-manual-y").value = Math.round((state.base.dims[1] - 1) / 2);
  $("init-manual-z").value = Math.round((state.base.dims[2] - 1) / 2);
  setRangeOutputs();
  updateMethodNote();
  updatePreprocessingVisibility();
  updateInitMethodVisibility();
  regenerate();
}
initialize();
