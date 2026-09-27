const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

const downloads = "C:/Users/user/Downloads";
const output = "C:/Users/user/Documents/git_clones/algo/recovered-phase2";

function readJson(name) {
  return JSON.parse(fs.readFileSync(path.join(downloads, name), "utf8"));
}

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonicalize(value[key])]),
    );
  }
  return value;
}

function sha256(value) {
  return crypto
    .createHash("sha256")
    .update(JSON.stringify(canonicalize(value)))
    .digest("hex");
}

function writeJson(name, value) {
  fs.writeFileSync(path.join(output, name), `${JSON.stringify(value, null, 2)}\n`);
}

const runState = readJson("run_state.json");
const contract = readJson("experiment_contract.json");
const sourceSpec = runState.fingerprint_payload.spec;
const sourceAdaptive = sourceSpec.adaptive_power_blend_experiment;

const manifestName = "adaptive_power_blend_chicken_dataset_manifest.recovered.json";
const specName = "adaptive_power_blend_chicken_phase02_spec.recovered.json";
const manifest = {
  schema_version: sourceAdaptive.schema_version,
  manifest_id: sourceAdaptive.manifest_id,
  datasets: sourceAdaptive.datasets.map((dataset) => dataset.declaration),
};

const spec = { ...sourceSpec };
delete spec.datasets;
spec.adaptive_power_blend_experiment = {
  dataset_manifest: manifestName,
  design: sourceAdaptive.design,
  evaluation: sourceAdaptive.evaluation,
};

const manifestHash = sha256(manifest);
if (manifestHash !== contract.manifest_sha256) {
  throw new Error(`Manifest hash mismatch: ${manifestHash} != ${contract.manifest_sha256}`);
}

fs.mkdirSync(output, { recursive: true });
writeJson(manifestName, manifest);
writeJson(specName, spec);
writeJson("RECOVERY.json", {
  source_run_state: path.join(downloads, "run_state.json"),
  sealed_manifest_sha256: manifestHash,
  sealed_manifest_sha256_expected: contract.manifest_sha256,
  sealed_spec_sha256_expected: contract.spec_sha256,
  original_output_dir: sourceSpec.output_dir,
  original_manifest_path: sourceAdaptive.manifest_path,
  recovered_files: [manifestName, specName],
});

console.log(JSON.stringify({
  output,
  manifest_sha256: manifestHash,
  expected_manifest_sha256: contract.manifest_sha256,
  expected_spec_sha256: contract.spec_sha256,
  manifest: path.join(output, manifestName),
  spec: path.join(output, specName),
}, null, 2));
