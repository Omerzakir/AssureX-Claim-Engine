'use strict';
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');

const LABELS = ['Invalid Claim', 'Manual Review', 'Valid Claim'];
let cached = null;

async function loadModel(modelDir) {
  const dir = path.resolve(modelDir);
  const modelPath = path.join(dir, 'model.json');
  const metadataPath = path.join(dir, 'metadata.json');
  const weightsPath = path.join(dir, 'weights.bin');
  for (const p of [modelPath, metadataPath, weightsPath]) if (!fs.existsSync(p)) throw new Error(`Teachable Machine artifact missing: ${p}`);
  const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
  const labels = Array.isArray(metadata.labels) ? metadata.labels.map(String) : LABELS;
  if (labels.length !== 3 || labels.join('|') !== LABELS.join('|')) throw new Error(`Unexpected Teachable Machine labels: ${labels.join(', ')}`);
  const model = await tf.loadLayersModel('file://' + modelPath);
  cached = {dir, model, labels, metadata};
  return cached;
}

async function predictImage(input, modelDir) {
  const state = cached && cached.dir === path.resolve(modelDir) ? cached : await loadModel(modelDir);
  const buffer = Buffer.isBuffer(input) ? input : fs.readFileSync(input);
  const result = tf.tidy(() => {
    let image = tf.node.decodeImage(buffer, 3);
    image = tf.image.resizeBilinear(image, [224, 224]);
    image = image.toFloat().div(255.0).expandDims(0);
    const output = state.model.predict(image);
    return Array.from(output.dataSync());
  });
  const sum = result.reduce((a,b)=>a+b,0) || 1;
  const probabilities = result.map(x=>x/sum);
  const index = probabilities.indexOf(Math.max(...probabilities));
  return {label:state.labels[index], class_id:index, confidence:probabilities[index], probabilities:Object.fromEntries(state.labels.map((l,i)=>[l,probabilities[i]])), model_version:'21.0.0', image_size:224};
}

module.exports = {loadModel, predictImage};
