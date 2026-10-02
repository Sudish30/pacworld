// pacworld in the browser: the exported sampler (web/model_fp16.onnx) run with ONNX Runtime Web, and the same
// context rule as dataset.History (strided offsets, clamped at the start of the history).
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";   // version: configs/web_demo.yaml

export const SIZE = 64;
const PIX = 3 * SIZE * SIZE;

export async function loadJSON(url) { return (await fetch(url)).json(); }
export async function loadBin(url) { return (await fetch(url)).arrayBuffer(); }

// Standard normal samples (Box-Muller) into a Float32Array.
export function randn(out) {
  for (let i = 0; i < out.length; i += 2) {
    const u = 1 - Math.random(), v = Math.random();
    const r = Math.sqrt(-2 * Math.log(u)), t = 2 * Math.PI * v;
    out[i] = r * Math.cos(t);
    if (i + 1 < out.length) out[i + 1] = r * Math.sin(t);
  }
  return out;
}

// n standard normals from mulberry32 + Box-Muller: the same numbers as tools/export_onnx.py prng_normal(seed, n)
export function seededNormal(seed, n) {
  const out = new Float32Array(n), m = (n + 1) >> 1;
  const uni = (i) => {                                   // mulberry32 output number i (1-based) for this seed
    let a = (seed + Math.imul(0x6D2B79F5, i)) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1) >>> 0;
    t = (t ^ (t + Math.imul(t ^ (t >>> 7), t | 61))) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let k = 0; k < m; k++) {
    const u1 = 1 - uni(2 * k + 1), u2 = uni(2 * k + 2);
    const r = Math.sqrt(-2 * Math.log(u1)), th = 2 * Math.PI * u2;
    out[2 * k] = r * Math.cos(th);
    if (2 * k + 1 < n) out[2 * k + 1] = r * Math.sin(th);
  }
  return out;
}

export async function createSession(modelUrl, onProgress) {
  const hasGPU = !!navigator.gpu && !!(await navigator.gpu.requestAdapter().catch(() => null));
  const forced = new URLSearchParams(location.search).get("ep");          // ?ep=wasm forces the CPU backend (debugging)
  const providers = forced ? [forced] : hasGPU ? ["webgpu"] : ["wasm"];
  const resp = await fetch(modelUrl);
  const total = +resp.headers.get("content-length") || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (onProgress) onProgress(got, total);
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  const session = await ort.InferenceSession.create(buf, { executionProviders: providers, graphOptimizationLevel: "all" });
  return { session, backend: providers[0] };
}

// dataset.History for one rollout: the last `reach` frames (Float32Array, CHW, [-1, 1]) and their actions;
// actions[j] is the action taken after frames[j]; the last slot is the action given to context().
export class History {
  constructor(frames, actions, offsets) {
    this.offsets = offsets;
    this.reach = -Math.min(...offsets);
    this.frames = frames.slice(-this.reach);
    this.actions = actions.concat([0]).slice(-this.reach);
  }
  // indices of the context frames: max(T + offset, 0), exactly dataset.gather_context
  indices() { const T = this.frames.length; return this.offsets.map(o => Math.max(T + o, 0)); }
  context(action, ctxOut, actOut) {
    this.actions[this.actions.length - 1] = action;
    this.indices().forEach((i, k) => {
      ctxOut.set(this.frames[i], k * PIX);
      actOut[k] = BigInt(this.actions[i]);
    });
  }
  push(frame) {
    this.frames.push(frame);
    this.actions.push(0);
    if (this.frames.length > this.reach) { this.frames.shift(); this.actions.shift(); }
  }
}

export class World {
  constructor(session, meta, startsBuf) {
    this.session = session;
    this.meta = meta;
    this.K = meta.offsets.length;
    this.ctx = new Float32Array(this.K * PIX);
    this.acts = new BigInt64Array(this.K);
    this.noise = new Float32Array(PIX);
    this.ctxNoise = new Float32Array(this.K * PIX);
    this.sigma = meta.ctx_sigma;          // fixed in the exported graph; the page only adds the matching noise
    this.startsBuf = startsBuf;
    this.reset(0);
  }
  reset(which) {
    const { reach, size } = this.meta;
    const per = reach * size * size * 3 + (reach - 1) * 4;
    const base = (which % this.meta.n) * per;
    const u8 = new Uint8Array(this.startsBuf, base, reach * size * size * 3);
    const acts = new Int32Array(this.startsBuf.slice(base + reach * size * size * 3, base + per));
    const frames = [];
    for (let t = 0; t < reach; t++) {                      // HWC uint8 -> CHW float in [-1, 1]
      const f = new Float32Array(PIX), off = t * size * size * 3;
      for (let p = 0; p < size * size; p++)
        for (let c = 0; c < 3; c++) f[c * size * size + p] = u8[off + p * 3 + c] / 127.5 - 1;
      frames.push(f);
    }
    this.hist = new History(frames, Array.from(acts), this.meta.offsets);
    this.last = frames[frames.length - 1];
  }
  // one frame: gather the context, add the context noise, run the 3-step sampler, push the result
  // The context noise is drawn fresh every step and never stored; the history keeps the float output frame.
  // seeds = [ctxNoiseSeed, noiseSeed] makes the step reproducible (the rollout check); omit it for play.
  async step(action, seeds) {
    this.hist.context(action, this.ctx, this.acts);
    const cn = seeds ? seededNormal(seeds[0], this.ctx.length) : randn(this.ctxNoise);
    for (let i = 0; i < this.ctx.length; i++) this.ctx[i] += this.sigma * cn[i];
    const out = await this.session.run({
      ctx: new ort.Tensor("float32", this.ctx, [1, 3 * this.K, SIZE, SIZE]),
      actions: new ort.Tensor("int64", this.acts, [1, this.K]),
      noise: new ort.Tensor("float32", seeds ? seededNormal(seeds[1], this.noise.length) : randn(this.noise), [1, 3, SIZE, SIZE]),
    });
    const frame = new Float32Array(out.frame.data);
    this.hist.push(frame);
    this.last = frame;
    return frame;
  }
}

// CHW float [-1, 1] -> uint8 value, as dataset.to_uint8
// held keys -> ALE action, from the table tools/export_onnx.py writes with serve.server.keys_to_action
export const keysToAction = (keymap, k) => keymap[(k.up ? 1 : 0) | (k.down ? 2 : 0) | (k.left ? 4 : 0) | (k.right ? 8 : 0)];

export const toU8 = v => Math.round((Math.min(1, Math.max(-1, v)) + 1) * 127.5);

export function draw(canvasCtx, frame, image) {
  const n = SIZE * SIZE, d = image.data;
  for (let p = 0; p < n; p++) {
    d[4 * p] = toU8(frame[p]); d[4 * p + 1] = toU8(frame[n + p]); d[4 * p + 2] = toU8(frame[2 * n + p]); d[4 * p + 3] = 255;
  }
  canvasCtx.putImageData(image, 0, 0);
}

export { ort };
