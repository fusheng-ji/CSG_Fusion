// 3D Gaussian viewer of the CSG-Fusion project page: WebGL2, no dependencies.
// Reads the files of scripts/viz/export_viewer_scenes.py. One canvas, one viewport per method, one shared camera.
(() => {
  // ---------------------------------------------------------------- worker: decode the file, cull and sort by depth
  const workerSource = `
    let pos, radius, kept, n, bits, order;
    const f1 = new Float32Array(1), u1 = new Uint32Array(f1.buffer);
    const half = (x) => {
      f1[0] = x;
      const u = u1[0], s = (u >>> 16) & 0x8000, e = ((u >>> 23) & 255) - 112, m = u & 0x7fffff;
      if (e >= 31) return s | 0x7bff;
      if (e > 0) return s + (e << 10) + ((m + 0x1000) >>> 13);
      return e < -10 ? s : s + (((m | 0x800000) + (1 << (13 - e))) >>> (14 - e));
    };
    async function decode(buffer, meta) {
      const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"));
      const data = new Uint8Array(await new Response(stream).arrayBuffer());
      const S = meta.size, P = S * S, perView = 13;
      n = meta.views.length * P;
      pos = new Float32Array(3 * n); radius = new Float32Array(n); kept = new Uint8Array(n).fill(1);
      bits = new Uint16Array(n); order = new Uint32Array(n);
      const tex = new Uint32Array(8 * n), f = new Float32Array(tex.buffer), q = [];
      for (let c = 0; c < perView; c++) q.push(new Int32Array(P));
      let at = 0;
      for (let v = 0; v < meta.views.length; v++) {
        for (let c = 0; c < perView; c++) {
          const [predictor, planes] = meta.channels[v * perView + c], x = q[c];
          for (let i = 0; i < P; i++) x[i] = data[at + i];
          for (let k = 1; k < planes; k++) for (let i = 0, o = at + k * P; i < P; i++) x[i] |= data[o + i] << (8 * k);
          at += planes * P;
          for (let y = 0, i = 0; y < S; y++) for (let u = 0; u < S; u++, i++) {
            const z = x[i], r = (z >>> 1) ^ -(z & 1);
            let p = 0;
            if (y && u) {
              const a = x[i - 1], b = x[i - S], d = x[i - S - 1];
              p = predictor ? (d >= Math.max(a, b) ? Math.min(a, b) : d <= Math.min(a, b) ? Math.max(a, b) : a + b - d) : a + b - d;
            } else if (u) p = x[i - 1]; else if (y) p = x[i - S];
            x[i] = r + p;
          }
        }
        const M = meta.views[v], h = 1 << (meta.oct_bits - 1), st = meta.chol_step;
        for (let i = 0, g = v * P; i < P; i++, g++) {
          let a = q[0][i] / h, b = q[1][i] / h;
          const c = 1 - Math.abs(a) - Math.abs(b);
          if (c < 0) { a += a >= 0 ? c : -c; b += b >= 0 ? c : -c; }
          const r = Math.exp(q[2][i] * meta.log_step) / Math.hypot(a, b, c), t = 8 * g;
          for (let k = 0; k < 3; k++) pos[3 * g + k] = f[t + k] = M[k][3] + r * (M[k][0] * a + M[k][1] * b + M[k][2] * c);
          tex[t + 3] = (q[3][i] | q[4][i] << 8 | q[5][i] << 16 | q[6][i] << 24) >>> 0;
          const A = Math.exp(st * q[7][i]), B = Math.exp(st * q[8][i]), C = Math.exp(st * q[9][i]);
          const D = Math.sinh(st * q[10][i]) * A, E = Math.sinh(st * q[11][i]) * A, F = Math.sinh(st * q[12][i]) * B;
          const xx = A * A, xy = A * D, xz = A * E, yy = D * D + B * B, yz = D * E + B * F, zz = E * E + F * F + C * C;
          radius[g] = 3 * Math.sqrt(xx + yy + zz);
          const w = 1024;                                  // keeps millimetre-sized variances in half-float range
          tex[t + 4] = (half(xx * w) | half(xy * w) << 16) >>> 0;
          tex[t + 5] = (half(xz * w) | half(yy * w) << 16) >>> 0;
          tex[t + 6] = (half(yz * w) | half(zz * w) << 16) >>> 0;
        }
      }
      if (meta.mask) for (let i = 0; i < n; i++) { kept[i] = data[at + i]; tex[8 * i + 7] = 1 - kept[i]; }
      if (at + (meta.mask ? n : 0) !== data.length) throw new Error("unexpected file length");
      for (let i = 0; i < n; i++) if (!(tex[8 * i + 3] >>> 24)) radius[i] = -1;     // fully transparent
      postMessage({ tex, n }, [tex.buffer]);
    }
    function sort(m) {
      const [a, b, c, d, e, f, g, h, i9] = m.R, [tx, ty, tz] = m.t, ax = m.tan[0], ay = m.tan[1];
      // a Gaussian reaches at most its 3-sigma radius times this beyond its centre (Jacobian clamp of the shader included)
      const kx = Math.sqrt(1 + 1.69 * ax * ax), ky = Math.sqrt(1 + 1.69 * ay * ay), count = new Uint32Array(65537);
      f1[0] = 0.02;
      const base = u1[0];
      let visible = 0;
      for (let i = 0; i < n; i++) {
        if (radius[i] < 0 || !(kept[i] || m.removed)) continue;
        const x = pos[3 * i] - tx, y = pos[3 * i + 1] - ty, w = pos[3 * i + 2] - tz, z = c * x + f * y + i9 * w;
        if (z <= 0.02) continue;
        const s = radius[i] + 0.02 * z;
        if (Math.abs(a * x + d * y + g * w) - s * kx > ax * z || Math.abs(b * x + e * y + h * w) - s * ky > ay * z) continue;
        f1[0] = z;                                         // float bits are a logarithmic depth key: 4096 steps per octave
        const key = 65535 - Math.min(65535, (u1[0] - base) >>> 11);
        bits[visible] = key; order[visible++] = i; count[key + 1]++;
      }
      for (let i = 0; i < 65536; i++) count[i + 1] += count[i];
      const out = new Uint32Array(visible);
      for (let i = 0; i < visible; i++) out[count[bits[i]]++] = order[i];
      postMessage({ order: out, id: m.id }, [out.buffer]);
    }
    onmessage = ({ data: m }) => { if (m.buffer) decode(m.buffer, m.meta).catch((e) => postMessage({ error: e.message })); else sort(m); };`;

  const vertexShader = `#version 300 es
    precision highp float; precision highp int; precision highp usampler2D;
    uniform usampler2D uData; uniform mat3 uR; uniform vec3 uT; uniform vec2 uFocal, uCenter, uSize; uniform float uRgbMax;
    layout(location = 0) in vec2 corner; layout(location = 1) in uint index;
    out vec2 vP; out vec4 vColor;
    void main() {
      ivec2 at = ivec2(int(index % 1024u) * 2, int(index / 1024u));
      uvec4 t0 = texelFetch(uData, at, 0), t1 = texelFetch(uData, at + ivec2(1, 0), 0);
      vec3 c = uR * (uintBitsToFloat(t0.xyz) - uT);
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vP = vec2(0.0); vColor = vec4(0.0);
      if (c.z <= 0.02) return;                                              // the rasterizer's near cut
      vec2 c0 = unpackHalf2x16(t1.x) / 1024.0, c1 = unpackHalf2x16(t1.y) / 1024.0, c2 = unpackHalf2x16(t1.z) / 1024.0;
      mat3 S = uR * mat3(c0.x, c0.y, c1.x, c0.y, c1.y, c2.x, c1.x, c2.x, c2.y) * transpose(uR);
      // as in the rasterizer, the Jacobian is taken at most 30% outside the image: Gaussians beside the camera would smear
      vec2 lim = 0.65 * uSize / uFocal, g = clamp(c.xy / c.z, -lim, lim);
      vec3 jx = vec3(uFocal.x / c.z, 0.0, -uFocal.x * g.x / c.z), jy = vec3(0.0, uFocal.y / c.z, -uFocal.y * g.y / c.z);
      float a = dot(jx, S * jx) + 0.3, d = dot(jy, S * jy) + 0.3, b = dot(jx, S * jy);   // + 0.3 px^2, as in the rasterizer
      float m = 0.5 * (a + d), r = length(vec2(0.5 * (a - d), b)), l1 = m + r, l2 = max(m - r, 0.05);
      vec2 e = abs(b) < 1e-6 ? (a >= d ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) : normalize(vec2(b, l1 - a));
      vec2 px = uFocal * c.xy / c.z + uCenter + 3.0 * (corner.x * sqrt(l1) * e + corner.y * sqrt(l2) * vec2(-e.y, e.x));
      gl_Position = vec4(px.x / uSize.x * 2.0 - 1.0, 1.0 - px.y / uSize.y * 2.0, 0.0, 1.0);
      vP = 3.0 * corner;
      vColor = vec4(vec3(t0.w & 255u, (t0.w >> 8) & 255u, (t0.w >> 16) & 255u) / 255.0 * uRgbMax, float(t0.w >> 24) / 255.0);
      if (t1.w == 1u) vColor.rgb = mix(vColor.rgb, vec3(1.0, 0.12, 0.15), 0.75);          // removed by the filter
    }`;
  const fragmentShader = `#version 300 es
    precision highp float; in vec2 vP; in vec4 vColor; out vec4 color;
    void main() { float r = dot(vP, vP); if (r > 9.0) discard; float a = vColor.a * exp(-0.5 * r); color = vec4(vColor.rgb * a, a); }`;
  const copyVertex = `#version 300 es
    out vec2 vUv; uniform float uScale;
    void main() { vec2 p = vec2(gl_VertexID & 1, gl_VertexID >> 1) * 2.0; vUv = p * uScale; gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;
  const copyFragment = `#version 300 es
    precision highp float; in vec2 vUv; uniform sampler2D uImage; out vec4 color;
    void main() { color = vec4(clamp(texture(uImage, vUv).rgb, 0.0, 1.0), 1.0); }`;

  function program(gl, vs, fs) {
    const p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  // ---------------------------------------------------------------- one set of Gaussians on the GPU
  class GaussianSet {
    constructor(gl, { buffer, meta }, onReady, onSorted, onError) {
      this.count = 0; this.busy = false; this.sorted = 0; this.rgbMax = meta.rgb_max;
      const url = URL.createObjectURL(new Blob([workerSource], { type: "text/javascript" }));
      this.worker = new Worker(url);
      URL.revokeObjectURL(url);
      this.worker.onmessage = ({ data }) => {
        if (data.error) { onError(new Error(data.error)); return; }
        if (data.tex) {
          this.texture = gl.createTexture();
          gl.bindTexture(gl.TEXTURE_2D, this.texture);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, 2048, data.n / 1024, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, data.tex);
          this.buffer = gl.createBuffer();
          this.total = data.n;
          onReady();
          return;
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
        gl.bufferData(gl.ARRAY_BUFFER, data.order, gl.DYNAMIC_DRAW);
        this.count = data.order.length; this.busy = false; this.sorted = data.id;
        onSorted();
      };
      this.worker.onerror = (e) => onError(new Error(e.message || "the worker could not start"));
      this.worker.postMessage({ buffer, meta }, [buffer]);
    }
    // exact: always sort for this camera; otherwise only once the camera has moved enough to matter
    sort(cam, id, tan, removed, distance, exact) {
      if (!this.total || this.busy || this.sorted === id) return;
      const last = this.last;
      if (!exact && last && last.removed === removed) {
        const turn = 1 - (cam.R[2] * last.R[2] + cam.R[5] * last.R[5] + cam.R[8] * last.R[8]);
        const move = Math.hypot(cam.t[0] - last.t[0], cam.t[1] - last.t[1], cam.t[2] - last.t[2]);
        if (turn < 2e-5 && move < 0.004 * distance) return;                 // 0.36 degrees
      }
      this.busy = true;
      this.last = { R: cam.R, t: cam.t, removed };
      this.worker.postMessage({ R: cam.R, t: cam.t, tan, removed, id });
    }
  }

  // ---------------------------------------------------------------- small pose maths (row-major 3x3, c2w)
  const mul = (A, B) => [0, 1, 2].flatMap((i) => [0, 1, 2].map((j) => A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j]));
  const apply = (A, v) => [0, 1, 2].map((i) => A[3 * i] * v[0] + A[3 * i + 1] * v[1] + A[3 * i + 2] * v[2]);
  const toQuat = (R) => {
    const [a, b, c, d, e, f, g, h, i] = R, t = a + e + i;
    const q = t > 0 ? [t + 1, h - f, c - g, d - b] : a > e && a > i ? [h - f, 1 + a - e - i, b + d, c + g]
      : e > i ? [c - g, b + d, 1 + e - a - i, f + h] : [d - b, c + g, f + h, 1 + i - a - e];
    const n = Math.hypot(...q);
    return q.map((x) => x / n);
  };
  const toMat = ([w, x, y, z]) => [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z),
    1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
  const mix = (a, b, s) => {
    let q = b.q, dot = a.q.reduce((t, v, i) => t + v * q[i], 0);
    if (dot < 0) { q = q.map((v) => -v); dot = -dot; }
    const th = Math.acos(Math.min(1, dot)), k = Math.sin(th);
    const u = k < 1e-5 ? 1 - s : Math.sin((1 - s) * th) / k, v = k < 1e-5 ? s : Math.sin(s * th) / k;
    const r = a.q.map((x, i) => u * x + v * q[i]), n = Math.hypot(...r);
    return { q: r.map((x) => x / n), t: a.t.map((x, i) => x + s * (b.t[i] - x)) };
  };
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const still = window.matchMedia("(prefers-reduced-motion: reduce)");

  // ---------------------------------------------------------------- viewer
  class Viewer {
    constructor(root) {
      this.root = root; this.stage = root.querySelector(".viewer-stage"); this.button = root.querySelector(".viewer-load");
      this.note = root.querySelector(".viewer-note"); this.names = root.dataset.sets.split(","); this.base = root.dataset.scene + "/";
      this.button.addEventListener("click", () => this.load());
      if ("auto" in root.dataset && !(navigator.connection && navigator.connection.saveData)) {
        const seen = new IntersectionObserver((e) => { if (e.some((x) => x.isIntersecting)) { seen.disconnect(); this.load(); } });
        seen.observe(root);
      }
    }
    async fetchSet(name) {
      const meta = this.scene.sets[name], response = await fetch(this.base + meta.file);
      if (!response.ok) throw new Error(`${meta.file}: HTTP ${response.status}`);
      // .txt files hold base64 (hosts that do not serve binary files)
      const buffer = meta.file.endsWith(".txt") ? Uint8Array.from(atob(await response.text()), (ch) => ch.charCodeAt(0)).buffer
        : await response.arrayBuffer();
      if (buffer.byteLength !== meta.bytes) throw new Error(`${name}: ${buffer.byteLength} of ${meta.bytes} bytes`);
      return { buffer, meta };
    }
    async load() {
      if (this.loading) return;
      this.loading = true; this.button.disabled = true;
      try {
        const canvas = document.createElement("canvas");
        const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, preserveDrawingBuffer: true });
        if (!gl) throw new Error("this browser does not support WebGL2");
        if (!window.DecompressionStream) throw new Error("this browser is too old to unpack the scene");
        this.canvas = canvas; this.gl = gl;
        const reply = await fetch(this.base + "scene.json");
        if (!reply.ok) throw new Error(`scene.json: HTTP ${reply.status}`);
        this.scene = await reply.json();
        this.poses = this.scene.poses.map((m) => ({ q: toQuat(m.slice(0, 3).flatMap((r) => r.slice(0, 3))), t: m.slice(0, 3).map((r) => r[3]) }));
        this.button.textContent = "Loading…";
        this.raw = await Promise.all(this.names.map((name) => this.fetchSet(name)));
        this.setup();
      } catch (error) { this.fail(error); }
    }
    fail(error) {
      (this.sets || []).forEach((s) => s.worker.terminate());
      this.sets = null;
      const lose = this.gl && this.gl.getExtension("WEBGL_lose_context");
      this.gl = null;
      if (lose) lose.loseContext();
      this.button.disabled = false; this.loading = false; this.button.textContent = "Try again";
      this.note.textContent = `The 3D view could not be loaded: ${error.message}.`;
    }
    setup() {
      const gl = this.gl;
      this.main = program(gl, vertexShader, fragmentShader);
      this.copy = program(gl, copyVertex, copyFragment);
      this.float = !!gl.getExtension("EXT_color_buffer_float");
      this.u = Object.fromEntries(["uData", "uR", "uT", "uFocal", "uCenter", "uSize", "uRgbMax"].map((n) => [n, gl.getUniformLocation(this.main, n)]));
      this.vao = gl.createVertexArray();
      gl.bindVertexArray(this.vao);
      const quad = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(1); gl.vertexAttribDivisor(1, 1);
      let ready = 0;
      const onReady = () => { if (++ready === this.names.length) this.start(); };
      this.scaleAt = gl.getUniformLocation(this.copy, "uScale");
      this.sets = this.raw.map((raw) => new GaussianSet(gl, raw, onReady, () => { this.dirty = true; this.wake(); }, (e) => this.fail(e)));
      this.raw = null; this.target = null;
    }
    start() {
      this.state = { base: this.poses[0], yaw: 0, pitch: 0, zoom: 1 };
      this.version = 1; this.frame = 0;
      const c = this.canvas;
      c.tabIndex = 0;
      c.setAttribute("role", "application");
      c.addEventListener("webglcontextlost", () => {                       // e.g. too many contexts on a phone: start over
        if (!this.gl) return;
        this.visible = false; this.stop();
        c.remove(); this.stage.append(this.button);
        this.root.classList.remove("is-live");
        this.root.querySelectorAll(".viewer-controls button").forEach((b) => { b.disabled = true; });
        this.fail(new Error("the browser released its graphics memory"));
      });
      const focused = document.activeElement === this.button;
      c.setAttribute("aria-label", "3D view. Drag or use the arrow keys to orbit, plus and minus to zoom.");
      this.stage.querySelectorAll("img, .viewer-load").forEach((e) => e.remove());
      this.stage.prepend(c);
      if (focused) c.focus({ preventScroll: true });
      this.root.querySelectorAll(".viewer-controls button").forEach((b) => { b.disabled = false; });
      if (!this.bound) {
        new ResizeObserver(() => this.gl && this.resize()).observe(this.stage);
        document.addEventListener("visibilitychange", () => this.wake());
        this.bindButtons();
        this.bound = true;
      }
      this.bind();
      this.resize();
      this.sway = this.poses.length === 2 && !still.matches ? performance.now() : null;
      this.removed = false; this.quality = 1; this.spacing = 16;
      // nothing is drawn while the view is off screen or the tab is hidden
      this.visible = true;
      new IntersectionObserver((e) => { this.visible = !!this.gl && e[e.length - 1].isIntersecting; this.wake(); }).observe(this.stage);
      this.root.classList.add("is-live");
      this.wake();
    }
    bind() {
      const c = this.canvas, points = new Map();
      let pinch = 0;
      const orbit = (dx, dy) => { this.stop(); this.state.yaw = clamp(this.state.yaw + dx, -0.6, 0.6); this.state.pitch = clamp(this.state.pitch + dy, -0.4, 0.4); this.change(); };
      const zoom = (k) => { this.stop(); this.state.zoom = clamp(this.state.zoom * k, 0.5, 1.6); this.change(); };
      c.addEventListener("pointerdown", (e) => { c.setPointerCapture(e.pointerId); points.set(e.pointerId, [e.clientX, e.clientY]); pinch = 0; c.focus({ preventScroll: true }); });
      c.addEventListener("pointermove", (e) => {
        const old = points.get(e.pointerId);
        if (!old) return;
        points.set(e.pointerId, [e.clientX, e.clientY]);
        if (points.size === 2) {
          const [a, b] = [...points.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
          if (pinch) zoom(pinch / d);
          pinch = d;
        } else orbit(-(e.clientX - old[0]) * 0.004, (e.clientY - old[1]) * 0.004);
      });
      for (const type of ["pointerup", "pointercancel"]) c.addEventListener(type, (e) => { points.delete(e.pointerId); pinch = 0; });
      // the wheel scrolls the page unless the view has been clicked
      c.addEventListener("wheel", (e) => { if (document.activeElement !== c && !e.ctrlKey) return; e.preventDefault(); zoom(Math.exp(e.deltaY * 0.001)); }, { passive: false });
      c.addEventListener("keydown", (e) => {
        const move = { ArrowLeft: [-0.05, 0], ArrowRight: [0.05, 0], ArrowUp: [0, 0.05], ArrowDown: [0, -0.05] }[e.key];
        if (move) orbit(...move); else if (e.key === "+" || e.key === "=") zoom(0.9); else if (e.key === "-") zoom(1.1); else return;
        e.preventDefault();
      });
    }
    bindButtons() {
      this.root.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => this.go(+b.dataset.view)));
      const toggle = this.root.querySelector("[data-removed]");
      if (toggle) toggle.addEventListener("click", () => this.toggleRemoved(toggle));
    }
    stop() {
      if (this.sway !== null) this.state = { ...this.state };
      this.sway = null; this.flight = null;
    }
    go(k) {
      this.stop();
      if (still.matches) { this.state = { base: this.poses[k], yaw: 0, pitch: 0, zoom: 1 }; this.change(); return; }
      this.flight = { from: { ...this.state }, to: this.poses[k], at: performance.now() };
      this.wake();
    }
    change() { this.version++; this.dirty = true; this.moved = performance.now(); this.wake(); }
    wake() {
      if (this.frame || !this.visible || document.hidden || !this.sets) return;
      this.frame = requestAnimationFrame((now) => { this.frame = 0; if (this.sets && this.visible && !document.hidden && this.step(now)) this.wake(); });
    }
    toggleRemoved(button) {
      this.removed = !this.removed;
      button.setAttribute("aria-pressed", this.removed);
      this.change();
    }
    camera() {
      const { base, yaw, pitch, zoom } = this.state, d = this.scene.distance, R0 = toMat(base.q);
      const [cy, sy, cx, sx] = [Math.cos(yaw), Math.sin(yaw), Math.cos(pitch), Math.sin(pitch)];
      const R = mul(R0, mul([cy, 0, sy, 0, 1, 0, -sy, 0, cy], [1, 0, 0, 0, cx, -sx, 0, sx, cx]));
      const f0 = apply(R0, [0, 0, d]), f = apply(R, [0, 0, d * zoom]);
      return { R, t: base.t.map((v, i) => v + f0[i] - f[i]) };
    }
    // one animation step; returns whether another one is needed
    step(now) {
      const gap = this.last ? now - this.last : 16;
      this.last = now;
      if (this.sway !== null) {
        if (now - (this.swayed || 0) < 30) return true;                     // the slow sway does not need 60 frames a second
        this.clock = (this.clock || 0) + Math.min(now - (this.swayed || now), 100);
        this.swayed = now;
        this.state.base = mix(this.poses[0], this.poses[1], 0.5 - 0.5 * Math.cos(this.clock / 5000 * Math.PI));
        this.version++; this.dirty = true; this.moved = now;
      } else if (this.flight) {
        const s = clamp((now - this.flight.at) / 700, 0, 1), e = s * s * (3 - 2 * s), { from, to } = this.flight;
        this.state = { base: mix(from.base, to, e), yaw: from.yaw * (1 - e), pitch: from.pitch * (1 - e), zoom: from.zoom + (1 - from.zoom) * e };
        if (s === 1) this.flight = null;
        this.version++; this.dirty = true; this.moved = now;
      }
      const moving = now - (this.moved || 0) < 200;
      // slow machines draw fewer pixels while the camera moves, and the full image once it rests
      if (moving && gap < 500) {
        this.spacing += 0.1 * (gap - this.spacing);
        if (this.spacing > 30 && this.quality > 0.5) { this.quality = Math.max(0.5, this.quality - 0.1); this.spacing = 22; }
      } else if (!moving && this.quality < 1) { this.quality = 1; this.dirty = true; }
      const cam = this.camera(), [, , w, h] = this.viewports()[0], tan = [w / h * this.tangent(0), this.tangent(1)];
      for (const s of this.sets) s.sort(cam, this.version, tan, this.removed, this.scene.distance, !moving);
      if (this.dirty) { this.dirty = false; this.paint(cam); }
      return this.sway !== null || this.flight || moving || this.dirty || this.quality < 1 || this.sets.some((s) => s.sorted !== this.version);
    }
    // half the field of view of the 512 px image; like the rasterizer, only the angle counts (centred principal point)
    tangent(k) {
      const K = this.scene.K, size = this.scene.size;
      return Math.tan((Math.atan(K[k][2] / K[k][k]) + Math.atan((size - K[k][2]) / K[k][k])) / 2);
    }
    resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2), box = this.stage.getBoundingClientRect();
      this.canvas.width = Math.max(1, Math.round(box.width * dpr)); this.canvas.height = Math.max(1, Math.round(box.height * dpr));
      this.version++; this.dirty = true; this.wake();
    }
    // viewports in pixels, top-left origin: side by side when wide, stacked when narrow
    viewports(scale = 1) {
      const W = Math.round(this.canvas.width * scale), H = Math.round(this.canvas.height * scale), n = this.sets.length, gap = n > 1 ? Math.round(W * 0.008) : 0;
      if (W >= H) { const w = Math.floor((W - gap * (n - 1)) / n); return this.sets.map((_, i) => [i * (w + gap), 0, w, H]); }
      const h = Math.floor((H - gap * (n - 1)) / n);
      return this.sets.map((_, i) => [0, i * (h + gap), W, h]);
    }
    framebuffer() {
      const gl = this.gl, { width, height } = this.canvas;
      if (this.target && this.target.width === width && this.target.height === height) return this.target;
      const texture = gl.createTexture(), fbo = gl.createFramebuffer();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, width, height);
      for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.LINEAR);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      if (this.target) { gl.deleteTexture(this.target.texture); gl.deleteFramebuffer(this.target.fbo); }
      this.target = { width, height, texture, fbo };
      return this.target;
    }
    paint(cam = this.camera()) {
      const gl = this.gl, target = this.float ? this.framebuffer() : null, scale = target ? this.quality : 1;
      const H = Math.round(this.canvas.height * scale);
      gl.bindFramebuffer(gl.FRAMEBUFFER, target && target.fbo);
      gl.disable(gl.SCISSOR_TEST);
      gl.clearColor(1, 1, 1, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.SCISSOR_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.useProgram(this.main); gl.bindVertexArray(this.vao);
      // uR = world-to-camera; GLSL matrices are column-major, so the c2w rows are its columns
      gl.uniformMatrix3fv(this.u.uR, false, cam.R); gl.uniform3fv(this.u.uT, cam.t); gl.uniform1i(this.u.uData, 0);
      this.viewports(scale).forEach(([x, y, w, h], i) => {
        const set = this.sets[i];
        gl.viewport(x, H - y - h, w, h); gl.scissor(x, H - y - h, w, h);
        gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
        if (!set.count) return;
        gl.uniform2f(this.u.uFocal, h / 2 / this.tangent(0), h / 2 / this.tangent(1));
        gl.uniform2f(this.u.uCenter, w / 2, h / 2);
        gl.uniform2f(this.u.uSize, w, h); gl.uniform1f(this.u.uRgbMax, set.rgbMax);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, set.texture);
        gl.bindBuffer(gl.ARRAY_BUFFER, set.buffer); gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, 0);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, set.count);
      });
      gl.disable(gl.SCISSOR_TEST); gl.disable(gl.BLEND);
      if (target) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.useProgram(this.copy); gl.uniform1f(this.scaleAt, scale); gl.bindTexture(gl.TEXTURE_2D, target.texture); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
      this.drawn = this.version;
    }
  }
  document.querySelectorAll(".splat-viewer").forEach((root) => { root.viewer = new Viewer(root); });
})();
