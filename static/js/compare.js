// Tabs (view overlap, scene), and for "Comparisons with Baselines" one zoom shared by all renders of a scene.
// Markup: <div data-tabs><div role="tablist"><button role="tab">…</div> then one role="tabpanel" child per tab.
for (const root of document.querySelectorAll("[data-tabs]")) {
  const tabs = [...root.querySelectorAll(":scope > [role=tablist] > [role=tab]")];
  const panels = [...root.querySelectorAll(":scope > [role=tabpanel]")];
  const select = (i) => {
    tabs.forEach((t, k) => { t.setAttribute("aria-selected", k === i); t.tabIndex = k === i ? 0 : -1; });
    panels.forEach((p, k) => { p.hidden = k !== i; p.querySelectorAll(".is-zoomed").forEach((e) => e.classList.remove("is-zoomed")); p.classList.remove("is-zoomed"); });
  };
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => select(i));
    tab.addEventListener("keydown", (e) => {
      const n = tabs.length, k = { ArrowRight: (i + 1) % n, ArrowLeft: (i + n - 1) % n, Home: 0, End: n - 1 }[e.key];
      if (k === undefined) return;
      e.preventDefault();
      select(k);
      tabs[k].focus();
    });
  });
  root.classList.add("is-ready");
  select(0);
}

for (const panel of document.querySelectorAll(".baseline-panel")) {
  const reset = () => panel.classList.remove("is-zoomed");
  // every tile shows the same held-out view, so one origin in percent is the same spot in all of them
  const zoom = (x, y) => {
    panel.style.setProperty("--x", `${100 * x}%`);
    panel.style.setProperty("--y", `${100 * y}%`);
    panel.classList.add("is-zoomed");
  };
  const zoomAt = (tile, e) => {
    const r = tile.getBoundingClientRect(), clamp = (v) => Math.min(Math.max(v, 0), 1);
    zoom(clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height));
  };
  for (const tile of panel.querySelectorAll(".zoom-tile")) {
    let pointer = "";
    tile.addEventListener("pointerdown", (e) => { pointer = e.pointerType; });
    tile.addEventListener("pointermove", (e) => { if (e.pointerType === "mouse") zoomAt(tile, e); });
    tile.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse") reset(); });
    tile.addEventListener("click", (e) => {
      if (e.detail && pointer === "mouse") return;              // a mouse zooms by hovering
      if (panel.classList.contains("is-zoomed")) reset();
      else if (e.detail) zoomAt(tile, e);
      else zoom(0.5, 0.5);                                      // keyboard
    });
  }
  panel.addEventListener("keydown", (e) => { if (e.key === "Escape") reset(); });
}
