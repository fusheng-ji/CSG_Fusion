(() => {
  const lazyVideos = document.querySelectorAll("video[data-src]");

  const loadVideo = (video) => {
    if (video.dataset.loaded === "true") return;
    video.src = video.dataset.src;
    video.dataset.loaded = "true";
    video.load();
  };

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          loadVideo(entry.target);
          observer.unobserve(entry.target);
        });
      },
      { rootMargin: "400px 0px" }
    );

    lazyVideos.forEach((video) => observer.observe(video));
  } else {
    lazyVideos.forEach(loadVideo);
  }
})();

(() => {
  const desktopZoom = window.matchMedia("(any-hover: hover) and (any-pointer: fine)");
  const maxZoom = 2.5;
  const defaultLensWidth = 320;
  const defaultLensHeight = 150;
  const minimumLensWidth = 220;
  const minimumLensHeight = 104;
  let teardownZoom = () => {};

  const initializeZoom = () => {
    const cleanups = Array.from(document.querySelectorAll(".figure-zoom")).map((container) => {
      const image = container.querySelector("img");
      if (!image) return () => {};

      const lens = document.createElement("span");
      lens.className = "figure-zoom-lens";
      lens.setAttribute("aria-hidden", "true");
      container.append(lens);

      let frameId = null;
      let pointerPosition = null;

      const renderLens = () => {
        frameId = null;
        if (!pointerPosition) return;

        const imageRect = image.getBoundingClientRect();
        if (!imageRect.width || !imageRect.height) return;

        const x = Math.min(Math.max(pointerPosition.x - imageRect.left, 0), imageRect.width);
        const y = Math.min(Math.max(pointerPosition.y - imageRect.top, 0), imageRect.height);
        const naturalScale =
          image.naturalWidth && image.naturalHeight
            ? Math.min(
                image.naturalWidth / imageRect.width,
                image.naturalHeight / imageRect.height
              )
            : maxZoom;
        const zoom = Math.max(1, Math.min(maxZoom, naturalScale));
        const lensWidth = Math.min(
          imageRect.width,
          Math.max(minimumLensWidth, Math.min(defaultLensWidth, imageRect.width * 0.72))
        );
        const lensHeight = Math.min(
          imageRect.height,
          Math.max(
            minimumLensHeight,
            Math.min(defaultLensHeight, imageRect.height * 0.82, lensWidth * 0.5)
          )
        );
        const lensLeft = Math.min(
          Math.max(x, lensWidth / 2),
          imageRect.width - lensWidth / 2
        );
        const lensTop = Math.min(
          Math.max(y, lensHeight / 2),
          imageRect.height - lensHeight / 2
        );
        const source = image.currentSrc || image.src;

        container.style.setProperty("--lens-width", `${lensWidth}px`);
        container.style.setProperty("--lens-height", `${lensHeight}px`);
        lens.style.left = `${lensLeft}px`;
        lens.style.top = `${lensTop}px`;
        lens.style.backgroundImage = `url(${JSON.stringify(source)})`;
        lens.style.backgroundSize = `${imageRect.width * zoom}px ${imageRect.height * zoom}px`;
        lens.style.backgroundPosition =
          `${lensWidth / 2 - x * zoom}px ${lensHeight / 2 - y * zoom}px`;
      };

      const queueRender = (event) => {
        pointerPosition = { x: event.clientX, y: event.clientY };
        if (frameId === null) frameId = window.requestAnimationFrame(renderLens);
      };

      const showLens = (event) => {
        container.classList.add("is-zooming");
        queueRender(event);
      };

      const hideLens = () => {
        container.classList.remove("is-zooming");
        pointerPosition = null;
        if (frameId !== null) {
          window.cancelAnimationFrame(frameId);
          frameId = null;
        }
      };

      container.addEventListener("mouseenter", showLens);
      container.addEventListener("mousemove", queueRender);
      container.addEventListener("mouseleave", hideLens);

      return () => {
        hideLens();
        container.removeEventListener("mouseenter", showLens);
        container.removeEventListener("mousemove", queueRender);
        container.removeEventListener("mouseleave", hideLens);
        container.style.removeProperty("--lens-width");
        container.style.removeProperty("--lens-height");
        lens.remove();
      };
    });

    return () => cleanups.forEach((cleanup) => cleanup());
  };

  const syncZoomBehavior = () => {
    teardownZoom();
    const zoomEnabled = desktopZoom.matches || navigator.maxTouchPoints === 0;
    document.documentElement.classList.toggle("figure-zoom-enabled", zoomEnabled);
    teardownZoom = zoomEnabled ? initializeZoom() : () => {};
  };

  syncZoomBehavior();

  if (typeof desktopZoom.addEventListener === "function") {
    desktopZoom.addEventListener("change", syncZoomBehavior);
  } else {
    desktopZoom.addListener(syncZoomBehavior);
  }
})();
