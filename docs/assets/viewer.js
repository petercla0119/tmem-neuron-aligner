
(function () {
  function initViewer(viewer) {
    const frames = Array.from(viewer.querySelectorAll(".viewer-frame"));
    if (!frames.length) return;
    const slider = viewer.querySelector("[data-viewer-slider]");
    const label = viewer.querySelector("[data-viewer-label]");
    const playButton = viewer.querySelector('[data-viewer-action="play"]');
    const blinkButton = viewer.querySelector('[data-viewer-action="blink"]');
    const onionButton = viewer.querySelector('[data-viewer-action="onion"]');
    const speed = viewer.querySelector("[data-viewer-speed]");
    const opacity = viewer.querySelector("[data-viewer-opacity]");
    const dayButtons = Array.from(viewer.querySelectorAll("[data-day-index]"));
    let index = 0;
    let timer = null;
    let blink = false;
    let blinkToggle = false;
    function show(nextIndex) {
      index = (nextIndex + frames.length) % frames.length;
      frames.forEach((frame, frameIndex) => {
        frame.classList.toggle("active", frameIndex === index);
        frame.classList.toggle("previous", frameIndex === ((index - 1 + frames.length) % frames.length));
        frame.style.opacity = frameIndex === index ? (opacity ? opacity.value : "1") : "";
      });
      dayButtons.forEach((button, buttonIndex) => button.classList.toggle("active", buttonIndex === index));
      if (slider) slider.value = String(index);
      if (label) label.textContent = "Day " + (frames[index].dataset.day || String(index + 1));
    }
    function step() {
      if (blink && frames.length > 1) {
        blinkToggle = !blinkToggle;
        show(blinkToggle ? index : index + 1);
      } else {
        show(index + 1);
      }
    }
    function stop() {
      if (timer) window.clearInterval(timer);
      timer = null;
      if (playButton) playButton.textContent = "Play";
    }
    function play() {
      stop();
      timer = window.setInterval(step, Number(speed ? speed.value : 700));
      if (playButton) playButton.textContent = "Pause";
    }
    viewer.addEventListener("click", function (event) {
      const target = event.target.closest("button");
      if (!target) return;
      if (target.dataset.dayIndex) {
        stop();
        show(Number(target.dataset.dayIndex));
      }
      const action = target.dataset.viewerAction;
      if (action === "prev") {
        stop();
        show(index - 1);
      }
      if (action === "next") {
        stop();
        show(index + 1);
      }
      if (action === "play") {
        timer ? stop() : play();
      }
      if (action === "blink") {
        blink = !blink;
        target.textContent = blink ? "Blink on" : "Blink off";
      }
      if (action === "onion") {
        viewer.classList.toggle("onion");
        target.textContent = viewer.classList.contains("onion") ? "Onion skin on" : "Onion skin off";
      }
    });
    if (slider) {
      slider.addEventListener("input", function () {
        stop();
        show(Number(slider.value));
      });
    }
    if (opacity) {
      opacity.addEventListener("input", function () {
        frames[index].style.opacity = opacity.value;
      });
    }
    if (speed) {
      speed.addEventListener("change", function () {
        if (timer) play();
      });
    }
    show(0);
  }
  function initPlateFilters() {
    const search = document.querySelector("[data-plate-search]");
    const condition = document.querySelector('[data-plate-filter="condition"]');
    const qc = document.querySelector('[data-plate-filter="qc"]');
    const wells = Array.from(document.querySelectorAll(".plate-well.active"));
    function applyFilters() {
      const query = search ? search.value.trim().toUpperCase() : "";
      const conditionValue = condition ? condition.value : "";
      const qcValue = qc ? qc.value : "";
      wells.forEach((well) => {
        const wellId = well.dataset.well || "";
        const cond = well.dataset.condition || "";
        const qcLabel = well.dataset.qc || "";
        const matchesSearch = !query || wellId.includes(query);
        const matchesCondition = !conditionValue ||
          (conditionValue === "reporter_control" && cond.includes("reporter_control")) ||
          (conditionValue === "tmem106b" && cond.includes("TMEM106B"));
        const matchesQc = !qcValue || qcLabel === qcValue;
        well.classList.toggle("hidden-by-filter", !(matchesSearch && matchesCondition && matchesQc));
      });
    }
    [search, condition, qc].forEach((control) => {
      if (control) control.addEventListener("input", applyFilters);
      if (control) control.addEventListener("change", applyFilters);
    });
  }
  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-tmem-viewer]").forEach(initViewer);
    initPlateFilters();
  });
})();
