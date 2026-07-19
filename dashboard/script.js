/* =========================================================
   script.js — PC_MAP dashboard
   Required files (in ./data/):
     - hospitals_confirmed.csv
     - coverage_lookup.json          (JSONL: {pc_id, radius_km, pop})
     - coverage_distance.json        ({ "0.1": people, ... })
     - coverage_distance_public.json ({ "0.1": people, ... })
   Required libs in index.html: leaflet, papaparse, chart.js
   ========================================================= */

(() => {
  // -----------------------------
  // CONFIG
  // -----------------------------
  const PATH_FACILITIES = "data/hospitals_confirmed.csv";
  const PATH_COV_LOOKUP_JSONL = "data/coverage_lookup.json";
  const PATH_COV_DISTANCE_JSON = "data/coverage_distance.json";
  const PATH_COV_DISTANCE_PUBLIC_JSON = "data/coverage_distance_public.json";

  const TH_BOUNDS = L.latLngBounds([5.6, 97.3], [20.6, 105.7]);

  // Chart/UI colors — keep in sync with the CSS custom properties in style.css
  const COLOR = {
    total: "#3987e5",
    public: "#199e70",
    gap: "#9085e9",
    gridline: "rgba(255,255,255,0.08)",
    textSecondary: "rgba(255,255,255,0.75)",
    marker: "rgba(255,255,255,0.55)",
  };

  const TYPE_COLOR = {
    public: { line: "#3987e5", fill: "rgba(57,135,229,0.18)" },
    pharm: { line: "#d95926", fill: "rgba(217,89,38,0.18)" },
    nurse: { line: "#e66767", fill: "rgba(230,103,103,0.18)" },
    doctor: { line: "#008300", fill: "rgba(0,131,0,0.20)" },
    other: { line: "#898781", fill: "rgba(137,135,129,0.14)" },
  };

  // -----------------------------
  // DOM
  // -----------------------------
  const slider = document.getElementById("radius-slider");
  const labelTop = document.getElementById("radius-label");
  const labelSide = document.getElementById("sidebar-radius");
  const kpiCoverage = document.getElementById("coverage");
  const kpiCoveragePublic = document.getElementById("coverage-public");
  const kpiCoverageGap = document.getElementById("coverage-gap");
  const chartCanvas = document.getElementById("coverage-chart");

  const cbPublic = document.getElementById("cb-public");
  const cbPharm = document.getElementById("cb-pharm");
  const cbNurse = document.getElementById("cb-nurse");
  const cbDoctor = document.getElementById("cb-doctor");
  const cbOther = document.getElementById("cb-other");

  if (!slider || !labelTop || !labelSide || !kpiCoverage) {
    console.error("Missing required DOM elements (#radius-slider/#radius-label/#sidebar-radius/#coverage).");
    return;
  }

  // -----------------------------
  // STATE
  // -----------------------------
  let radii = [];              // available radii (km), ascending — read from coverage_distance.json
  let currentRadiusKm = 1.0;

  let coverageLookup = {};       // pc_id -> { "0.1": pop, ... }
  let coverageDistance = null;   // { "0.1": n, ... }
  let coverageDistancePublic = null;

  const LAYERS = {
    public: L.layerGroup(),
    pharm: L.layerGroup(),
    nurse: L.layerGroup(),
    doctor: L.layerGroup(),
    other: L.layerGroup(),
  };

  const facilityItems = []; // { circle, marker, pcId, name, ctype }

  let chartCoverage = null;
  let chartLabels = [];
  let chartValuesAll = [];
  let chartValuesPublic = [];
  let chartValuesGap = [];

  // -----------------------------
  // UTILS
  // -----------------------------
  const fmtInt = (n) => Number(n).toLocaleString("en-US", { maximumFractionDigits: 0 });
  const kmKey = (km) => Number(km).toFixed(1);
  const kmToM = (km) => km * 1000;
  const fmtCompact = (n) => (n === 0 ? "0" : `${(n / 1_000_000).toFixed(0)}M`);

  function safeGet(obj, keys) {
    for (const k of keys) if (obj[k] !== undefined && obj[k] !== null && obj[k] !== "") return obj[k];
    return null;
  }

  function normalizePcId(x) {
    if (x === null || x === undefined) return null;
    return String(x).replace(/\.0$/, "").padStart(5, "0");
  }

  function layerKeyFromType(ctype) {
    const t = String(ctype || "").toLowerCase();
    if (t.includes("public")) return "public";
    if (t.includes("pharm")) return "pharm";
    if (t.includes("nurse")) return "nurse";
    if (t.includes("doctor")) return "doctor";
    return "other";
  }

  function typeColor(ctype) {
    return TYPE_COLOR[layerKeyFromType(ctype)];
  }

  // -----------------------------
  // MAP INIT
  // -----------------------------
  const map = L.map("map", {
    minZoom: 5,
    maxZoom: 19,
    maxBounds: TH_BOUNDS,
    maxBoundsViscosity: 1.0,
    zoomControl: false,
  });

  map.fitBounds(TH_BOUNDS, { padding: [20, 20] });
  L.control.zoom({ position: "bottomright" }).addTo(map);

  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap &copy; CARTO",
  }).addTo(map);

  Object.values(LAYERS).forEach((lg) => lg.addTo(map));

  function setLayerVisible(key, visible) {
    const lg = LAYERS[key];
    if (!lg) return;
    if (visible) map.addLayer(lg);
    else map.removeLayer(lg);
  }

  cbPublic?.addEventListener("change", () => setLayerVisible("public", cbPublic.checked));
  cbPharm?.addEventListener("change", () => setLayerVisible("pharm", cbPharm.checked));
  cbNurse?.addEventListener("change", () => setLayerVisible("nurse", cbNurse.checked));
  cbDoctor?.addEventListener("change", () => setLayerVisible("doctor", cbDoctor.checked));
  cbOther?.addEventListener("change", () => setLayerVisible("other", cbOther.checked));

  [
    ["public", cbPublic],
    ["pharm", cbPharm],
    ["nurse", cbNurse],
    ["doctor", cbDoctor],
    ["other", cbOther],
  ].forEach(([key, cb]) => cb && setLayerVisible(key, cb.checked));

  // -----------------------------
  // COVERAGE LOOKUP (JSONL)
  // -----------------------------
  function parseCoverageJsonl(text) {
    const lookup = {};
    const lines = text.split("\n");

    for (const raw of lines) {
      const line = raw.trim();
      if (!line) continue;

      let rec;
      try { rec = JSON.parse(line); } catch { continue; }

      const pc = normalizePcId(rec.pc_id ?? rec.hcode);
      const r = rec.radius_km;
      const pop = rec.pop;

      if (!pc || r === null || r === undefined || pop === null || pop === undefined) continue;

      const key = kmKey(r);
      const val = Number(pop);
      if (!Number.isFinite(val)) continue;

      lookup[pc] ??= {};
      lookup[pc][key] = val;
    }
    return lookup;
  }

  function nearestAtOrBelow(dict, radiusKm) {
    if (!dict) return null;
    const exactKey = kmKey(radiusKm);
    if (dict[exactKey] !== undefined) return dict[exactKey];

    const want = Number(radiusKm);
    const keys = Object.keys(dict).map(Number).filter(Number.isFinite).sort((a, b) => a - b);
    let val = null;
    for (const k of keys) if (k <= want) val = dict[kmKey(k)];
    return val;
  }

  function getCoverageForPc(pcId, radiusKm) {
    return nearestAtOrBelow(coverageLookup[String(pcId)], radiusKm);
  }
  function getTotalCoverage(radiusKm) {
    const v = nearestAtOrBelow(coverageDistance, radiusKm);
    return v === null ? null : Number(v);
  }
  function getPublicCoverage(radiusKm) {
    const v = nearestAtOrBelow(coverageDistancePublic, radiusKm);
    return v === null ? null : Number(v);
  }

  // -----------------------------
  // FACILITIES CSV
  // -----------------------------
  function loadFacilitiesCsv() {
    return new Promise((resolve, reject) => {
      Papa.parse(PATH_FACILITIES, {
        download: true,
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => resolve(results.data || []),
        error: reject,
      });
    });
  }

  function renderPopup({ name, ctype, pcId }) {
    const cov = pcId ? getCoverageForPc(pcId, currentRadiusKm) : null;
    const covTxt = typeof cov === "number" ? fmtInt(cov) : "N/A";
    return `
      <b>${name}</b><br/>
      Type: ${ctype || "N/A"}<br/>
      ID: ${pcId || "N/A"}<br/>
      Radius: ${kmKey(currentRadiusKm)} km<br/>
      Coverage: <b>${covTxt}</b> people
    `;
  }

  function plotFacilities(rows) {
    const bounds = L.latLngBounds([]);
    const radiusM = kmToM(currentRadiusKm);

    for (const row of rows) {
      const lat = safeGet(row, ["lat", "latitude", "LAT", "y", "Y"]);
      const lon = safeGet(row, ["lon", "lng", "longitude", "LON", "x", "X"]);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;

      const pcId = normalizePcId(safeGet(row, ["pc_id", "PC_ID", "hcode", "HCODE"]));
      const name = safeGet(row, ["pc_name", "name", "NAME"]) || "Facility";
      const ctype = safeGet(row, ["clinic_type", "type", "facility_type"]) || "";

      const layerKey = layerKeyFromType(ctype);
      const colors = typeColor(ctype);

      const circle = L.circle([lat, lon], {
        radius: radiusM,
        color: colors.line,
        fillColor: colors.fill,
        fillOpacity: 1,
        weight: 1.2,
      }).addTo(LAYERS[layerKey]);

      const marker = L.circleMarker([lat, lon], {
        radius: 3,
        color: "#ffffff",
        weight: 1.2,
        fillColor: colors.line,
        fillOpacity: 1,
      }).addTo(LAYERS[layerKey]);

      const item = { circle, marker, pcId, name, ctype };

      circle.bindPopup(() => renderPopup(item));
      marker.bindPopup(() => renderPopup(item));

      facilityItems.push(item);
      bounds.extend([lat, lon]);
    }

    if (bounds.isValid()) map.fitBounds(bounds, { padding: [30, 30] });
  }

  // -----------------------------
  // CHART
  // -----------------------------
  // Draws a vertical line at the currently-selected radius, in place of the
  // (unused, never-actually-registered) chartjs-plugin-annotation dependency.
  const currentRadiusLinePlugin = {
    id: "currentRadiusLine",
    afterDraw(chart) {
      const idx = chartLabels.indexOf(kmKey(currentRadiusKm));
      if (idx === -1) return;
      const { ctx, chartArea, scales } = chart;
      const x = scales.x.getPixelForValue(idx);
      ctx.save();
      ctx.strokeStyle = COLOR.marker;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();
      ctx.restore();
    },
  };

  function initChart() {
    if (!chartCanvas || !coverageDistance) return;

    chartLabels = radii.map(kmKey);
    chartValuesAll = chartLabels.map((k) => Number(coverageDistance[k] ?? 0));
    const hasPublic = !!coverageDistancePublic;
    chartValuesPublic = hasPublic ? chartLabels.map((k) => Number(coverageDistancePublic[k] ?? 0)) : [];
    chartValuesGap = hasPublic ? chartValuesAll.map((v, i) => Math.max(v - chartValuesPublic[i], 0)) : [];

    const datasets = [
      {
        label: "All facility types",
        data: chartValuesAll,
        borderColor: COLOR.total,
        backgroundColor: COLOR.total,
        borderWidth: 2,
        tension: 0.25,
        pointRadius: 0,
      },
    ];

    if (hasPublic) {
      datasets.push(
        {
          label: "Public only",
          data: chartValuesPublic,
          borderColor: COLOR.public,
          backgroundColor: COLOR.public,
          borderWidth: 2,
          tension: 0.25,
          pointRadius: 0,
        },
        {
          label: "Non-public gap",
          data: chartValuesGap,
          borderColor: COLOR.gap,
          backgroundColor: COLOR.gap,
          borderWidth: 1.5,
          tension: 0.25,
          pointRadius: 0,
          borderDash: [5, 4],
        }
      );
    }

    const ctx = chartCanvas.getContext("2d");
    if (chartCoverage) {
      try { chartCoverage.destroy(); } catch (e) { console.warn("Destroy skipped:", e); }
    }

    chartCoverage = new Chart(ctx, {
      type: "line",
      data: { labels: chartLabels, datasets },
      plugins: [currentRadiusLinePlugin],
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: {
            title: { display: true, text: "Radius (km)", color: COLOR.textSecondary },
            grid: { color: COLOR.gridline },
            ticks: {
              color: COLOR.textSecondary,
              autoSkip: true,
              maxTicksLimit: 8,
              callback: (_, index) => chartLabels[index],
            },
          },
          y: {
            title: { display: true, text: "Population", color: COLOR.textSecondary },
            grid: { color: COLOR.gridline },
            ticks: { color: COLOR.textSecondary, callback: fmtCompact },
          },
        },
        plugins: {
          legend: { display: false }, // legend is rendered as HTML in the sidebar
          tooltip: {
            backgroundColor: "#1a1a19",
            borderColor: "rgba(255,255,255,0.12)",
            borderWidth: 1,
            titleColor: "#ffffff",
            bodyColor: "rgba(255,255,255,0.85)",
            padding: 10,
            callbacks: {
              label: (item) => `${item.dataset.label}: ${fmtInt(item.parsed.y)}`,
            },
          },
        },
        onClick(evt, _elements, chart) {
          const pts = chart.getElementsAtEventForMode(evt, "nearest", { intersect: true }, true);
          if (!pts.length) return;
          const idx = pts[0].index;
          const rKm = Number(chartLabels[idx]);
          if (!Number.isFinite(rKm)) return;
          setRadiusByValue(rKm);
        },
      },
    });
  }

  // -----------------------------
  // SLIDER (index-based over the published radius steps, so every
  // position on the slider maps to a real, published data point)
  // -----------------------------
  function setupSlider() {
    slider.min = 0;
    slider.max = radii.length - 1;
    slider.step = 1;
    slider.value = radii.indexOf(1.0) !== -1 ? radii.indexOf(1.0) : Math.floor(radii.length / 2);
  }

  function setRadiusByValue(km) {
    const idx = radii.indexOf(Number(km.toFixed(1)));
    if (idx === -1) return;
    slider.value = idx;
    applyRadius(radii[idx]);
  }

  slider.addEventListener("input", (e) => {
    const idx = Number(e.target.value);
    applyRadius(radii[idx]);
  });

  // -----------------------------
  // APPLY RADIUS (single source of truth)
  // -----------------------------
  function applyRadius(radiusKm) {
    if (!Number.isFinite(radiusKm)) return;
    currentRadiusKm = radiusKm;

    const kmTxt = Number(radiusKm).toLocaleString("en-US", {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    });
    labelTop.textContent = kmTxt;
    labelSide.textContent = kmTxt;

    const total = getTotalCoverage(radiusKm);
    const pub = getPublicCoverage(radiusKm);
    kpiCoverage.textContent = typeof total === "number" ? fmtInt(total) : "—";
    if (kpiCoveragePublic) kpiCoveragePublic.textContent = typeof pub === "number" ? fmtInt(pub) : "—";
    if (kpiCoverageGap) {
      const gap = typeof total === "number" && typeof pub === "number" ? Math.max(total - pub, 0) : null;
      kpiCoverageGap.textContent = typeof gap === "number" ? fmtInt(gap) : "—";
    }

    const rM = kmToM(radiusKm);
    for (const it of facilityItems) {
      it.circle.setRadius(rM);
      const pop1 = it.circle.getPopup();
      if (pop1 && pop1.isOpen()) it.circle.setPopupContent(renderPopup(it));
      const pop2 = it.marker.getPopup();
      if (pop2 && pop2.isOpen()) it.marker.setPopupContent(renderPopup(it));
    }

    if (chartCoverage) chartCoverage.update();
  }

  // -----------------------------
  // MAIN
  // -----------------------------
  async function main() {
    const resAll = await fetch(PATH_COV_DISTANCE_JSON);
    if (!resAll.ok) throw new Error(`Failed to load ${PATH_COV_DISTANCE_JSON} (${resAll.status})`);
    coverageDistance = await resAll.json();

    radii = Object.keys(coverageDistance).map(Number).filter(Number.isFinite).sort((a, b) => a - b);

    const resPublic = await fetch(PATH_COV_DISTANCE_PUBLIC_JSON);
    if (!resPublic.ok) {
      console.warn(`Failed to load ${PATH_COV_DISTANCE_PUBLIC_JSON} (${resPublic.status})`);
      coverageDistancePublic = null;
    } else {
      coverageDistancePublic = await resPublic.json();
    }

    setupSlider();
    initChart();

    {
      const res = await fetch(PATH_COV_LOOKUP_JSONL);
      if (!res.ok) throw new Error(`Failed to load ${PATH_COV_LOOKUP_JSONL}`);
      coverageLookup = parseCoverageJsonl(await res.text());
    }

    {
      const rows = await loadFacilitiesCsv();
      applyRadius(radii[Number(slider.value)]); // set currentRadiusKm before plotting circles
      plotFacilities(rows);
    }

    applyRadius(radii[Number(slider.value)]);
  }

  document.addEventListener("DOMContentLoaded", () => {
    main().catch((err) => {
      console.error(err);
      alert(
        "Load data failed.\n" +
        "Tip: run a local server, e.g. python -m http.server\n" +
        "Ensure these files exist under dashboard/data/:\n" +
        `- ${PATH_FACILITIES}\n` +
        `- ${PATH_COV_LOOKUP_JSONL}\n` +
        `- ${PATH_COV_DISTANCE_JSON}`
      );
    });
  });
})();
