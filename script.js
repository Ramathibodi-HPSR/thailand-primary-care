/* =========================================================
   script.js — Integrated (km slider + marker + checkbox + chart)
   Required files:
     - hospitals_confirmed.csv
     - coverage_lookup.json   (JSONL: {pc_id/hcode, radius_km, pop})
     - coverage_distance.json (JSON: {"0.1": n, ...})
   Required libs in index.html:
     - leaflet
     - papaparse
     - chart.js + chartjs-plugin-annotation
   ========================================================= */

(() => {
  // -----------------------------
  // CONFIG
  // -----------------------------
  const PATH_FACILITIES = "hospitals_confirmed.csv";
  const PATH_COV_LOOKUP_JSONL = "coverage_lookup.json";
  const PATH_COV_DISTANCE_JSON = "coverage_distance.json";

  const TH_BOUNDS = L.latLngBounds([5.6, 97.3], [20.6, 105.7]);

  // -----------------------------
  // DOM
  // -----------------------------
  const slider = document.getElementById("radius-slider");     // km
  const labelTop = document.getElementById("radius-label");    // km
  const labelSide = document.getElementById("sidebar-radius"); // km
  const kpiCoverage = document.getElementById("coverage");     // people
  const chartCanvas = document.getElementById("coverage-chart");

  // Checkboxes (optional but expected)
  const cbPublic = document.getElementById("cb-public");
  const cbPharm  = document.getElementById("cb-pharm");
  const cbNurse  = document.getElementById("cb-nurse");
  const cbDoctor = document.getElementById("cb-doctor");
  const cbOther  = document.getElementById("cb-other");

  if (!slider || !labelTop || !labelSide || !kpiCoverage) {
    console.error("Missing required DOM elements (#radius-slider/#radius-label/#sidebar-radius/#coverage).");
    return;
  }

  // -----------------------------
  // STATE
  // -----------------------------
  let currentRadiusKm = Number(slider.value || 1.0); // km

  let coverageLookup = {};      // pc_id -> { "0.1": pop, "1.0": pop, ... }
  let coverageDistance = null;  // { "0.1": n, ... }

  // Layer groups controlled by checkboxes
  const LAYERS = {
    public: L.layerGroup(),
    pharm:  L.layerGroup(),
    nurse:  L.layerGroup(),
    doctor: L.layerGroup(),
    other:  L.layerGroup()
  };

  // Keep references so we can update radius + popup content on slider move
  // items: { circle, marker, pcId, name, ctype }
  const facilityItems = [];

  // Chart state
  let chart = null;
  let chartLabels = []; // ["0.1","0.2",...]
  let chartValues = [];

  // -----------------------------
  // UTILS
  // -----------------------------
  const fmtInt = (n) => Number(n).toLocaleString("en-US", { maximumFractionDigits: 0 });
  const kmKey  = (km) => Number(km).toFixed(1);     // "1.0"
  const kmToM  = (km) => km * 1000;

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
    if (t.includes("pharm"))  return "pharm";
    if (t.includes("nurse"))  return "nurse";
    if (t.includes("doctor")) return "doctor";
    return "other";
  }

  function typeColor(ctype) {
    const t = String(ctype || "").toLowerCase();
    if (t.includes("pharm"))  return { line: "rgba(255,140,0,0.85)",  fill: "rgba(255,140,0,0.18)" };
    if (t.includes("nurse"))  return { line: "rgba(220,53,69,0.85)",  fill: "rgba(220,53,69,0.18)" };
    if (t.includes("doctor")) return { line: "rgba(40,167,69,0.85)",  fill: "rgba(40,167,69,0.18)" };
    if (t.includes("public")) return { line: "rgba(0,102,255,0.85)",  fill: "rgba(0,102,255,0.18)" };
    return { line: "rgba(200,200,200,0.75)", fill: "rgba(200,200,200,0.12)" };
  }

  // -----------------------------
  // MAP INIT
  // -----------------------------
  const map = L.map("map", {
    minZoom: 5,
    maxZoom: 19,
    maxBounds: TH_BOUNDS,
    maxBoundsViscosity: 1.0,
    zoomControl: false
  });

  map.fitBounds(TH_BOUNDS, { padding: [20, 20] });
  L.control.zoom({ position: "bottomright" }).addTo(map);

  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap &copy; CARTO"
  }).addTo(map);

  // Add all layers initially
  Object.values(LAYERS).forEach(lg => lg.addTo(map));

  function setLayerVisible(key, visible) {
    const lg = LAYERS[key];
    if (!lg) return;
    if (visible) map.addLayer(lg);
    else map.removeLayer(lg);
  }

  // Hook checkbox events (if elements exist)
  cbPublic?.addEventListener("change", () => setLayerVisible("public", cbPublic.checked));
  cbPharm ?.addEventListener("change", () => setLayerVisible("pharm",  cbPharm.checked));
  cbNurse ?.addEventListener("change", () => setLayerVisible("nurse",  cbNurse.checked));
  cbDoctor?.addEventListener("change", () => setLayerVisible("doctor", cbDoctor.checked));
  cbOther ?.addEventListener("change", () => setLayerVisible("other",  cbOther.checked));

  // If checkboxes exist, apply initial state
  if (cbPublic) setLayerVisible("public", cbPublic.checked);
  if (cbPharm)  setLayerVisible("pharm",  cbPharm.checked);
  if (cbNurse)  setLayerVisible("nurse",  cbNurse.checked);
  if (cbDoctor) setLayerVisible("doctor", cbDoctor.checked);
  if (cbOther)  setLayerVisible("other",  cbOther.checked);

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
      const r  = rec.radius_km;
      const pop = rec.pop;

      if (!pc || r === null || r === undefined || pop === null || pop === undefined) continue;

      const key = kmKey(r);     // "1.0"
      const val = Number(pop);
      if (!Number.isFinite(val)) continue;

      lookup[pc] ??= {};
      lookup[pc][key] = val;
    }
    return lookup;
  }

  function getCoverageForPc(pcId, radiusKm) {
    const rec = coverageLookup[String(pcId)];
    if (!rec) return null;

    const want = Number(radiusKm);
    const exactKey = kmKey(want);
    if (rec[exactKey] !== undefined) return rec[exactKey];

    // fallback nearest <=
    const keys = Object.keys(rec).map(Number).filter(Number.isFinite).sort((a,b)=>a-b);
    let val = null;
    for (const k of keys) if (k <= want) val = rec[kmKey(k)];
    return val;
  }

  // -----------------------------
  // COVERAGE DISTANCE (sum by radius)
  // -----------------------------
  function getTotalCoverage(radiusKm) {
    if (!coverageDistance) return null;

    const exactKey = kmKey(radiusKm);
    if (coverageDistance[exactKey] !== undefined) return Number(coverageDistance[exactKey]);

    // fallback nearest <=
    const want = Number(radiusKm);
    const keys = Object.keys(coverageDistance).map(Number).filter(Number.isFinite).sort((a,b)=>a-b);
    let val = null;
    for (const k of keys) if (k <= want) val = Number(coverageDistance[kmKey(k)]);
    return val;
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
        error: reject
      });
    });
  }

  function renderPopup({ name, ctype, pcId }) {
    const cov = pcId ? getCoverageForPc(pcId, currentRadiusKm) : null;
    const covTxt = (typeof cov === "number") ? fmtInt(cov) : "N/A";
    return `
      <b>${name}</b><br/>
      Type: ${ctype || "N/A"}<br/>
      ID: ${pcId || "N/A"}<br/>
      Radius: ${kmKey(currentRadiusKm)} km<br/>
      Coverage: <b>${covTxt}</b> คน
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

      // Circle coverage
      const circle = L.circle([lat, lon], {
        radius: radiusM,
        color: colors.line,
        fillColor: colors.fill,
        fillOpacity: 1,
        weight: 1.2
      }).addTo(LAYERS[layerKey]);

      // Marker center
      const marker = L.circleMarker([lat, lon], {
        radius: 0.3,
        color: "#ffffff",
        weight: 1.2,
        fillColor: colors.line,
        fillOpacity: 1
      }).addTo(LAYERS[layerKey]);

      const item = { circle, marker, pcId, name, ctype };

      // Shared popup
      circle.bindPopup(() => renderPopup(item));
      marker.bindPopup(() => renderPopup(item));

      facilityItems.push(item);
      bounds.extend([lat, lon]);
    }

    if (bounds.isValid()) map.fitBounds(bounds, { padding: [30, 30] });
  }

  // -----------------------------
  // CHART (interactive)
  // -----------------------------
  function initChart() {
    if (!chartCanvas || !coverageDistance) return;

    const radii = Object.keys(coverageDistance)
      .map(Number)
      .filter(Number.isFinite)
      .sort((a,b)=>a-b);

    chartLabels = radii.map(r => kmKey(r)); // strings "0.1", ...
    chartValues = chartLabels.map(k => Number(coverageDistance[k] ?? 0));

    // register annotation plugin
    if (window["chartjs-plugin-annotation"]) {
      Chart.register(window["chartjs-plugin-annotation"]);
    }

    chart = new Chart(chartCanvas, {
      type: "line",
      data: {
        labels: chartLabels,
        datasets: [{
          data: chartValues,
          tension: 0.25,
          pointRadius: 2,
          pointHoverRadius: 5,
          fill: true,
          borderColor: "#d14f4fff",
          backgroundColor: "rgba(209, 79, 79, 0.12)",
          pointBackgroundColor: "#d14f4fff"
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },

        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "rgba(20,20,20,0.95)",
            titleColor: "#ffffff",
            bodyColor: "#ffffff",
            borderColor: "rgba(255,255,255,0.2)",
            borderWidth: 1,
            callbacks: {
              title: (items) => `รัศมี ${items[0].label} กิโลเมตร`,
              label: (item) => `ครอบคลุม ${fmtInt(item.raw)} คน`
            }
          },
          annotation: {
            annotations: {
              currentRadiusLine: {
                type: "line",
                xMin: kmKey(currentRadiusKm),
                xMax: kmKey(currentRadiusKm),
                borderWidth: 2,
                borderColor: "rgba(255,255,255,0.55)"
              }
            }
          }
        },

        scales: {
          x: {
            type: "category",
            title: { display: true, text: "รัศมี (กม.)", color: "rgba(255,255,255,0.85)" },
            grid: { color: "rgba(255,255,255,0.08)" },
            ticks: {
              color: "rgba(255,255,255,0.8)",
              autoSkip: true,
              maxTicksLimit: 12,
              // show only integer km labels: 1.0 2.0 3.0 ...
              callback: (_, index) => {
                const lbl = chartLabels[index];
                const n = Number(lbl);
                return Number.isInteger(n) ? n.toFixed(1) : "";
              }
            }
          },
          y: {
            title: { display: true, text: "จำนวนคน", color: "rgba(255,255,255,0.85)" },
            grid: { color: "rgba(255,255,255,0.08)" },
            ticks: {
              color: "rgba(255,255,255,0.8)",
              callback: (v) => (v === 0 ? "0" : `${(v / 1_000_000).toFixed(0)}M`)
            }
          }
        },

        // Click point -> set slider
        onClick: (evt) => {
          const pts = chart.getElementsAtEventForMode(evt, "nearest", { intersect: true }, true);
          if (!pts.length) return;
          const idx = pts[0].index;
          const rKm = Number(chartLabels[idx]);
          slider.value = kmKey(rKm);
          applyRadius(rKm);
        }
      }
    });

    updateChartMarker(currentRadiusKm);
  }

  function updateChartMarker(radiusKm) {
    if (!chart) return;
    const key = kmKey(radiusKm);
    const ann = chart.options.plugins.annotation.annotations.currentRadiusLine;
    ann.xMin = key;
    ann.xMax = key;
    chart.update("none");
  }

  // -----------------------------
  // APPLY RADIUS (single source of truth)
  // -----------------------------
  function applyRadius(radiusKm) {
    if (!Number.isFinite(radiusKm)) return;
    currentRadiusKm = radiusKm;

    const kmTxt = Number(radiusKm).toLocaleString("en-US", {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1
    });

    // labels
    labelTop.textContent = kmTxt;
    labelSide.textContent = kmTxt;

    // KPI (from coverage_distance)
    const total = getTotalCoverage(radiusKm);
    kpiCoverage.textContent = (typeof total === "number") ? fmtInt(total) : "—";

    // update circle radius (km -> meters)
    const rM = kmToM(radiusKm);
    for (const it of facilityItems) {
      it.circle.setRadius(rM);
      // refresh popup content if open
      const pop1 = it.circle.getPopup();
      if (pop1 && pop1.isOpen()) it.circle.setPopupContent(renderPopup(it));
      const pop2 = it.marker.getPopup();
      if (pop2 && pop2.isOpen()) it.marker.setPopupContent(renderPopup(it));
    }

    updateChartMarker(radiusKm);
  }

  // Slider listener (km)
  slider.addEventListener("input", (e) => applyRadius(Number(e.target.value)));

  // -----------------------------
  // MAIN
  // -----------------------------
  async function main() {
    // 1) coverage_distance.json
    {
      const res = await fetch(PATH_COV_DISTANCE_JSON);
      if (!res.ok) throw new Error(`Failed to load ${PATH_COV_DISTANCE_JSON}`);
      coverageDistance = await res.json();
    }

    // 2) init chart
    initChart();

    // 3) coverage_lookup.json (JSONL) for per-facility popup
    {
      const res = await fetch(PATH_COV_LOOKUP_JSONL);
      if (!res.ok) throw new Error(`Failed to load ${PATH_COV_LOOKUP_JSONL}`);
      coverageLookup = parseCoverageJsonl(await res.text());
    }

    // 4) facilities
    {
      const rows = await loadFacilitiesCsv();
      plotFacilities(rows);
    }

    // 5) initial apply
    applyRadius(Number(slider.value || 1.0));
  }

  document.addEventListener("DOMContentLoaded", () => {
    main().catch(err => {
      console.error(err);
      alert(
        "Load data failed.\n" +
        "Tip: run local server: python -m http.server\n" +
        "Ensure these files exist next to index.html:\n" +
        `- ${PATH_FACILITIES}\n` +
        `- ${PATH_COV_LOOKUP_JSONL}\n` +
        `- ${PATH_COV_DISTANCE_JSON}`
      );
    });
  });
})();
