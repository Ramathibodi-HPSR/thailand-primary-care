/* Dashboard v2: scroll story over one map and one chart.
   Every number in the text comes from data/facts.json (built by scripts/v2/20_dashboard_bundle.py). */
(() => {
  "use strict";

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const COLOR = {
    P: css("--c-P"), nc: css("--c-nc"), ph: css("--c-ph"), dc: css("--c-dc"),
    before: css("--n-before"), peak: css("--n-peak"), now: css("--n-now"), planner: css("--planner"),
    text: css("--text"), text2: css("--text-2"), muted: css("--muted"), hair: "rgba(255,255,255,.10)", bg: css("--bg"),
  };
  const TYPE = { P: "Class P unit", nc: "Nurse clinic", ph: "Pharmacy", dc: "Medical clinic" };
  const TYPE_TH = { P: "หน่วย class P", nc: "คลินิกการพยาบาล", ph: "ร้านยา", dc: "คลินิกเวชกรรม" };
  const YEARS = ["FY 2020", "FY 2021", "FY 2022", "FY 2023", "FY 2024", "FY 2025 (peak)", "Nov 2026"];
  const QUINTILES = ["Poorest fifth", "2nd", "3rd", "4th", "Richest fifth"];
  const TOUCH = window.matchMedia("(hover: none)").matches;
  const COARSE = window.matchMedia("(pointer: coarse)").matches;  // phones and iPads, whatever the user agent says
  const NARROW = window.matchMedia("(max-width: 900px)").matches;
  const NET = NARROW ? { before: "Before", peak: "Peak", now: "Now" } : { before: "Before FY2024", peak: "Peak, Nov 2025", now: "Now, Nov 2026" };
  const YEAR_TICKS = [["2020"], ["2021"], ["2022"], ["2023"], ["2024"], ["2025", "(peak)"], ["Nov", "2026"]];

  const fmt = (v, kind) => {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    switch (kind) {
      case "int": return Math.round(v).toLocaleString("en-US");
      case "pct": return `${v.toFixed(1)}%`;
      case "pct0": return `${Math.round(v)}%`;
      case "dec1": return v.toFixed(1);
      case "dec2": return v.toFixed(2);
      case "dec3": return v.toFixed(3);
      case "m": return `${v.toFixed(2)} million`;
      case "mth": return `${v.toFixed(2)} ล้าน`;
      default: return String(v);
    }
  };

  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };

  // Thai text on or off
  const langBtn = document.getElementById("lang");
  const setLang = (on) => {
    document.body.classList.toggle("no-th", !on);
    langBtn.setAttribute("aria-pressed", String(on));
    store.set("pcmap-th", on ? "1" : "0");
  };
  langBtn.addEventListener("click", () => setLang(langBtn.getAttribute("aria-pressed") !== "true"));
  setLang(store.get("pcmap-th") !== "0");

  // Tooltip: hover on desktop, tap on touch screens
  const tip = document.getElementById("tip");
  let tipTimer = null;
  const showTip = (html, e) => {
    tip.innerHTML = html; tip.hidden = false;
    const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    const y = Math.min(e.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
    tip.style.left = `${Math.max(8, x)}px`; tip.style.top = `${Math.max(8, y)}px`;
    clearTimeout(tipTimer);
    if (TOUCH) tipTimer = setTimeout(hideTip, 4000);
  };
  const hideTip = () => { tip.hidden = true; };
  const bindTip = (layer, html) => {
    layer.on("mouseover", (e) => showTip(html(), e.originalEvent));
    layer.on("mouseout", hideTip);
    layer.on("click", (e) => showTip(html(), e.originalEvent));
  };

  // no-cache: revalidate, so a rebuilt bundle is never mixed with an old one
  const get = (p) => fetch(p, { cache: "no-cache" }).then((r) => { if (!r.ok) throw new Error(p); return r; });
  const getJSON = (p) => get(p).then((r) => r.json());
  const getCSV = (p) => get(p).then((r) => r.text()).then((t) => Papa.parse(t, { header: true, dynamicTyping: true, skipEmptyLines: true }).data);

  Promise.all([getJSON("data/facts.json"), getJSON("data/charts.json"), getJSON("data/hex6.json"),
    getCSV("data/units.csv"), getCSV("data/sites.csv"), getJSON("data/provinces.geojson")]).then(start).catch((err) => {
    document.getElementById("error").hidden = false;
    console.error(err);
  });

  function start([facts, charts, hex, units, sites, provinces]) {
    document.querySelectorAll("[data-f]").forEach((el) => { el.textContent = fmt(facts[el.dataset.f], el.dataset.fmt); });

    /* ---------- map ---------- */
    // The page scrolls past the map, so the wheel and one-finger drags must scroll the page;
    // zoom with the +/- buttons.
    const map = L.map("map", {
      zoomControl: true, attributionControl: true, preferCanvas: true, minZoom: 3, maxZoom: 13,
      zoomSnap: 0.25, zoomDelta: 0.5, scrollWheelZoom: false, dragging: !COARSE, touchZoom: false,
    });
    const THAILAND = [[5.6, 97.3], [20.5, 105.7]];
    const fit = () => { map.invalidateSize(); map.fitBounds(THAILAND, { padding: [12, 12] }); };
    fit();
    // Refit whenever the map box itself changes size (layout settling, rotation, chart-only
    // steps). The box is sized in svh, so the iOS address bar showing or hiding does not move it.
    let fitTimer = null; let lastBox = "";
    new ResizeObserver(([e]) => {
      const box = `${Math.round(e.contentRect.width)}x${Math.round(e.contentRect.height)}`;
      if (box === lastBox || e.contentRect.height === 0) return;
      lastBox = box;
      clearTimeout(fitTimer); fitTimer = setTimeout(fit, 120);
    }).observe(document.getElementById("map"));
    const canvas = L.canvas({ padding: 0.3 });
    map.attributionControl.setPrefix(false);
    L.geoJSON(provinces, {
      renderer: canvas, interactive: false, attribution: "Boundaries: OCHA/RTSD via HDX",
      style: { color: "#33415a", weight: 0.8, fillColor: "#151d2a", fillOpacity: 1 },
    }).addTo(map);
    const zoom = () => Math.round(map.getZoom());
    const radius = () => { const z = zoom(); return z <= 5 ? 1.4 : z === 6 ? 2 : z === 7 ? 2.8 : z === 8 ? 3.6 : 4.6; };

    // Units: class P first in slate, the innovation units on top in their type colours
    const unitLayer = L.layerGroup().addTo(map);
    const markers = units.map((u) => {
      const m = L.circleMarker([u.lat, u.lon], { renderer: canvas, radius: radius(), weight: 0, fillOpacity: 0.9, fillColor: COLOR[u.t] });
      m.u = u;
      bindTip(m, () => {
        const status = u.s === "c" ? "In the network, Nov 2026" : u.s === "z" ? "Moved to class Z by Nov 2026" : "No longer an innovation unit by Nov 2026";
        return `<div class="t-name">${u.hname}</div><div>${TYPE[u.t]} · ${TYPE_TH[u.t]}</div>` +
          `<div class="t-sub">${u.y ? `Entry FY ${u.y} · ` : ""}${status}${u.c ? "<br>Placed at the tambon centre (no checked coordinate)" : ""}</div>`;
      });
      return m;
    });
    const isEntrant = (u) => u.a === 1 && u.t !== "P" && (u.y === 2024 || u.y === 2025);
    const yearInput = document.getElementById("year");
    const yearLabel = document.getElementById("year-label");
    let unitMode = "network";
    function drawUnits() {
      const idx = +yearInput.value;
      yearLabel.textContent = YEARS[idx];
      unitLayer.clearLayers();
      const r = radius();
      const base = []; const top = [];
      for (const m of markers) {
        const u = m.u;
        if (unitMode === "where") {
          if (u.a === 1 && u.t === "P") base.push([m, { radius: r * 0.8, fillColor: COLOR.P, fillOpacity: 0.35 }]);
          else if (isEntrant(u)) top.push([m, { radius: r + 0.4, fillColor: COLOR[u.t], fillOpacity: 0.95 }]);
        } else if (unitMode === "left") {
          // what left: the units that are no longer innovation units, over the network that stayed
          if (u.s === "c") base.push([m, { radius: r, fillColor: COLOR.P, fillOpacity: 0.35 }]);
          else if (u.a === 1 && u.t !== "P") top.push([m, { radius: r + 0.5, fillColor: COLOR[u.t], fillOpacity: 0.95 }]);
        } else if (idx < 6) {
          if (u.a !== 1) continue;
          if (u.t === "P") base.push([m, { radius: r, fillColor: COLOR.P, fillOpacity: 0.5 }]);
          else if (u.y && u.y <= 2020 + idx) top.push([m, { radius: r, fillColor: COLOR[u.t], fillOpacity: 0.9 }]);
          else if (idx === 5 && !u.y) top.push([m, { radius: r, fillColor: COLOR[u.t], fillOpacity: 0.9 }]);
        } else if (u.s === "c") {
          // the network of November 2026, in the same encoding as the years before
          if (u.t === "P") base.push([m, { radius: r, fillColor: COLOR.P, fillOpacity: 0.5 }]);
          else top.push([m, { radius: r, fillColor: COLOR[u.t], fillOpacity: 0.9 }]);
        }
      }
      for (const [m, s] of [...base, ...top]) { m.setStyle({ ...s, weight: 0 }); unitLayer.addLayer(m); }
      legendUnits();
    }
    yearInput.addEventListener("input", () => { drawUnits(); if (current === "network" || current === "exit") drawChart(current); });

    // Hexagons: resolution 6 (about 36 km2) when zoomed in, merged to resolution 5
    // (about 250 km2) for the national view, where a resolution-6 cell is about a pixel
    const cols = Object.fromEntries(hex.columns.map((c, i) => [c, i]));
    const MEANS = ["wealth_rank", "within2_before", "within2_peak", "within2_current", "walk_minutes", "availability"];
    const parents = new Map();
    for (const row of hex.rows) {
      const id = h3.cellToParent(row[cols.h3], 5);
      let a = parents.get(id);
      if (!a) { a = { people: 0, beyond2_current: 0, s: {}, w: {} }; parents.set(id, a); }
      const w = row[cols.people];
      a.people += w; a.beyond2_current += row[cols.beyond2_current];
      for (const k of MEANS) {
        const v = row[cols[k]];
        if (v !== null) { a.s[k] = (a.s[k] || 0) + v * w; a.w[k] = (a.w[k] || 0) + w; }
      }
    }
    const rows5 = [...parents].map(([id, a]) => {
      const r = new Array(hex.columns.length).fill(null);
      r[cols.h3] = id; r[cols.people] = a.people; r[cols.beyond2_current] = a.beyond2_current;
      for (const k of MEANS) if (a.w[k]) r[cols[k]] = a.s[k] / a.w[k];
      return r;
    });
    const makePolys = (rows, layer) => rows.map((row) => {
      const p = L.polygon(h3.cellToBoundary(row[cols.h3]), { renderer: canvas, weight: 0.6, fillOpacity: 1 });
      p.row = row;
      bindTip(p, () => hexTip(row));
      layer.addLayer(p);
      return p;
    });
    const hex5 = L.layerGroup(); const hex6 = L.layerGroup();
    const polys = [...makePolys(rows5, hex5), ...makePolys(hex.rows, hex6)];
    const hexLayer = L.layerGroup();
    const pickHexRes = () => { hexLayer.clearLayers(); hexLayer.addLayer(zoom() >= 7 ? hex6 : hex5); };
    const AREA = new Map([...rows5, ...hex.rows].map((r) => [r[cols.h3], h3.cellArea(r[cols.h3], "km2")]));
    pickHexRes();

    // One rule for every hexagon map: brighter and warmer means worse access
    const WORSE = ["#25324a", "#3b4c6b", "#5f6f8f", "#b06a4a", "#e8875a", "#ffc78f"];
    const NODATA = COLOR.before;  // drawn as an outline only
    const RAMP = {
      walk: { stops: [10, 20, 30, 45, 60], colors: WORSE, key: "walk_minutes",
        title: "Walking minutes to the nearest unit", labels: ["< 10", "10–20", "20–30", "30–45", "45–60", "60 +"] },
      avail: { stops: [0.01, 1, 2, 4, 8], colors: [...WORSE].reverse(), key: "availability",
        title: "Units per 10,000 people (2 km catchment)", labels: ["none within 2 km", "< 1", "1–2", "2–4", "4–8", "8 +"] },
      beyond: { stops: [1, 10, 50, 200, 1000], colors: WORSE, value: (row) => row[cols.beyond2_current] / AREA.get(row[cols.h3]),
        title: "People more than 2 km from a unit, per km²", labels: ["< 1", "1–10", "10–50", "50–200", "200–1,000", "1,000 +"] },
      // the poorest fifth brightest, so the eye goes where access is thinnest
      wealth: { stops: [0.2, 0.4, 0.6, 0.8], colors: ["#b4c3dc", "#7f8fb3", "#5f6491", "#4d4470", "#3b2f4f"], key: "wealth_rank",
        title: "Wealth of the area (Meta RWI), people in fifths", labels: ["Poorest fifth", "2nd", "3rd", "4th", "Richest fifth"] },
    };
    const colorFor = (ramp, v) => {
      if (v === null || v === undefined) return NODATA;
      let i = 0; while (i < ramp.stops.length && v >= ramp.stops[i]) i++;
      return ramp.colors[i];
    };
    function hexTip(row) {
      const v = (k) => row[cols[k]];
      const area = h3.getResolution(row[cols.h3]) === 5 ? "about 250 km²" : "about 36 km²";
      return `<div><strong>${fmt(v("people"), "int")}</strong> people in this hexagon (${area})</div>` +
        `<div class="t-sub">Within 2 km of a unit: ${fmt(v("within2_current") * 100, "pct0")} (${fmt(v("beyond2_current"), "int")} people farther)` +
        `<br>Walking to the nearest unit: ${v("walk_minutes") === null ? "–" : `${fmt(v("walk_minutes"), "dec1")} min`}` +
        `<br>Units per 10,000 people: ${fmt(v("availability"), "dec2")}` +
        `<br>Wealth rank: ${v("wealth_rank") === null ? "–" : `${Math.round(v("wealth_rank") * 100)} of 100`}</div>`;
    }
    function styleHex(metric) {
      const ramp = RAMP[metric];
      for (const p of polys) {
        const v = ramp.value ? ramp.value(p.row) : p.row[cols[ramp.key]];
        if (v === null || v === undefined) { p.setStyle({ fillOpacity: 0, color: NODATA, opacity: 0.8 }); continue; }
        const c = colorFor(ramp, v);
        p.setStyle({ fillColor: c, color: c, fillOpacity: 1, opacity: 1 });
      }
    }

    // Planner sites, drawn in the planner colour
    const siteLayer = L.layerGroup();
    let siteK = 100;
    const maxAdded = Math.max(...sites.map((s) => s.people_added));
    function drawSites() {
      siteLayer.clearLayers();
      const z = zoom();
      for (const s of sites) {
        if (s.order > siteK) continue;
        const m = L.circleMarker([s.lat, s.lon], {
          renderer: canvas, radius: (z <= 6 ? 2.5 : 4) + 6 * Math.sqrt(s.people_added / maxAdded),
          color: COLOR.bg, weight: 1.5, fillColor: COLOR.planner, fillOpacity: 0.95,
        });
        bindTip(m, () => `<div><strong>Site ${s.order}</strong> · ${s.ADM3_EN}, ${s.ADM2_EN}, ${s.ADM1_EN}</div>` +
          `<div class="t-name">${s.ADM3_TH} ${s.ADM2_TH} ${s.ADM1_TH}</div>` +
          `<div class="t-sub">Brings ${fmt(s.people_added, "int")} more people within 2 km</div>`);
        siteLayer.addLayer(m);
      }
    }
    document.querySelectorAll("#sites-control button").forEach((b) => b.addEventListener("click", () => {
      document.querySelectorAll("#sites-control button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      siteK = +b.dataset.k;
      drawSites();
      drawChart("next");
    }));
    map.on("zoomend", () => {
      pickHexRes();
      if (map.hasLayer(unitLayer)) drawUnits();
      if (map.hasLayer(siteLayer)) drawSites();
    });

    /* ---------- legend ---------- */
    const legend = document.getElementById("map-legend");
    const dot = (c) => `<span class="sw" style="background:${c}"></span>`;
    function legendUnits() {
      if (unitMode === "where") {
        legend.innerHTML = `<div class="ttl">Units that joined in fiscal 2024–25</div>` +
          ["nc", "ph", "dc"].map((t) => `<div class="row">${dot(COLOR[t])}${TYPE[t]}</div>`).join("") +
          `<div class="row">${dot(COLOR.P)}Class P unit (for reference)</div>`;
      } else if (unitMode === "left") {
        legend.innerHTML = `<div class="ttl">No longer innovation units by Nov 2026</div>` +
          ["nc", "ph", "dc"].map((t) => `<div class="row">${dot(COLOR[t])}${TYPE[t]}</div>`).join("") +
          `<div class="row">${dot(COLOR.P)}Still in the network</div>`;
      } else {
        legend.innerHTML = `<div class="ttl">${+yearInput.value === 6 ? "Network in Nov 2026" : "Primary-care units"}</div>` +
          `<div class="row">${dot(COLOR.P)}Class P unit</div>` +
          ["nc", "ph", "dc"].map((t) => `<div class="row">${dot(COLOR[t])}${TYPE[t]}</div>`).join("");
      }
    }
    function legendHex(metric, extra = "") {
      const r = RAMP[metric];
      legend.innerHTML = `<div class="ttl">${r.title}</div>` +
        r.colors.map((c, i) => `<div class="row"><span class="sw sq" style="background:${c}"></span>${r.labels[i]}</div>`).join("") +
        `<div class="row"><span class="sw sq none"></span>No data</div>` + extra;
    }

    /* ---------- chart ---------- */
    Chart.defaults.font.family = "DM Sans";
    Chart.defaults.color = COLOR.text2;
    Chart.defaults.borderColor = COLOR.hair;
    Chart.defaults.animation = false;
    if (NARROW) {
      Chart.defaults.font.size = 10;
      Chart.defaults.scale.ticks.maxRotation = 0;
      Chart.defaults.scale.ticks.autoSkipPadding = 6;
    }
    const ctx = document.getElementById("chart");
    const chartTitle = document.getElementById("chart-title");
    const chartNote = document.getElementById("chart-note");
    let chart = null;

    // Value labels on bars, and text labels at data points ("notes"), without a plugin library
    const labelPlugin = {
      id: "labels",
      // the options hold plain functions and arrays, not Chart.js scriptable or per-dataset options
      descriptors: { _scriptable: false, _indexable: false },
      afterDatasetsDraw(c, _a, opts) {
        const { ctx: g } = c;
        g.save();
        g.font = "500 12px DM Sans"; g.fillStyle = COLOR.text; g.textAlign = "center";
        if (opts.values) {
          c.data.datasets.forEach((ds, i) => c.getDatasetMeta(i).data.forEach((el, j) => {
            g.fillText(opts.values(ds.data[j]), el.x, el.y - 6);
          }));
        }
        for (const n of opts.notes || []) {
          const x = c.scales.x.getPixelForValue(n.x); const y = c.scales.y.getPixelForValue(n.y);
          g.textAlign = n.align || "left"; g.fillStyle = n.color || COLOR.text;
          g.fillText(n.text, x + (n.dx || 0), y + (n.dy || 0));
        }
        g.restore();
      },
    };
    Chart.register(labelPlugin);

    const pctAxis = (min = 0, max = 100) => ({ min, max, ticks: { callback: (v) => `${v}%` }, grid: { color: COLOR.hair } });
    const legendOpts = { position: "top", align: "start", labels: { boxWidth: NARROW ? 8 : 10, boxHeight: NARROW ? 8 : 10 } };
    function bars(labels, sets, yAxis, extra = {}) {
      return {
        type: "bar",
        data: { labels, datasets: sets.map(([label, data, color]) => ({ label, data, backgroundColor: color, borderRadius: 3, maxBarThickness: 36 })) },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: extra.noLegend ? { display: false } : legendOpts, labels: extra.labels || {},
            tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${extra.tipFmt ? extra.tipFmt(c.parsed.y) : c.parsed.y}` } } },
          scales: { x: { grid: { display: false } }, y: yAxis },
        },
      };
    }
    const lineOpts = (scales, extra = {}) => ({
      responsive: true, maintainAspectRatio: false, layout: { padding: { right: 16, top: 6 } },
      plugins: {
        legend: extra.noLegend ? { display: false }
          : { ...legendOpts, labels: { ...legendOpts.labels, ...(extra.lineLegend ? { usePointStyle: true, pointStyle: "line" } : {}),
            filter: (item) => !["Gap", "Crossover"].includes(item.text) } },
        labels: extra.labels || {}, tooltip: extra.tooltip || {},
      },
      scales,
    });

    const CHARTS = {
      network() {
        const y = charts.years;
        const idx = +yearInput.value;
        const last = (ctx2) => (ctx2.p1DataIndex === 6 ? [4, 4] : undefined);
        const pt = YEARS.map((_, i) => (i === idx ? 6 : 3));
        return {
          title: "People with a unit within 1 km and 2 km, as the network grew",
          note: "Class P units of November 2025 plus the innovation units that had joined by each fiscal year; FY 2025 is the whole network of November 2025. Units that left before then are not seen, so early years undercount. The dashed segment joins the peak to the network of November 2026.",
          cfg: {
            type: "line",
            data: {
              labels: YEAR_TICKS,
              datasets: [
                { label: "Within 1 km", data: [...y.within1, y.current_within1], borderColor: COLOR.peak, backgroundColor: COLOR.peak, tension: 0.15, pointRadius: pt, segment: { borderDash: last } },
                { label: "Within 2 km", data: [...y.within2, y.current_within2], borderColor: COLOR.before, backgroundColor: COLOR.before, borderDash: [6, 4], tension: 0.15, pointRadius: pt, segment: { borderDash: (c2) => (c2.p1DataIndex === 6 ? [2, 4] : [6, 4]) } },
              ],
            },
            options: lineOpts({ x: { grid: { display: false }, title: { display: true, text: "Fiscal year" }, ticks: { maxRotation: 0 } }, y: pctAxis(30, 90) },
              { lineLegend: true, tooltip: { callbacks: { title: (items) => YEARS[items[0].dataIndex], label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(1)}%` } } }),
          },
        };
      },
      where() {
        const u = charts.units_per_100k;
        return {
          title: "New units per 100,000 people, by the wealth of the area they opened in (fiscal 2024–25)",
          note: `Wealth fifths hold equal numbers of people, ranked by Meta's Relative Wealth Index. For scale, class P units run from ${fmt(u.class_P_2025[0], "dec1")} per 100,000 in the poorest fifth to ${fmt(u.class_P_2025[4], "dec1")} in the richest.`,
          cfg: bars(QUINTILES, [["Nurse clinics", u.entrants_nurse_clinic, COLOR.nc], ["Pharmacies", u.entrants_pharmacy, COLOR.ph],
            ["Medical clinics", u.entrants_doctor_clinic, COLOR.dc]], { grid: { color: COLOR.hair }, beginAtZero: true }, { tipFmt: (v) => v.toFixed(1) }),
        };
      },
      exit() {
        const y = charts.years;
        return {
          title: "People with a unit within 1 km",
          note: "Before: class P plus the innovation units that joined before FY2024. Peak: the network of November 2025. Now: November 2026, without class Z.",
          cfg: bars([NET.before, NET.peak, NET.now], [
            ["Within 1 km", [y.within1[3], y.within1[5], y.current_within1], [COLOR.before, COLOR.peak, COLOR.now]],
          ], pctAxis(0, 70), { noLegend: true, tipFmt: (v) => `${v.toFixed(1)}%`, labels: { values: (v) => `${v.toFixed(1)}%` } }),
        };
      },
      equity() {
        const q = charts.within2_by_quintile;
        return {
          title: "People with a unit within 2 km, by wealth fifth",
          note: "Straight-line distance to the nearest class P or innovation unit. The map shows the wealth fifths.",
          cfg: bars(QUINTILES, [[NET.before, q.before, COLOR.before], [NET.peak, q.peak, COLOR.peak], [NET.now, q.current, COLOR.now]],
            pctAxis(0, 100), { tipFmt: (v) => `${v.toFixed(1)}%` }),
        };
      },
      far() {
        const q = charts.walk30_by_quintile;
        return {
          title: "People who can walk to a unit within 30 minutes, by wealth fifth",
          note: "Walking at 5 km/h along the OpenStreetMap road network. By motor vehicle almost everyone is within 15 minutes.",
          cfg: bars(QUINTILES, [[NET.before, q.before, COLOR.before], [NET.peak, q.peak, COLOR.peak], [NET.now, q.current, COLOR.now]],
            pctAxis(0, 100), { tipFmt: (v) => `${v.toFixed(1)}%` }),
        };
      },
      avail() {
        const a = charts.availability_by_quintile;
        return {
          title: "Units per 10,000 people within reach (2 km), by wealth fifth",
          note: "Gaussian two-step floating catchment; every unit has the same capacity.",
          cfg: bars(QUINTILES, [[NET.before, a.before, COLOR.before], [NET.peak, a.peak, COLOR.peak], [NET.now, a.current, COLOR.now]],
            { grid: { color: COLOR.hair }, beginAtZero: true }, { tipFmt: (v) => v.toFixed(2) }),
        };
      },
      planner() {
        const p = charts.planner;
        const market = facts.market_newly_within2_m;
        const planner = facts.planner_newly_within2_m;
        const k = facts.entrants;
        const match = facts.planner_sites_to_match;
        return {
          title: "People newly within 2 km, by number of new units: planner and market",
          note: "Planner: greedy maximal covering from the network before FY2024. Market: the units that actually joined in fiscal 2024–25.",
          cfg: {
            type: "line",
            data: {
              datasets: [
                { label: "Planner", data: p.k.map((kk, i) => ({ x: kk, y: p.people[i] })), borderColor: COLOR.planner, backgroundColor: COLOR.planner, pointRadius: 0, tension: 0.1 },
                { label: "Market (the units that joined)", data: [{ x: k, y: market }], borderColor: COLOR.now, backgroundColor: COLOR.now, pointRadius: 8, showLine: false, clip: false },
                { label: "Same gain as the market", data: [{ x: 0, y: market }, { x: k, y: market }], borderColor: COLOR.muted, borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
                { label: "Gap", data: [{ x: k, y: market }, { x: k, y: planner }], borderColor: COLOR.now, borderDash: [3, 3], pointRadius: 0, borderWidth: 1.5 },
                { label: "Crossover", data: [{ x: match, y: market }], borderColor: COLOR.planner, backgroundColor: COLOR.bg, pointRadius: 5, pointBorderWidth: 2, showLine: false },
              ],
            },
            options: lineOpts({
              x: { type: "linear", min: 0, max: k, title: { display: true, text: "New units" }, grid: { display: false }, ticks: { callback: (v) => v.toLocaleString("en-US") } },
              y: { min: 0, title: { display: true, text: "Million people" }, grid: { color: COLOR.hair } },
            }, {
              tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.parsed.x, "int")} units, ${c.parsed.y.toFixed(2)} million people` } },
              labels: { notes: [
                { x: match, y: market, text: `${fmt(match, "int")} sites`, dx: 10, dy: 18 },
                { x: k, y: (market + planner) / 2, text: `${fmt(facts.planner_over_market, "dec1")}×`, dx: -10, align: "right", color: COLOR.now },
              ] },
            }),
          },
        };
      },
      next() {
        const p = charts.policy;
        const sel = p.k.indexOf(siteK);
        return {
          title: "People brought within 2 km by new public units placed by the planner",
          note: `Starting from the network of November 2026. ${fmt(facts.beyond2_current_m, "m")} people are now beyond 2 km.`,
          cfg: {
            type: "line",
            data: {
              datasets: [
                { label: "New public units", data: p.k.map((kk, i) => ({ x: kk, y: p.people[i] })), borderColor: COLOR.planner, backgroundColor: COLOR.planner,
                  pointRadius: p.k.map((_, i) => (i === sel ? 6 : 0)), tension: 0.1 },
              ],
            },
            options: lineOpts({
              x: { type: "linear", min: 0, max: 500, title: { display: true, text: "New public units" }, grid: { display: false } },
              y: { min: 0, title: { display: true, text: "Million people" }, grid: { color: COLOR.hair } },
            }, {
              noLegend: true,
              tooltip: { callbacks: { label: (c) => `${fmt(c.parsed.x, "int")} units: ${c.parsed.y.toFixed(2)} million people` } },
              labels: { notes: sel >= 0 ? [{ x: siteK, y: p.people[sel], text: `${siteK} units: ${p.people[sel].toFixed(2)}${NARROW ? " M" : " million"}`, dx: siteK > 300 ? -10 : 10, dy: 18, align: siteK > 300 ? "right" : "left" }] : [] },
            }),
          },
        };
      },
    };
    function drawChart(view) {
      const spec = CHARTS[view]();
      chartTitle.textContent = spec.title;
      chartNote.textContent = spec.note;
      ctx.setAttribute("aria-label", spec.title);
      if (chart) chart.destroy();
      chart = new Chart(ctx, spec.cfg);
    }

    /* ---------- views ---------- */
    const stage = document.querySelector(".stage");
    const yearCtl = document.getElementById("year-control");
    const sitesCtl = document.getElementById("sites-control");
    const exitCtl = document.getElementById("exit-control");
    const exitBtns = [...exitCtl.querySelectorAll("button")];
    const setExitShow = (show) => {
      exitBtns.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.show === show)));
      unitMode = show === "left" ? "left" : "network";
      drawUnits();
    };
    exitBtns.forEach((b) => b.addEventListener("click", () => setExitShow(b.dataset.show)));
    const stepCount = document.getElementById("step-count");
    const VIEWS = {
      network: { map: "units", mode: "network", year: 5 },
      where: { map: "units", mode: "where" },
      exit: { map: "units", mode: "network", year: 6, exit: true },
      equity: { map: "wealth" },
      far: { map: "walk" },
      avail: { map: "avail" },
      planner: { map: null },
      next: { map: "sites" },
    };
    const ORDER = Object.keys(VIEWS);
    let current = null;
    function activate(view) {
      if (view === current) return;
      current = view;
      const v = VIEWS[view];
      stage.classList.toggle("chart-only", !v.map);
      [unitLayer, hexLayer, siteLayer].forEach((l) => map.removeLayer(l));
      yearCtl.hidden = !(v.map === "units" && v.mode === "network" && !v.exit);
      exitCtl.hidden = !v.exit;
      sitesCtl.hidden = v.map !== "sites";
      if (v.map === "units") {
        unitMode = v.mode;
        if (v.year !== undefined) yearInput.value = v.year;
        if (v.exit) exitBtns.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.show === "now")));
        drawUnits(); unitLayer.addTo(map);
      } else if (v.map === "sites") {
        styleHex("beyond"); hexLayer.addTo(map); drawSites(); siteLayer.addTo(map);
        legendHex("beyond", `<div class="row"><span class="sw" style="background:${COLOR.planner};border:1px solid ${COLOR.bg}"></span>Planner site (size = people added)</div>`);
      } else if (v.map) {
        styleHex(v.map); hexLayer.addTo(map); legendHex(v.map);
      }
      drawChart(view);
      stepCount.textContent = `${ORDER.indexOf(view) + 1} / ${ORDER.length}`;
    }

    const steps = [...document.querySelectorAll(".step")];
    // The active step is the one under the reading line. On phones the stage covers the
    // top of the screen, so the line sits lower. A scroll listener rather than an
    // IntersectionObserver: it also catches fast jumps (menu links, keyboard).
    const tocLinks = [...document.querySelectorAll(".toc a")];
    let ticking = false;
    function pickStep() {
      ticking = false;
      const line = window.innerHeight * (NARROW ? 0.66 : 0.5);
      if (steps[0].getBoundingClientRect().top > line) {
        steps.forEach((s2) => s2.classList.remove("on"));
        tocLinks.forEach((a) => a.classList.remove("on"));
        return;
      }
      let on = steps[0];
      for (const st of steps) if (st.getBoundingClientRect().top <= line) on = st;
      steps.forEach((s2) => s2.classList.toggle("on", s2 === on));
      activate(on.dataset.view);
      tocLinks.forEach((a) => a.classList.toggle("on", a.dataset.steps.split(" ").includes(current)));
    }
    window.addEventListener("scroll", () => {
      if (!ticking) { ticking = true; setTimeout(pickStep, 60); }
    }, { passive: true });
    activate("network");
    pickStep();
  }
})();
