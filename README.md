# Who the market reached: primary-care access in Thailand, 2024–2026

In fiscal years 2024–25 Thailand's Universal Coverage Scheme grew its primary-care
network by contracting private nurse clinics, pharmacies and medical clinics
("innovation units") instead of building public units. This repository holds the
pipeline and the interactive dashboard that test where those units went, who they
reached, and how a coverage-maximising planner would have done with the same
number of sites.

**Dashboard:** https://ramathibodi-hpsr.github.io/thailand-primary-care/
(published from [`dashboard/`](dashboard/) by GitHub Actions on every push to `main`).

Peerasit Sitthirat et al., Health Policy and Systems Research, Ramathibodi.
All numbers are preliminary; a working paper is in preparation.

## What is measured

| Step | Question | Method |
|---|---|---|
| F | Are the published numbers right and reproducible? | Every coordinate checked against the unit's registered tambon; nearest unit for every populated 100 m cell (KD-tree on the unit sphere) in place of the earlier ArcGIS buffer run |
| A1 | Where did the units that joined in FY2024–25 go, and which ones left by November 2026? | Networks before FY2024, at the November 2025 peak and in November 2026; distance to class P units; population-density fifths |
| A2 | Did access and the new units follow area wealth? | Meta Relative Wealth Index fifths; concentration index |
| A3 | How long does it take to reach a unit? | Shortest paths on the OpenStreetMap road graph, by motor vehicle and on foot |
| A4 | Are there enough units for the people around them? | Gaussian two-step floating catchment, equal capacity |
| A5 | How many more people would a planner have reached? Where should new public units go? | Greedy maximal covering location model |

Primary-care units are NHSO class P plus the contracted pharmacies, nurse clinics
and medical clinics (codes R0211/R0311, R0213/R0313, R0207/R0307). Prison health
units and class Z are not counted.

## Repository layout

```
data/raw/            NHSO registry snapshots (29 Nov 2025, 20 Nov 2026) and the earlier
                     ArcGIS outputs, used only to check the new pipeline (Git LFS)
scripts/v2/          the pipeline, steps 10–20 (config.py holds paths and rules)
dashboard/           the static site: index.html, style.css, app.js and data/
data/interim/, data/processed/   pipeline outputs, not in git
```

## Inputs not in this repository

Put these in a folder `PC_GIS/` next to this repository, or point `PC_GIS_DIR` at it
(paths in `scripts/v2/config.py`):

| File | Source | Licence |
|---|---|---|
| `Dashboard/Data/tha_ppp_2020_UNadj_constrained.tif` | WorldPop 2020, 100 m, UN-adjusted, constrained | CC BY 4.0 |
| `data/raw/tha_adm/tha_admbnda_adm1_rtsd_20220121.shp`, `tha_admbnda_adm3_rtsd_20250628.shp` | OCHA/RTSD administrative boundaries via HDX | see HDX |
| `data/raw/osm/thailand-latest.osm.pbf` | OpenStreetMap via Geofabrik | ODbL 1.0 |
| `data/raw/rwi/tha_relative_wealth_index.csv` | Meta Relative Wealth Index via HDX (Chi et al. 2022, PNAS 119(3): e2113658119) | CC BY-NC 4.0 |

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/v2/build_v2.py
```

About 12 minutes on a laptop, most of it the road graph (step 16) and the
travel-time runs (step 17). Step 20 writes the dashboard bundle to
`dashboard/data/`; every number quoted in the page text comes from
`dashboard/data/facts.json`.

To preview the dashboard:

```bash
cd dashboard && python -m http.server
```

## Earlier version

The first map (a radius slider over ArcGIS buffer results) and its pipeline are in
the git history at commit `c50356c`. The new pipeline reproduces its national
totals to within 1.7% (the difference comes from the population base).

## Cite

Sitthirat P, et al. Who the market reached: contracted private primary care and
spatial access in Thailand, 2024–2026. Working paper in preparation; 2026.

## License

Code: MIT, see [LICENSE](LICENSE). Data keep their own licences (above).
