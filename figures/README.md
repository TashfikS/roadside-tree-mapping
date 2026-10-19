# Figures

| Script | Paper figure | Needs |
|---|---|---|
| `make_fig_overview.py` | Data, method and references overview (`fig_overview`) and `graphical_abstract.png` | `results/`, `data/`, plus the Bangladesh images (`BD_DATASET_DIR`), Pasadena panoramas (`PASADENA_DIR`) and highway photos (`HIGHWAY_PHOTO_DIR`) |
| `make_fig_maps.py` | Study areas (`fig_study_area`), Pasadena street excerpt (`fig_pasadena_map`), highway maps (`fig_highway_map`) | `results/`, `data/` (basemap, field sites, highway photos) and the Pasadena benchmark (`PASADENA_DIR`) |
| `make_fig_examples.py` | Detection examples (`fig_examples`) and trunk-base accuracy (`fig_trunk`) | `results/`, `data/`, Bangladesh images (`BD_DATASET_DIR`) and highway photos (`HIGHWAY_PHOTO_DIR`) |
| `make_fig_validation.py` | Validation against the references (`fig_validation`) | `results/` only |
| `make_fig_highway.py` | Highway density, temperature and canopy (`fig_highway`) | `results/` only |

Each script writes its PNG (and PDF where used in the paper) next to itself, e.g.
`PASADENA_DIR=/path/to/PasadenaUrbanTrees_Detection python figures/make_fig_maps.py`.
Shared style settings are in `figstyle.py`. Boundaries, rivers and city points come from Natural Earth (public domain),
subset in `data/basemap_bangladesh_naturalearth.geojson`.
