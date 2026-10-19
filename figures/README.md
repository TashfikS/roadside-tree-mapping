# Figures

| Script | Figure | Needs |
|---|---|---|
| `make_fig_overview.py` | Fig. 1 (data, method and answer keys) and `graphical_abstract.png` | `results/`, `data/`, plus the Bangladesh images (`BD_DATASET_DIR`), Pasadena panoramas (`PASADENA_DIR`) and highway photos (`HIGHWAY_PHOTO_DIR`) |
| `make_fig_validation.py` | Fig. 2 (validation against the answer keys) | `results/` only |
| `make_fig_highway.py` | Fig. 3 (Dhaka–Chattogram highway) | `results/` only |

`python figures/make_fig_validation.py` writes the PNG next to the script. Shared style settings are in `figstyle.py`.
