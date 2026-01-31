# Exposure-Adjusted Bicycle Crash Risk Estimation and Safer Routing in Berlin
This is the repository for the 2025 Data Literacy course project of
[Eric Berger](), [Edward Eichhorn](), [Liaisan Faidrakhmanova](), [Luise Grasl](), [Tobias Schnarr]().

## Setup
1) Clone the repository.
2) Download the input datasets and store them in `data/input/counting_stations`:
    * from the [German Accident Atlas](https://unfallatlas.statistikportal.de/?BL=BB%20=%20Brandenburg) select _Accident Atlas and Open Data_, then download the csv-files for years 2019-2023 and unzip;
    * download [Strava data for Berlin](https://zenodo.org/records/15332147/files/berlin_data.parquet?download=1) and [graph geometry data for Berlin](https://zenodo.org/records/15332147/files/berlin_graph_geometry.parquet?download=1);
    * download the [Berlin bike counting stations data](https://www.berlin.de/sen/uvk/_assets/verkehr/verkehrsplanung/radverkehr/weitere-radinfrastruktur/zaehlstellen-und-fahrradbarometer/gesamtdatei-stundenwerte.xlsx?ts=1752674590).
4) You can set up a virtual environment and install the requirements as follows:
   ```bash
   pip install uv
   uv sync
   ```
5) To reproduce all results, run notebooks `01` through `04` in the `notebooks/` directory sequentially.

## Project Structure
```
project_root_dir/                                   <--- root directory of the project
├── pyproject.toml                                  <--- Python project + dependencies
├── README.md                                       <--- project overview and quickstart
├── src/                                            <--- all Python source code (importable package)
│   ├── __init__.py                                 <--- package marker
│   ├── accidents.py                                <--- crash data loading + matching to segments
│   ├── nodes.py                                    <--- junction/node utilities
│   ├── panels.py                                   <--- panel data helpers
│   ├── risk_estimates.py                           <--- Empirical Bayes risk estimation
│   ├── routing_graph.py                            <--- build risk-annotated graph from segments
│   ├── routing_algorithm.py                        <--- safety-aware routing under detour constraints
│   ├── segments.py                                 <--- segment utilities and transformations
│   ├── strava_exposure.py                          <--- Strava-derived exposure ingestion + checks
│   └── utils.py                                    <--- path helpers and project root
│
├── notebooks/                                      <--- exploratory work and analysis notebooks
│   ├── 01_data_preparation.ipynb                   <--- data loading, inspection, aggregation, and merging
│   ├── 02_risk_estimation.ipynb                    <--- segment/junction risk estimation
│   ├── 03_graph_and_routing.ipynb                  <--- graph construction + routing
│   ├── 04_paper.ipynb                              <--- figures for the report
│   └── ...                                     
│
├── data/                                           <--- input datasets
│   ├── accidents/                                  <--- Unfallatlas crash CSVs
│   ├── counting_stations/                          <--- official counters
│   ├── csv/                                        <--- auxiliary CSVs
│   ├── panel/                                      <--- panel-format exports
│   └── strava/                                     <--- Strava features + metadata
│
├── report/                                         <--- report and related files
│   ├── report_template.tex                         <--- main manuscript
│   ├── bibliography.bib                            <--- references
│   ├── figs/                                       <--- figures used in the report
│   └── ...                                     
│
└── .gitignore                                      <--- git ignore rules
```

