# 06 — Kaggle Datasets: Research and Integration Plan

| Field | Value |
|---|---|
| Document role | **Source of truth for external data**: which Kaggle datasets are used, why, under which licence, and exactly how they enter scenarios and experiments. |
| Status | **Provisional selection** (decision 7, revision 2). Research done during design (Issue #1, 2026-11-17). Integration and final verification in **Issue #7**. |
| Verification status | **What was checked:** on 2026-11-17, an automated fetch read each dataset's *public metadata record* from Kaggle's API (title, licence label, total size, description text). The HTML pages did not render without JavaScript. **What was not checked:** no dataset files, data cards, column dictionaries or full terms of use have been downloaded or read. None are in the project files. File lists, exact schemas, row counts, column semantics (e.g. whether Olist's estimated delivery date is a usable urgency signal) and licence terms beyond the label are **unverified**. The selection below is therefore provisional. It becomes final only after the verification gate in §5.1 (first task of Issue #7). |

---

## 1. Role of data in this project

No Kaggle dataset contains "multi-robot coordination" data with our grid semantics. The project's own surveys found no Kaggle-hosted multi-agent path-finding or warehouse-robot benchmark. The value of Kaggle data is therefore to make **scenarios realistic and diverse** in the dimensions the specification names:

| Scenario dimension | Synthetic default | Data-driven replacement |
|---|---|---|
| *Where* tasks appear (pickup/drop-off spatial distribution) | Uniform over free cells | Empirical origin–destination (OD) samples from real delivery and trip data |
| *When* tasks appear (arrival process) | Bernoulli/Poisson with constant rate | Non-homogeneous Poisson process with an empirical time-of-day intensity profile |
| *How urgent* tasks are (priority) | Fixed categorical distribution | Priority derived from a real urgency proxy (promised delivery slack) |
| *What maps look like* | Random density, rooms, aisles | Externally generated maze maps (perfect and imperfect), sizes up to 150×150 cells |
| *How dynamic obstacles move* | Random walk, patrol, bounce | (Conditional) correlated random walk calibrated on real pedestrian trajectories |

Data enters through **distributions and maps**, not as training labels. This makes the integration genuine (real structure changes the difficulty and character of scenarios) without pretending the data is something it is not (assumption A-9 in `01-requirements.md`).

---

## 2. Evaluation criteria

Each candidate was scored on:

1. **Relevance**: does it inform a specific scenario dimension above?
2. **Structure and quality**: needed fields present (coordinates, timestamps), size and cleanliness.
3. **Licence**: academic, non-commercial use permitted; rules for derived data.
4. **Integration cost**: download size, preprocessing effort.
5. **Genuine contribution**: does it change experimental conclusions or just decorate?

---

## 3. Candidates and decisions

| # | Dataset (Kaggle) | Metadata facts (2026-11-17; labels as stated, not independently verified) | Decision |
|---|---|---|---|
| K1 | **Brazilian E-Commerce Public Dataset by Olist** — `kaggle.com/datasets/olistbr/brazilian-ecommerce` | Licence label **CC BY-NC-SA 4.0**. ~126 MB. Description: ~100 k real, anonymised orders 2016–2018, provided by Olist; multiple linked tables incl. orders with timestamps and a zip-code geolocation table. *Column names unverified.* | **Provisionally selected** |
| K2 | **NYC Taxi Trip Duration** — `kaggle.com/datasets/yasserh/nyc-taxi-trip-duration` (mirror of the Kaggle competition `nyc-taxi-trip-duration` data) | Licence label **CC0: Public Domain** (uploader's label; the competition's own data rules are *not verified*). ~200 MB. Columns *as stated in the description*: `pickup_datetime`, `dropoff_datetime`, `pickup_/dropoff_longitude/latitude`, `trip_duration`, … Stated source: 2016 NYC Yellow Cab records (NYC TLC), sampled and cleaned for the competition. | **Provisionally selected** |
| K3 | **🧩 Maze Dataset** — `kaggle.com/datasets/mexwell/maze-dataset` | Licence label **CC0: Public Domain**. ~197 MB. Description: 3 000 mazes (≈1 500 perfect, ≈1 500 imperfect), 10×10 to 150×150. *File format unverified.* | **Provisionally selected** |
| K4 | **Stanford Drone Dataset** — `kaggle.com/datasets/brendanalvey/stanford-drone-dataset` | ~72 GB total (videos + annotations). Kaggle licence label CC BY-SA 3.0, **but** the description states the original **CC BY-NC-SA 3.0**. Re-upload of Robicquet et al., ECCV 2016. | **Conditional (optional)** |
| K5 | Swiggy-Warehouse-Level-Picklist-Optimization — `kaggle.com/datasets/abhaygupta619/swiggy-warehouse-level-picklist-optimization` | Uploader's licence label Apache 2.0. Anonymised hackathon data; columns include zone, bin rank, priority. | **Rejected** |
| K6 | Logistics Warehouse Dataset — `kaggle.com/datasets/ziya07/logistics-warehouse-dataset` | Synthetic, ~3 200 records, no timestamps. | **Rejected** |
| — | MovingAI grid benchmarks (Sturtevant, 2012) | Not found on Kaggle. Hosted by the Moving AI Lab. | **Not a Kaggle source**: optional external cross-check only |

### 3.1 Justification of decisions

The justifications below rest on the dataset descriptions. Each is conditional on the verification gate (§5.1).

- **K1 Olist (provisionally selected).** If the described fields exist as expected, it is the only candidate with a *real urgency signal*. Each order would have a purchase timestamp and a promised (estimated) delivery date, so *slack* = promised − purchase gives a priority proxy. It would also provide seller → customer OD pairs, mapping to pickup → drop-off, and purchase-time patterns for arrivals. The licence label (CC BY-NC-SA 4.0) indicates academic non-commercial use is permitted; full terms are reviewed at the gate. We do not redistribute derived data (§4.1).
- **K2 NYC Taxi (provisionally selected).** As described, it has dense, timestamped pickup and drop-off events with coordinates within one city. That suits estimating a time-varying arrival intensity $\lambda(t)$ and an intra-city OD distribution with spatial hotspots. Hotspots create *contention*, where coordination and priority matter. The CC0 label is the mirror uploader's statement. The original competition's data rules and the TLC source terms are checked at the gate. If the mirror's status is doubtful, the competition data itself is used under its rules.
- **K3 Maze Dataset (provisionally selected).** Independently generated maps with very different topology from our generators: long corridors, dead ends, and loops in the imperfect mazes. These are the conditions where A\* and RRT\* are expected to differ most (hypothesis H1). External maps reduce the risk that our own generator favours one planner.
- **K4 Stanford Drone (conditional).** Real pedestrian motion would ground the dynamic-obstacle model. However, the dataset is 72 GB and only the small annotation files are needed. Its licence labelling is inconsistent, so we must assume the stricter NC-SA terms. It will be used **only if** annotation files can be downloaded individually (`kaggle datasets download -f <file>`), and only as aggregate statistics with attribution. Otherwise the synthetic motion models remain, and this fallback is documented.
- **K5 Swiggy (rejected).** No geometry (bins are ranks within zones, not coordinates), so it cannot inform spatial distributions. Redistribution rights of the original hackathon data are unclear.
- **K6 (rejected).** Synthetic data adds no realism over our own generators.

---

## 4. Integration design

### 4.1 Pipeline

```text
data/raw/<dataset>/...          (downloaded by the student; git-ignored)
        │  python -m robocoord data prepare <dataset>      (deterministic, seeded, logged)
        ▼
data/processed/<dataset>/       derived files: histograms, OD samples, .map files (git-ignored)
data/manifests/<dataset>.json   provenance manifest (committed): source URL, licence label and
                                terms reviewed, download date, raw-file SHA-256 hashes, row counts,
                                filters, parameters, processed-file hashes
        ▼
scenario samplers (src/robocoord/scenario + data/)  →  ScenarioSpec  →  engine / training / experiments
```

**Repository policy (decision 8, ADR-9).**
- **Raw and derived datasets are not committed by default**: `data/raw/` and `data/processed/` are git-ignored.
- Committed: the acquisition instructions (§5), the preprocessing code, the manifests, attribution (§6), and small **synthetic** test fixtures (`tests/fixtures/data/`). Fixtures contain invented rows in each dataset's schema, not real records.
- Anyone with the raw downloads can regenerate the processed files deterministically, and the manifests' hashes confirm identical results.
- Redistribution of any derived data is considered only after the dataset's licence and terms are verified, and only with the student's explicit approval. CC BY-NC-SA sources (K1, K4) would require share-alike licensing and attribution.
- Consequence: data-driven experiments (E1 maze maps, E7) run locally, not in CI. CI uses only synthetic fixtures.

### 4.2 Spatial mapping (K1, K2): coordinates → grid cells

1. Restrict to a coherent region: São Paulo municipality for Olist; Manhattan bounding box for NYC. Drop rows with missing or out-of-range coordinates.
2. Project with a local equirectangular projection: $X = R\,\Delta\lambda \cos\varphi_0$, $Y = R\,\Delta\varphi$. Over a city-sized region the error is negligible for our purpose.
3. Clip to the 1st–99th percentile box to remove outliers, then apply the affine map onto $[0, W) \times [0, H)$ and floor to cells.
4. If the cell is an obstacle, snap to the nearest free cell by BFS. The snapping distance is recorded as a quality statistic.
5. Pickup and drop-off may not coincide after discretisation; such samples are re-drawn.

The map itself comes from the selected map family (open city-block layout, warehouse, or maze). Data provides *where* on that map demand concentrates.

### 4.3 Temporal mapping: arrivals

- Estimate an hour-of-week intensity profile $\hat\lambda(h)$ from timestamps (purchase time for K1, pickup time for K2), smoothed and normalised to mean 1.
- Simulation intensity: $\lambda(t) = \lambda_0 \cdot \hat\lambda(h(t))$, where $h(t)$ maps steps to hours through a configurable time compression (e.g. 1 simulated hour = 50 steps).
- **Load calibration.** Real arrival magnitudes are unrelated to our fleet size, so $\lambda_0$ is set from a target utilisation $u$: $\lambda_0 = u \cdot N / \bar s$, where $\bar s$ is the mean task service time in steps, estimated from the OD distances. The *shape* comes from the data and the *magnitude* is an experimental factor. This is stated explicitly in all results.

### 4.4 Priority mapping (K1)

- Slack $\sigma$ = `order_estimated_delivery_date − order_purchase_timestamp` (days). The exact column names are verified at download.
- Priority levels by slack terciles: shortest third → $\pi = 3$, middle → 2, longest → 1. Tercile boundaries are stored in the manifest.
- For K2 there is no urgency signal. Priority uses the configured synthetic distribution, and this is stated in K2 scenario descriptions rather than invented from unrelated fields.

### 4.5 Maps (K3)

- Convert each maze to MovingAI `.map` format. Walls → `@`, open → `.`. The exact source encoding is verified at download; the generator described on the dataset page uses $(2d+1) \times (2d+1)$ arrays with 1 = wall.
- Select stratified subsets by size and type (perfect/imperfect) for E1 and E7. The selection is defined by a seeded rule recorded in the manifest. The converted maps stay in `data/processed/` (git-ignored).
- Validation: the free space is connected (perfect mazes) or the largest component is used; the converted map round-trips through our loader.

### 4.6 Dynamic obstacle calibration (K4, conditional)

- From annotation tracks, compute per-agent speed and heading-change distributions, converted to grid units at the scenario's cell scale.
- Fit a *correlated random walk*: per-step move probability $p_{\text{move}}$ and probability $p_{\text{turn}}$ of changing direction, by maximum likelihood on discretised tracks.
- Use as a fourth obstacle motion model ("data-driven"), alongside random walk, patrol and bounce.

### 4.7 Fidelity validation (X-8 tests and E7)

A data-driven generator is accepted only if generated scenarios reproduce the source distributions within tolerance:
- arrivals: two-sample Kolmogorov–Smirnov test of hour-of-day of simulated vs source events;
- spatial: correlation of 2-D histograms of pickup cells vs projected source points;
- priority: level frequencies vs terciles.

The results of these checks are recorded in the verification log.

---

## 5. Acquisition procedure (performed by the student)

Kaggle downloads require the student's own Kaggle account. Claude never handles Kaggle credentials.

**Option A — Browser.** Open each dataset page, accept any terms, click **Download**, and extract the archive into `data/raw/<dataset-slug>/`.

**Option B — Kaggle CLI.** Create an API token on kaggle.com → *Settings* → *API*, and store it as described in Kaggle's documentation (never inside the repository). Then run, for example:

```bash
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw/olist --unzip
```

```bash
kaggle datasets download -d yasserh/nyc-taxi-trip-duration -p data/raw/nyc-taxi --unzip
```

```bash
kaggle datasets download -d mexwell/maze-dataset -p data/raw/mazes --unzip
```

The exact commands, expected file lists and checksums are finalised in Issue #7 after the first download.

### 5.1 Verification gate (first task of Issue #7)

For each provisionally selected dataset, before any integration code depends on it:

1. **Licence and terms:** read the dataset page's licence section and any linked terms (and, for K2, the original competition's data rules). Record the exact text or link in the manifest. If use for an academic project is not clearly permitted, drop the dataset.
2. **Files and schema:** list files, sizes, columns and dtypes. Confirm that the fields required by §4 exist:
   - K1: purchase timestamp, estimated delivery date, seller and customer zip prefixes, geolocation table;
   - K2: pickup/drop-off timestamps and coordinates;
   - K3: an encoding convertible to occupancy grids.
3. **Suitability:** check value ranges, missing-value rates, and whether the proposed proxies are meaningful. For K1: is the slack distribution non-degenerate? For K2: are coordinates dense enough inside the chosen bounding box?
4. **Record** the findings in this document, replacing "provisional" with "verified" or "dropped", and in the verification log.
5. If fewer than two datasets pass, select replacements from the §3 candidates or new research, with the student's approval, before continuing (acceptance criterion AC-9).

---

## 6. Attribution and licences (to be included in the final README)

| Source | Licence (as labelled) | Attribution |
|---|---|---|
| Olist, *Brazilian E-Commerce Public Dataset by Olist*, Kaggle | CC BY-NC-SA 4.0 | Olist; Kaggle dataset `olistbr/brazilian-ecommerce` |
| NYC Taxi Trip Duration (Kaggle mirror) | CC0 (mirror); original data NYC TLC 2016 via Kaggle competition *New York City Taxi Trip Duration* | NYC Taxi & Limousine Commission; Kaggle |
| Maze Dataset | CC0 | Kaggle dataset `mexwell/maze-dataset` (perfect-maze generator credited by the uploader to M. Gold) |
| Stanford Drone Dataset (if used) | Treated as CC BY-NC-SA 3.0 | A. Robicquet, A. Sadeghian, A. Alahi, S. Savarese, *Learning Social Etiquette*, ECCV 2016 |

---

## 7. Sources consulted during research

- Kaggle dataset pages and metadata for K1–K6 (URLs in §3), accessed 2026-11-17.
- R. Stern et al., *Multi-Agent Pathfinding: Definitions, Variants, and Benchmarks*, SoCS 2019 (arXiv:1906.08291), for the MovingAI MAPF benchmark context.
- N. Sturtevant, *Benchmarks for Grid-Based Pathfinding*, IEEE TCIAIG 2012.
- A. Robicquet et al., *Learning Social Etiquette: Human Trajectory Prediction in Crowded Scenes*, ECCV 2016.
