# End-to-End Solution Approach: Walmart M5 Demand Forecasting

**Internal Hackathon | Cognizant Evaluation Criteria**

Our approach is built on a simple philosophy: **every other team will build a model. We are building a product.** We selected the top 6 unique insights from our data analysis and wove them into a coherent, end-to-end solution that speaks to every single criterion the Cognizant judges are evaluating.

---

## 🎯 The 6 Best Ideas We Are Keeping (And Why)

After analysis, we shortlisted these 6 because they are **data-proven, technically credible, and visually demonstrable** in the UI:

| # | Idea | Why It Wins |
|---|---|---|
| 1 | **Ghost Stockout Detection** | Directly addresses the core problem: "balancing stockouts vs. overstocking" |
| 2 | **Leading Zero Trap Cleanup** | Proves dataset mastery; most teams will fall for this trap |
| 3 | **Phantom Promotion Engineering** | Turns a dataset weakness (no promo data) into a feature strength |
| 4 | **SNAP + Weekend Shock Features** | Hard numbers (10.2% SNAP, 31% Weekend) make the pitch credible |
| 5 | **Christmas Closure Domain Override** | Shows that the team applied domain logic, not just brute-force ML |
| 6 | **Agentic AI Supply Chain Copilot** | The "wow" factor; GenAI + ML on the UI will electrify judges |

---

## 🗺️ End-to-End Approach

### PHASE 0 — Data Audit (Before Writing Any Model Code)
**Goal:** Understand what you are actually working with. This is where most teams skip and pay the price.

**Step 1 — Verify the "Leading Zero" Trap**
Run a cross-check between `sell_prices.csv` and `sales_train_evaluation.csv`. For every item, find the first week a price appears. Mark every sales day before that week as `pre_launch = True`. These are **not real zeros** — the product didn't exist on the shelf yet. Remove them from the training set entirely.
> 🔎 **Proof:** `FOODS_3_595` in `CA_1` had 1,841 days of zero sales before any price appeared. Training on those zeros poisons the model.

**Step 2 — Tag "Ghost Stockouts"**
For every item in the dataset, compute a 28-day rolling average of sales. If the rolling average is >3 units/day AND the current day has `0` sales AND the previous 3 days are all `0`, flag it as a likely stockout (`is_ghost_stockout = True`). This row should be excluded from training (it is not zero demand — it is a data quality issue).

**Step 3 — Apply the Christmas Override**
Hard-code a lookup table of Walmart closure dates (all December 25ths, 2011–2016). For any forecast that falls on these dates, override the predicted value to `0` post-modeling. Do not let the model "learn" Christmas — let domain logic handle it.

---

### PHASE 1 — Data Pipeline & Feature Engineering
**Goal:** Transform 5 raw CSVs into a rich ML-ready training table.

**Step 1 — Melt Wide to Long**
The sales file has `d_1` to `d_1969` as column headers. `pd.melt()` it into a long format: one row per `(item_id, store_id, day)`. This creates ~30 million rows but is essential for ML.

**Step 2 — Join Calendar & Prices**
Merge the melted sales table with `calendar.csv` (on the day column) to get dates, event flags, and SNAP flags. Then merge `sell_prices.csv` (on `store_id`, `item_id`, `wm_yr_wk`) to get weekly pricing.

**Step 3 — Build the Core Features**

| Feature | Type | Rationale |
|---|---|---|
| `lag_7`, `lag_28` | Lag | Sales 7 and 28 days ago (weekly/monthly momentum) |
| `rolling_mean_7`, `rolling_mean_28` | Rolling | Capture recent trend and volatility |
| `rolling_zero_count_7` | Rolling | Measure how "sparse" this item is (intermittency) |
| `day_of_week`, `month`, `day_of_month` | Calendar | Weekly heartbeat + monthly payday effect |
| `is_weekend` | Calendar | Captures the **31% weekend surge** |
| `snap_CA`, `snap_TX`, `snap_WI` | Covariate | SNAP disbursement flags per state |
| `is_food_and_is_snap` | Interaction | **10.2% SNAP shockwave** — critical interaction term |
| `price_pct_change` | Price | Relative price momentum (avoids the **$0.01 penny trap**) |
| `is_phantom_promo` | Engineered | Flag weeks where price dropped >5% vs. previous week |
| `is_ghost_stockout` | Engineered | Flag probable stockouts (exclude from training) |
| `cat_id`, `dept_id`, `store_id` | Categorical | Hierarchical embeddings for the global model |

---

### PHASE 2 — The Forecasting Model
**Goal:** Build a single, highly accurate global model using LightGBM with Tweedie loss.

**Why Tweedie Loss?**
The `FOODS` category (which drives **69.56% of total sales volume**) is highly intermittent and zero-inflated. The Tweedie distribution is the mathematical "sweet spot" between Poisson (for counts) and Gamma (for continuous positive values), making it the industry standard for this exact pattern. No other loss function handles this correctly.

**Global Model Strategy**
Instead of training 42,840 separate models (one per series), we train **one single global LightGBM model** on all data simultaneously. Categorical features like `cat_id`, `dept_id`, and `store_id` act as implicit item embeddings, letting the model share learnings across similar items. This is especially powerful for sparse `HOBBIES` items that have too little data to train individual models.

**Foods-First Tuning**
Because `FOODS` = 70% of the competition metric (WRMSSE is volume-weighted), we tune hyperparameters specifically to minimize error on the Foods category first. Hobbies/Household models run on default parameters.

**Hierarchical Reconciliation**
After generating bottom-level (Store-Item) predictions, apply **Bottom-Up reconciliation**: higher-level forecasts (Department, Category, State, Total) are simply the sum of Store-Item predictions. This guarantees mathematical coherence across all 12 hierarchy levels — no contradictions in the forecast.

---

### PHASE 3 — The Agentic AI Layer (The "Wow" Factor)
**Goal:** Wrap the LightGBM output in a GenAI Copilot that turns numbers into business decisions.

After the model produces `F1–F28` predictions, a lightweight Python function:
1. Scans for items where predicted sales are more than **1.5x the rolling average** (anomaly detection).
2. Checks if the spike coincides with a SNAP day, weekend, or phantom promo.
3. Sends the context to a Gemini/OpenAI LLM with a structured prompt.
4. The LLM generates a human-readable alert that appears in the UI.

**Example Copilot Output:**
> *🚨 ALERT — CA_3 Store, FOODS_3_090:*
> *Predicted volume spike of 43% in Week 2 of forecast. Likely driven by SNAP disbursement period overlapping with weekend. Current 7-day average: 12 units/day. Forecasted peak: 17 units/day.*
> *Recommendation: Increase safety stock by 35 units before Day F8 to prevent stockout.*

---

### PHASE 4 — Backend API
**Goal:** Serve model predictions on demand via a clean REST API.

- **Framework:** FastAPI (Python)
- **Endpoints:**
  - `POST /forecast` → Accepts `store_id`, `item_id`, returns `F1–F28` predictions + confidence bounds
  - `GET /anomalies` → Returns the list of Copilot alerts generated for the current forecast window
  - `GET /stockout-history/{item_id}` → Returns historical ghost stockout flags for a given item

---

### PHASE 5 — UI Dashboard
**Goal:** A production-quality dashboard that judges can click through in the presentation.

**4 Panels:**
1. **Forecast View** — Select any State → Store → Item → See 28-day forecast line chart with historical sales overlaid. Includes confidence bands (safety stock zone in amber, stockout risk zone in red).
2. **Copilot Alerts Panel** — Live list of AI-generated plain-English supply chain alerts.
3. **Stockout Vulnerability Map** — A heatmap showing which stores/departments historically had the most ghost stockouts.
4. **SNAP Impact View** — Bar chart showing SNAP vs. Non-SNAP sales volume for the selected state's FOODS category.

---

### PHASE 6 — Deployment & CI/CD
**Goal:** Satisfy the judges' technical rigor criteria.

- **Docker:** The FastAPI backend and the UI are each containerized in separate Dockerfiles.
- **docker-compose.yml:** One command to spin up the entire stack locally for the demo.
- **GitHub Actions:** On every push — auto-run pytest unit tests for the data pipeline and API endpoints.
- **Cloud Hosting:** Deploy both containers to a free-tier cloud (Render.com, Hugging Face Spaces, or AWS EC2) so the judges can access a live URL during the presentation.

---

### PHASE 7 — Presentation Strategy
**Goal:** Make the judges feel like they just saw an enterprise product, not a Kaggle notebook.

**The Story Arc (8 minutes):**
1. **(1 min) The Problem** — Show a real-world headline about a retailer losing millions to stockouts. Quantify the pain.
2. **(1 min) The Data** — Show the 5 CSVs, explain the hierarchy (bottom-up), and immediately mention the two traps (Leading Zeros + Christmas) that everyone else missed.
3. **(2 min) The Model** — Explain Tweedie Loss in one sentence, show the feature table, and emphasize the 70% Foods focus.
4. **(2 min) Live Demo** — Open the live UI. Click through a real forecast. Show the Copilot alert firing for a SNAP-day spike.
5. **(1 min) The Metrics** — Show WRMSSE score on the validation set. Show the Ghost Stockout detection recall.
6. **(1 min) The Roadmap** — How this scales to real-time streaming data with Kafka + MLflow model registry.

---

## ✅ Judging Criteria Checklist

| Criterion | How We Satisfy It |
|---|---|
| Use Case Understanding | Ghost Stockout, SNAP, Leading Zero analysis proves deep domain knowledge |
| Solution Architecture | 6-layer architecture (Data → Features → Model → API → UI → Cloud) |
| Innovation & AI/ML | Tweedie Loss + Agentic AI Copilot (GenAI integration) |
| UI & UX | 4-panel live dashboard with interactive forecasts and Copilot alerts |
| Technical Implementation | Modular Python codebase, PEP-8, GitHub Actions CI |
| Model Performance | WRMSSE metric on hold-out validation set; Foods-first tuning |
| Deployment & Integration | Dockerized, cloud-hosted, live URL for judges |
| Estimation & Roadmap | Kafka streaming + MLflow versioning as next steps |
| Documentation | README + Architecture diagram + inline code comments |
