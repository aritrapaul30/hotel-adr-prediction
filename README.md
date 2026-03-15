This project predicts nightly hotel room rates (ADR in RM) using a 6,630-booking hospitality dataset with 26 features spanning customer demographics, property attributes, and market dynamics. Five custom features are engineered: SpendPerNight, DemandPressure, EventDistanceInteraction, PriceCompetitiveness, and CustomerValueIndex to capture real-world pricing behaviour. Three models are compared (Linear Regression R²=0.98, Decision Tree R²=0.99, Random Forest R²=0.9964), with the tuned Random Forest selected via 5-Fold RandomizedSearchCV as the production model.

## 📒 Notebook Sections

---

### 1️⃣ 🏨 Business Problem
> Moving hotels from reactive guesswork to proactive, data-driven pricing

- 📊 **Competitor Alignment** — Price rooms in sync with real-time market rates
- 💹 **Revenue Forecasting** — Predict nightly ADR to plan financials accurately
- 👥 **Customer Segmentation** — Tailor pricing strategies to different guest profiles

---

### 2️⃣ 📦 Data Loading
> Authentic hotel bookings & performance data, individual stay granularity

| Detail | Value |
|---|---|
| 📊 Observations | 6,630 rows |
| 📋 Variables | 26 columns |
| 🎯 Target Variable | `ADR_RM` : Average Daily Rate in RM |
| 🏨 Granularity | One row = One individual hotel booking |

---

### 3️⃣ 🩹 Missing Value Handling
> Tailored imputation strategy based on variable type

| Variable | Missing Count | Strategy | Why |
|---|---|---|---|
| `MealsIncluded` | 158 | 🔵 Mode | Preserve most common category |
| `PaymentType` | 261 | 🔵 Mode | Preserve most common category |
| `CustomerAge` | 101 | 🟡 Mean | Retain central tendency |
| `ReviewScore` | 141 | 🟡 Mean | Retain central tendency |
| `CompetitorAvgRateRM` | — | 🟡 Mean | Avoid pricing model bias |
| `ADR_RM` | 31 | 🟡 Mean | Target variable : minimal intervention |

---

### 4️⃣ 🧼 Text Hygiene
> Standardising all categorical text before encoding

**Columns cleaned:** `Country` · `BrandTier` · `RoomType` · `BookingChannel` · `MealsIncluded` · `PaymentType`

**Steps applied:**
- 🔡 Convert all text → **lowercase** (e.g. `Thailand` → `thailand`)
- ✂️ Strip **leading & trailing whitespace**
- 🚫 Replace `"nan"`, `"none"`, `"na"`, `""` → `NaN` for proper imputation
- ✅ Result: Uniform, consistent categories across all 6 columns

---

### 5️⃣ 🔢 Label Encoding
> Converting text categories into numbers for ML models

| Column | Example Before | Example After |
|---|---|---|
| `PaymentType` | `"card"` | `0` |
| `BrandTier` | `"luxury"` | `1` |
| `BookingChannel` | `"online"` | `2` |
| `RoomType` | `"suite"` | `3` |


> ⚙️ Tool used: `sklearn.preprocessing.LabelEncoder` — applied independently per column

---

### 6️⃣ ⚙️ Feature Engineering
> Creating 5 new business-meaningful variables to boost model performance

| 🆕 Feature | 📐 Formula | 💡 Business Meaning |
|---|---|---|
| `SpendPerNight` | `TotalSpend / NightsStayed` | Fair comparison across stay lengths |
| `EventDistanceInteraction` | `EventsIndex × DistanceToCenter` | Proximity to event hubs drives price surges |
| `DemandPressure` | `OccupancyRate × EventsIndex` | Tight supply = higher ADR |
| `PriceCompetitiveness` | `ADR_RM / CompetitorAvgRate` | How the hotel is positioned vs market |
| `CustomerValueIndex` | `(TotalSpend + ExtraSpend) / NightsStayed` | Profitability per guest per night |

---

### 7️⃣ 📏 Normalization
> Scaling all numeric features to prevent large-magnitude variables from dominating

- 🔧 Tool: `StandardScaler` from `sklearn.preprocessing`
- 🎯 Target `ADR_RM` was **separated first** — scaled independently
- 📐 All other numeric features transformed to: **Mean = 0 · Std Dev = 1**
- ✅ Result: Every feature contributes equally to model learning

---

### 8️⃣ 🤖 Model Training
> Three supervised regression algorithms benchmarked on the same dataset

| Model | R² Score | MAE (RM) | RMSE (RM) | Verdict |
|---|---|---|---|---|
| 📏 Linear Regression | 0.9829 | 11.69 | 20.46 | ✅ Great baseline, assumes linearity |
| 🌿 Decision Tree | 0.9923 | 7.65 | 13.77 | ✅ Captures non-linear patterns |
| 🌲 **Random Forest** | **0.9964** | **4.38** | **9.34** | ⭐ Best : stable & robust |

> 💡 Random Forest wins because it combines hundreds of decision trees, reducing overfitting and capturing complex interactions between demand, occupancy, and competitor pricing.

---

### 9️⃣ 🎛️ Hyperparameter Tuning
> Systematic optimisation using RandomizedSearchCV with 5-Fold Cross Validation

**Search Space Explored:**

n_estimators → 200 to 700 (number of trees)
max_depth → 5 to 30 (tree depth)
min_samples_split → 2 to 12
min_samples_leaf → 1 to 8
max_features → sqrt | log2 | 0.5 | 0.8

**🏆 Best Parameters Found — Random Forest:**

| Parameter | Best Value |
|---|---|
| `n_estimators` | **684** |
| `max_depth` | **27** |
| `max_features` | **0.8** |
| `min_samples_split` | **4** |
| `min_samples_leaf` | **1** |
| Best 5-Fold CV R² | **1.00** ✅ |
