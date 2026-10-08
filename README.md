# MapleFreight — Delivery Delay Prediction & Slack Alerts

**AML 3303 · Assessment 1** · Arti · Student ID `<your_id>`

This project predicts which in-transit shipments will miss their promised delivery window. It posts the highest-risk ones to a `#dispatch-alerts` Slack channel, so dispatch can re-route or warn the customer before the window closes.

## Problem

MapleFreight moves about 8,000 shipments a week. On-time performance has dropped from 94% to 88%, and a late delivery is only discovered when the customer calls. By then the service credit is owed and the complaint has already reached the account manager. The goal is to flag likely-late shipments **at dispatch** and alert the team automatically.

## Dataset

`maplefreight_delivery_delay_dataset.csv` is included in this repository.

- **Size:** 6,035 rows × 23 columns
- **Target:** `delivered_late` (20.6% late)

Problems found and fixed in the raw export:

| Issue | Fix |
|---|---|
| 32 exact duplicate rows, plus 3 repeated IDs that differ only in `service_level` casing | Normalise casing first, then keep one row per `shipment_id`, leaving 6,000 shipments |
| `service_level` spelled six ways (`Standard` / `standard`, …) | Title-case it, leaving 3 levels |
| 40 weights between 170,900 and 2,999,300 kg, with an empty gap above 5,044 kg | Recorded in grams, so divide by 1,000 |
| Missing `fuel_cost_cad` (4.0%), `driver_experience_years` (3.5%), `weather_condition` (2.5%) and `traffic_index` (2.0%) | Weather becomes `Unknown`; numbers get a median imputer inside the model pipeline (fitted on training folds only) |
| `actual_transit_hours` defines the label exactly (late ⇔ actual > 1.10 × scheduled) but is recorded on delivery | Dropped, because it leaks the target |

## Repository contents

```
MapleFreight_Delivery_Delay_Prediction_<your_id>.ipynb   full analysis, run top to bottom with outputs
maplefreight_delivery_delay_dataset.csv                   raw data
screenshots/slack_alert.png                               the alert as it appeared in Slack
requirements.txt
.env.example                                              template for the Slack webhook (no real URL)
.gitignore                                                includes .env
```

## How to run

```bash
git clone <this repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then paste your webhook URL into .env
jupyter notebook                                       # open the notebook → Kernel → Restart & Run All
```

### Slack setup

1. Create a free Slack workspace and a channel called **#dispatch-alerts**.
2. Go to <https://api.slack.com/apps> → **Create New App** → *From scratch* → pick the workspace.
3. Open **Incoming Webhooks**, switch it on, then **Add New Webhook to Workspace** → `#dispatch-alerts` → *Allow*.
4. Paste the URL into `.env` as `SLACK_WEBHOOK_URL=...`.

The notebook loads the URL with `python-dotenv` and never prints it. Error messages show only the exception type, because `requests` errors include the full URL. If `.env` is missing, the notebook still runs top to bottom and just prints a preview of the message.

## Approach

1. **Understand and clean** the data: duplicates, spellings, units, missing values and leakage.
2. **Explore** late rates by category, numeric distributions by outcome, chi-square tests, single-feature AUCs, correlations, and a carrier × weather interaction.
3. **Engineer and test features** with 3 × 5-fold cross-validation.
   - Dropping the two city columns helped, because they carry no signal.
   - Only `is_winter` (Dec–Feb) among the six engineered features added anything.
4. **Model.**
   - Two baselines: majority class, and plain logistic regression at threshold 0.5.
   - Handle the 21% / 79% imbalance with class weights.
   - Compare weighted logistic regression, random forest and gradient boosting.
   - Tune the winner.
5. **Choose the alert threshold** from out-of-fold predictions on the training set, translated into alerts per week.
6. **Evaluate once** on a 20% held-out test set (1,200 shipments). Report precision, recall, F1 and confusion matrices, with bootstrap confidence intervals.
7. **Explain** the model with odds ratios per practical unit and permutation importance, plus a per-shipment reason ("storm forecast, picked up 81 min late") that goes into each Slack alert.
8. **Alert:** score the in-transit batch and post a phone-friendly Slack message with the count and the top 5 shipments (ID, destination, carrier, risk score, reason).

## Results (test set, 1,200 shipments)

| Model | Precision | Recall | F1 | Late caught (of 247) |
|---|---|---|---|---|
| Majority class ("always on time", 79.4% accurate) | 0.00 | 0.00 | 0.00 | 0 |
| Baseline: logistic regression, threshold 0.5 | 0.70 | 0.35 | 0.47 | 87 |
| **Final: weighted logistic regression, threshold 0.60** | **0.52** | **0.65** | **0.58** | **160** |

- **Final model confusion matrix:** 805 true on-time, 148 false alarms, 87 missed, 160 caught.
- **F1 gain over the baseline:** +0.11. The 95% bootstrap interval is +0.05 to +0.16, and the gain was positive in every resample.
- **PR-AUC:** 0.61, against 0.21 for random guessing.
- **Tree models did worse.** Random forest and gradient boosting did not beat logistic regression in cross-validation (PR-AUC 0.52 and 0.53 vs 0.56). The drivers act roughly additively, which a linear model captures well, and it is also easier to explain to dispatch.

## Chosen threshold: 0.60

The threshold is the value that maximises F1 on out-of-fold predictions; the test set was never used to pick it. At 8,000 shipments a week it means:

- about **2,100 alerts a week**, of which close to half are genuinely late
- about **62% of late shipments caught** in cross-validation (65% on test), against 33% for the baseline.

**Why not lower?** At 0.40–0.45, recall rises to about 80%, but 41–46% of all shipments get flagged and fewer than 4 alerts in 10 are real. Dispatch would mostly chase false alarms and soon ignore the channel.

**Why not higher?** At 0.70 and above, fewer than half the late shipments are caught.

**Why F1?** The F1 curve is flat between about 0.50 and 0.65, and 0.60 sits in the middle of that plateau. The brief gives no dollar costs, so F1, which weighs a miss and a false alarm equally, is the neutral choice. With real costs for a credit plus re-delivery versus a dispatcher's check, the threshold table in Section 9.1 shows where to move it. The alert lists shipments by risk, so dispatch always sees the most urgent ones first.

## Key findings

| Driver | Late rate | Odds ratio from the model |
|---|---|---|
| Storm forecast | 46% (vs 14% clear) | ≈ 10.7× vs clear |
| Snow forecast | 34% | ≈ 4.3× vs clear |
| Contract owner-operators | 36% (vs 13% own fleet) | ≈ 5.3× vs own fleet |
| Owner-operators in snow or storm | 55–56% | — |
| Economy service | 27% (vs 15% Express) | ≈ 2.6× vs Express |
| Each extra intermediate stop | 12% at 0 stops → 29% at 5 | ≈ 1.36× per stop |
| +30 min pickup delay | — | ≈ 1.50× |
| +10 traffic index points | — | ≈ 1.36× |
| +5 years driver experience | — | ≈ 0.73× (protective) |

- **Smaller effects:** cross-border trips, carrier late history, vehicle age, route congestion, refrigerated goods and Dec–Feb pickups.
- **No signal:** customer tier, origin and destination city, weight and fuel cost.

## Recommendations

1. **Weather-aware promises.** Add buffer to the promised transit time, or re-plan non-urgent loads, when snow or a storm is forecast, and warn the customer up front.
2. **Owner-operator freight.** Keep Gold and Express loads on the own fleet, especially on bad-weather days. Put owner-operators on an on-time scorecard tied to rate.
3. **Late pickup as a trigger.** Re-score each shipment at the pickup scan, and send a proactive heads-up when a pickup is more than an hour late.
4. **Fewer stops on long or congested routes.** Also time departures around the traffic index.
5. **Re-set Economy windows.** They are too tight for how those loads actually move.
6. **Match newer drivers to simpler routes,** especially in winter.
7. **Run the alert hourly in production.** Log the outcome of each alert, and use that log to retrain monthly and re-tune the threshold against real costs.

## Limitations

- The in-transit batch is the held-out test set standing in for a live feed.
- Risk scores are a ranking, not calibrated probabilities, because class weighting inflates them.
- The threshold uses F1 because no cost data was available.
- Live GPS, ETA and en-route weather data would likely raise the ceiling (PR-AUC 0.61).

## AI usage

**Assistant used:** Claude (Anthropic), via claude.ai.

**What I used it for:**
- structuring the notebook
- drafting the cleaning, modelling and Slack code
- checking that the webhook URL can't leak into notebook output.

**How I checked the output:** I ran every cell and checked each figure in this README against the notebook output.

**One case where the output was wrong.** The first version of the function that writes the "why" reason for each Slack alert (`describe()` in Section 10.2) crashed with `ValueError: Unknown format code 'f' for object of type 'str'`.
- **Cause:** it built every message template in one dictionary of f-strings. Python evaluates all of them before picking one, so a numeric format like `{v:.0f}` was applied to text values such as the weather forecast.
- **Fix:** each template was wrapped in a `lambda`, so only the template that matches the feature is formatted.
