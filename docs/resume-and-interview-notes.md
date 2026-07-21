# Resume Bullets & Interview Story

## Resume bullets (pick 2-3)

**Version A -- balanced (ML + systems):**
Built an end-to-end predictive maintenance system (CMAPSS turbofan RUL)
spanning physics-informed feature engineering, 5 model families
(XGBoost/LSTM/CNN/autoencoder/ensemble), and a live FastAPI + Docker
deployment on Render, with automated drift/robustness monitoring and a
cost-based maintenance decision layer.

**Version B -- leads with a concrete result:**
Engineered physics-informed signal features (FFT, wavelets,
thermal-fatigue indices) and tuned an XGBoost/LSTM/CNN model suite (best
RMSE 12.8, CMAPSS score 267) for turbofan RUL prediction; found and fixed
a train/validation leakage bug via engine-level splitting that had been
masking true generalization error.

**Version C -- leads with the deployment/robustness angle:**
Deployed a predictive maintenance ML pipeline to production (FastAPI,
Docker, CI/CD, live cloud hosting), including drift detection (PSI/KS-test),
adversarial robustness testing, and a maintenance-cost decision layer
estimating ~$1.6M in simulated savings vs. reactive maintenance across
100 test assets.

Recommended combination for general software/ML roles: **B + C**.

---

## STAR-format interview story

**Situation:** I wanted a portfolio project that went beyond a notebook
with an accuracy score -- something that reflected how predictive
maintenance actually gets adopted in industry, where "does it forecast
well" is only one of several questions that matter.

**Task:** Build a full RUL prediction system for turbofan engines (NASA
CMAPSS dataset) that could genuinely answer: which model fits which
deployment constraint, how it degrades under real-world sensor problems,
and what it should actually recommend a maintenance team do.

**Action:** I engineered physics-informed features tied to real failure
mechanisms (bearing fault frequencies, thermal stress, fatigue
accumulation), trained and rigorously compared five model families, and
-- critically -- caught and fixed a subtle data leakage bug where
overlapping sliding windows from the same engine were split across
train/validation, which had been making validation performance look far
better than true test performance. I then built a model-selector that
picks the right model per deployment constraint (cloud vs. edge vs.
real-time), tested robustness against sensor drift, noise, and missing
data (finding and partially fixing a real weakness -- zero-imputation
hurting more than proportional noise), and layered a cost-based decision
system on top translating predictions into actual maintenance
recommendations. I containerized and deployed the whole thing live via
FastAPI, Docker, and Render, with a monitoring dashboard polling it in
real time.

**Result:** A working, live, publicly-deployed system where every claim
is backed by a re-runnable script against real held-out test data --
including honest negative results (an ensemble that didn't beat the best
single model) reported directly rather than massaged into a positive
story, which is what actually built credibility on this project.

---

## Practice checklist

- [ ] Explain the project out loud, unaided, in under 3 minutes
- [ ] Be ready to answer: "why split by engine instead of randomly?"
- [ ] Be ready to answer: "why didn't the ensemble help?"
- [ ] Be ready to answer: "what would you do differently with more time?"
