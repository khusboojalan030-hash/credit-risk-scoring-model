# Credit Risk Scoring Model - PD, LGD, EAD -> Expected Loss

A project that builds out the full Expected Loss framework banks use
for credit risk - not just "will this loan default" but combining
Probability of Default, Loss Given Default, and Exposure at Default
into a single portfolio-level risk number.

Target role: Credit Risk Analyst / Model Validator

## What it teaches

- **The three components of Expected Loss** - PD, LGD, EAD - and how
  they combine into `EL = PD x LGD x EAD`
- **Logistic Regression for PD estimation** - the industry-standard
  baseline model for default prediction
- **Gradient Boosting / Random Forest as alternatives** - compared
  directly against Logistic Regression using AUC
- **ROC curve and AUC** - why this is the right metric here instead of
  plain accuracy (defaults are rare, so a model that just predicts
  "no default" every time would still score ~95% accuracy and be
  completely useless)
- **Cross-validation** - and why judging a model by its in-sample
  (training) AUC is close to meaningless, since the model has already
  seen every one of those data points

## How it's built

1. Simulate a realistic loan portfolio - 5,000 loans, with default
   generated from a logistic function of real risk drivers plus noise
   (so the relationship is realistic but not perfectly learnable, same
   as real data)
2. Engineer features: credit score, DTI ratio, employment years, late
   payments in the last 2 years, loan amount, interest rate
3. Train Logistic Regression, Random Forest, and Gradient Boosting
4. Compare all three using in-sample AUC, held-out test AUC, and
   5-fold cross-validation AUC
5. Estimate LGD and EAD per loan
6. Use the best-performing model's PD estimates to calculate portfolio
   Expected Loss

## Why in-sample AUC is misleading

A model evaluated on the same data it was trained on can partially
memorize noise instead of learning the real pattern - so in-sample AUC
is almost always higher than how the model actually performs on new
data. The gap between in-sample AUC and test/CV AUC tells you how much
a model is overfitting.

In this project, Gradient Boosting shows the clearest example: it hits
a very high in-sample AUC but drops noticeably on the held-out test set
and 5-fold CV - a textbook overfitting signature. Logistic Regression,
despite being the simplest model here, generalizes the best on this
data. That's a genuinely common result in credit risk modelling:
simpler, well-regularized models often outperform more complex ones
once you actually check generalization instead of trusting the
training score.

## The Expected Loss calculation

```
EL = PD x LGD x EAD
```

- **PD (Probability of Default)** - estimated by the best-performing
  classifier from the comparison step
- **LGD (Loss Given Default)** - the % of exposure not recovered if
  the borrower defaults; modeled here as a function of loan size plus
  randomness (LGD is genuinely hard to predict precisely in practice)
- **EAD (Exposure at Default)** - how much is actually outstanding at
  the moment of default, approximated as a fraction of the original
  loan amount

The script totals EL across the whole portfolio and reports it both in
dollar terms and as a % of total exposure - the number a bank would
actually use to size its loan loss reserves.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python credit_scoring.py
```

## Sample output

```
Model                 In-sample AUC   Test AUC    5-fold CV AUC
----------------------------------------------------------------------
Logistic Regression   0.7375          0.7662      0.7283
Random Forest         0.8885          0.7439      0.7118
Gradient Boosting     0.9820          0.7331      0.6705

Total exposure (sum of EAD):    $67,027,900
Average PD:                     4.63%
Average LGD:                    55.16%
Total Expected Loss:            $1,582,209
EL as % of total exposure:      2.36%
```
(exact numbers change slightly each run since the portfolio is randomly
simulated - re-run with a fixed seed for reproducible results)

## Project structure

```
credit-risk-scoring/
├── credit_scoring.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Possible extensions

- Add SHAP values to explain individual loan-level PD predictions
- Calibrate PD estimates (Platt scaling / isotonic regression) since
  raw classifier probabilities aren't always well-calibrated
- Segment the portfolio (e.g. by credit score band) and compute EL per
  segment instead of only at the portfolio level
- Compare against a real-world dataset (e.g. Lending Club) instead of
  simulated data
