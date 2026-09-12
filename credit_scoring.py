"""
Credit Risk Scoring Model - PD, LGD, EAD -> Expected Loss
------------------------------------------------------------
Built this to work through how banks actually estimate credit risk on
a loan portfolio - not just "will this person default" but the full
Expected Loss framework: EL = PD x LGD x EAD.

WHAT THIS COVERS
- The three components of Expected Loss (PD, LGD, EAD) and how they combine
- Logistic Regression for PD estimation
- Gradient Boosting / Random Forest as more powerful alternatives, compared via AUC
- ROC curve and AUC - the right metric here since defaults are rare
  (imbalanced classes), accuracy alone would be misleading
- 5-fold cross-validation, and why judging a model on its in-sample
  (training) AUC is basically meaningless - it always looks better
  than it really is

HOW IT'S BUILT
1. simulate a realistic loan portfolio (5000 loans)
2. engineer features: DTI ratio, credit score, employment years, late payments
3. train Logistic Regression, Random Forest, and Gradient Boosting
4. compare all three with AUC + 5-fold cross-validation
5. estimate LGD and EAD per loan, then calculate portfolio Expected Loss
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
N_LOANS = 5000


# ---------------------------------------------------------------------
# 1. SIMULATE A REALISTIC LOAN PORTFOLIO
# ---------------------------------------------------------------------

def simulate_loan_portfolio(n=N_LOANS):
    """
    Simulate loan-level data with features that actually drive default
    risk in real underwriting - debt-to-income ratio, credit score,
    employment history, and late payment history. Default probability
    is generated from a logistic function of these features plus noise,
    so the "ground truth" relationship is realistic but not perfectly
    learnable (like real data).
    """
    credit_score = np.clip(np.random.normal(680, 80, n), 300, 850)
    dti_ratio = np.clip(np.random.normal(0.35, 0.15, n), 0.02, 0.95)  # debt-to-income
    employment_years = np.clip(np.random.exponential(5, n), 0, 40)
    late_payments_2yr = np.random.poisson(0.8, n)
    loan_amount = np.clip(np.random.lognormal(9.5, 0.6, n), 2000, 200000)
    interest_rate = np.clip(np.random.normal(0.11, 0.04, n), 0.03, 0.30)

    # true underlying default "logit" - higher DTI, lower credit score,
    # more late payments, less employment history -> higher default risk
    logit = (
        -6.0
        + 4.5 * dti_ratio
        - 0.006 * (credit_score - 680)
        + 0.35 * late_payments_2yr
        - 0.04 * employment_years
        + 8.0 * interest_rate
        + np.random.normal(0, 0.6, n)  # noise so it isn't perfectly separable
    )
    true_pd = 1 / (1 + np.exp(-logit))
    default = np.random.binomial(1, true_pd)

    df = pd.DataFrame({
        "credit_score": credit_score,
        "dti_ratio": dti_ratio,
        "employment_years": employment_years,
        "late_payments_2yr": late_payments_2yr,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "default": default,
    })
    return df


# ---------------------------------------------------------------------
# 2. TRAIN PD MODELS + COMPARE WITH AUC / CROSS-VALIDATION
# ---------------------------------------------------------------------

def train_and_compare_models(df):
    features = ["credit_score", "dti_ratio", "employment_years",
                "late_payments_2yr", "loan_amount", "interest_rate"]
    X = df[features]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
    }

    print("=" * 70)
    print("MODEL COMPARISON: in-sample AUC vs held-out test AUC vs 5-fold CV AUC")
    print("=" * 70)
    print(f"{'Model':<22}{'In-sample AUC':<16}{'Test AUC':<12}{'5-fold CV AUC':<16}")
    print("-" * 70)

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        # logistic regression benefits from scaled features, trees don't need it
        # but scaling doesn't hurt them either, so use scaled data for all
        model.fit(X_train_scaled, y_train)

        in_sample_pred = model.predict_proba(X_train_scaled)[:, 1]
        in_sample_auc = roc_auc_score(y_train, in_sample_pred)

        test_pred = model.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, test_pred)

        # cross-validation on the full scaled training set, refitting each fold
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
        cv_auc = cv_scores.mean()

        results[name] = {
            "model": model, "in_sample_auc": in_sample_auc,
            "test_auc": test_auc, "cv_auc": cv_auc, "cv_scores": cv_scores,
        }
        print(f"{name:<22}{in_sample_auc:<16.4f}{test_auc:<12.4f}{cv_auc:<16.4f}")

    print()
    print("Why in-sample AUC is misleading: the model has already SEEN every")
    print("one of these points during training, so it can partly memorize noise")
    print("rather than learn the real pattern. That's why in-sample AUC is")
    print("almost always higher than test/CV AUC - the gap between them tells")
    print("you how much the model is overfitting. 5-fold CV is more reliable")
    print("than a single test split because it averages performance across 5")
    print("different train/test partitions, so it isn't just luck of one split.")
    print()

    return results, X_test, X_test_scaled, y_test, features, scaler


def plot_roc_curves(results, X_test_scaled, y_test):
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, r in results.items():
        probs = r["model"].predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {r['test_auc']:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - PD Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig("roc_curves.png", dpi=150)
    print("saved ROC curve chart to roc_curves.png\n")


# ---------------------------------------------------------------------
# 3. LGD AND EAD ESTIMATION
# ---------------------------------------------------------------------

def estimate_lgd_ead(df):
    """
    LGD (Loss Given Default) - the % of the exposure you don't recover
    if the borrower defaults. Modeled here as a function of collateral/
    loan size, with realistic randomness (LGD is notoriously hard to
    predict precisely in practice, this keeps it simple but not silly).

    EAD (Exposure at Default) - how much is actually outstanding at the
    moment of default. Approximated here as a fraction of the original
    loan amount to reflect partial paydown before default.
    """
    n = len(df)
    # larger loans tend to have somewhat better recovery structures (more
    # likely secured) - simple illustrative relationship, not a real model
    base_lgd = 0.55 - 0.05 * np.log(df["loan_amount"] / df["loan_amount"].median())
    lgd = np.clip(base_lgd + np.random.normal(0, 0.08, n), 0.10, 0.95)

    ead_fraction = np.clip(np.random.normal(0.85, 0.10, n), 0.4, 1.0)
    ead = df["loan_amount"] * ead_fraction

    return lgd, ead


# ---------------------------------------------------------------------
# 4. EXPECTED LOSS: EL = PD x LGD x EAD
# ---------------------------------------------------------------------

def calculate_expected_loss(df, pd_estimates, lgd, ead):
    df = df.copy()
    df["PD"] = pd_estimates
    df["LGD"] = lgd
    df["EAD"] = ead
    df["expected_loss"] = df["PD"] * df["LGD"] * df["EAD"]

    total_exposure = df["EAD"].sum()
    total_el = df["expected_loss"].sum()

    print("=" * 70)
    print("PORTFOLIO EXPECTED LOSS  (EL = PD x LGD x EAD)")
    print("=" * 70)
    print(f"Number of loans:               {len(df):,}")
    print(f"Total exposure (sum of EAD):    ${total_exposure:,.0f}")
    print(f"Average PD:                     {df['PD'].mean():.2%}")
    print(f"Average LGD:                    {df['LGD'].mean():.2%}")
    print(f"Total Expected Loss:            ${total_el:,.0f}")
    print(f"EL as % of total exposure:      {total_el/total_exposure:.2%}")
    print()
    print("Top 5 riskiest loans by Expected Loss:")
    print(df.sort_values("expected_loss", ascending=False)
            [["credit_score", "dti_ratio", "PD", "LGD", "EAD", "expected_loss"]]
            .head(5).to_string(index=False))
    print()
    return df


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("Simulating loan portfolio...\n")
    df = simulate_loan_portfolio()
    print(f"Portfolio: {len(df)} loans, default rate = {df['default'].mean():.2%}\n")

    results, X_test, X_test_scaled, y_test, features, scaler = train_and_compare_models(df)
    plot_roc_curves(results, X_test_scaled, y_test)

    # use the best model (by test AUC) to score PD for the WHOLE portfolio
    best_name = max(results, key=lambda k: results[k]["test_auc"])
    best_model = results[best_name]["model"]
    print(f"Using best model ({best_name}) to score full portfolio PD...\n")

    X_full_scaled = scaler.transform(df[features])
    pd_estimates = best_model.predict_proba(X_full_scaled)[:, 1]

    lgd, ead = estimate_lgd_ead(df)
    scored_df = calculate_expected_loss(df, pd_estimates, lgd, ead)
    scored_df.to_csv("scored_portfolio.csv", index=False)
    print("saved full scored portfolio to scored_portfolio.csv")


if __name__ == "__main__":
    main()
