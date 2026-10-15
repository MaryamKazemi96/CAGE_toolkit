"""Survey analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SESSION_DENSITY = {1: "low", 2: "low", 3: "low", 4: "low", 5: "high", 6: "high", 7: "high", 8: "low"}
RPS_REVERSE = ["Pre_RPS_Q3_SafetyFirst", "Pre_RPS_Q4_AvoidRisks", "Pre_RPS_Q5_NoUnnecessaryRisks", "Pre_RPS_Q7_DislikeUnpredictable"]
RPS_DIRECT = ["Pre_RPS_Q6_TakeRisksRegularly", "Pre_RPS_Q8_CrowdsAsChallenge", "Pre_RPS_Q9_AssertiveNavigator"]
TLX = ["TLX_Mental", "TLX_Physical", "TLX_Temporal", "TLX_Performance", "TLX_Effort", "TLX_Frustration"]
RELATED = {
    "Robot Intent Legibility": ("PostS3_Q2_DelivLegibility", "PostS4_Q1_ProjClarity"),
    "Path Adaptation / Hesitation": ("PostS3_Q3_Hesitation", "PostS4_Q3_EarlyPathAlter"),
    "Perceived Safety": ("PostS3_Q4_ForcedToYield", "PostS4_Q4_ReducedAnxiety"),
}
SPEARMAN = {
    "Risk propensity vs implicit hesitation": ("PRS_recomputed", "PostS3_Q3_Hesitation"),
    "Risk propensity vs forced yield": ("PRS_recomputed", "PostS3_Q4_ForcedToYield"),
    "Prior trust vs implicit legibility": ("Pre_Q2_Trust", "PostS3_Q2_DelivLegibility"),
    "Prior trust vs explicit clarity": ("Pre_Q2_Trust", "PostS4_Q1_ProjClarity"),
}


def load_survey(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data[[name for name in data if not name.startswith("Unnamed")]].copy()
    for name in data.columns:
        if name not in {"Session_ID", "PostS4_Comments"}:
            data[name] = pd.to_numeric(data[name], errors="coerce")
    data["session"] = data.Session_ID.astype(str).str.removeprefix("s").astype(int)
    data["density"] = data.session.map(SESSION_DENSITY)
    data["PRS_recomputed"] = pd.concat([data[RPS_REVERSE].apply(lambda col: 10 - col), data[RPS_DIRECT]], axis=1).mean(axis=1)
    data["TLX_recomputed"] = data[TLX].mean(axis=1)
    return data


def analyze_survey(path: Path, output: Path) -> pd.DataFrame:
    """Write cleaned survey data and summary tables."""
    data = load_survey(path)
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "cleaned_survey.csv", index=False)
    low = data.loc[data.density == "low", "TLX_recomputed"].dropna()
    high = data.loc[data.density == "high", "TLX_recomputed"].dropna()
    rows = [{"family": "descriptive", "analysis": "TLX low density", "n": len(low), "mean": low.mean(), "median": low.median()},
            {"family": "descriptive", "analysis": "TLX high density", "n": len(high), "mean": high.mean(), "median": high.median()}]
    if len(low) >= 2 and len(high) >= 2:
        statistic, pvalue = stats.mannwhitneyu(low, high, alternative="two-sided")
        rows.append({"family": "between-group", "analysis": "TLX density Mann-Whitney U", "n": len(low) + len(high), "statistic": statistic, "p_value": pvalue})
    for construct, columns in RELATED.items():
        for condition, column in zip(("implicit (S3)", "explicit (S4)"), columns):
            values = data[column].dropna().to_numpy(float)
            row = {"family": "one-sample", "analysis": f"{construct}: {condition} vs neutral (3)", "n": len(values),
                   "mean": values.mean() if len(values) else np.nan, "median": np.median(values) if len(values) else np.nan}
            if len(values) >= 2:
                row["statistic"], row["p_value"] = stats.ttest_1samp(values, 3.0)
                if np.any(values != 3):
                    row["wilcoxon_statistic"], row["wilcoxon_p_value"] = stats.wilcoxon(values - 3.0)
            rows.append(row)
    for label, (left, right) in SPEARMAN.items():
        values = data[[left, right]].dropna()
        row = {"family": "exploratory", "analysis": label, "n": len(values)}
        if len(values) >= 2:
            row["statistic"], row["p_value"] = stats.spearmanr(values[left], values[right])
        rows.append(row)
    # Bowker's test of symmetry for paired nominal robot preferences.
    preference = data[["PostS3_Q5_S3Preference", "PostS4_Q5_FinalPreference"]].dropna().astype(int)
    categories = [1, 2, 3, 4]
    table = pd.crosstab(preference.iloc[:, 0], preference.iloc[:, 1]).reindex(index=categories, columns=categories, fill_value=0)
    chi_square, degrees = 0.0, 0
    for first in range(len(categories)):
        for second in range(first + 1, len(categories)):
            forward, reverse = table.iloc[first, second], table.iloc[second, first]
            if forward + reverse:
                chi_square += (forward - reverse) ** 2 / (forward + reverse)
                degrees += 1
    rows.append({"family": "paired nominal", "analysis": "Robot preference S3 vs S4: Bowker symmetry", "n": len(preference),
                 "statistic": chi_square if degrees else np.nan, "degrees_freedom": degrees,
                 "p_value": stats.chi2.sf(chi_square, degrees) if degrees else np.nan})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "survey_summary.csv", index=False)
    return summary
