"""Serve the local portfolio demo and score customers with the saved pipeline."""
import argparse
import csv
import io
import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from src import feature_engineering as fe, preprocessing as prep, risk_scoring as risk

APP_DIR = Path(__file__).resolve().parent
MAX_BYTES = 2_000_000
MAX_ROWS = 10_000
CHARGE_CUTS = [-np.inf, 35.50, 70.35, 89.85, np.inf]
FIELDS = prep.ORIGINAL_CATEGORICAL + prep.ORIGINAL_NUMERIC
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/main.js": ("main.js", "text/javascript; charset=utf-8"),
}


@lru_cache(maxsize=1)
def model():
    return risk.load_and_verify_model()


@lru_cache(maxsize=1)
def settings():
    encoder = model().named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    categories = {
        name: [str(value) for value in values]
        for name, values in zip(prep.CATEGORICAL_FEATURES, encoder.categories_, strict=True)
        if name in prep.ORIGINAL_CATEGORICAL
    }
    example = {
        "gender": "Male", "SeniorCitizen": "0", "Partner": "No", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "tenure": 3, "MonthlyCharges": 84.50,
        "TotalCharges": 253.50,
    }
    return {"categories": categories, "example": example, "threshold": risk.FLAG_THRESHOLD}


def score(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [field for field in FIELDS if field not in frame.columns]
    if missing:
        raise ValueError("Missing fields: " + ", ".join(missing))
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError(f"Provide between 1 and {MAX_ROWS:,} customer rows.")

    features = frame[FIELDS].copy()
    for column, values in settings()["categories"].items():
        cleaned = features[column].astype("string").str.strip()
        invalid = ~cleaned.isin(values)
        if invalid.any():
            row = np.flatnonzero(invalid)[0] + 1
            raise ValueError(f"Customer row {row}: {column} must be one of {', '.join(values)}.")
        features[column] = cleaned.astype("int64") if column == "SeniorCitizen" else cleaned
    for column in prep.ORIGINAL_NUMERIC:
        cleaned = features[column].astype("string").str.strip()
        if column == "TotalCharges":
            zero_tenure = features["tenure"].eq(0)
            cleaned = cleaned.mask(zero_tenure & (cleaned.isna() | cleaned.eq("")), "0")
        numeric = pd.to_numeric(cleaned, errors="coerce").astype(float)
        invalid = ~np.isfinite(numeric) | numeric.lt(0)
        if column == "tenure":
            invalid |= numeric.ne(numeric.round())
        if invalid.any():
            row = np.flatnonzero(invalid)[0] + 1
            units = "whole number" if column == "tenure" else "number"
            raise ValueError(f"Customer row {row}: {column} must be a nonnegative {units}.")
        features[column] = numeric

    # Fixed cutpoints reproduce the feature categories used by the frozen model.
    features["TenureGroup"] = pd.cut(features["tenure"], fe.TENURE_BINS, labels=fe.TENURE_LABELS)
    features["MonthlyChargeBand"] = pd.cut(features["MonthlyCharges"], CHARGE_CUTS, labels=fe.MC_BAND_LABELS)
    features, _ = fe.add_service_count(features)
    features = fe.add_has_security_support(features)
    features, _ = fe.add_avg_monthly_charge(features)
    scores = model().predict_proba(features[prep.ALL_CANDIDATE_FEATURES])[:, 1]
    result = pd.DataFrame({
        "risk_score": scores,
        "risk_segment": risk.assign_risk_segment(scores),
        "flag_at_0_40": (scores >= risk.FLAG_THRESHOLD).astype(int),
    })
    if prep.ID_COL in frame.columns:
        result.insert(0, prep.ID_COL, frame[prep.ID_COL].fillna("").to_numpy())
    return result


def batch(body: bytes) -> pd.DataFrame:
    rows = csv.reader(io.StringIO(body.decode("utf-8-sig"), newline=""), strict=True)
    headers = [name.strip() for name in next(rows, [])]
    if not headers or len(set(headers)) != len(headers):
        raise ValueError("Provide a CSV with unique column names in the header.")
    records = []
    for row in rows:
        if not row:
            continue
        if len(row) != len(headers):
            raise ValueError(f"Customer row {len(records) + 1} has the wrong number of columns.")
        records.append(row)
        if len(records) > MAX_ROWS:
            raise ValueError(f"CSV contains more than {MAX_ROWS:,} customer rows.")
    return score(pd.DataFrame(records, columns=headers))


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, content_type, body, filename=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def json_reply(self, value, status=200):
        self.reply(status, "application/json; charset=utf-8", json.dumps(value).encode("utf-8"))

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/settings":
            self.json_reply(settings())
        elif path == "/example.csv":
            frame = pd.DataFrame([settings()["example"]], columns=FIELDS)
            self.reply(200, "text/csv; charset=utf-8", frame.to_csv(index=False).encode("utf-8"), "customer-example.csv")
        elif path in STATIC:
            file, content_type = STATIC[path]
            self.reply(200, content_type, (APP_DIR / file).read_bytes())
        else:
            self.json_reply({"error": "Page not found."}, 404)

    def do_POST(self):
        path = urlsplit(self.path).path
        if path not in {"/api/score", "/api/batch"}:
            self.json_reply({"error": "Page not found."}, 404)
            return
        expected = "application/json" if path == "/api/score" else "text/csv"
        if self.headers.get_content_type() != expected:
            self.json_reply({"error": f"Send the request as {expected}."}, 415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= MAX_BYTES:
                self.json_reply({"error": "Provide a request under 2 MB."}, 413)
                return
            body = self.rfile.read(length)
            if path == "/api/score":
                record = json.loads(body)
                if not isinstance(record, dict):
                    raise ValueError("Provide one customer profile.")
                self.json_reply(score(pd.DataFrame([record])).iloc[0].to_dict())
                return
            result = batch(body)
            if prep.ID_COL in result.columns:
                # Make identifiers safe to open in spreadsheet programs.
                result[prep.ID_COL] = result[prep.ID_COL].map(
                    lambda value: "'" + value if value.lstrip().startswith(("=", "+", "-", "@"))
                    or value.startswith(("\t", "\r", "\n")) else value
                )
            self.reply(200, "text/csv; charset=utf-8", result.to_csv(index=False).encode("utf-8"), "customer-risk-scores.csv")
        except (ValueError, csv.Error) as error:
            self.json_reply({"error": str(error)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    settings()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Customer Churn Scoring is running at http://127.0.0.1:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
