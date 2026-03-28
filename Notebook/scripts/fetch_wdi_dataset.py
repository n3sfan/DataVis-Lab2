import pandas as pd
import requests
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from typing import TypedDict

# ============================================================
# 1. GLOBAL CONFIG – thêm/sửa/xóa chỉ số TẠI ĐÂY
# ============================================================
COUNTRIES = [
    "VNM", "THA", "IDN", "PHL", "IND", "BGD", "PAK", "CHN", "JPN", "KOR",
    "DEU", "FRA", "GBR", "SWE", "USA", "CAN", "BRA", "MEX", "ZAF", "NGA",
]

YEARS = list(range(2000, 2024))          # 2000 – 2023 (recent years, <= 2023)

# INDICATOR_GROUPS: concept_name → [primary_code, fallback1, fallback2, ...]
# Chương trình sẽ dùng code đầu tiên có dữ liệu; ghi nhận fallback ra file audit.
INDICATOR_GROUPS = {
    # Giáo dục
    "School enrollment, tertiary (% gross)":                     ["SE.TER.ENRR"],
    # # # Nguồn nhân lực
    # "Unemployment, total (% of total labor force) (modeled ILO estimate)": ["SL.UEM.TOTL.ZS"],
    # "Employment in services (% of total employment) (modeled ILO estimate)": ["SL.SRV.EMPL.ZS"],
    # # Nữ giới trong chính trị
    # "Proportion of seats held by women in national parliaments (%)":  ["SG.GEN.PARL.ZS"],
    # # Đầu tư giáo dục & y tế
    # "Government expenditure on education, total (% of GDP)":          ["SE.XPD.TOTL.GD.ZS", "SE.XPD.TOTL.GB.ZS"],
    # "Domestic general government health expenditure (% of general government expenditure)": ["SH.XPD.GHED.GE.ZS", "SH.XPD.GHED.GD.ZS", "SH.XPD.CHEX.GD.ZS"],
    # # Kinh tế
    # "GDP per capita (current US$)":                              ["NY.GDP.PCAP.CD"],
}
# ============================================================


BASE_URL  = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"
BATCH_SIZE = 5          # số quốc gia / request để tránh timeout
MAX_RETRIES = 3
REQ_TIMEOUT = (10, 120)  # (connect, read) giây


def _fetch_page(countries_batch, indicator, retries=MAX_RETRIES):
    """Fetch one page, retry on timeout."""
    url = BASE_URL.format(countries=";".join(countries_batch), indicator=indicator)
    params = {"format": "json", "per_page": 5000}
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=REQ_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ReadTimeout:
            if attempt < retries - 1:
                continue
            raise


def fetch_indicator(country_list, indicator_code):
    """Fetch all records for one indicator across all countries (batch to avoid timeout)."""
    records = []
    for i in range(0, len(country_list), BATCH_SIZE):
        batch = country_list[i : i + BATCH_SIZE]
        payload = _fetch_page(batch, indicator_code)
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            continue
        for rec in payload[1]:
            year = int(rec["date"])
            if year not in YEARS:
                continue
            records.append({
                "Country Name":  rec["country"]["value"],
                "Country Code":  rec["countryiso3code"],
                "Series Name":   rec["indicator"]["value"],
                "Series Code":   rec["indicator"]["id"],
                "Year":         year,
                "Value":        rec["value"],
            })
    return pd.DataFrame(records)


def build_dataset():
    """Pull all indicators, resolve fallbacks, reshape to wide format."""
    all_long = []

    for concept, codes in INDICATOR_GROUPS.items():
        for code in codes:
            df = fetch_indicator(COUNTRIES, code)
            if len(df) > 0:
                all_long.append(df)

    if not all_long:
        raise RuntimeError("Khong lay duoc du lieu tu API. Kiem tra ket noi mang.")

    raw = pd.concat(all_long, ignore_index=True)

    # Bảng map Country Code → Country Name (lấy từ dữ liệu thực)
    name_map = (
        raw[["Country Code", "Country Name"]]
        .dropna()
        .drop_duplicates("Country Code")
        .set_index("Country Code")["Country Name"]
        .to_dict()
    )

    # Chọn code có dữ liệu cho từng country × concept
    out_rows, fallback_records = [], []

    for concept, codes in INDICATOR_GROUPS.items():
        for cc in COUNTRIES:
            chosen = None
            for code in codes:
                sub = raw[(raw["Country Code"] == cc) & (raw["Series Code"] == code)]
                if sub["Value"].notna().sum() > 0:
                    chosen = (sub, code, sub["Series Name"].dropna().iloc[0] if sub["Series Name"].notna().any() else concept)
                    break

            if chosen is None:
                chosen = (
                    pd.DataFrame({"Year": YEARS, "Value": [None] * len(YEARS)}),
                    codes[0], concept,
                )

            sub_df, used_code, used_name = chosen
            if used_code != codes[0]:
                fallback_records.append({
                    "Country Code":  cc,
                    "Concept":       concept,
                    "Requested":     codes[0],
                    "Resolved":      used_code,
                })

            year_val = {int(r.Year): r.Value for _, r in sub_df.iterrows()}

            row = {
                "Country Name": name_map.get(cc, cc),
                "Country Code": cc,
                "Series Name":  used_name,
                "Series Code":  used_code,
            }
            for y in YEARS:
                row[f"{y} [YR{y}]"] = year_val.get(y)
            out_rows.append(row)

    out_df = (
        pd.DataFrame(out_rows)
        .sort_values(["Country Name", "Series Code"])
        .reset_index(drop=True)
    )
    fallback_df = pd.DataFrame(fallback_records)
    return out_df, fallback_df


class ValidationRow(TypedDict):
    check: str
    value: int | float
    expected: int | str
    status: str


def build_validation_report(out_df, fallback_df):
    """Kiểm tra shape và chất lượng dữ liệu."""
    year_cols = [f"{y} [YR{y}]" for y in YEARS]
    rows: list[ValidationRow] = []

    # Shape
    rows.append({"check": "row_count",         "value": len(out_df),                   "expected": 20 * len(INDICATOR_GROUPS), "status": ""})
    rows.append({"check": "country_count",     "value": int(out_df["Country Code"].nunique()), "expected": 20,                  "status": ""})
    rows.append({"check": "series_count",      "value": int(out_df["Series Code"].nunique()), "expected": len(INDICATOR_GROUPS),      "status": ""})
    rows.append({"check": "year_col_count",    "value": sum(c in out_df.columns for c in year_cols), "expected": len(YEARS), "status": ""})
    dup = int(out_df.duplicated(subset=["Country Code", "Series Code"]).sum())
    rows.append({"check": "duplicate_rows",     "value": dup,                    "expected": 0,  "status": ""})

    # Missing %
    for code in out_df["Series Code"].unique():
        sub = out_df[out_df["Series Code"] == code][year_cols].apply(pd.to_numeric, errors="coerce")
        pct = sub.isna().mean().mean() * 100
        rows.append({"check": f"missing_pct::{code}", "value": float(Decimal(str(pct)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)), "expected": "informational", "status": ""})

    for r in rows:
        r["status"] = "ok" if (isinstance(r["expected"], int) and r["value"] == r["expected"]) else ("warn" if r["expected"] != "informational" else "info")

    for r in rows:
        if r["check"].startswith("missing_pct") and r["value"] > 40:
            r["status"] = "warn"

    rows.append({"check": "fallback_count", "value": len(fallback_df), "expected": "informational", "status": "info"})

    return pd.DataFrame(rows)


def evaluate_missing(out_df):
    """
    Đánh giá tỷ lệ giá trị thiếu trong dataset đã tải.
    Xuất báo cáo chi tiết theo từng chỉ số và theo từng năm.

    Phù hợp với cell 1.4 "Tỷ lệ giá trị thiếu" trong Notebook.
    """
    year_cols = [f"{y} [YR{y}]" for y in YEARS]
    df_years = out_df[year_cols].apply(pd.to_numeric, errors="coerce")

    # --- Toàn cục ---
    total_cells   = df_years.size
    total_missing = int(df_years.isnull().sum().sum())
    overall_pct   = float(Decimal(str(total_missing / total_cells * 100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    print("=" * 65)
    print("BÁO CÁO TỶ LỆ GIÁ TRỊ THIẾU – WDI Dataset")
    print("=" * 65)
    print(f"  Tổng số ô dữ liệu : {total_cells}")
    print(f"  Số ô bị thiếu     : {total_missing}")
    print(f"  Tỷ lệ thiếu       : {overall_pct}%")
    print()

    # --- Theo từng chỉ số (Series Code) ---
    series_rows = []
    print("Theo từng chỉ số:")
    print("-" * 65)
    for code, grp in out_df.groupby("Series Code"):
        series_name = grp["Series Name"].iloc[0]
        vals      = grp[year_cols].apply(pd.to_numeric, errors="coerce")
        n_missing = int(vals.isnull().sum().sum())
        n_total   = vals.size
        pct       = float(Decimal(str(n_missing / n_total * 100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        bar_len   = int(pct // 2)
        status    = "WARN" if pct > 40 else "OK"
        print(f"  [{code}] {series_name}")
        print(f"    Thiếu: {n_missing}/{n_total} ô ({pct}%)  {'#' * bar_len}")
        series_rows.append({
            "Series Code":        code,
            "Series Name":        series_name,
            "Missing Count":       n_missing,
            "Total Cells":         n_total,
            "Missing %":           pct,
            "Status":              status,
        })
        print()
    print()

    # --- Theo từng năm (cột) ---
    missing_per_col = df_years.isnull().sum()
    pct_per_col     = (missing_per_col / len(df_years) * 100).round(2)

    print("Theo từng năm (cột):")
    print("-" * 65)
    year_rows = []
    for col, n, p in zip(year_cols, missing_per_col, pct_per_col):
        bar_len = int(p // 2)
        status  = "WARN" if p > 4 else "OK"
        print(f"  {col} | Thiếu: {n}/{len(df_years)} chỉ số ({p}%)  {'#' * bar_len}")
        year_rows.append({
            "Year Column":     col,
            "Missing Count":   int(n),
            "Total Countries": len(df_years),
            "Missing %":       float(Decimal(str(float(p))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
            "Status":          status,
        })

    print()
    print("=" * 65)
    print("Ghi chú: STATUS = WARN nếu tỷ lệ thiếu > 4%")
    print("=" * 65)

    # Trả về DataFrame để lưu ra CSV nếu cần
    return (
        pd.DataFrame(series_rows),
        pd.DataFrame(year_rows),
    )


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Bat dau lay du lieu WDI cho 20 quoc gia, 7 chi so, 2000-2023...")
    out_df, fallback_df = build_dataset()

    report_df = build_validation_report(out_df, fallback_df)

    # --- Đánh giá tỷ lệ thiếu (cell 1.4) ---
    series_missing_df, year_missing_df = evaluate_missing(out_df)

    out_csv         = data_dir / "dataset.csv"
    qa_csv          = data_dir / "wdi_validation_report.csv"
    fallback_csv    = data_dir / "wdi_fallback_audit.csv"
    missing_by_series_csv = data_dir / "wdi_missing_by_series.csv"
    missing_by_year_csv   = data_dir / "wdi_missing_by_year.csv"

    out_df.to_csv(out_csv,      index=False, encoding="utf-8-sig")
    report_df.to_csv(qa_csv,    index=False, encoding="utf-8-sig")
    fallback_df.to_csv(fallback_csv, index=False, encoding="utf-8-sig")
    series_missing_df.to_csv(missing_by_series_csv, index=False, encoding="utf-8-sig")
    year_missing_df.to_csv(missing_by_year_csv,     index=False, encoding="utf-8-sig")

    print(f"\n[OK] dataset.csv                  – {len(out_df)} dong")
    print(f"[OK] wdi_validation_report.csv    – {len(report_df)} dong (xem cot 'status' de phat hien warn)")
    print(f"[OK] wdi_fallback_audit.csv       – {len(fallback_df)} dong (chi so duoc giai quyet bang fallback)")
    print(f"[OK] wdi_missing_by_series.csv   – {len(series_missing_df)} dong (ty le thieu theo chi so)")
    print(f"[OK] wdi_missing_by_year.csv      – {len(year_missing_df)} dong (ty le thieu theo nam)")


if __name__ == "__main__":
    main()
