"""Build the editable Power BI project from the existing churn snapshot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "powerbi_dataset.csv"
REPORT = ROOT / "Churn Retention.Report"
MODEL = ROOT / "Churn Retention.SemanticModel"
DEFINITION = REPORT / "definition"
SCHEMAS = "https://developer.microsoft.com/json-schemas/fabric/"
REPORT_SCHEMA = "item/report/definition/"
TABLE = "powerbi_dataset"
WIDTH, HEIGHT = 1280, 780
CSV_PLACEHOLDER = r"C:\path\to\powerbi_dataset.csv"

INTEGER = {"Churn", "predicted_churn_at_0.40", "tenure", "SeniorCitizen", "ServiceCount", "HasSecuritySupport"}
DECIMAL = {"MonthlyCharges", "TotalCharges", "AvgMonthlyCharge", "churn_probability"}
COUNT = "#,0"
PERCENT = "0.00%"
MEASURES = {
    "Customers": (f"COUNTROWS ( {TABLE} )", COUNT),
    "Churned customers": (f"CALCULATE ( [Customers], KEEPFILTERS ( {TABLE}[Churn] = 1 ) )", COUNT),
    "Observed churn rate": ("DIVIDE ( [Churned customers], [Customers] )", PERCENT),
    "Flagged at 0.40": (f"CALCULATE ( [Customers], KEEPFILTERS ( {TABLE}[predicted_churn_at_0.40] = 1 ) )", COUNT),
    "Observed churn among flags": (
        f"DIVIDE ( CALCULATE ( [Churned customers], KEEPFILTERS ( {TABLE}[predicted_churn_at_0.40] = 1 ) ), [Flagged at 0.40] )",
        PERCENT,
    ),
    "High and Critical customers": (
        f'CALCULATE ( [Customers], KEEPFILTERS ( {TABLE}[risk_segment] IN {{ "High", "Critical" }} ) )',
        COUNT,
    ),
    "High and Critical monthly billings": (
        f'CALCULATE ( SUM ( {TABLE}[MonthlyCharges] ), KEEPFILTERS ( {TABLE}[risk_segment] IN {{ "High", "Critical" }} ) )',
        '$#,0.00',
    ),
}
DISPLAY = {
    "customerID": "Customer ID", "risk_segment": "Risk band", "churn_probability": "Risk score",
    "tenure": "Tenure (months)", "InternetService": "Internet service", "MonthlyCharges": "Monthly charges",
    "predicted_churn_at_0.40": "Flag at 0.40", "TenureGroup": "Tenure group",
    "PaymentMethod": "Payment method", "ServiceCount": "Service count", "OnlineSecurity": "Online security",
    "TechSupport": "Tech support", "MonthlyChargeBand": "Monthly charge band",
}


def write(path, values, schema=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if schema:
        values = {"$schema": SCHEMAS + schema + "/schema.json", **values}
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def m_string(value):
    return '"' + value.replace('"', '""') + '"'


def csv_path_expression(path):
    return m_string(str(path)) + ' meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]'


def configure_csv():
    path = MODEL / "model.bim"
    definition = json.loads(path.read_text(encoding="utf-8"))
    parameter = next(entry for entry in definition["model"]["expressions"] if entry["name"] == "ChurnCsvPath")
    parameter["expression"] = csv_path_expression(CSV)
    write(path, definition)


def build_model():
    with CSV.open(encoding="utf-8-sig", newline="") as stream:
        columns = next(csv.reader(stream))
    extra = ["Historical status", "Tenure order", "Risk order", "Charge band order"]
    names = columns + extra
    integer = INTEGER | set(extra[1:])
    model_columns = []
    for name in names:
        data_type = "int64" if name in integer else "double" if name in DECIMAL else "string"
        column = {"name": name, "sourceColumn": name, "dataType": data_type, "summarizeBy": "none"}
        if name in extra[1:]:
            column["isHidden"] = True
        if name in {"TenureGroup", "risk_segment", "MonthlyChargeBand"}:
            column["sortByColumn"] = dict(zip(("TenureGroup", "risk_segment", "MonthlyChargeBand"), extra[1:]))[name]
        if name == "churn_probability":
            column["formatString"] = "0.000"
        if name == "MonthlyCharges":
            column["formatString"] = "$#,0.00"
        model_columns.append(column)
    m_types = {"int64": "Int64.Type", "double": "type number", "string": "type text"}
    types = ", ".join(
        "{" + m_string(column["name"]) + ", " + m_types[column["dataType"]] + "}"
        for column in model_columns if column["name"] in columns
    )
    transform = [
        "let",
        f"    Raw = Csv.Document(File.Contents(ChurnCsvPath), [Delimiter=\",\", Columns={len(columns)}, Encoding=65001, QuoteStyle=QuoteStyle.Csv]),",
        "    Headers = Table.PromoteHeaders(Raw, [PromoteAllScalars=true]),",
        '    Typed = Table.TransformColumnTypes(Headers, {' + types + '}, "en-US"),',
        '    Outcome = Table.AddColumn(Typed, "Historical status", each if [Churn] = 1 then "Churned" else "Retained", type text),',
    ]
    orders = [
        ("TenureGroup", "Tenure order", ["≤6 months", "7–12 months", "13–24 months", "25+ months"]),
        ("risk_segment", "Risk order", ["Low", "Medium", "High", "Critical"]),
        ("MonthlyChargeBand", "Charge band order", ["Low", "Medium-Low", "Medium-High", "High"]),
    ]
    previous = "Outcome"
    for index, (field, name, labels) in enumerate(orders, 1):
        label_list = "{" + ", ".join(map(m_string, labels)) + "}"
        step = f"Sorted{index}"
        transform.append(f"    {step} = Table.AddColumn({previous}, {m_string(name)}, each List.PositionOf({label_list}, [{field}]), Int64.Type),")
        previous = step
    transform[-1] = transform[-1].removesuffix(",")
    transform.extend(["in", f"    {previous}"])
    measures = [{"name": name, "expression": dax, "formatString": fmt} for name, (dax, fmt) in MEASURES.items()]
    partition = {"name": TABLE, "mode": "import", "source": {"type": "m", "expression": transform}}
    parameter = {
        "name": "ChurnCsvPath", "kind": "m",
        "expression": csv_path_expression(CSV_PLACEHOLDER),
    }
    write(MODEL / "definition.pbism", {"version": "4.2", "settings": {"qnaEnabled": False}},
          "item/semanticModel/definitionProperties/1.0.0")
    write(MODEL / "model.bim", {
        "name": "Churn Retention", "compatibilityLevel": 1600,
        "model": {
            "culture": "en-US", "defaultPowerBIDataSourceVersion": "powerBI_V3",
            "sourceQueryCulture": "en-US", "expressions": [parameter],
            "tables": [{"name": TABLE, "columns": model_columns, "measures": measures, "partitions": [partition]}],
        },
    })


def expression(kind, name):
    return {kind: {"Expression": {"SourceRef": {"Entity": TABLE}}, "Property": name}}


def project(kind, name, active=False):
    label = DISPLAY.get(name, name)
    result = {"field": expression(kind, name), "queryRef": f"{TABLE}.{name}", "nativeQueryRef": label}
    if label != name:
        result["displayName"] = label
    if active:
        result["active"] = True
    return result


def literal(value):
    return {"expr": {"Literal": {"Value": value}}}


def color(value):
    return {"solid": {"color": literal(f"'{value}'")}}


def properties(**values):
    return [{"properties": values}]


def stable_id(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


class Page:
    def __init__(self, title):
        self.name = stable_id(title)
        self.root = DEFINITION / "pages" / self.name
        self.index = 0
        write(self.root / "page.json", {
            "name": self.name, "displayName": title, "displayOption": "FitToPage",
            "width": WIDTH, "height": HEIGHT,
            "objects": {"background": properties(color=color("#EEF3F1"), transparency=literal("0D"))},
        }, REPORT_SCHEMA + "page/2.0.0")

    def add(self, key, bounds, visual_type, roles=None, title=None, sort=None, objects=None):
        self.index += 1
        name = stable_id(f"{self.name}/{key}")
        x, y, width, height = bounds
        visual = {"visualType": visual_type}
        if roles:
            query = {"queryState": {role: {"projections": fields} for role, fields in roles.items()}}
            if sort:
                kind, field, direction = sort
                query["sortDefinition"] = {
                    "sort": [{"field": expression(kind, field), "direction": direction}],
                    "isDefaultSort": False,
                }
            visual["query"] = query
        if objects:
            visual["objects"] = objects
        if title:
            visual["visualContainerObjects"] = {
                "title": properties(show=literal("true"), text=literal("'" + title.replace("'", "''") + "'"),
                                    bold=literal("true"), fontColor=color("#122B2B")),
                "background": properties(show=literal("true"), color=color("#FFFFFF"), transparency=literal("0D")),
            }
        write(self.root / "visuals" / name / "visual.json", {
            "name": name,
            "position": {"x": x, "y": y, "width": width, "height": height,
                         "z": self.index * 1000, "tabOrder": self.index * 1000},
            "visual": visual,
        }, REPORT_SCHEMA + "visualContainer/2.9.0")

    def header(self, title, subtitle):
        paragraphs = []
        for value, size, text_color, font in ((title, 29, "#122B2B", "Segoe UI Semibold"),
                                               (subtitle, 15, "#49615D", "Segoe UI")):
            paragraphs.append({"textRuns": [{"value": value, "textStyle": {
                "fontFamily": font, "fontSize": f"{size}px", "color": text_color,
            }}], "horizontalTextAlignment": "left"})
        self.add("header", (24, 17, 1232, 72), "textbox", objects={"general": properties(paragraphs=paragraphs)})

    def cards(self, *measures):
        self.add("metrics", (24, 94, 1232, 103), "cardVisual", {"Data": [project("Measure", name) for name in measures]})

    def slicer(self, column, bounds):
        self.add("slicer-" + column, bounds, "slicer", {"Values": [project("Column", column, True)]},
                 title=DISPLAY.get(column, column),
                 objects={"data": properties(mode=literal("'Dropdown'"))})

    def chart(self, column, measure, bounds, title, *, rate_order=False, donut=False):
        roles = {"Category": [project("Column", column, True)], "Y": [project("Measure", measure)]}
        if measure == "Observed churn rate":
            roles["Tooltips"] = [project("Measure", "Customers"), project("Measure", "Churned customers")]
        sort = ("Measure", measure, "Descending") if rate_order else ("Column", column, "Ascending")
        visual_type = "donutChart" if donut else "clusteredColumnChart" if column == "ServiceCount" else "clusteredBarChart"
        self.add(column, bounds, visual_type, roles, title=title, sort=sort)


def build_report():
    write(ROOT / "Churn Retention.pbip", {
        "version": "1.0", "artifacts": [{"report": {"path": REPORT.name}}],
        "settings": {"enableAutoRecovery": True},
    }, "pbip/pbipProperties/1.0.0")
    write(REPORT / "definition.pbir", {
        "version": "4.0", "datasetReference": {"byPath": {"path": "../" + MODEL.name}},
    }, "item/report/definitionProperties/2.0.0")
    theme = "CY26SU02"
    theme_path = f"BaseThemes/{theme}.json"
    write(REPORT / "StaticResources" / "SharedResources" / theme_path, {
        "name": theme,
        "dataColors": ["#17674F", "#D28A46", "#648F8C", "#325463", "#A85B47", "#627653"],
        "background": "#EEF3F1", "foreground": "#122B2B", "tableAccent": "#17674F",
    })
    write(DEFINITION / "report.json", {
        "themeCollection": {"baseTheme": {
            "name": theme, "type": "SharedResources",
            "reportVersionAtImport": {"visual": "2.9.0", "report": "3.2.0", "page": "2.0.0"},
        }},
        "resourcePackages": [{"name": "SharedResources", "type": "SharedResources", "items": [
            {"name": theme, "path": theme_path, "type": "BaseTheme"},
        ]}],
        "settings": {"defaultFilterActionIsDataFilter": True},
    }, REPORT_SCHEMA + "report/3.2.0")
    write(DEFINITION / "version.json", {"version": "2.0.0"}, REPORT_SCHEMA + "versionMetadata/1.0.0")

    overview = Page("Executive Overview")
    overview.header("Customer churn overview", "Historical telecom snapshot • Filter by contract, tenure group, internet service or risk band.")
    overview.cards("Customers", "Churned customers", "Observed churn rate", "High and Critical customers")
    for index, column in enumerate(("Contract", "TenureGroup", "InternetService", "risk_segment")):
        overview.slicer(column, (24 + 312 * index, 208, 296, 62))
    overview.chart("Historical status", "Customers", (24, 284, 608, 228), "Historical outcomes", donut=True)
    overview.chart("risk_segment", "Customers", (648, 284, 608, 228), "Customers by risk band", donut=True)
    overview.chart("Contract", "Observed churn rate", (24, 528, 608, 228), "Observed churn by contract", rate_order=True)
    overview.chart("TenureGroup", "Observed churn rate", (648, 528, 608, 228), "Observed churn by tenure")

    drivers = Page("Churn Drivers")
    drivers.header("Churn drivers", "Observed associations in the historical snapshot. Use them to choose hypotheses for a retention test.")
    for index, (column, label) in enumerate((
        ("Contract", "contract"), ("TenureGroup", "tenure group"), ("InternetService", "internet service"),
        ("PaymentMethod", "payment method"), ("MonthlyChargeBand", "monthly charge band"),
        ("ServiceCount", "service count"), ("OnlineSecurity", "online security"), ("TechSupport", "tech support"),
    )):
        drivers.chart(column, "Observed churn rate",
                      (24 + (index % 2) * 624, 94 + (index // 2) * 169, 608, 153),
                      f"Observed churn by {label}", rate_order=column not in {"TenureGroup", "MonthlyChargeBand", "ServiceCount"})

    customer = Page("Customer Risk")
    customer.header("Customer risk ranking", "Historical scores include training rows. Scores are uncalibrated and the table is a retrospective view.")
    customer.cards("High and Critical customers", "Flagged at 0.40", "Observed churn among flags", "High and Critical monthly billings")
    customer.slicer("risk_segment", (24, 208, 296, 62))
    customer.slicer("Historical status", (336, 208, 296, 62))
    customer.slicer("Contract", (648, 208, 296, 62))
    customer.slicer("InternetService", (960, 208, 296, 62))
    customer.chart("risk_segment", "Customers", (24, 284, 350, 472), "Customers by risk band")
    customer.add("ranked-customers", (390, 284, 866, 472), "tableEx", {"Values": [
        project("Column", name) for name in ("customerID", "risk_segment", "churn_probability",
                                            "predicted_churn_at_0.40", "Historical status", "Contract",
                                            "tenure", "InternetService", "MonthlyCharges")
    ]}, "Historical customers by score", ("Column", "churn_probability", "Descending"),
        objects={"columnHeaders": properties(autoSizeColumnWidth=literal("true"),
                                             columnAdjustment=literal("'growToFit'"))})

    write(DEFINITION / "pages" / "pages.json", {
        "pageOrder": [overview.name, drivers.name, customer.name], "activePageName": overview.name,
    }, REPORT_SCHEMA + "pagesMetadata/1.0.0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configure-csv", action="store_true", help="Set the CSV path for this checkout.")
    args = parser.parse_args()
    if args.configure_csv:
        configure_csv()
        print(f"ChurnCsvPath set to {CSV}")
    else:
        build_model()
        build_report()
        print(f"Power BI project written to {ROOT / 'Churn Retention.pbip'}")
