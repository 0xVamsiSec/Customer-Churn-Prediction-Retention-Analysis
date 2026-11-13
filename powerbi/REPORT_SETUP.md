# Power BI report

Keep the `Churn Retention.Report` and `Churn Retention.SemanticModel` folders beside `Churn Retention.pbip`. Use a current version of Power BI Desktop with enhanced report format support.

1. From the repository root, run `python powerbi/build_report.py --configure-csv` to set the `ChurnCsvPath` Power Query parameter for your checkout. You can also set it in Desktop under **Home → Transform data → Edit parameters**.
2. Open `Churn Retention.pbip` and select **Home → Refresh** to load `powerbi_dataset.csv`.
3. Clear the slicers and check the totals below. Save in Desktop after refresh to retain the local data cache.

Running the builder without `--configure-csv` regenerates the supplied model and report layout with a placeholder CSV path. Use the flag for normal local setup.

| Page | What to check |
|---|---|
| Executive Overview | 7,043 customers, 1,869 historical churn events, 26.54% observed churn, and 2,854 High or Critical scores. Select a contract to see the visuals update. |
| Churn Drivers | Month-to-month contracts at 42.71%, two-year contracts at 2.83%, and the first six months of tenure at 52.94% observed churn. |
| Customer Risk | 3,465 flags at 0.40, 47.24% observed churn among flagged rows, and $218,856.40 in monthly billings for High and Critical rows. The customer table starts at the highest score. Use the historical status slicer to inspect retained or churned records. |

The customer table describes the historical snapshot, which contains both churned and retained records. The score is an uncalibrated ranking, and the billings card sums monthly charges for the selected segment. All chart measures apply to the current filter context.

The checked-in PBIP is editable source. The report build uses the CSV, a tabular semantic model, and native Power BI visuals. The [Microsoft project documentation](https://learn.microsoft.com/en-us/power-bi/developer/projects/projects-overview) describes how Desktop opens and saves these project files.
