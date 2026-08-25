# Assignment 2 Report — Carvana Used Car Listings

## Dataset
Carvana used-car listings from Kaggle (`data/raw/carvana.csv`), 22,000 rows with four
columns: `Name` (model), `Year`, `Miles`, and `Price`.

## Question explored
How do a car's **age** and **mileage** jointly affect its resale price — does mileage
keep dragging the price down at the same rate no matter how many miles are already on
the car, or does the effect taper off?

## What I found
After cleaning, price falls fairly steadily as mileage rises through about 80,000
miles, then the drop almost flattens out from 80,000 to 130,000 miles. Chart 1
(`reports/a2_chart1.png`) shows this: a steep slope at low mileage that gradually
levels off.

Age tells a similar story: average price drops the most in the earlier age bands (0-2
to 4-6 years), and the decline slows down for older cars. Chart 2
(`reports/a2_chart2.png`) shows this step-down by age band.

The age x mileage pivot table (Task 3) shows these two effects overlap — mileage is
doing at least as much work as age in setting price, not just standing in for it.

## Cleaning decisions (summary)
- `Name`: stripped leading whitespace present on every value.
- `Year`: ~13% of rows had corrupted values (5-8 digits, e.g. `20173`) that lined up
  with truncated model names (e.g. `"BMW X"` instead of `"BMW X3"`). Truncating to the
  first 4 digits recovered a value inside the same 2009-2023 range for every corrupted
  row, so that was the fix rather than dropping ~2,850 rows.
- Duplicates: 12,683 fully duplicated rows were dropped, leaving 9,317 rows.

## Limitation
The `Name` field is truncated for a subset of rows (trim/model info missing), which was
the root cause of the Year corruption. That means some model-level comparisons in Task
3 (`model_avg_price`) group different trims of the same base model together.

## Reflection
See `reports/weekend-a2-reflection.md`.