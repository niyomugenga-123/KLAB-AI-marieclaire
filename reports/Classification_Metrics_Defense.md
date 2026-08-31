# Classification Metrics Defense: Premium Car Listing Predictor

## Problem Statement
Carvana has 9,317 cleaned car listings, with 9.8% premium cars (≥$30,000) and 90.2% regular cars. We built a logistic regression classifier using Age and Miles to predict which cars are premium.

## The Accuracy Trap
A baseline model that always predicts "regular" achieves 94.2% accuracy but finds zero premium cars. Our model achieves 94.2% accuracy, but for the right reasons — it actually identifies premium listings.

## Results
- **Accuracy:** 94.2% (misleading on imbalanced data)
- **Precision:** 80% (when it says premium, it's correct 4 out of 5 times)
- **Recall:** 1.5% (finds only 4 out of 260 actual premium cars)
- **F1:** 3% (low due to precision-recall tradeoff)
- **ROC-AUC:** 80.4% (good ranking quality)

## Metric Choice: Recall

**Why Recall Matters Most for This Business Problem:**

1. **Misses are expensive:** A premium car incorrectly classified as regular gets underpriced and undersold. Missing even one high-value listing directly costs the dealership revenue and customer satisfaction.

2. **False alarms are cheap:** When the model predicts "premium" (80% accuracy), a human reviewer can quickly verify it. Checking 5 predictions to confirm 4 correct premium cars is a small operational cost.

3. **Inventory discovery impact:** With only 260 premium cars in the test set, improving recall from 1.5% to even 5% would mean finding 13 instead of 4 premium listings — a significant increase in high-value inventory visibility.

## Confusion Matrix Analysis
- True Negatives: 4,139
- False Positives: 1
- False Negatives: 256
- True Positives: 4

The 256 false negatives (missed premium cars) represent lost revenue opportunities, while the single false positive is easily caught by a quick human check.

## Conclusion
Recall should be the priority metric because in e-commerce, missing high-value inventory is more costly than occasional false alarms. A dealership would rather verify a few extra predictions than risk losing premium listings to competitors.

---
**Dataset:** Carvana car listings (21,000 samples)
**Model:** Logistic Regression | **Features:** Age, Miles
**Date:** Week 2, Day 3 - TekHer AI Bootcamp