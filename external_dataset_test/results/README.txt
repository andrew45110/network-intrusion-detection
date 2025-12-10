Test Results Directory
======================

All evaluation results and visualizations will be saved here.

After running test_external.py, you'll find:

Files Generated:
----------------
1. evaluation_results_[timestamp].txt
   - Detailed metrics report
   - Accuracy, AUC, precision, recall
   - Confusion matrix
   - Classification report

2. confusion_matrix_external.png
   - Visual heatmap of predictions vs actual
   - Shows true/false positives/negatives

3. roc_curve_external.png
   - ROC curve with AUC score
   - Shows model discrimination ability

4. precision_recall_external.png
   - Precision-Recall curve
   - Important for imbalanced datasets

Reading the Results:
--------------------

GOOD PERFORMANCE:
✓ Accuracy > 85%
✓ AUC > 0.90
✓ Balanced precision & recall
✓ Low false negatives (missed attacks)

POOR PERFORMANCE:
✗ Accuracy < 70%
✗ AUC < 0.80
✗ High confusion matrix off-diagonal values
✗ Predicts only one class

What It Means:
--------------
- High accuracy on external data = Model generalizes well ✓
- Low accuracy = Model overfit to training data ✗
- Compare with training results (99.96%) to assess generalization

