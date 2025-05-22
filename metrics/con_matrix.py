import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows
import os

# Define confusion matrices for each exercise
conf_matrices = {
    'Cylindrical Grasp': {'TP': 37, 'TN': 33, 'FP': 7, 'FN': 3},
    'OpenHand Pose': {'TP': 36, 'TN': 33, 'FP': 7, 'FN': 4},
    'Sequential Thumb-to-Finger Opposition Task': {'TP': 28, 'TN': 40, 'FP': 0, 'FN': 12},
}

# Storage for calculated metrics
metrics = {}

# Calculate classification metrics
for name, values in conf_matrices.items():
    TP = values['TP']
    TN = values['TN']
    FP = values['FP']
    FN = values['FN']
    total = TP + TN + FP + FN

    accuracy = (TP + TN) / total
    misclassification_rate = (FP + FN) / total
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    metrics[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1 Score': f1_score,
        'Misclassification Rate': misclassification_rate,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Total': total
    }

# Create DataFrames for export
metrics_df = pd.DataFrame(metrics).T
confusion_matrix_df = pd.DataFrame({
    'Exercise': list(conf_matrices.keys()),
    'TP': [conf_matrices[ex]['TP'] for ex in conf_matrices.keys()],
    'TN': [conf_matrices[ex]['TN'] for ex in conf_matrices.keys()],
    'FP': [conf_matrices[ex]['FP'] for ex in conf_matrices.keys()],
    'FN': [conf_matrices[ex]['FN'] for ex in conf_matrices.keys()]
})

# Export to Excel with multiple sheets
output_filename = 'classification_analysis_results.xlsx'

with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    # Sheet 1: Complete Metrics Summary
    metrics_df.round(4).to_excel(writer, sheet_name='Metrics Summary', index=True)
    
    # Sheet 2: Confusion Matrix Raw Data
    confusion_matrix_df.to_excel(writer, sheet_name='Confusion Matrices', index=False)
    
    # Sheet 3: Individual Exercise Details
    detailed_results = []
    for name, values in conf_matrices.items():
        detailed_results.append({
            'Exercise': name,
            'True Positives (TP)': values['TP'],
            'True Negatives (TN)': values['TN'],
            'False Positives (FP)': values['FP'],
            'False Negatives (FN)': values['FN'],
            'Total Samples': values['TP'] + values['TN'] + values['FP'] + values['FN'],
            'Accuracy': metrics[name]['Accuracy'],
            'Precision': metrics[name]['Precision'],
            'Recall (Sensitivity)': metrics[name]['Recall'],
            'Specificity': metrics[name]['Specificity'],
            'F1 Score': metrics[name]['F1 Score'],
            'Misclassification Rate': metrics[name]['Misclassification Rate']
        })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_excel(writer, sheet_name='Detailed Results', index=False)

# Format the Excel file
wb = Workbook()
wb = pd.ExcelFile(output_filename).book if os.path.exists(output_filename) else wb

print(f"Results exported to '{output_filename}' with multiple sheets:")
print("- 'Metrics Summary': All calculated metrics for each exercise")
print("- 'Confusion Matrices': Raw confusion matrix values")
print("- 'Detailed Results': Comprehensive results table")

# --- Grouped Bar Chart ---
metric_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score']
x = np.arange(len(metric_names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, (exercise, metric) in enumerate(metrics.items()):
    values = [metric[m] for m in metric_names]
    ax.bar(x + i * width, values, width, label=exercise, color=colors[i], alpha=0.8)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Comparison of Classification Metrics Across Exercises', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metric_names, rotation=45, ha='right')
ax.set_ylim(0, 1.1)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('metrics_comparison_bar_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Line Chart for Comparison ---
metrics_df_line = metrics_df[metric_names]

plt.figure(figsize=(12, 7))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']

for i, exercise in enumerate(metrics_df_line.index):
    plt.plot(metric_names, metrics_df_line.loc[exercise], 
             marker=markers[i], label=exercise, linewidth=2.5, 
             markersize=8, color=colors[i])

plt.title('Metric Trends Across Exercises', fontsize=14, fontweight='bold')
plt.xlabel('Metric', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1.1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('metrics_trend_line_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Confusion Matrix Heatmaps ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, values) in enumerate(conf_matrices.items()):
    matrix = np.array([[values['TP'], values['FN']],
                       [values['FP'], values['TN']]])
    labels = ['Positive', 'Negative']
    df_cm = pd.DataFrame(matrix, index=labels, columns=labels)

    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', 
                ax=axes[idx], cbar_kws={'shrink': 0.8})
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_title(f'Confusion Matrix\n{name}', fontsize=10)

plt.tight_layout()
plt.savefig('confusion_matrices_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Print summary table ---
print("\nClassification Metrics Summary:")
print("=" * 80)
print(metrics_df[metric_names].round(3).to_string())

print(f"\n\nFiles created:")
print(f"- {output_filename} (Excel file with all results)")
print("- metrics_comparison_bar_chart.png")
print("- metrics_trend_line_chart.png") 
print("- confusion_matrices_heatmap.png")

# Display basic statistics
print(f"\nBasic Statistics:")
print(f"- Number of exercises analyzed: {len(conf_matrices)}")
print(f"- Total samples across all exercises: {sum(metrics[ex]['Total'] for ex in metrics.keys())}")
print(f"- Average accuracy across exercises: {metrics_df['Accuracy'].mean():.3f}")
print(f"- Best performing exercise (by F1 Score): {metrics_df['F1 Score'].idxmax()}")
print(f"- Highest F1 Score: {metrics_df['F1 Score'].max():.3f}")