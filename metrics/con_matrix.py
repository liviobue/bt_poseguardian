import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows
import os

# Set global matplotlib parameters for better readability
plt.rcParams.update({
    'font.size': 16,           # Base font size
    'axes.titlesize': 22,      # Title font size
    'axes.labelsize': 20,      # Axis label font size
    'xtick.labelsize': 16,     # X-tick font size
    'ytick.labelsize': 16,     # Y-tick font size
    'legend.fontsize': 16,     # Legend font size
    'figure.titlesize': 24,    # Figure title font size
    'font.family': 'DejaVu Sans'  # Clean, readable font
})

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

# --- IMPROVED Grouped Bar Chart ---
metric_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score']
exercise_names = ['Cylindrical\nGrasp', 'OpenHand\nPose', 'Sequential Thumb\nOpposition']  # Shortened names

x = np.arange(len(metric_names))
width = 0.25

fig, ax = plt.subplots(figsize=(16, 10))
# Modern, accessible color palette
colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
edge_colors = ['#1B5B7A', '#7A2951', '#C4730A']

bars = []
for i, (exercise, metric) in enumerate(metrics.items()):
    values = [metric[m] for m in metric_names]
    bar = ax.bar(x + i * width, values, width, 
                label=exercise_names[i], 
                color=colors[i], 
                edgecolor=edge_colors[i],
                linewidth=1.5,
                alpha=0.85)
    bars.append(bar)
    
    # Add value labels on top of bars
    for j, v in enumerate(values):
        ax.text(x[j] + i * width, v + 0.02, f'{v:.2f}', 
               ha='center', va='bottom', fontweight='bold', fontsize=14)

ax.set_ylabel('Score', fontweight='bold', fontsize=22)
ax.set_title('Classification Metrics Comparison Across Hand Exercises', 
            fontweight='bold', fontsize=26, pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(metric_names, fontweight='bold')
ax.set_ylim(0, 1.2)

# Improved legend
ax.legend(loc='lower right', 
         frameon=True, fancybox=True, shadow=True,
         borderpad=1, columnspacing=1)

# Enhanced grid
ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=1)
ax.set_axisbelow(True)

# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

plt.tight_layout()
plt.savefig('metrics_comparison_bar_chart.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

# --- IMPROVED Line Chart for Comparison ---
metrics_df_line = metrics_df[metric_names]

plt.figure(figsize=(16, 10))
colors = ['#2E86AB', '#A23B72', '#F18F01']
markers = ['o', 's', '^']
linestyles = ['-', '--', '-.']

for i, exercise in enumerate(metrics_df_line.index):
    plt.plot(metric_names, metrics_df_line.loc[exercise], 
             marker=markers[i], 
             label=exercise_names[i], 
             linewidth=4, 
             markersize=12, 
             color=colors[i],
             linestyle=linestyles[i],
             markeredgecolor='white',
             markeredgewidth=2)
    
    # Add value labels on points
    for j, value in enumerate(metrics_df_line.loc[exercise]):
        plt.annotate(f'{value:.2f}', 
                    (j, value), 
                    textcoords="offset points", 
                    xytext=(0,15), 
                    ha='center',
                    fontsize=13,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor=colors[i], 
                             alpha=0.7,
                             edgecolor='white'))

plt.title('Performance Metrics Trend Across Hand Exercises', 
         fontweight='bold', fontsize=26, pad=20)
plt.xlabel('Classification Metrics', fontweight='bold', fontsize=22)
plt.ylabel('Score', fontweight='bold', fontsize=22)
plt.ylim(-0.05, 1.15)

# Enhanced grid
plt.grid(True, linestyle='--', alpha=0.4, linewidth=1)
plt.gca().set_axisbelow(True)

# Improved legend
plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True,
          borderpad=1, columnspacing=1)

# Remove top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.tight_layout()
plt.savefig('metrics_trend_line_chart.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()

# --- IMPROVED Confusion Matrix Heatmaps ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Confusion Matrices for Hand Exercise Classification', 
            fontsize=26, fontweight='bold', y=1.02)

cmap = sns.color_palette("Blues", as_cmap=True)

for idx, (name, values) in enumerate(conf_matrices.items()):
    matrix = np.array([[values['TP'], values['FN']],
                       [values['FP'], values['TN']]])
    labels = ['Positive', 'Negative']
    df_cm = pd.DataFrame(matrix, index=labels, columns=labels)

    # Create heatmap with improved styling
    sns.heatmap(df_cm, 
               annot=True, 
               fmt='d', 
               cmap=cmap,
               ax=axes[idx], 
               cbar_kws={'shrink': 0.8},
               annot_kws={'fontsize': 18, 'fontweight': 'bold'},
               square=True,
               linewidths=2,
               linecolor='white')
    
    axes[idx].set_xlabel('Predicted', fontweight='bold', fontsize=20)
    axes[idx].set_ylabel('Actual', fontweight='bold', fontsize=20)
    axes[idx].set_title(f'{exercise_names[idx]}', fontweight='bold', fontsize=20, pad=10)
    
    # Style tick labels
    axes[idx].tick_params(labelsize=16)
    axes[idx].set_xticklabels(axes[idx].get_xticklabels(), fontweight='bold')
    axes[idx].set_yticklabels(axes[idx].get_yticklabels(), fontweight='bold', rotation=0)

plt.tight_layout()
plt.savefig('confusion_matrices_heatmap.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()

# --- Print summary table ---
print("\n" + "="*100)
print("CLASSIFICATION METRICS SUMMARY".center(100))
print("="*100)

# Create a more readable table format
summary_table = metrics_df[metric_names].round(3)
print(summary_table.to_string(float_format='%.3f'))

print(f"\n{'='*100}")
print("PERFORMANCE INSIGHTS".center(100))
print("="*100)

print(f"📊 Number of exercises analyzed: {len(conf_matrices)}")
print(f"📈 Total samples across all exercises: {sum(metrics[ex]['Total'] for ex in metrics.keys())}")
print(f"🎯 Average accuracy across exercises: {metrics_df['Accuracy'].mean():.3f}")
print(f"🏆 Best performing exercise (by F1 Score): {metrics_df['F1 Score'].idxmax()}")
print(f"⭐ Highest F1 Score: {metrics_df['F1 Score'].max():.3f}")

print(f"\nFiles created:")
print(f"   • {output_filename} (Excel file with all results)")
print("   • metrics_comparison_bar_chart.png")
print("   • metrics_trend_line_chart.png") 
print("   • confusion_matrices_heatmap.png")

# Display basic statistics

# Reset matplotlib parameters to default
plt.rcdefaults()