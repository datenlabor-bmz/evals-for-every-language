import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.special import logit

df = pd.read_json("../results.json")

df = df[df["metric"] != "chrf"]
df = df.groupby(["task", "metric", "bcp_47"]).agg({"score": "mean"}).reset_index()

# Apply logit transformation to classification scores to reduce skewness
def transform_classification_scores(row):
    if row['task'] == 'classification':
        # Avoid division by zero and infinite values by clipping
        score = np.clip(row['score'], 0.001, 0.999)
        # Apply logit transformation (log(p/(1-p)))
        return logit(score)
    else:
        return row['score']

df['score'] = df.apply(transform_classification_scores, axis=1)

# Create a pivot table with tasks as columns and languages as rows
pivot_df = df.pivot_table(
    values='score', 
    index='bcp_47', 
    columns='task', 
    aggfunc='mean'
)

# Calculate correlation matrix
correlation_matrix = pivot_df.corr()

# Create the correlation plot
plt.figure(figsize=(12, 10))
# Create mask for upper triangle including diagonal to show only lower triangle  
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Create a heatmap
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='coolwarm', 
    center=0,
    square=True,
    mask=mask,
    cbar_kws={"shrink": .8},
    fmt='.3f'
)

plt.title('Task Performance Correlation Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Tasks', fontsize=12)
plt.ylabel('Tasks', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save the plot
plt.savefig('task_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Print correlation values for reference
print("Correlation Matrix:")
print("Note: Classification scores have been logit-transformed to reduce skewness")
print(correlation_matrix.round(3))

# Also create a scatter plot matrix for pairwise relationships with highlighted languages
highlighted_languages = ['en', 'zh', 'hi', 'es', 'ar']

# Create color mapping
def get_color_and_label(lang_code):
    if lang_code in highlighted_languages:
        color_map = {'en': 'red', 'zh': 'blue', 'hi': 'green', 'es': 'orange', 'ar': 'purple'}
        return color_map[lang_code], lang_code
    else:
        return 'lightgray', 'Other'

# Create custom scatter plot matrix
tasks = pivot_df.columns.tolist()
n_tasks = len(tasks)

fig, axes = plt.subplots(n_tasks, n_tasks, figsize=(15, 12))
fig.suptitle('Pairwise Task Performance (Highlighted Languages)', fontsize=16, fontweight='bold')

# Create legend elements
legend_elements = []
for lang in highlighted_languages:
    color, _ = get_color_and_label(lang)
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=lang))
legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=8, label='Other'))

for i, task_y in enumerate(tasks):
    for j, task_x in enumerate(tasks):
        ax = axes[i, j]
        
        if i == j:
            # Diagonal: histogram
            task_data = pivot_df[task_y].dropna()
            colors = [get_color_and_label(lang)[0] for lang in task_data.index]
            ax.hist(task_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'{task_y}', fontsize=10)
        else:
            # Off-diagonal: scatter plot
            for lang_code in pivot_df.index:
                if pd.notna(pivot_df.loc[lang_code, task_x]) and pd.notna(pivot_df.loc[lang_code, task_y]):
                    color, _ = get_color_and_label(lang_code)
                    alpha = 0.8 if lang_code in highlighted_languages else 0.3
                    size = 50 if lang_code in highlighted_languages else 20
                    ax.scatter(pivot_df.loc[lang_code, task_x], pivot_df.loc[lang_code, task_y], 
                             c=color, alpha=alpha, s=size)
        
        # Set labels
        if i == n_tasks - 1:
            ax.set_xlabel(task_x, fontsize=10)
        if j == 0:
            ax.set_ylabel(task_y, fontsize=10)
        
        # Remove tick labels except for edges
        if i != n_tasks - 1:
            ax.set_xticklabels([])
        if j != 0:
            ax.set_yticklabels([])

# Add legend
fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig('task_scatter_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
