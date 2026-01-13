import matplotlib.pyplot as plt
import numpy as np

# Set up the figure with dark background
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#1e3a8a')

# Data
models = ['BF16\n(Baseline)', '4-bit', 'Mixed 4/8']
accuracy = [0.4035, 0.4154, 0.4182]
colors = ['#94a3b8', '#3b82f6', '#10b981']

# Create bars
bars = ax.bar(models, accuracy, color=colors, width=0.6, edgecolor='none')

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracy)):
    height = bar.get_height()
    if i == 0:
        label = f'{acc:.4f}'
    elif i == 1:
        label = f'{acc:.4f}\n(+2.95%)'
    else:
        label = f'{acc:.4f}\n(+3.64%)'
    
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
            label, ha='center', va='bottom', color='white', 
            fontsize=13, fontweight='bold')

# Styling
ax.set_ylabel('Accuracy', fontsize=16, color='white', fontweight='bold')
ax.set_ylim(0.39, 0.425)
ax.tick_params(colors='white', labelsize=13)
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, color='white', linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Title
fig.suptitle('Efficient Compression.', 
             fontsize=32, fontweight='bold', color='white', y=.97)
# subtitle = '75% smaller, yet +1.18% more accurate'
# fig.text(0.5, 0.90, subtitle, ha='center', fontsize=20, color='#bfdbfe')

# Add metric boxes at bottom
metric_y = 0.84
box_props = dict(boxstyle='round,pad=0.5', facecolor='#1e3a8a', 
                 edgecolor='white', alpha=0.3, linewidth=1.5)

# Size Reduction
fig.text(0.22, metric_y, '75%', ha='center', fontsize=26, 
         fontweight='bold', color='#34d399', bbox=box_props)
fig.text(0.22, metric_y - 0.05, 'Size Reduction', ha='center', 
         fontsize=11, color='#d1d5db')

# Accuracy Gain
fig.text(0.5, metric_y, '+2.95%', ha='center', fontsize=26, 
         fontweight='bold', color='#60a5fa', bbox=box_props)
fig.text(0.5, metric_y - 0.05, 'Accuracy Gain', ha='center', 
         fontsize=11, color='#d1d5db')

# Quantization
fig.text(0.78, metric_y, '4-bit', ha='center', fontsize=26, 
         fontweight='bold', color='#c084fc', bbox=box_props)
fig.text(0.78, metric_y - 0.05, 'Quantization', ha='center', 
         fontsize=11, color='#d1d5db')

# Footer
# fig.text(0.5, 0.04, 'Efficient compression', 
#          ha='center', fontsize=14, color='#9ca3af', style='italic')

plt.tight_layout(rect=[0, 0.10, 1, 0.88])

# Save as PNG
plt.savefig('model_compression_teaser.png', dpi=300, 
            facecolor='#0f172a', bbox_inches='tight')
print("Plot saved as 'model_compression_teaser.png'")

# Display the plot
plt.show()