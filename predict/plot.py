#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json
from pathlib import Path

# load metrics.json
path = Path(
    r'E:\tbrain\aicup1\submissions\20191212\eval-3067\test15-pw171-lr8-b8-pe-e704x2-d512-c0-r0-d512_aicup1-v7_TITANRTX_8086K1-1.2@3067-metrics.json'
)
with path.open('r') as f:
    metrics = json.load(f)

tp = np.array(metrics['precision_recall_at_equal_thresholds'][0])
fp = np.array(metrics['precision_recall_at_equal_thresholds'][1])
tn = np.array(metrics['precision_recall_at_equal_thresholds'][2])
fn = np.array(metrics['precision_recall_at_equal_thresholds'][3])
precision = np.array(metrics['precision_recall_at_equal_thresholds'][4])
recall = np.array(metrics['precision_recall_at_equal_thresholds'][5])
thresholds = np.array(metrics['precision_recall_at_equal_thresholds'][6])
plt.plot(thresholds, recall)
plt.plot(thresholds, precision)
plt.plot(thresholds, 2 * precision * recall / (precision + recall))
plt.plot(1 - tn / (tn + fp), tp / (tp + fn))
plt.show()
max(
    zip(thresholds, 2 * precision * recall / (precision + recall)),
    key=lambda x: x[1]
)

# %%
