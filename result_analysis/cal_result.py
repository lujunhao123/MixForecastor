import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, average_precision_score, confusion_matrix
)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from imblearn.metrics import geometric_mean_score
import warnings
from sklearn.metrics import matthews_corrcoef

warnings.filterwarnings("ignore")

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def evaluate_classification(pred, true):
    pred_prob = softmax(pred, axis=-1)  # shape: [sample, pred_len, dim, n_classes]
    pred_class = np.argmax(pred_prob, axis=-1)

    true_flat = true.flatten()
    pred_flat = pred_class.flatten()

    true_bin = np.eye(pred_prob.shape[-1])[true_flat]  # one-hot
    pred_prob_2d = pred_prob.reshape(-1, pred_prob.shape[-1])

    metrics = {
        'accuracy': accuracy_score(true_flat, pred_flat),
        'f1_weighted': f1_score(true_flat, pred_flat, average='weighted', zero_division=0),
        'f1_macro': f1_score(true_flat, pred_flat, average='macro', zero_division=0),
        'precision': precision_score(true_flat, pred_flat, average='weighted', zero_division=0),
        'recall': recall_score(true_flat, pred_flat, average='weighted', zero_division=0),
        'mcc': matthews_corrcoef(true_flat, pred_flat),
        'prauc_weighted': average_precision_score(true_bin, pred_prob_2d, average='weighted'),
        'prauc_macro': average_precision_score(true_bin, pred_prob_2d, average='macro')
    }
    return metrics


def evaluate_regression(pred, true):
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    mse = mean_squared_error(true_flat, pred_flat)
    mae = mean_absolute_error(true_flat, pred_flat)
    rmse = np.sqrt(mse)
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }

def average_metrics(metrics_list):
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: np.mean([m[k] for m in metrics_list if k in m]) for k in keys}

# ========================== 主流程 ==========================
result_dir = "test_results"
all_classification_metrics = []
all_regression_metrics = []

classification_rows = []
regression_rows = []

for folder in os.listdir(result_dir):
    subdir = os.path.join(result_dir, folder)
    if not os.path.isdir(subdir):
        continue

    print(f"Processing {folder}...")

    # Classification
    try:
        pred_dis = np.load(os.path.join(subdir, 'pred_dis.npy'))
        true_dis = np.load(os.path.join(subdir, 'true_dis.npy')).astype(np.int64)
        class_metrics = evaluate_classification(pred_dis, true_dis)
        class_metrics['folder'] = folder
        classification_rows.append(class_metrics)
        print(f"  Classification OK - F1_weighted: {class_metrics['f1_weighted']:.4f}")
    except Exception as e:
        print(f"  [!] Classification error: {e}")

    # Regression
    try:
        pred_con = np.load(os.path.join(subdir, 'pred_con.npy'))
        true_con = np.load(os.path.join(subdir, 'true_con.npy'))
        reg_metrics = evaluate_regression(pred_con, true_con)
        reg_metrics['folder'] = folder
        regression_rows.append(reg_metrics)
        print(f"  Regression OK - RMSE: {reg_metrics['rmse']:.4f}")
    except Exception as e:
        print(f"  [!] Regression error: {e}")

# ========================== 保存 CSV ==========================
if classification_rows:
    df_class = pd.DataFrame(classification_rows)
    df_class = df_class[['folder'] + [col for col in df_class.columns if col != 'folder']]  # folder 放第一列
    df_class.to_csv('classification_results.csv', index=False)
    print("\nSaved classification_results.csv")

if regression_rows:
    df_reg = pd.DataFrame(regression_rows)
    df_reg = df_reg[['folder'] + [col for col in df_reg.columns if col != 'folder']]  # folder 放第一列
    df_reg.to_csv('regression_results.csv', index=False)
    print("Saved regression_results.csv")

# ========================== Summary ==========================
print("\n=== Summary ===")
if classification_rows:
    avg_class = average_metrics(classification_rows)
    print("Classification Avg:")
    for k, v in avg_class.items():
        if k != 'folder':
            print(f"  {k}: {v:.4f}")

if regression_rows:
    avg_reg = average_metrics(regression_rows)
    print("Regression Avg:")
    for k, v in avg_reg.items():
        if k != 'folder':
            print(f"  {k}: {v:.4f}")
