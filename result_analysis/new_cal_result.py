import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    average_precision_score
)
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
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

def process_folder(subdir):
    folder = os.path.basename(subdir)
    results = {'folder': folder}
    classification_metrics = None
    regression_metrics = None

    # 分类指标
    try:
        pred_dis = np.load(os.path.join(subdir, 'pred_dis.npy'))
        true_dis = np.load(os.path.join(subdir, 'true_dis.npy')).astype(np.int64)
        classification_metrics = evaluate_classification(pred_dis, true_dis)
        results.update({f'class_{k}': v for k, v in classification_metrics.items()})
    except Exception as e:
        results['class_error'] = str(e)

    # 回归指标
    try:
        pred_con = np.load(os.path.join(subdir, 'pred_con.npy'))
        true_con = np.load(os.path.join(subdir, 'true_con.npy'))
        regression_metrics = evaluate_regression(pred_con, true_con)
        results.update({f'reg_{k}': v for k, v in regression_metrics.items()})
    except Exception as e:
        results['reg_error'] = str(e)

    return results

def average_metrics(metrics_list, prefix):
    if not metrics_list:
        return {}
    keys = [k for k in metrics_list[0].keys() if k.startswith(prefix)]
    avg = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m]
        avg[k] = np.mean(vals)
    return avg

# =============== 主流程 ===============
result_dir = "test_results"
subdirs = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, f))]

classification_rows = []
regression_rows = []

with ProcessPoolExecutor() as executor:
    futures = {executor.submit(process_folder, sd): sd for sd in subdirs}
    for future in as_completed(futures):
        res = future.result()
        folder = res['folder']
        print(f"Processed {folder}")
        if any(k.startswith('class_') for k in res.keys()):
            class_metrics = {k[6:]: v for k, v in res.items() if k.startswith('class_')}
            class_metrics['folder'] = folder
            classification_rows.append(class_metrics)
            print(f"  Classification F1_weighted: {class_metrics.get('f1_weighted', float('nan')):.4f}")
        else:
            print(f"  Classification error: {res.get('class_error', 'Unknown')}")

        if any(k.startswith('reg_') for k in res.keys()):
            reg_metrics = {k[4:]: v for k, v in res.items() if k.startswith('reg_')}
            reg_metrics['folder'] = folder
            regression_rows.append(reg_metrics)
            print(f"  Regression RMSE: {reg_metrics.get('rmse', float('nan')):.4f}")
        else:
            print(f"  Regression error: {res.get('reg_error', 'Unknown')}")

# 保存 CSV
if classification_rows:
    df_class = pd.DataFrame(classification_rows)
    cols = ['folder'] + sorted([c for c in df_class.columns if c != 'folder'])
    df_class = df_class[cols]
    df_class.to_csv('classification_results.csv', index=False)
    print("\nSaved classification_results.csv")

if regression_rows:
    df_reg = pd.DataFrame(regression_rows)
    cols = ['folder'] + sorted([c for c in df_reg.columns if c != 'folder'])
    df_reg = df_reg[cols]
    df_reg.to_csv('regression_results.csv', index=False)
    print("Saved regression_results.csv")

# Summary
print("\n=== Summary ===")
if classification_rows:
    avg_class = average_metrics(classification_rows, 'f1_')  # 这里举例只平均f1相关指标，也可以改成全指标
    print("Classification Avg F1 Scores:")
    for k, v in avg_class.items():
        print(f"  {k}: {v:.4f}")

if regression_rows:
    avg_reg = average_metrics(regression_rows, 'rmse')  # 也可改成平均所有回归指标
    print("Regression Avg:")
    for k, v in average_metrics(regression_rows, 'reg_').items():
        print(f"  {k}: {v:.4f}")
