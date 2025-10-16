import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix
)
from imblearn.metrics import geometric_mean_score
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings("ignore")

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def compute_csi_per_class(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    csi_list = []
    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        denom = TP + FP + FN
        csi = TP / denom if denom > 0 else 0.0
        csi_list.append(csi)
    return csi_list

def evaluate_classification(pred, true):
    pred_prob = softmax(pred, axis=-1)
    pred_class = np.argmax(pred_prob, axis=-1)
    true_flat = true.flatten()
    pred_flat = pred_class.flatten()
    num_classes = pred_prob.shape[-1]

    true_bin = np.eye(num_classes)[true_flat]
    pred_prob_2d = pred_prob.reshape(-1, num_classes)

    precision_each = precision_score(true_flat, pred_flat, average=None, zero_division=0)
    recall_each = recall_score(true_flat, pred_flat, average=None, zero_division=0)
    f1_each = f1_score(true_flat, pred_flat, average=None, zero_division=0)
    prauc_each = average_precision_score(true_bin, pred_prob_2d, average=None)
    csi_each = compute_csi_per_class(true_flat, pred_flat, num_classes)

    metrics = {
        'accuracy': accuracy_score(true_flat, pred_flat),
        'f1_weighted': f1_score(true_flat, pred_flat, average='weighted', zero_division=0),
        'f1_macro': f1_score(true_flat, pred_flat, average='macro', zero_division=0),
        'precision_weighted': precision_score(true_flat, pred_flat, average='weighted', zero_division=0),
        'recall_weighted': recall_score(true_flat, pred_flat, average='weighted', zero_division=0),
        'gmean': geometric_mean_score(true_flat, pred_flat, average='weighted'),
        'prauc_macro': average_precision_score(true_bin, pred_prob_2d, average='macro'),
        'csi_macro': np.mean(csi_each),
    }

    for i in range(num_classes):
        metrics[f'precision_class{i}'] = precision_each[i]
        metrics[f'recall_class{i}'] = recall_each[i]
        metrics[f'f1_class{i}'] = f1_each[i]
        metrics[f'csi_class{i}'] = csi_each[i]
        metrics[f'prauc_class{i}'] = prauc_each[i]

    return metrics

def process_folder(folder_path_tuple):
    folder, root_dir = folder_path_tuple
    subdir = os.path.join(root_dir, folder)
    try:
        pred_dis = np.load(os.path.join(subdir, 'pred_dis.npy'))
        true_dis = np.load(os.path.join(subdir, 'true_dis.npy')).astype(np.int64)
        metrics = evaluate_classification(pred_dis, true_dis)
        metrics['folder'] = folder
        print(f"[✓] {folder} done")
        return metrics
    except Exception as e:
        print(f"[✗] {folder} error: {e}")
        return None

def average_metrics(metrics_list):
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: np.mean([m[k] for m in metrics_list if m and k in m]) for k in keys if k != 'folder'}

# ========================== 主函数 ==========================
if __name__ == "__main__":
    result_dir = "test_results"
    folders = [f for f in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, f))]

    print(f"🔄 Starting multiprocessing with {cpu_count()} cores...\n")

    with Pool(processes=cpu_count()) as pool:
        folder_paths = [(folder, result_dir) for folder in folders]
        results = pool.map(process_folder, folder_paths)

    # 过滤掉 None（出错的）
    results = [r for r in results if r is not None]

    if results:
        df = pd.DataFrame(results)
        df = df[['folder'] + [col for col in df.columns if col != 'folder']]
        df.to_csv('per_class_results.csv', index=False)
        print("\n✅ Saved classification_results.csv")

        # 输出平均值
        print("\n📊 Summary:")
        avg = average_metrics(results)
        for k, v in avg.items():
            print(f"  {k}: {v:.4f}")
