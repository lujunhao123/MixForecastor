# Mixed time series pattern learning for multi-task wind power and ramp event forecasting

## Abstract
Within the field of wind energy prediction, existing studies mainly focus on modeling continuous-variable time series such as wind power, while often overlooking discrete-variable time series that represent ramp event states. Yet in wind power systems, mixed time series comprising continuous and discrete variables are in fact widespread. Integrating these heterogeneous spatiotemporal modalities across wind farms remains challenging due to distributional heterogeneity and the difficulty of extracting categorical temporal features. To address these challenges, this paper proposes MixForecastor, a unified multi-task learning framework for mixed time series forecasting, where both inputs and outputs comprise continuous and discrete variables. The model jointly predicts wind power and ramp event risks by fusing heterogeneous inputs within a shared representation space, which is implemented via a novel heterogeneous fusion module. In addition, the model incorporates a new difference-aware auxiliary module that reinforces the coupling between power variations and ramp dynamics. To handle severe class imbalance and ensure task fairness, MixForecastor is driven by a customized bi-level optimization strategy, wherein a multi-objective imbalance controller dynamically selects optimal strategies for each wind farm, and a fairness-aware multi-task optimizer mitigates conflicts between tasks. Extensive experiments on real-world wind power datasets demonstrate that MixForecastor achieves superior performance in forecasting both continuous power outputs and discrete ramp event states, providing a modeling framework that broadens the scope beyond traditional inputs based on continuous-variable time series.

<p align="center"> 
    <img src="./pic/MixForecastor.png" width="600">
</p>

## ⚠️ Note to Users

This version of the code allows you to run three types of experiments and includes a dataset for testing:

1. **Single-Task Learning**
2. **Multi-Task Learning**
3. **Simple Mixed Sequence Modeling (MixRamp)**

From these experiments, two important observations emerge:

1. In regression and classification tasks, Single-Task Learning is less effective than Multi-Task Learning. Even with PCGrad, standard Multi-Task Learning struggles to handle task conflicts.
2. Multi-Task Learning performs better than simple mixed sequence modeling (MixRamp) on regression tasks but underperforms on classification tasks. This highlights the need for mixed sequence modeling. However, MixRamp does not address heterogeneity, which motivates the development of **MixForecastor**, our main contribution.


These results demonstrate the necessity of our work. The detailed prediction results are available in the [`test_results/`](./test_results) folder. You can find the final evaluation metrics in [`result.txt`](./result.txt) and the training logs in [`test.log`](./test.log).

| Experiment Type | Script | Task | MAE | MSE | RMSE | Accuracy | Precision | Recall | F1 |
|------------------|---------|------|------|------|-------|-------|-------|-----------|-----------|
| **Single-Task Learning (STL)** | `exp_ramp_normalS.py` | Regression | 22.2660 | 777.9416 | 27.8916  | – | – | – |
| **Single-Task Learning (STL)** | `exp_ramp_normalS.py` | Classification | – | – | – | 0.6598 | 0.5332 | 0.6598 | 0.5634 |
| **Multi-Task Learning (MTL)** | `exp_ramp_normalM.py` | Regression | 26.4925 | 977.6452 | 31.2673 | – | – | – |
| **Multi-Task Learning (MTL)** | `exp_ramp_normalM.py` | Classification | – | – | – | 0.6393 | 0.4087 | 0.6393 | 0.4987 |
| **Simple Mixed Sequence Modeling (MixRamp)** | `exp_ramp_normal.py` | Regression | 29.5477 | 1221.1166 | 34.9445 | – | – | – | – |
| **Simple Mixed Sequence Modeling (MixRamp)** | `exp_ramp_normal.py` | Classification | – | – | – | 0.8538 | 0.8522 | 0.8538 | 0.8525 |
