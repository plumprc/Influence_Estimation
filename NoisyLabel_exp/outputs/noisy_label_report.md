# Noisy-Label Experiment Report

Generated: 2026-09-07 13:22 UTC

## Completion

| Family | Included | Complete | Available / expected |
|---|---|---|---:|
| main | yes | yes | 15 / 15 |
| architecture | yes | yes | 9 / 9 |

## Main Detection

Values are mean +/- sample standard deviation over three training seeds.

| rho | Specification | AUPRC | P@10% |
|---:|---|---:|---:|
| 0.05 | GradSim / loss | 0.177 +/- 0.019 | 0.158 +/- 0.004 |
| 0.05 | GradSim / logit | 0.055 +/- 0.007 | 0.041 +/- 0.004 |
| 0.05 | GradSim / margin | 0.104 +/- 0.002 | 0.084 +/- 0.006 |
| 0.05 | TracIn / loss | 0.110 +/- 0.014 | 0.137 +/- 0.006 |
| 0.05 | TracIn / logit | 0.329 +/- 0.032 | 0.227 +/- 0.006 |
| 0.05 | TracIn / margin | 0.259 +/- 0.031 | 0.204 +/- 0.010 |
| 0.05 | LiSSA / loss | 0.289 +/- 0.006 | 0.210 +/- 0.000 |
| 0.05 | LiSSA / logit | 0.080 +/- 0.004 | 0.053 +/- 0.003 |
| 0.05 | LiSSA / margin | 0.128 +/- 0.011 | 0.077 +/- 0.007 |
| 0.05 | Random | 0.050 +/- 0.000 | 0.049 +/- 0.001 |
| 0.05 | RepSim | 0.062 +/- 0.001 | 0.065 +/- 0.004 |
| 0.05 | NTK | 0.044 +/- 0.001 | 0.044 +/- 0.002 |
| 0.10 | GradSim / loss | 0.238 +/- 0.013 | 0.280 +/- 0.022 |
| 0.10 | GradSim / logit | 0.153 +/- 0.030 | 0.142 +/- 0.028 |
| 0.10 | GradSim / margin | 0.223 +/- 0.026 | 0.229 +/- 0.020 |
| 0.10 | TracIn / loss | 0.210 +/- 0.020 | 0.293 +/- 0.024 |
| 0.10 | TracIn / logit | 0.452 +/- 0.042 | 0.492 +/- 0.029 |
| 0.10 | TracIn / margin | 0.371 +/- 0.079 | 0.416 +/- 0.063 |
| 0.10 | LiSSA / loss | 0.377 +/- 0.018 | 0.394 +/- 0.017 |
| 0.10 | LiSSA / logit | 0.214 +/- 0.028 | 0.192 +/- 0.029 |
| 0.10 | LiSSA / margin | 0.352 +/- 0.046 | 0.311 +/- 0.040 |
| 0.10 | Random | 0.101 +/- 0.001 | 0.101 +/- 0.002 |
| 0.10 | RepSim | 0.125 +/- 0.001 | 0.139 +/- 0.004 |
| 0.10 | NTK | 0.096 +/- 0.002 | 0.107 +/- 0.001 |
| 0.20 | GradSim / loss | 0.401 +/- 0.017 | 0.511 +/- 0.019 |
| 0.20 | GradSim / logit | 0.345 +/- 0.022 | 0.389 +/- 0.028 |
| 0.20 | GradSim / margin | 0.402 +/- 0.068 | 0.514 +/- 0.100 |
| 0.20 | TracIn / loss | 0.349 +/- 0.027 | 0.537 +/- 0.037 |
| 0.20 | TracIn / logit | 0.602 +/- 0.031 | 0.831 +/- 0.024 |
| 0.20 | TracIn / margin | 0.468 +/- 0.038 | 0.704 +/- 0.059 |
| 0.20 | LiSSA / loss | 0.535 +/- 0.025 | 0.651 +/- 0.034 |
| 0.20 | LiSSA / logit | 0.421 +/- 0.013 | 0.492 +/- 0.017 |
| 0.20 | LiSSA / margin | 0.587 +/- 0.020 | 0.781 +/- 0.022 |
| 0.20 | Random | 0.198 +/- 0.001 | 0.197 +/- 0.004 |
| 0.20 | RepSim | 0.216 +/- 0.008 | 0.219 +/- 0.016 |
| 0.20 | NTK | 0.197 +/- 0.004 | 0.207 +/- 0.012 |
| 0.30 | GradSim / loss | 0.524 +/- 0.013 | 0.669 +/- 0.027 |
| 0.30 | GradSim / logit | 0.463 +/- 0.023 | 0.561 +/- 0.038 |
| 0.30 | GradSim / margin | 0.465 +/- 0.008 | 0.568 +/- 0.001 |
| 0.30 | TracIn / loss | 0.478 +/- 0.013 | 0.729 +/- 0.018 |
| 0.30 | TracIn / logit | 0.722 +/- 0.033 | 0.937 +/- 0.019 |
| 0.30 | TracIn / margin | 0.534 +/- 0.014 | 0.787 +/- 0.027 |
| 0.30 | LiSSA / loss | 0.636 +/- 0.014 | 0.803 +/- 0.030 |
| 0.30 | LiSSA / logit | 0.541 +/- 0.019 | 0.695 +/- 0.036 |
| 0.30 | LiSSA / margin | 0.668 +/- 0.010 | 0.860 +/- 0.010 |
| 0.30 | Random | 0.295 +/- 0.001 | 0.295 +/- 0.008 |
| 0.30 | RepSim | 0.295 +/- 0.002 | 0.276 +/- 0.007 |
| 0.30 | NTK | 0.285 +/- 0.010 | 0.295 +/- 0.021 |
| 0.40 | GradSim / loss | 0.643 +/- 0.041 | 0.805 +/- 0.069 |
| 0.40 | GradSim / logit | 0.569 +/- 0.020 | 0.711 +/- 0.035 |
| 0.40 | GradSim / margin | 0.607 +/- 0.029 | 0.753 +/- 0.051 |
| 0.40 | TracIn / loss | 0.586 +/- 0.073 | 0.801 +/- 0.064 |
| 0.40 | TracIn / logit | 0.800 +/- 0.003 | 0.951 +/- 0.009 |
| 0.40 | TracIn / margin | 0.533 +/- 0.015 | 0.708 +/- 0.008 |
| 0.40 | LiSSA / loss | 0.713 +/- 0.021 | 0.892 +/- 0.031 |
| 0.40 | LiSSA / logit | 0.632 +/- 0.021 | 0.854 +/- 0.056 |
| 0.40 | LiSSA / margin | 0.738 +/- 0.017 | 0.925 +/- 0.017 |
| 0.40 | Random | 0.395 +/- 0.001 | 0.396 +/- 0.006 |
| 0.40 | RepSim | 0.367 +/- 0.004 | 0.324 +/- 0.010 |
| 0.40 | NTK | 0.367 +/- 0.005 | 0.360 +/- 0.007 |

## Removal Utility

Test-accuracy change after removing the top 10% and retraining. Values are percentage points over three source seeds.

| Specification | Test accuracy change | Selected noise rate |
|---|---:|---:|
| GradSim / loss | 0.623 +/- 1.194 | 0.511 +/- 0.019 |
| GradSim / logit | -0.447 +/- 0.867 | 0.389 +/- 0.028 |
| GradSim / margin | 0.243 +/- 2.025 | 0.514 +/- 0.100 |
| TracIn / loss | 1.063 +/- 0.915 | 0.537 +/- 0.037 |
| TracIn / logit | 3.200 +/- 0.217 | 0.831 +/- 0.024 |
| TracIn / margin | 2.233 +/- 0.946 | 0.704 +/- 0.059 |
| LiSSA / loss | 1.397 +/- 0.790 | 0.651 +/- 0.034 |
| LiSSA / logit | 0.113 +/- 0.670 | 0.492 +/- 0.017 |
| LiSSA / margin | 2.213 +/- 0.955 | 0.781 +/- 0.022 |
| Random | -1.673 +/- 0.443 | 0.197 +/- 0.004 |
| RepSim | -2.703 +/- 0.441 | 0.219 +/- 0.016 |
| NTK | -2.007 +/- 0.748 | 0.207 +/- 0.012 |

## Architecture Extension

AUPRC at rho = 0.2. Values are mean +/- sample standard deviation over three training seeds.

| Model | Specification | AUPRC | P@10% |
|---|---|---:|---:|
| ResNet-50 | GradSim / loss | 0.306 +/- 0.038 | 0.395 +/- 0.051 |
| ResNet-50 | GradSim / logit | 0.295 +/- 0.107 | 0.338 +/- 0.146 |
| ResNet-50 | GradSim / margin | 0.324 +/- 0.105 | 0.387 +/- 0.154 |
| ResNet-50 | TracIn / loss | 0.501 +/- 0.138 | 0.725 +/- 0.162 |
| ResNet-50 | TracIn / logit | 0.746 +/- 0.045 | 0.868 +/- 0.047 |
| ResNet-50 | TracIn / margin | 0.621 +/- 0.137 | 0.829 +/- 0.106 |
| ResNet-50 | LiSSA / loss | 0.409 +/- 0.090 | 0.515 +/- 0.112 |
| ResNet-50 | LiSSA / logit | 0.357 +/- 0.111 | 0.423 +/- 0.148 |
| ResNet-50 | LiSSA / margin | 0.421 +/- 0.134 | 0.518 +/- 0.188 |
| VGG-19-BN | GradSim / loss | 0.308 +/- 0.020 | 0.391 +/- 0.023 |
| VGG-19-BN | GradSim / logit | 0.216 +/- 0.020 | 0.251 +/- 0.026 |
| VGG-19-BN | GradSim / margin | 0.243 +/- 0.049 | 0.283 +/- 0.062 |
| VGG-19-BN | TracIn / loss | 0.441 +/- 0.026 | 0.679 +/- 0.037 |
| VGG-19-BN | TracIn / logit | 0.801 +/- 0.012 | 0.934 +/- 0.008 |
| VGG-19-BN | TracIn / margin | 0.678 +/- 0.056 | 0.895 +/- 0.028 |
| VGG-19-BN | LiSSA / loss | 0.307 +/- 0.015 | 0.388 +/- 0.016 |
| VGG-19-BN | LiSSA / logit | 0.218 +/- 0.020 | 0.253 +/- 0.027 |
| VGG-19-BN | LiSSA / margin | 0.246 +/- 0.050 | 0.285 +/- 0.061 |
| MobileNetV2 | GradSim / loss | 0.236 +/- 0.003 | 0.268 +/- 0.001 |
| MobileNetV2 | GradSim / logit | 0.195 +/- 0.002 | 0.194 +/- 0.006 |
| MobileNetV2 | GradSim / margin | 0.213 +/- 0.006 | 0.216 +/- 0.010 |
| MobileNetV2 | TracIn / loss | 0.350 +/- 0.035 | 0.525 +/- 0.059 |
| MobileNetV2 | TracIn / logit | 0.647 +/- 0.012 | 0.812 +/- 0.007 |
| MobileNetV2 | TracIn / margin | 0.494 +/- 0.019 | 0.675 +/- 0.030 |
| MobileNetV2 | LiSSA / loss | 0.237 +/- 0.003 | 0.269 +/- 0.001 |
| MobileNetV2 | LiSSA / logit | 0.196 +/- 0.001 | 0.194 +/- 0.006 |
| MobileNetV2 | LiSSA / margin | 0.214 +/- 0.007 | 0.217 +/- 0.014 |
