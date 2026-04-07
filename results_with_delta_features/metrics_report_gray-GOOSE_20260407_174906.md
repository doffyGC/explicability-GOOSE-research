# Metrics Report - XGBOOST

**Dataset:** gray-GOOSE
**Date/Time:** 07/04/2026 17:49:06

---

## 📊 Cross-Validation (K-Fold)

Results of cross-validation with **95% confidence intervals** (CI 95%).

### Metrics by Class

| Class | F1-Score | Precision | Recall |
|--------|----------|-----------|--------|
| **SAG.DB** | 0.9973 ± 0.0003 | 0.9949 ± 0.0005 | 0.9998 ± 0.0001 |
| **FRG** | 0.8707 ± 0.0013 | 0.8524 ± 0.0125 | 0.8899 ± 0.0109 |
| **SAG.PB** | 0.9737 ± 0.0011 | 0.9675 ± 0.0019 | 0.9801 ± 0.0020 |
| **SAG.PBM** | 0.7938 ± 0.0018 | 0.7084 ± 0.0049 | 0.9027 ± 0.0092 |
| **Normal** | 0.9679 ± 0.0006 | 0.9972 ± 0.0004 | 0.9402 ± 0.0011 |

### Global Metrics (CV)

- **Accuracy (CV - Mean ± CI):** 0.9413 ± 0.0005
- **Cohen's Kappa:** 0.9055 ± 0.0007

---

## 🧾 Confusion Matrix (CV - Aggregated)

```
Predicted →          SAG.DB          FRG       SAG.PB      SAG.PBM       Normal 
Real ↓
SAG.DB             102449            2           15            4            0 
FRG                    56        90984          295         9628         1273 
SAG.PB                389           41        98377         1368          198 
SAG.PBM                57         7978         1771        91998          106 
Normal                 25         7754         1229        26874       564118 
```

---

## 📈 Interpretation

- **Best Performance (CV):** Class `SAG.DB` with F1-Score of **0.9973**
- **Cohen's Kappa (CV):** Agreement **almost perfect** (0.9055 ± 0.0007)

---

*Report generated automatically by the XGBOOST training pipeline*
