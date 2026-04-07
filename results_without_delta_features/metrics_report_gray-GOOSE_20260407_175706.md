# Metrics Report - XGBOOST

**Dataset:** gray-GOOSE
**Date/Time:** 07/04/2026 17:57:06

---

## 📊 Cross-Validation (K-Fold)

Results of cross-validation with **95% confidence intervals** (CI 95%).

### Metrics by Class

| Class | F1-Score | Precision | Recall |
|--------|----------|-----------|--------|
| **SAG.DB** | 0.9932 ± 0.0003 | 0.9864 ± 0.0006 | 1.0000 ± 0.0000 |
| **FRG** | 0.6364 ± 0.0040 | 0.5246 ± 0.0033 | 0.8090 ± 0.0144 |
| **SAG.PB** | 0.7437 ± 0.0035 | 0.7028 ± 0.0119 | 0.7899 ± 0.0161 |
| **SAG.PBM** | 0.6948 ± 0.0041 | 0.5839 ± 0.0070 | 0.8581 ± 0.0233 |
| **Normal** | 0.8692 ± 0.0014 | 0.9747 ± 0.0005 | 0.7843 ± 0.0020 |

### Global Metrics (CV)

- **Accuracy (CV - Mean ± CI):** 0.8168 ± 0.0007
- **Cohen's Kappa:** 0.7231 ± 0.0010

---

## 🧾 Confusion Matrix (CV - Aggregated)

```
Predicted →          SAG.DB          FRG       SAG.PB      SAG.PBM       Normal 
Real ↓
SAG.DB             102469            0            0            1            0 
FRG                   347        82711         3378         7120         8680 
SAG.PB                389         1803        79281        15543         3357 
SAG.PBM               426         9150         4709        87450          175 
Normal                249        64011        25468        39695       470577 
```

---

## 📈 Interpretation

- **Best Performance (CV):** Class `SAG.DB` with F1-Score of **0.9932**
- **Cohen's Kappa (CV):** Agreement **substantial** (0.7231 ± 0.0010)

---

*Report generated automatically by the XGBOOST training pipeline*
