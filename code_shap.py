import pandas as pd
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# 1. Carregar o dataset
df = pd.read_csv("./data/CSV files/randomicMessage.csv")  # ajuste o nome do arquivo

# 2. Definir a classe positiva (1) e negativa (0)
y = (df["class"] == "RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE").astype(int)

# 3. Features (remove a coluna alvo)
X = df.drop(columns=["class"])

# 4. Transformar categóricas em numéricas (one-hot simples)
X = pd.get_dummies(X, drop_first=True)

# 5. Treinar XGBoost (binário, simples)
model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

print("F1 Score on training data:", f1_score(y, model.predict(X)))

# 6. SHAP com TreeExplainer
explainer = shap.Explainer(model)
shap_values = explainer(X)

print("Gerando gráficos SHAP...")

# 7. Summary plot tipo beeswarm
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X,show=False)
plt.tight_layout()
plt.savefig("results/SOG.PBM/Summary_PBM_plot.png", dpi=300)
plt.close()

print("Summary plot salvo.")

# 8. Force plot para uma instância de ataque
idx_attack_instance = y[y == 1].index[0]  # pega o primeiro
plt.title(f"Force Plot of PBM Dataset (Attack Instance)")
shap.plots.force(shap_values[idx_attack_instance], show=False, matplotlib=True)
plt.tight_layout()  
plt.savefig("results/SOG.PBM/Force_PBM_plot.png", dpi=300, bbox_inches='tight')
plt.close()