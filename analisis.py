import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/supply_chain_clean.csv")
sns.set_theme(style="darkgrid")
fig, axes=plt.subplots(2,2,figsize=(16,12))
fig.suptitle("Supply Chain - Análisis Exploratorio", fontsize=16, fontweight="bold")

#=== Gráfica 1 - Estado de envío ===
delivery_counts = df["Delivery Status"].value_counts()
axes[0,0].bar(delivery_counts.index, delivery_counts.values,
              color = ["#e74c3c","#2ecc71","#3498db","#e67e22"])
axes[0,0].set_title("Distribución de Delivery Status")
axes[0,0].set_xlabel("Estado")
axes[0,0].set_ylabel("Cantidad de órdenes")
axes[0,0].tick_params(axis="x", rotation=15)

#=== Gráfica 2 - Retraso promedio por Shipping Mode ===
delay_by_mode = df.groupby("Shipping Mode")["shipping_delay_days"].mean().sort_values(ascending=False)
axes[0,1].barh(delay_by_mode.index, delay_by_mode.values, color="#e74c3c")
axes[0,1].set_title("Retraso Promedio por Shipping Mode")
axes[0,1].set_xlabel("Días de retraso promedio")
axes[0,1].axvline(x=0, color="black", linewidth = 0.8)

#=== Gráfica 3 - Top 10 categorías con más retraso ===
delay_by_cat = df.groupby("Category Name")["shipping_delay_days"].mean().sort_values(ascending=False).head(10)
axes[1,0].barh(delay_by_cat.index, delay_by_cat.values, color="#e67e22")
axes[1,0].set_title("Top 10 Categorías con mayor retraso")
axes[1,0].set_xlabel("Días de retraso promedio")
axes[1,0].axvline(x=0, color="black",linewidth=0.8)

#=== Gráfica 4 - Distribución de días de retraso ===
axes[1,1].hist(df["shipping_delay_days"], bins=20, color="#3498bd", edgecolor = "white")
axes[1,1].set_title("Distribución de días de retraso")
axes[1,1].set_xlabel("Días de retraso")
axes[1,1].set_ylabel("Frecuencia")
axes[1,1].axvline(x=0, color="red", linewidth=1.5, linestyle="--",label = "Sin retraso")
axes[1,1].legend()

plt.tight_layout()
plt.savefig("data/analisis_exploratorio.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfica guardada en data/analisis_exploratorio.png")