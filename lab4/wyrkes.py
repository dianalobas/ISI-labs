import matplotlib.pyplot as plt
from cluster import fit_articles, elbow

articles, vectorizer, matrix, model = fit_articles(k=5, max_features=300, min_df=2)

elbow_data = elbow(matrix)
k_values = [item["k"] for item in elbow_data]
inertia_values = [item["inertia"] for item in elbow_data]

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
plt.title("Metoda Łokcia (Elbow Method) dla K-Means", fontsize=14)
plt.xlabel("Liczba klastrów (k)", fontsize=12)
plt.ylabel("Inertia (Suma kwadratów odległości)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

chosen_k = 5
chosen_inertia = [item["inertia"] for item in elbow_data if item["k"] == chosen_k][0]
plt.annotate(f'Wybrany k={chosen_k}',
             xy=(chosen_k, chosen_inertia),
             xytext=(chosen_k + 0.5, chosen_inertia + 1.5),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6),
             fontsize=11, color='red', fontweight='bold')

plt.show()
