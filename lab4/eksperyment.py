"""py eksperyment.py: porównanie ustawień bez zmieniania CSV i kodu API."""
from math import isclose
from sklearn.metrics import adjusted_rand_score
from cluster import clusters, elbow, fit_articles

if __name__ == "__main__":
    for features, minimum in ((100, 2), (300, 2), (300, 1)):
        articles, vectorizer, matrix, model = fit_articles(max_features=features, min_df=minimum)
        groups = clusters(articles, vectorizer, model)
        assert matrix.shape[0] == len(articles) == 42 #40
        assert matrix.shape[1] <= features
        assert sum(group["count"] for group in groups) == 42 #40
        assert len({doc["id"] for group in groups for doc in group["articles"]}) == 42 #40
        # Niezależnie odczytujemy odległość każdego punktu do jego centroida.
        distances = model.transform(matrix)
        measured = sum(distances[i, label] ** 2 for i, label in enumerate(model.labels_))
        assert isclose(measured, model.inertia_, rel_tol=1e-9, abs_tol=1e-9)
        ari = adjusted_rand_score([article["source"] for article in articles], model.labels_)
        assert isclose(ari, adjusted_rand_score([article["source"] for article in articles], model.labels_ + 20))
        print(f"max_features={features}, min_df={minimum}, shape={matrix.shape}, ARI={ari:.3f}, inertia={model.inertia_:.3f}")
        if (features, minimum) == (100, 2):
            print("Elbow (ta sama macierz):", elbow(matrix))
