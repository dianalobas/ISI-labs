"""Lab 4: polskie artykuły -> TF-IDF -> K-Means. Uruchom: python cluster.py."""
import csv
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path(__file__).with_name("articles.csv")
STOP = ["i", "w", "na", "do", "z", "o", "po", "dla", "oraz", "nad", "przed", "że", "się", "jest", "ma", "są"]


def load_articles():
    with DATA.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def fit_articles(k=5, max_features=300, min_df=2):
    articles = load_articles()
    vectorizer = TfidfVectorizer(stop_words=STOP, max_features=max_features, min_df=min_df)
    matrix = vectorizer.fit_transform([f"{a['title']} {a['text']}" for a in articles])
    model = KMeans(n_clusters=k, n_init=20, random_state=142).fit(matrix)
    return articles, vectorizer, matrix, model


def clusters(articles, vectorizer, model):
    words = vectorizer.get_feature_names_out()
    result = []
    for cluster_id in range(model.n_clusters):
        keywords = [words[i] for i in model.cluster_centers_[cluster_id].argsort()[-6:][::-1]]
        docs = [dict(id=int(a["id"]), title=a["title"], text=a["text"])
                for a, label in zip(articles, model.labels_) if label == cluster_id]
        result.append(dict(cluster_id=cluster_id, keywords=keywords, articles=docs, count=len(docs)))
    return result


def elbow(matrix):
    return [{"k": k, "inertia": round(float(KMeans(n_clusters=k, n_init=10, random_state=42).fit(matrix).inertia_), 3)}
            for k in range(2, 9)]


if __name__ == "__main__":
    articles, vectorizer, matrix, model = fit_articles()
    print(f"{matrix.shape[0]} artykułów, {matrix.shape[1]} cech")
    for group in clusters(articles, vectorizer, model):
        print(group["cluster_id"], group["count"], ", ".join(group["keywords"][:4]))
    print("Elbow:", elbow(matrix))
