from flask import Flask, jsonify, send_from_directory
import os

# Importujemy logikę klastrowania bezpośrednio z pliku cluster.py (lub recommender.py)
if __package__:
    from .cluster import clusters, elbow, fit_articles
else:
    from cluster import clusters, elbow, fit_articles

app = Flask(__name__, static_folder="../assets", static_url_path="/assets")

articles, vectorizer, matrix, model = fit_articles()
groups = clusters(articles, vectorizer, model)


@app.get("/")
def home():
    if os.path.exists("index.html"):
        return send_from_directory(".", "index.html")
    return "API Klastrów działa! Użyj /api/clusters lub /api/elbow"


@app.get("/api/clusters")
def api_clusters():
    return jsonify(groups)


@app.get("/api/clusters/<int:cluster_id>")
def api_cluster(cluster_id):
    group = next((g for g in groups if g["cluster_id"] == cluster_id), None)
    return jsonify(group) if group else (jsonify({"error": "nie ma takiego klastra"}), 404)


@app.get("/api/elbow")
def api_elbow():
    return jsonify(elbow(matrix))


if __name__ == "__main__":
    app.run(debug=True, port=5004)