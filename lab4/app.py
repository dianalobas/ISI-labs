from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Uruchom: python app.py; potem otwórz http://127.0.0.1:5004/api/clusters."""
from flask import Flask, jsonify, send_from_directory
from course import materials, page
if __package__:
    from .cluster import clusters, elbow, fit_articles
else:
    from cluster import clusters, elbow, fit_articles

app = Flask(__name__, static_folder="../assets", static_url_path="/assets")

articles, vectorizer, matrix, model = fit_articles()
groups = clusters(articles, vectorizer, model)


@app.get("/")
def home():
    return send_from_directory(app.root_path, "index.html")


@app.get("/<name>")
def source(name):
    return page(4, name)


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


app.register_blueprint(materials)


if __name__ == "__main__":
    app.run(debug=True, port=5004)
