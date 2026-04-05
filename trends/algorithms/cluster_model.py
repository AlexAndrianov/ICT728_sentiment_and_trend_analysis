import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    return text

def build_cluster_model(posts, n_clusters=10):

    clean_posts = [clean_text(p) for p in posts]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vectorizer.fit_transform(clean_posts)

    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(X)

    score = silhouette_score(X, clusters)

    return clusters, score
