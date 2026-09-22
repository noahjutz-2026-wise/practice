import argparse
import json
from pathlib import Path
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "super_productivity.json"


def load_data(filepath: Path):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_cluster_labels(task_texts, cluster_labels, num_clusters=10):
    """
    Extracts a readable label for each cluster using TF-IDF.
    """
    # Group texts by cluster
    cluster_docs = {i: [] for i in range(num_clusters)}
    for text, label in zip(task_texts, cluster_labels):
        cluster_docs[label].append(text)
        
    # Join texts for each cluster to form a "document" per cluster
    docs = [" ".join(cluster_docs[i]) for i in range(num_clusters)]
    
    # Use TF-IDF to find top words per cluster
    # exclude common English stop words
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    
    generated_labels = {}
    for i in range(num_clusters):
        # Get top 2-3 keywords for this cluster
        row = tfidf_matrix.getrow(i).toarray()[0]
        top_indices = row.argsort()[-3:][::-1]
        top_words = [feature_names[idx].capitalize() for idx in top_indices if row[idx] > 0]
        
        if top_words:
            generated_labels[i] = " & ".join(top_words)
        else:
            generated_labels[i] = f"Misc {i}"
            
    return generated_labels


def generate_treemaps(input_json: Path, output_dir: Path):
    data = load_data(input_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    projects = {
        p["id"]: p.get("title", "") 
        for p in data.get("project", {}).get("entities", {}).values()
    }

    all_tasks = {}
    for src in [data.get("task", {}), data.get("archiveYoung", {}).get("task", {}), data.get("archiveOld", {}).get("task", {})]:
        if src and isinstance(src, dict):
            for tid, t in src.get("entities", {}).items():
                all_tasks[tid] = t

    leaf_tasks = [t for t in all_tasks.values() if not t.get("subTaskIds")]
    
    # Filter out tasks with zero time spent to avoid cluttering the visual with empty tasks
    leaf_tasks = [t for t in leaf_tasks if t.get("timeSpent", 0) > 0]

    print("Loading embedding model...")
    # Use SentenceTransformer to embed tasks for dynamic clustering
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"Processing {len(leaf_tasks)} tasks...")
    task_records = []
    contexts = []

    for t in leaf_tasks:
        proj_id = t.get("projectId")
        proj = projects.get(proj_id, "Uncategorized")
        title = t.get("title", "Untitled")
        par_id = t.get("parentId")
        parent_title = all_tasks.get(par_id, {}).get("title", "") if par_id else ""

        # Rich context for embedding
        context = f"{proj}: {parent_title} - {title}"
        contexts.append(context)

        hours = t.get("timeSpent", 0) / 3600000.0
        task_records.append({
            "id": t["id"],
            "title": title,
            "project": proj,
            "hours": hours,
            "context": context
        })

    # 1. Embed contexts using HuggingFace model
    print("Computing embeddings...")
    embeddings = model.encode(contexts)

    # 2. Cluster into 10 bins
    num_bins = min(10, len(leaf_tasks))  # Ensure we don't request more bins than tasks
    print(f"Clustering into {num_bins} dynamic bins...")
    kmeans = KMeans(n_clusters=num_bins, random_state=42)
    cluster_assignments = kmeans.fit_predict(embeddings)

    # 3. Generate candidate labels for each cluster using TF-IDF
    cluster_names = get_cluster_labels(contexts, cluster_assignments, num_clusters=num_bins)

    # Assign dynamic bins to tasks
    for r, cluster_id in zip(task_records, cluster_assignments):
        r["bin"] = cluster_names[cluster_id]

    # Calculate bin totals
    bin_totals = {}
    for r in task_records:
        b = r["bin"]
        bin_totals[b] = bin_totals.get(b, 0.0) + r["hours"]

    # Build Treemap data (Total -> Bin -> Task)
    ids = ["Total"]
    labels = ["Total Portfolio"]
    parents = [""]
    values = [sum(bin_totals.values())]

    # Add Bins
    for b in sorted(bin_totals.keys()):
        if bin_totals[b] > 0:
            ids.append(f"bin_{b}")
            labels.append(f"<b>{b}</b><br>{bin_totals[b]:.2f} hrs")
            parents.append("Total")
            values.append(bin_totals[b])

    # Add Tasks
    for r in task_records:
        b = r["bin"]
        tid = r["id"]
        h = r["hours"]
        
        # Ensure task IDs are unique in the treemap
        task_id = f"task_{tid}"
        ids.append(task_id)
        labels.append(r["title"])
        parents.append(f"bin_{b}")
        values.append(h)

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        textinfo="label+value",
        maxdepth=3,
        pathbar=dict(visible=True),
    ))

    fig.update_layout(
        title=dict(
            text="<b>SuperProductivity Tasks Breakdown (AI Clustered Bins)</b>",
            font=dict(size=18),
        ),
        margin=dict(t=50, l=15, r=15, b=15),
        height=900,
    )

    out_file = output_dir / "super_productivity_treemap_tasks.html"
    fig.write_html(str(out_file), include_plotlyjs="cdn")
    print(f"Saved: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate AI-clustered Treemap")
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    generate_treemaps(input_json=args.input, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
