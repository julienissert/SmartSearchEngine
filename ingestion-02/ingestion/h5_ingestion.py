# scripts/ingestion/h5_ingestion.py
import h5py

def load_h5(path):
    docs = []
    try:
        with h5py.File(path, "r") as f:
            def visit(name, node):
                if isinstance(node, h5py.Dataset):
                    # Limite de sécurité (10 MB) pour éviter de saturer la RAM
                    if node.size * node.dtype.itemsize > 10 * 1024 * 1024:
                        content_summary = f"Large Dataset: shape={node.shape}, dtype={node.dtype}"
                    else:
                        try:
                            content_summary = str(node[()])
                        except:
                            content_summary = "Non-text binary data"

                    docs.append({
                        "source": path,
                        "type": "h5",
                        "dataset": name,
                        "content": content_summary
                    })
            f.visititems(visit)
    except Exception:
        pass 

    return docs