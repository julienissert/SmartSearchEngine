# src/ingestion/loaders/h5_loader.py
import h5py
from ingestion.loaders.base_loader import BaseLoader

class H5Loader(BaseLoader):
    def get_supported_extensions(self):
        return [".h5", ".hdf5"]

    def can_handle(self, extension: str) -> bool:
        return extension.lower() in self.get_supported_extensions()

    def load(self, path: str, valid_labels=None) -> list:
        docs = []
        try:
            with h5py.File(path, "r") as f:
                def visit(name, node):
                    if isinstance(node, h5py.Dataset):
                        if node.size * node.dtype.itemsize > 10 * 1024 * 1024:
                            content_summary = f"Large Dataset: shape={node.shape}, dtype={node.dtype}"
                        else:
                            try:
                                content_summary = str(node[()])
                            except:
                                content_summary = "Non-text binary data"
                        
                        attrs = {k: str(v) for k, v in node.attrs.items()}
                        suggested = attrs.get("label") or attrs.get("category") or attrs.get("class")
                        
                        docs.append({
                            "source": path,
                            "type": "h5",
                            "content": {name: content_summary},
                            "suggested_label": suggested 
                        })
                f.visititems(visit)
        except Exception:
            pass 
        return docs