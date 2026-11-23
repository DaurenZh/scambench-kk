"""Load non-pickle TF-IDF/SVM exports for local conversation experiments."""
import json
import pathlib

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


class PortableClassifier:
    def __init__(self, folder, target):
        root=pathlib.Path(folder)
        self.metadata=json.loads((root/f"{target}.json").read_text(encoding="utf-8"))
        arrays=np.load(root/f"{target}.npz",allow_pickle=False)
        self.coefficients=arrays["coefficients"]
        self.intercept=arrays["intercept"]
        self.indices=arrays["class_indices"]
        self.labels=self.metadata["labels"]
        self.vectorizers=[]
        for name in self.metadata["vectorizers"]:
            config=json.loads((root/f"{target}_{name}.json").read_text(encoding="utf-8"))
            params=config["parameters"]
            params["ngram_range"]=tuple(params["ngram_range"])
            vectorizer=TfidfVectorizer(vocabulary=config["vocabulary"],**params)
            vectorizer.idf_=arrays[f"idf_{name}"]
            self.vectorizers.append(vectorizer)

    def predict(self, texts):
        matrices=[v.transform(texts) for v in self.vectorizers]
        x=matrices[0] if len(matrices)==1 else hstack(matrices,format="csr")
        margins=np.asarray(x@self.coefficients.T)+self.intercept
        indices=(margins[:,0]>0).astype(int) if margins.shape[1]==1 else margins.argmax(axis=1)
        return [self.labels[int(self.indices[i])] for i in indices]


def flatten(messages):
    """Exactly the preprocessing used by the classifiers: contents in turn order."""
    if not isinstance(messages,list) or not messages:
        raise ValueError("Conversation must be a nonempty message list")
    if any(not isinstance(m,dict) or not isinstance(m.get("content"),str) for m in messages):
        raise ValueError("Each message needs a string content field")
    return "\n".join(m["content"] for m in messages)
