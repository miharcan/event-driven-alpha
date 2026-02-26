from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

class NewsEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_headlines(self, df, text_col="headline"):
        embeddings = self.model.encode(
            df[text_col].tolist(),
            show_progress_bar=True
        )
        return np.array(embeddings)

    def aggregate_daily(self, df, embeddings, date_col=None):
        """
        Aggregate headline embeddings to daily mean vectors.
        Works when df has DatetimeIndex (your case).
        """

        # Use index (your loader sets date as index)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex.")

        # Normalize to daily (remove time component if exists)
        date_only = df.index.normalize()

        # Create embedding dataframe WITHOUT carrying index name
        df_emb = pd.DataFrame(embeddings)
        df_emb["group_date"] = date_only.values  # use different name

        daily = df_emb.groupby("group_date").mean()

        # Ensure datetime index
        daily.index = pd.to_datetime(daily.index)

        return daily


def reduce_embeddings(embeddings_df, n_components=20):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings_df.values)
    
    reduced_df = pd.DataFrame(
        reduced,
        index=embeddings_df.index,
        columns=[f"news_pca_{i}" for i in range(n_components)]
    )
    return reduced_df