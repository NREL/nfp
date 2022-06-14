import numpy as np
import pandas as pd
from nfp.preprocessing.mol_preprocessor import SmilesPreprocessor
from tensorflow.keras.utils import get_file
from tqdm.auto import tqdm

from spektral.data import Dataset, Graph
from spektral.utils import sparse

# Load the YSI dataset and split into train and validation sets
ysi = pd.read_csv(
    get_file(
        "ysi.csv", "https://github.com/pstjohn/YSIs_for_prediction/raw/master/ysi.csv"
    )
)

valid = ysi.sample(50, random_state=1)
train = ysi[~ysi.index.isin(valid)].sample(frac=1.0, random_state=1)


preprocessor = SmilesPreprocessor(explicit_hs=False)


class YSIDataset(Dataset):
    def __init__(self, data, train=False):
        self.train = train
        self.data = data
        super().__init__()

    @staticmethod
    def inputs_to_graph(inputs, y):
        a, e = sparse.edge_index_to_matrix(
            edge_index=inputs["connectivity"],
            edge_weight=np.ones_like(inputs["bond"]),
            edge_features=inputs["bond"][:, np.newaxis],
        )

        x = inputs["atom"][:, np.newaxis]
        return Graph(x=x, a=a, e=e, y=y)

    def read(self):
        return [
            self.inputs_to_graph(preprocessor(row.SMILES, train=self.train), row.YSI)
            for _, row in tqdm(self.data.iterrows(), total=len(self.data))
        ]


dataset_tr = YSIDataset(train, train=True)
dataset_te = YSIDataset(valid, train=False)
