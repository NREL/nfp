import pandas as pd
import numpy as np

import warnings
from tqdm import tqdm

import gzip
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ForwardSDMolSupplier

from itertools import islice

from nfp.preprocessing import MolPreprocessor, GraphSequence
from sklearn.preprocessing import RobustScaler


df = pd.read_csv('../data/qm9.csv.gz')
df.index = df['index'].apply(lambda x: 'gdb_{}'.format(x))

f = gzip.open('../data/gdb9.sdf.gz')

mol_supplier = ForwardSDMolSupplier(f, removeHs=False)

mols = []
total_mols = len(df)

for mol in tqdm(mol_supplier, total=total_mols):
    if mol:
        mols += [(mol.GetProp('_Name'), mol, mol.GetNumAtoms())]

mols = pd.DataFrame(mols, columns=['mol_id', 'Mol', 'n_atoms'])

test = mols.sample(10000, random_state=0)
valid = mols[~mols.mol_id.isin(test.mol_id)].sample(10000, random_state=0)
train = mols[
    (~mols.mol_id.isin(test.mol_id) & ~mols.mol_id.isin(valid.mol_id))
            ].sample(frac=1., random_state=0)

train = train.set_index('mol_id')
valid = valid.set_index('mol_id')
test = test.set_index('mol_id')

df.reindex(train.index).join(train.n_atoms).to_csv('train.csv.gz', compression='gzip')
df.reindex(valid.index).join(valid.n_atoms).to_csv('valid.csv.gz', compression='gzip')
df.reindex(test.index).join(test.n_atoms).to_csv('test.csv.gz', compression='gzip')

# Preprocess molecules
def atomic_number_tokenizer(atom):
    return atom.GetAtomicNum()

preprocessor = MolPreprocessor(
    n_neighbors=100, atom_features=atomic_number_tokenizer)

inputs_train = preprocessor.fit(train.Mol)
inputs_valid = preprocessor.predict(valid.Mol)
inputs_test = preprocessor.predict(test.Mol)

import pickle
with open('processed_inputs.p', 'wb') as file:        
    pickle.dump({
        'inputs_train': inputs_train,
        'inputs_valid': inputs_valid,
        'inputs_test': inputs_test,
        'preprocessor': preprocessor,
    }, file)
