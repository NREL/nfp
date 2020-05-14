import pytest
import numpy as np
from nfp.preprocessing import SmilesPreprocessor


smiles = [
    'OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O',
    'Cc1occc1C(=O)Nc2ccccc2', 'CC(C)=CCCC(C)=CC(=O)',
    'c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43', 'c1ccsc1', 'c2ccc1scnc1c2',
    'Clc1cc(Cl)c(c(Cl)c1)c2c(Cl)cccc2Cl',
    'CC12CCC3C(CCc4cc(O)ccc34)C2CCC1O',
    'ClC4=C(Cl)C5(Cl)C3C1CC(C2OC12)C3C4(Cl)C5(Cl)Cl',
    'COc5cc4OCC3Oc2c1CC(Oc1ccc2C(=O)C3c4cc5OC)C(C)=C ', 'O=C1CCCN1',
    'Clc1ccc2ccccc2c1', 'CCCC=C', 'CCC1(C(=O)NCNC1=O)c2ccccc2',
    'CCCCCCCCCCCCCC', 'CC(C)Cl', 'CCC(C)CO', 'N#Cc1ccccc1',
    'CCOP(=S)(OCC)Oc1cc(C)nc(n1)C(C)C', 'CCCCCCCCCC(C)O',
    'Clc1ccc(c(Cl)c1)c2c(Cl)ccc(Cl)c2Cl ',
    'O=c2[nH]c1CCCc1c(=O)n2C3CCCCC3', 'CCOP(=S)(OCC)SCSCC',
    'CCOc1ccc(NC(=O)C)cc1', 'CCN(CC)c1c(cc(c(N)c1N(=O)=O)C(F)(F)F)N(=O)=O',
    'CCCCCCCO', 'Cn1c(=O)n(C)c2nc[nH]c2c1=O', 'CCCCC1(CC)C(=O)NC(=O)NC1=O',
    'ClC(Cl)=C(c1ccc(Cl)cc1)c2ccc(Cl)cc2', 'CCCCCCCC(=O)OC',
    'CCc1ccc(CC)cc1', 'CCOP(=S)(OCC)SCSC(C)(C)C',
    'COC(=O)Nc1cccc(OC(=O)Nc2cccc(C)c2)c1', 'ClC(=C)Cl',
    'Cc1cccc2c1Cc3ccccc32', 'CCCCC=O', 'N(c1ccccc1)c2ccccc2',
    'CN(C)C(=O)SCCCCOc1ccccc1', 'CCCOP(=S)(OCCC)SCC(=O)N1CCCCC1C',
    'CCCCCCCI', 'c1c(Cl)cccc1c2ccccc2', 'OCCCC=C',
    'O=C2NC(=O)C1(CCC1)C(=O)N2', 'CC(C)C1CCC(C)CC1O ', 'CC(C)OC=O',
    'CCCCCC(C)O', 'CC(=O)Nc1ccc(Br)cc1', 'c1ccccc1n2ncc(N)c(Br)c2(=O)',
    'COC(=O)C1=C(C)NC(=C(C1c2ccccc2N(=O)=O)C(=O)OC)C ',
    'c2c(C)cc1nc(C)ccc1c2', 'CCCCCCC#C', 'CCC1(C(=O)NC(=O)NC1=O)C2=CCCCC2',
    'c1ccc2c(c1)ccc3c4ccccc4ccc23', 'CCC(C)n1c(=O)[nH]c(C)c(Br)c1=O ',
    'Clc1cccc(c1Cl)c2c(Cl)c(Cl)cc(Cl)c2Cl']

@pytest.fixture(scope="module")
def get_2d_data():

    preprocessor = SmilesPreprocessor(explicit_hs=False)
    inputs = preprocessor.fit(smiles)
    y = np.random.rand(len(smiles))
    return inputs, y
