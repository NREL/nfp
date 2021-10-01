import nfp.preprocessing.crystal_preprocessor


def test_pymatgen_preprocessor(structure_inputs):

    preprocessor = nfp.preprocessing.crystal_preprocessor.PymatgenPreprocessor()
    inputs = preprocessor(structure_inputs[0], train=True)

