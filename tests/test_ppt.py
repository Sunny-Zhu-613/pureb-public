import pyqet

def test_is_positive_partial_transpose():
    # false positive example from https://qetlab.com/Main_Page
    # tiles UPB/BES
    dm_tiles = pyqet.upb.load_upb('tiles', return_bes=True)[1]
    assert pyqet.ppt.is_positive_partial_transpose(dm_tiles)
    # Determined to be entangled via the realignment criterion. Reference:
    # K. Chen and L.-A. Wu. A matrix realignment method for recognizing entanglement. Quantum Inf. Comput., 3:193-202, 2003

    dim = [2,3] #this is correct for PPT
    for _ in range(100):
        pyqet.ppt.is_positive_partial_transpose
        tmp0 = pyqet.random.rand_bipartitle_state(*dim, k=1, return_dm=True)
        assert pyqet.ppt.is_positive_partial_transpose(tmp0, dim)==True
        tmp0 = pyqet.random.rand_bipartitle_state(*dim, k=2, return_dm=True)
        assert pyqet.ppt.is_positive_partial_transpose(tmp0, dim)==False
