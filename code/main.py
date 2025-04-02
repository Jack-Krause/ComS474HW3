from helper import load_data, show_images
from solution import PCA, reconstruct_error
import numpy as np
import os

def test_pca():
    cwd = os.getcwd()
    dataloc = os.path.join(cwd, "data", "USPS.mat")
    output_dir = os.path.join(cwd, "outputs")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not os.path.isfile(dataloc):
        raise FileNotFoundError(f"file not found: {dataloc}")

    A = load_data(dataloc)
    ps = [10, 50, 100, 200]
    for p in ps:
        pca = PCA(A, p)
        Ap = pca.get_reduced()
        A_re = pca.reconstruction(Ap)
        error = reconstruct_error(A, A_re)
        print('Reconstruction error for p = %d is %.4f' % (p, error))
        show_images(A_re, p, 1, output_dir)
        show_images(A_re, p, 2, output_dir)

if __name__ == '__main__':
    test_pca()
