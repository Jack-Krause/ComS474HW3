from helper import load_data, show_images
from solution import PCA, reconstruct_error
import numpy as np
import os

def test_pca():
    cwd = os.getcwd()
    print(os.listdir(cwd + "/data"))
    print(cwd)
    dataloc = cwd + "/data/USPS.mat"
    output_dir = cwd + "/outputs/"

    if not os.path.isdir(cwd + output_dir):
        os.mkdir(cwd + "/outputs")

    if not os.path.isfile(dataloc):
        raise FileNotFoundError("path(s) not found")

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
