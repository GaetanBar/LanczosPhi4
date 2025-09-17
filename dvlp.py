from test import Lattice_Encoding, Energy_encoding
import sys
import scipy


def main(argv):

    test = Energy_encoding(Nmax= 4, n_lvl=1, L=5., m=1., lambd=1)
    test2 = Lattice_Encoding(L = 5., a = 1. , m = 1. ,lambd = 1, phi_max= 1 , n_q= 1)
    H1= test.build_full_hamiltonian()
    H2=test2.build_H()  
    print("H1=", H1)
    print(H1.shape)
    print("H2=", H2)
    print(H2.shape)

    eig1= scipy.linalg.eigvalsh(H1)
    eig2= scipy.linalg.eigvalsh(H2) 
    print('first eigenvalues of H1=', eig1[:3])
    print('first eigenvalues of H2=', eig2[:3])

if __name__ == "__main__":
    main(sys.argv)
