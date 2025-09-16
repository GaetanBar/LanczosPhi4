import numpy as np
import scipy as scp

class Lattice_Encoding():
    def __init__(self, L, a, m, lambd, phi_max, n_q):
        # Store parameters
        self.L = L
        self.a = a
        self.m = m
        self.lambd = lambd
        self.phi_max = phi_max
        self.n_q = n_q


        # Check divisibility
        if self.L % self.a != 0:
            raise ValueError("a must divide L exactly (L % a == 0)")

        self.n_sites = int(self.L // self.a)
        self.delta_phi = self.phi_max / (2**self.n_q - 1)
        self.N_phi=int(2*self.phi_max/self.delta_phi + 1)


    def phi(self):
        phi_grid = np.linspace(-self.phi_max, self.phi_max, self.N_phi)
        return phi_grid

    def single_site_kinetic_matrix(self):
        D = np.zeros((self.N_phi,self.N_phi ), dtype=float)
        for i in range(self.N_phi):
            if i - 1 >= 0:
                D[i, i-1] = 1.0
            D[i, i] = -2.0
            if i + 1 < self.N_phi:
                D[i, i+1] = 1.0
        D /= self.delta_phi**2
        K = -(1.0/(2.0*self.a)) * D
        return K

    
    def full_kinetic(self):
        K_single = self.single_site_kinetic_matrix()
        full_dim = self.N_phi**self.n_sites
        T = np.zeros((full_dim, full_dim), dtype=float)
        for site in range(self.n_sites):
            op = None
            for s in range(self.n_sites):
                mat = K_single if s == site else np.eye(self.N_phi)
                op = mat if op is None else np.kron(op, mat)
            T += op
        return T


    def potential_diagonal_with_phi4(self):
        """
        Build the potential energy operator (diagonal in amplitude basis),
        now including a phi^4 term: sum_i a*(lam/4!)*phi_i^4.
        Returns (V_diag_matrix, Mmat) where V_diag_matrix is diagonal in full Hilbert space.
        """
        # Lattice coupling matrix Mmat (N x N) from mass + nearest-neighbour gradient
        Mmat = np.zeros((self.n_sites, self.n_sites))
        diag = self.a * self.m**2 + 2.0 / self.a
        off = -1.0 / self.a
        for i in range(self.n_sites):
            Mmat[i, i] = diag
            Mmat[i, (i+1) % self.n_sites] = off
            Mmat[i, (i-1) % self.n_sites] = off

        # Enumerate all grid points (phi_0,...,phi_{N-1})
        grids = np.array(np.meshgrid(*([self.phi()]*self.n_sites), indexing='ij'))  # shape (N, M^N, ...)
        coords = grids.reshape(self.n_sites, -1)  # shape (N, M^N) columns are configurations



        # Quadratic part: 0.5 * phi^T Mmat phi for each config
        potentials_quad = 0.5 * np.einsum("ij,ji->i", coords.T @ Mmat, coords)  # length M^N

        # Phi^4 part: sum_i a*(lam/4!)*phi_i^4
        if self.lambd != 0.0:
            phi4_per_site = coords**4                    # shape (N, M^N)
            sum_phi4 = np.sum(phi4_per_site, axis=0)     # length M^N
            coeff = self.a * self.lambd / 24.0                       # lam/4! = lam/24
            potentials_phi4 = coeff * sum_phi4
        else:
            potentials_phi4 = np.zeros_like(potentials_quad)

        potentials_total = potentials_quad + potentials_phi4
        
        return np.diag(potentials_total)
 
    def build_H(self):
        # Placeholder for Hamiltonian construction
        T = self.full_kinetic()
        V = self.potential_diagonal_with_phi4()
        H = T + V
        return H


    def save_H_lattice(self, filename):
        H = self.build_H()
        np.savetxt(filename, H)

    


class Energy_enconding():
    def __init__(self, n_lvl, n_occ,m, lambd):
        self.L = L
        self.n_lvl = n_lvl
        self.n_occ = n_occ
        self.m = m
        self.lambd = lambdr
        
