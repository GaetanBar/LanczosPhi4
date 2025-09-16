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

    



class Energy_encoding():
    def __init__(self, Nmax, n_lvl,L, m, lambd):
        self.L = L
        self.n_lvl = n_lvl
        self.Nmax = Nmax
        self.N_modes = 2 * Nmax + 1
        self.m = m
        self.lambd = lambd
        

    def local_annihilation(self):
        """
        Local annihilation operator in the truncated Fock basis 0..nlvl.
        Dimension dloc = nlvl + 1.
        """
        d = self.n_lvl + 1
        a = np.zeros((d, d), dtype=float)
        for p in range(d-1):
            a[p, p+1] = np.sqrt(p+1)
        return a

    
    def build_mode_operators(self):
        """
        Build full-space annihilation/creation/number operators for modes n=-Nmax..Nmax.
        Returns:
        - n_list: array of integer mode indices (len N_modes = 2*Nmax+1)
        - omegas: np.array of frequencies length N_modes (matching order in n_list)
        - ops: dict with 'a' list, 'ad' list, 'n' list where each entry is a full-space matrix.
        - dloc: local dimension per mode
        """
        # mode integers and frequencies
        n_list = np.arange(-self.Nmax, self.Nmax+1)    # e.g. [-Nmax, ..., 0, ..., Nmax]
        ks = 2.0 * np.pi * n_list / self.L
        omegas = np.sqrt(ks**2 + self.m**2)

        # local truncated operators
        dloc = self.n_lvl + 1
        a_loc = self.local_annihilation()
        
        adag_loc = a_loc.T
        
        I_loc = np.eye(dloc)

        # full-space dimension
        dim = dloc ** self.N_modes

        # precompute Kronecker embeddings for each mode
        a_list = []
        adag_list = []
        n_list_ops = []
        for j in range(self.N_modes):
            op = None
            # build kron(I,...,I, a_loc, I,...,I)
            for s in range(self.N_modes):
                mat = a_loc if s == j else I_loc
                op = mat if op is None else np.kron(op, mat)
            op_a = op
            op_adag = op_a.T  # real matrices; in complex case use .conj().T
            op_n = op_adag @ op_a
            a_list.append(op_a)
            adag_list.append(op_adag)
            n_list_ops.append(op_n)


            ops = {"a": a_list, "ad": adag_list, "n": n_list_ops}
            return n_list, omegas, ops, dloc


    def build_free_hamiltonian(self):
        """
        Build free Hamiltonian H0 = sum_k omega_k (a_k^† a_k + 1/2).
        """
        n_list, omegas, ops, dloc = self.build_mode_operators()

        dim = ops['n'][0].shape[0]
        H0 = np.zeros((dim, dim), dtype=float)
        for j in range(self.N_modes):
            H0 += omegas[j] * (ops['n'][j])
        return H0

    def build_phi4_from_mode_sum(self):
        """
        Build H_int = (lambda/4!) * ∫ dx phi(x)^4 using the exact momentum-conserving mode sum.
        Implementation uses:
        phi(x) = sum_k (1/sqrt(2 L omega_k)) [ a_k e^{ikx} + a_k^† e^{-ikx} ]
        After integrating over x, the coefficient becomes:
        pref_global = lambda / (4! * (2L)^2) * L = lambda / (96 L)
        and the operator sum is over all (k1,k2,k3,k4) with k1+...+k4 = 0:
        H_int = pref_global * sum_{k1+k2+k3+k4=0} (1/sqrt(omega1*...*omega4))
                    * (a_{k1} + a_{-k1}^†) (a_{k2} + a_{-k2}^†) (a_{k3} + a_{-k3}^†) (a_{k4} + a_{-k4}^†)
        """
        n_list , omegas, ops, dloc = self.build_mode_operators()
        N_modes = len(n_list)
        # mapping from integer mode index (n) to position in arrays
        n_to_idx = {int(n_list[i]): i for i in range(N_modes)}

        pref_global = self.lambd / (96.0 * self.L)  # derived prefactor

        dim = ops['a'][0].shape[0]
        H_int = np.zeros((dim, dim), dtype=float)

        # For efficiency iterate over triples and compute required fourth mode
        # Loop over indices i,j,k and compute l = - (n_i + n_j + n_k)
        # If l present in n_to_idx, include the quartet (i,j,k,l).
        # This enumerates all ordered quartets consistent with momentum conservation.
        for i_idx, n_i in enumerate(n_list):
            for j_idx, n_j in enumerate(n_list):
                for k_idx, n_k in enumerate(n_list):
                    n_l = int(-(n_i + n_j + n_k))
                    # check if n_l is in our truncated set
                    if n_l not in n_to_idx:
                        continue
                    l_idx = n_to_idx[n_l]
                    # prepare each factor (a_{n} + a_{-n}^†)
                    # note: operator for -n is at index n_to_idx[-n]
                    # guard against missing -n in truncated set (should be present if symmetric truncation)
                    op_i = ops['a'][i_idx] + ops['ad'][ n_to_idx.get(-int(n_i), i_idx) ]
                    op_j = ops['a'][j_idx] + ops['ad'][ n_to_idx.get(-int(n_j), j_idx) ]
                    op_k = ops['a'][k_idx] + ops['ad'][ n_to_idx.get(-int(n_k), k_idx) ]
                    op_l = ops['a'][l_idx]    + ops['ad'][ n_to_idx.get(-int(n_l), l_idx) ]

                    # product operator (order matters) — we follow the given ordering
                    op_prod = op_i @ op_j @ op_k @ op_l

                    # multiply by 1/sqrt(omega_i * omega_j * omega_k * omega_l)
                    omega_prod_sqrt = np.sqrt(omegas[i_idx] * omegas[j_idx] * omegas[k_idx] * omegas[l_idx])
                    coeff = pref_global / omega_prod_sqrt

                    H_int += coeff * op_prod

        # H_int should be Hermitian (within numerical noise) — symmetrize to be safe
        H_int = 0.5 * (H_int + H_int.T)
        return H_int

    def build_full_hamiltonian(self):
        """
        Top-level builder that returns (H, H0, H_int, n_list, omegas, ops).
        """
        n_list, omegas, ops, dloc = self.build_mode_operators()
        H0 = self.build_free_hamiltonian()
        if lam != 0.0:
            H_int = self.build_phi4_from_mode_sum()
        else:
            H_int = np.zeros_like(H0)
        H = H0 + H_int
        return H, H0, H_int, n_list, omegas, ops