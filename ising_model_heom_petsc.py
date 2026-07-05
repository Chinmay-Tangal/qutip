import os
import time
from datetime import datetime
import qutip as qt
import numpy as np


from qutip.solver.heom import HEOMSolver, DrudeLorentzBath

def Ising_solve_petsc(N, g0, J0, tlist, max_depth, 
                 Nks, bath_coupling_points, temperatures):
    #N: number of spins
    #g0: energy splitting
    #J0: nearest neighbour coupling
    #gamma: single decay rate
    #tlist: list of time steps

    #max_depth: maximum depth of the hierarchy
    #bath_coupling_points: list of qubit indices that couple to baths
    #temperatures: list of temperatures for each bath

    # Setup operators for individual qubits
    g = g0 * np.ones(N) # Energy splitting term
    J = J0 * np.ones(N) # Interaction coefficients
    sx_list, sy_list, sz_list = [], [], []

    for i in range(N):
        op_list = [qt.qeye(2)] * N
        op_list[i] = qt.sigmax()
        sx_list.append(qt.tensor(op_list))
        op_list[i] = qt.sigmay()
        sy_list.append(qt.tensor(op_list))
        op_list[i] = qt.sigmaz()
        sz_list.append(qt.tensor(op_list))
 
    # Hamiltonian - Energy splitting terms
    H = 0.
    for i in range(N):
        H += g[i] * sz_list[i]

    # Interaction terms
    for n in range(N - 1):
        H += -J[n] * sx_list[n] * sx_list[n + 1]

    #########################################
    baths = []

    for site in range(N):
        if bath_coupling_points[site]==1:
            baths.append(DrudeLorentzBath(
            sz_list[site],
            lam=0.01,
            gamma=1,
            T=temperatures[site],
            Nk=Nks[site],
            ))
            
    state_list = [qt.basis(2, 0)] * (N)
    psi0 = qt.tensor(state_list)
    rho0 = qt.ket2dm(psi0)
    
    e_ops = [sz_list[-1]]
    
    # Configure solver to use PETSc backend with optimized options
    solver = HEOMSolver(
        H,
        baths,
        max_depth=max_depth,
        options={
            "nsteps": 10000, 
            "progress_bar": True,
            "backend": "petsc",
            "ts_type": "bdf",       # Use implicit solver for stiff ODEs
            "ts_adapt": "basic",    # Enable adaptive step sizing
        }
    )

    ado_count = len(solver.ados.labels)
    result = solver.run(rho0, tlist, e_ops=e_ops)
    

    print(
        f"Depth={max_depth}, "
        f"ADOs={ado_count}, "
        f"Runtime={result.stats['run time']:.3f} s"
    )

    return result, result.e_data[0]

#ising model spin paramaters:
N = 4 #number of spins in spin-chain
g0 = 1 
J0 = 1.4    

tlist = np.linspace(0, 10, 100)

#bath parameters:
MAX_DEPTH = 6
bath_coupling_points = [1, 1, 0, 0]  # Baths on qubits 0 and 3. This list should be length N, with 1 indicating a bath is coupled to that qubit, and 0 indicating no bath is coupled to that qubit.
temperatures = [1, 1.2, 0.7, 0.5]  # Temperatures (only qubit 0 is coupled to a bath, others dont matter as bath_coupling_points is 0 for them).  This list should be length N, with the temperature for each bath.  For qubits that are not coupled to a bath, the temperature value does not matter.
Nks = [4,4,4,4]  # Number of Matsubara terms for each bath (only qubit 0 is coupled to a bath, others dont matter as bath_coupling_points is 0 for them). This list should be length N, with the number of Matsubara terms for each bath.  For qubits that are not coupled to a bath, the Nks value does not matter.

print(f"starting N={N}, number of baths = {len([i for i in bath_coupling_points if i == 1])}")

start = time.time()
result, sz = Ising_solve_petsc(N, g0, J0, tlist, MAX_DEPTH, Nks, bath_coupling_points, temperatures)
et = time.time() - start
rt = result.stats['run time']
print(f"PETSc backend N={N}: runtime {rt} s, walltime {et}")
