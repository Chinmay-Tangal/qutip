import time
import numpy as np
import qutip as qt
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath

def run_benchmark():
    # System Parameters (Ising Model)
    N = 4 # number of spins in spin-chain
    g0 = 1 
    J0 = 1.4    

    MAX_DEPTH = 6
    bath_coupling_points = [1, 1, 0, 0]  # Baths on qubits 0 and 1
    temperatures = [1, 1.2, 0.7, 0.5]
    Nks = [4, 4, 4, 4]

    tlist = np.linspace(0, 10, 50)

    print(f"\nHEOM Traditional CSR Benchmark (Ising Model)")
    print(f"Spins (N) = {N}")
    print(f"Depth     = {MAX_DEPTH}")
    print(f"Baths     = {sum(bath_coupling_points)}")
    
    # Setup operators for individual qubits
    g = g0 * np.ones(N)
    J = J0 * np.ones(N)
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

    # Baths
    baths = []
    for site in range(N):
        if bath_coupling_points[site] == 1:
            baths.append(DrudeLorentzBath(
                sz_list[site],
                lam=0.01,
                gamma=1,
                T=temperatures[site],
                Nk=Nks[site],
            ))
            
    state_list = [qt.basis(2, 0)] * N
    psi0 = qt.tensor(state_list)
    rho0 = qt.ket2dm(psi0)
    
    e_ops = [sz_list[-1]]

    # Benchmark Standard 'csr' Backend
    print("\nRunning Standard 'csr' Backend")
    try:
        # Assembly
        t0 = time.time()
        solver_csr = HEOMSolver(
            H, baths, max_depth=MAX_DEPTH, 
            options={"backend": "csr", "store_ados": False, "progress_bar": None}
        )
        t_asm_csr = time.time() - t0
        print(f"   [csr] Assembly Time      : {t_asm_csr:.4f} s")
        print(f"   Total ADOs               : {len(solver_csr.ados.labels)}")
        
        # Time Evolution
        t0 = time.time()
        res_csr = solver_csr.run(rho0, tlist, e_ops=e_ops)
        t_evo_csr = time.time() - t0
        print(f"   [csr] Evolution Time     : {t_evo_csr:.4f} s")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"   [csr] Failed: {e}")

    print("\nBenchmark complete")

if __name__ == "__main__":
    run_benchmark()
