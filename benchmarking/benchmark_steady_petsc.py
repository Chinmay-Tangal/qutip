import time
import numpy as np
import qutip as qt
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    comm = None
    rank = 0
    size = 1

def print0(*args):
    if rank == 0:
        print(*args, flush=True)

def run_benchmark():
    # System Parameters (Ising Model)
    N = 4 # number of spins in spin-chain
    g0 = 1 
    J0 = 1.4    

    MAX_DEPTH = 6
    bath_coupling_points = [1, 1, 0, 0]  # Baths on qubits 0 and 1
    temperatures = [1, 1.2, 0.7, 0.5]
    Nks = [4, 4, 4, 4]

    print0(f"\nHEOM PETSc STEADY-STATE Benchmark (Ising Model)")
    print0(f"MPI Ranks = {size}")
    print0(f"Spins (N) = {N}")
    print0(f"Depth     = {MAX_DEPTH}")
    print0(f"Baths     = {sum(bath_coupling_points)}")
    
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

    # Benchmark Distributed 'petsc' Backend for Steady State
    print0(f"\nRunning Distributed 'petsc' Backend (KSP Solver)")
    try:
        # Assembly
        t0 = time.time()
        solver_petsc = HEOMSolver(
            H, baths, max_depth=MAX_DEPTH, 
            options={"backend": "petsc"}
        )
        if comm is not None: comm.Barrier()
        t_asm_petsc = time.time() - t0
        print0(f"   [petsc] Assembly Time    : {t_asm_petsc:.4f} s")
        print0(f"   Total ADOs               : {len(solver_petsc.ados.labels)}")
        
        # Steady State Solve
        t0 = time.time()
        # ** Here we call steady_state instead of run **
        steady_rho, steady_ados = solver_petsc.steady_state(tol=1e-8, atol=1e-8, ksp_type="gmres", pc_type="none")
        if comm is not None: comm.Barrier()
        t_solve_petsc = time.time() - t0
        print0(f"   [petsc] Solve Time       : {t_solve_petsc:.4f} s")
        
        # Verify trace is 1
        if rank == 0:
            print0(f"   Trace of steady state    : {steady_rho.tr():.6f}")

    except Exception as e:
        import traceback
        if rank == 0:
            traceback.print_exc()
        print0(f"   [petsc] Failed: {e}")
        
    print0("\nBenchmark complete")

if __name__ == "__main__":
    run_benchmark()
