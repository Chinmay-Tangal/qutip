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

    tlist = np.linspace(0, 10, 50)

    print0(f"\nHEOM PETSc Benchmark (Ising Model)")
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
            
    state_list = [qt.basis(2, 0)] * N
    psi0 = qt.tensor(state_list)
    rho0 = qt.ket2dm(psi0)
    
    e_ops = [sz_list[-1]]

    # Benchmark Distributed 'petsc' Backend
    print0(f"\nRunning Distributed 'petsc' Backend")
    try:
        # Assembly
        t0 = time.time()
        solver_petsc = HEOMSolver(
            H, baths, max_depth=MAX_DEPTH, 
            options={
                "backend": "petsc", 
                "store_ados": True, 
                "progress_bar": None,
                "store_states": True,
                "ts_type": "bdf",       # Use implicit solver for stiff ODEs
                "ts_adapt": "basic",    # Enable adaptive step sizing
                "atol": 1e-8,
                "rtol": 1e-6,
                "max_steps": 100000
            }
        )
        if comm is not None: comm.Barrier()
        t_asm_petsc = time.time() - t0
        print0(f"   [petsc] Assembly Time    : {t_asm_petsc:.4f} s")
        print0(f"   Total ADOs               : {len(solver_petsc.ados.labels)}")
        
        # Time Evolution
        t0 = time.time()
        res_petsc = solver_petsc.run(rho0, tlist, e_ops=e_ops)
        if comm is not None: comm.Barrier()
        t_evo_petsc = time.time() - t0
        print0(f"   [petsc] Evolution Time   : {t_evo_petsc:.4f} s")
        # print0("\nInitial State")
        # print0(res_petsc.states[0].full())
        # print0("\nfinal State")
        # print0(res_petsc.states[-1].full())        
        if rank == 0:
            # Print PETSc stats
            ts = solver_petsc._integrator.ts
            print0("\nPETSc Internal Stats")
            print0(f"   TS Type          : {ts.getType()}")
            print0(f"   TS Steps         : {ts.getStepNumber()}")
            print0(f"   Final dt         : {ts.getTimeStep():.2e}")
            print0(f"   SNES Iterations  : {ts.getSNESIterations()}")
            print0(f"   KSP Iterations   : {ts.getKSPIterations()}")
            
    except Exception as e:
        import traceback
        if rank == 0:
            traceback.print_exc()
        print0(f"   [petsc] Failed: {e}")
        
    print0("\nbenchmark complete")

if __name__ == "__main__":
    run_benchmark()
