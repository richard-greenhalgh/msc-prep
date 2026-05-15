# tool.py
# miscellaneous tools, for debugging/optimization
# NOT part of core functionality

import os
import numpy as np
from numba import njit

@njit
def dot_product_1024(a, b):
    """
    Dot product of two float32 vectors of length 1024
    """
    acc = 0.0  # will become float32 once used with arrays

    for i in range(1024):
        acc += a[i] * b[i]

    return acc

@njit(fastmath=True)
def dot_product_1024_fast(a, b):
    acc = np.float32(0.0)
    for i in range(1024):
        acc += a[i] * b[i]
    return acc

def dump_numba_inspection(func, out_dir="numba_dump", prefix=None):
    """
    Dump Numba inspection outputs (types, LLVM IR, ASM) to files.

    Parameters:
        func    : @njit function
        out_dir : directory to write files
        prefix  : optional filename prefix (defaults to func.__name__)
    """

    if prefix is None:
        prefix = func.__name__

    os.makedirs(out_dir, exist_ok=True)

    # Ensure function has been compiled at least once
    if not func.signatures:
        raise RuntimeError(
            f"{func.__name__} has no compiled signatures yet. "
            "Call it once with real inputs before dumping."
        )

    # --- 1. TYPES (single file, includes all signatures) ---
    types_path = os.path.join(out_dir, f"{prefix}_types.txt")
    with open(types_path, "w") as f:
        # inspect_types prints to stdout, so capture via context manager
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        try:
            buffer = StringIO()
            sys.stdout = buffer
            func.inspect_types()
            f.write(buffer.getvalue())
        finally:
            sys.stdout = old_stdout

    # --- 2. LLVM + ASM per signature ---
    for i, sig in enumerate(func.signatures):
        sig_str = str(sig).replace(" ", "").replace(",", "_").replace("(", "").replace(")", "")

        # LLVM
        llvm_path = os.path.join(out_dir, f"{prefix}_llvm_sig{i}.ll")
        with open(llvm_path, "w") as f:
            f.write(func.inspect_llvm(sig))

        # ASM
        asm_path = os.path.join(out_dir, f"{prefix}_asm_sig{i}.s")
        with open(asm_path, "w") as f:
            f.write(func.inspect_asm(sig))

    print(f"Numba inspection dumped to: {os.path.abspath(out_dir)}")

def dot_product_example(dot_product):
    # create inputs
    a = np.random.randn(1024).astype(np.float32)
    b = np.random.randn(1024).astype(np.float32)

    # trigger compilation
    res = dot_product(a, b)

    # dump inspection
    dump_numba_inspection(dot_product)

#==============================================================================

if __name__ == "__main__":
    dot_product_example(dot_product_1024_fast)
