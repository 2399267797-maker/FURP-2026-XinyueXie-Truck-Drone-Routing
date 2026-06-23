"""
ETRD-NL ALNS Solver Package
"""
try:
    from .alns_solver_drone import ETRD_ALNS_Collaborative_Solver
    __all__ = ['ETRD_ALNS_Collaborative_Solver']
except ImportError as e:
    __all__ = []
    print(f"Warning: Could not import drone solver: {e}")