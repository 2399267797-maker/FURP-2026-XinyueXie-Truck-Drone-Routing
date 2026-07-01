import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nsga2_vrp import NSGA2VRP

__all__ = ['NSGA2VRP']
