"""
import_auto_mpg.py

Importeer de auto MPG dataset uit de OpenML repo.
"""
from sklearn.datasets import fetch_openml

# Load Auto MPG dataset
auto_mpg = fetch_openml(name='autoMpg', version=1)
X, y = auto_mpg.data, auto_mpg.target
