"""
import_auto_mpg.py

Import the auto MPG dataset in case you do not have the download.
"""
from sklearn.datasets import fetch_openml

# Load Auto MPG dataset
auto_mpg = fetch_openml(name='autoMpg', version=1)
X, y = auto_mpg.data, auto_mpg.target
