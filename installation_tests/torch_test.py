"""
torch_test.py

Run this little script to see if PyTorch is installed properly.
The program should end with the following line in your terminal:

`Sum of 3.0 and 4.0 is 7.0`

After printing, the program should exit successfully.
"""
import torch

a = torch.tensor(3.0)
b = torch.tensor(4.0)
result = a + b

print(f"Sum of {a.item()} and {b.item()} is {result.item()}")