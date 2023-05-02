import torch

def det(matrix: torch.Tensor) -> float:
    
    return

def test_det():
    matrix = torch.tensor([[3,  1, 1],
                           [4, -2, 5],
                           [2,  8, 7]])
    assert det(matrix) == -144


# A -> Matrix. Can be represented in a tensor
# v -> Vector. Also can be represented as a tensor
# lambda -> Eigenvalue
# Av = λv
# Av - λv = 0
# v(A - λ) = 0?
def eigen_computation(tensor: torch.Tensor, vector: torch.Tensor) -> float:



def test_add():
    assert eigen_computation(2, 3) == 5
