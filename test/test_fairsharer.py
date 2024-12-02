import pytest
from src import fairsharer as fs

def test_typechecker():
    assert fs.typechecker(1, int) == "int"
    assert fs.typechecker(1.5, float) == "float"
    assert fs.typechecker("Check this out", str) == "str"
    assert fs.typechecker([1, 2, 3], str) == 
