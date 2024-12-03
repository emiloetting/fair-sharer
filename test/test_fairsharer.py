import sys; import os; sys.path.append(os.getcwd())
import pytest
import numpy as np
from src import fairsharer as fs

np.set_printoptions(suppress=True)


def test_listconverter():
    demo_array_1 = np.asarray([1,2,3])
    assert fs.list_converter(demo_array_1) == [1,2,3]

    demo_array_2 = np.matrix([1,2,3])
    assert fs.list_converter(demo_array_2) == [1,2,3]


    demo_array_3 = np.matrix([[1,2,3], [4,5,6]])
    assert fs.list_converter(demo_array_3) == [1,2,3,4,5,6]

    demo_array_4 = np.asarray([[1,2,3], [4,5,6]])
    assert fs.list_converter(demo_array_4) == [1,2,3,4,5,6]

def test_fairsharer():
    assert fs.fair_sharer(values=[0, 1000, 800, 0], num_iterations=1, share=0.1) == [100, 800, 900, 0]
    assert fs.fair_sharer(values=[0, 1000, 800, 0], num_iterations=2, share=0.1) == [100, 890, 720, 90]

    ndarray_1 = np.asarray([0, 1000, 800, 0])
    assert fs.fair_sharer(values=ndarray_1, num_iterations=1, share=0.1) == [100, 800, 900, 0]
    assert fs.fair_sharer(values=ndarray_1, num_iterations=2,share=0.1) == [100, 890, 720, 90]

    ndarray_2 = np.asarray([[0, 0], [1000, 0], [800, 0], [0, 0]])
    assert fs.fair_sharer(values=ndarray_2, num_iterations=1, share=0.1) == [0, 100, 800, 100, 800, 0, 0, 0]
    assert fs.fair_sharer(values=ndarray_2, num_iterations=2, share=0.1) == [0, 180, 640, 180, 800, 0, 0, 0]

    matrix_1 = np.matrix([0, 1000, 800, 0])
    assert fs.fair_sharer(values=matrix_1, num_iterations=1, share=0.1) == [100, 800, 900, 0]
    assert fs.fair_sharer(values=matrix_1, num_iterations=2, share=0.1) == [100, 890, 720, 90]
    
    matrix_2 = np.matrix([[0, 0], [1000, 0], [800, 0], [0, 0]])
    assert fs.fair_sharer(values=matrix_2, num_iterations=1, share=0.1) == [0, 100, 800, 100, 800, 0, 0, 0]
    assert fs.fair_sharer(values=matrix_2, num_iterations=2, share=0.1) == [0, 180, 640, 180, 800, 0, 0, 0]

    iterations = np.int64(2)
    assert fs.fair_sharer(values=[0, 1000, 800, 0], num_iterations=iterations, share=0.1) == [100, 890, 720, 90]

    with pytest.raises(TypeError) as err_info_2:
        fs.fair_sharer(values=[0, 1000, 800, 0], num_iterations="2", share=0.1)
    assert str(err_info_2.value) == "Object of unsupported type <class 'str'>."

    with pytest.raises(TypeError) as err_info_3:
        fs.fair_sharer(values=[0, 1000, 800, 0], num_iterations=2, share=["0.1"])
    assert str(err_info_3.value) == "Object of unsupported type <class 'list'>."

    iterations = "random_string"
    with pytest.raises(TypeError) as err_info_4:
        fs.fair_sharer(values=[0, 1000, 800, 0], num_iterations=iterations, share=0.1)
    assert str(err_info_4.value) == "Object of unsupported type <class 'str'>."

    with pytest.raises(TypeError) as err_info_4:
        fs.fair_sharer(values=[0, 1000, 800, 0], num_iterations=0.5, share=0.1)
    assert str(err_info_4.value) == "Object of unsupported type <class 'float'>."
