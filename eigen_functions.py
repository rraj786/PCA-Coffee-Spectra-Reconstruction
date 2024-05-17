'''
    The following script contains functions to numerically calculate eigenvalues
    and eigenvectors for a given matrix.
    Author: Rohit Rajagopal
'''


import numpy as np
from numpy.linalg import norm
import math


def power_method(A, v, tol):
     
    """
        Compute the largest eigenvalue and corresponding eigenvector for a given matrix.
        Inputs:
            - A: A 2D array of values
            - v: A list respresenting the initial unit vector
            - tol: Tolerance value to check for convergence
        Outputs:
            - eigenvalue: Largest eigenvalue
            - eigenvector: A list representing the largest eigenvector
    """

    # Normalize the vector and obtain eigenpair
    eigenvalue = norm(v)
    eigenvector = v/eigenvalue

    a = True
    while a == True:
        last = eigenvalue

        # Find dot product
        eigenvector = np.dot(A, eigenvector)

        # Find position of largest element in eigenvector
        pos = np.argmax(eigenvector)

        # Get eigenvalue estimate
        eigenvalue = norm(eigenvector)
        eigenvector = eigenvector/eigenvalue

        # Identify if convergence has occured (same eigenvalue as last iteration)
        if (abs((last - eigenvalue)/eigenvalue)) < tol:
            a = False

    # Check the sign of the eigenvalue
    if ((np.dot(A, eigenvector)[pos])/(eigenvector[pos])) < 0:
        eigenvalue = -eigenvalue
    else:
        eigenvalue = abs(eigenvalue)

    return eigenvalue, eigenvector


def deflate(A, eigenvalue, eigenvector):

    """
        Remove a given eigenvector from its corresponding matrix in place.
        Inputs:
            - A: A 2D array of values
            - eigenvalue: Largest eigenvalue
            - eigenvector: A list representing the largest eigenvector
        Outputs:
            - A_mod: A modified 2D array of values
    """

    eigenvector = np.array(eigenvector/norm(eigenvector))
    A_mod = np.array(A)

    # Perform matrix multiplication to obtain next largest eigenpair
    for i in range(len(eigenvector)):
        for j in range(len(eigenvector)):
            A_mod[i][j] = A[i][j] - (eigenvalue*eigenvector[i]*eigenvector[j])
    
    return A_mod


def test_power_method():

    """
        Test function to ensure the power_method function is working as expected.
    """    
   
    A = [[4.0, 6.0], [1.0, 3.0]]
    v = [1.0, 1.0]
    tol = 1e-7

    vect_soln = [0.94868417, 0.31622515]
    val_soln = 6
    
    eigenvalue, eigenvector = power_method(A, v, tol)
    assert abs(np.array(eigenvector)[0] - np.array(vect_soln)[0])/np.array(vect_soln)[0] < math.sqrt(tol)
    assert abs(np.array(eigenvector)[1] - np.array(vect_soln)[1])/np.array(vect_soln)[1] < math.sqrt(tol)
    assert abs(eigenvalue - val_soln)/val_soln < tol

    print("Power method test has passed")

    return 


def test_deflate():

    """
        Test function to ensure the deflate function is working as exptected.
    """

    A = [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 2.0]]
    eigenvalue = 3
    eigenvector = [1.0, 1.0, 0.0]
    tol = 1e-7

    C_soln = np.array([[-0.5, 0.5, 0], [0.5, -0.5, 0], [0, 0, 2]])

    C = deflate(A, eigenvalue, eigenvector)
    assert abs(norm(C - C_soln)) < tol

    print("Deflate test has passed")

    return
