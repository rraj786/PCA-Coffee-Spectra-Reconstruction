'''
    The following script contains functions to numerically calculate eigenvalues
    and eigenvectors for a given matrix.
    Author: Rohit Rajagopal
'''


import numpy as np
from numpy.linalg import norm


def center_matrix(A):
     
    """
        Calculate the centered matrix for a given input using its column means.
        Inputs:
            - A: A 2D array of values
        Outputs:
            - A_centered: A 2D centered matrix 
    """

    # Find column means
    means = A.mean(axis = 0)
    
    # Subtract column means from each column
    A_centered = A - means

    return A_centered


def projection(A, prin_comp):
     
    """
        Project an input matrix into Eigenspace for dimensionality reduction.
        Inputs:
            - A: A 2D array of values
            - prin_comp: A 2D array of Eigenvectors
        Outputs:
            - projected: A 2D array of the input matrix projected in Eigenspace
    """

    projected = np.matmul(A, prin_comp.transpose())

    return projected


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

    loop = True
    while loop:
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
