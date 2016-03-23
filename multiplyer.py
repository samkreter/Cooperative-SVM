# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:23:20 2016

@author: brendanmarsh
"""

import numpy as np
import cvxopt
import math


def CalculateLagrangeMultipliers(t,K,C):

    # cvxopt requires optimizations in the form min (1/2)x^T*P*x + q^T*x
    # Since the Lagrangian is something to be maximized, we flip the sign and
    # compute P and q as follows.

    # The P matrix is computed using np.outer for the training points t, which
    # computes all second order products of the elements of t, and K is the Gram matrix.
    P = cvxopt.matrix(K * np.outer(t,t))

    # The Q vector is simply -1 since the an's of first order have no other coefficients.
    (rows,columns) = t.shape
    n = columns
    q = cvxopt.matrix(-1 * np.eye(rows,columns))

    #cvxopt requires constraints in the form:
    # Gx <= h    and    Ax = b
    # 0 <= an <= C for some parameter C. This is really two constraints, -an <= 0 and an <= C.
    # The -an <= 0 is the normal separable constraint, and the C parameter arises from the slack variables.

    # Constraint -an <= 0
    G_seperable = cvxopt.matrix(-1 * np.eye(n))
    h_seperable = cvxopt.matrix(np.zeros(n))

    # Constraint an <= C
    G_slack = cvxopt.matrix(np.eye(n))
    h_slack = cvxopt.matrix(np.ones(n)*C)

    # To implement both of these constraints on an in a single matrix form, we stack the two matrices into one.
    G = cvxopt.matrix(np.vstack((G_seperable,G_slack)))
    h = cvxopt.matrix(np.vstack((h_seperable,h_slack)))

    # Implement the box constraints sum(an*tn) = 0
    #A = cvxopt.matrix(np.eye(n)@t)
    b = cvxopt.matrix(np.zeros(n))

    qp_result = cvxopt.solvers.qp(P,q)
    multipliers = qp_result['x']

    return multipliers


