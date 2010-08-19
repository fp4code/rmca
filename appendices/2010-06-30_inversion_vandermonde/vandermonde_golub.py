# -*- coding: utf-8 -*-

# algorithm 4.6.1 de Golub

from __future__ import division
import numpy as np
import scipy.linalg as sl

def vandermat(P,dx=1):
    """Returns Vandermonde matrix
    made of 2P+1 lines
      1 x x^2 x^3 ... x^2P
    where x are regularly spaced and centered around 0

    dx is the space between two consecutive x's

    >>> vandermat(2)
    array([[ 1, -2,  4, -8, 16],
           [ 1, -1,  1, -1,  1],
           [ 1,  0,  0,  0,  0],
           [ 1,  1,  1,  1,  1],
           [ 1,  2,  4,  8, 16]])
    """
    N = 2*P+1
    x = np.arange(-P,P+1).reshape(N,1) * dx
    m_v = np.ones((N,N), x.dtype)
    m_v[:,1:] = x
    m_v = m_v.cumprod(1)
    return m_v

def vandermatx(x):
    """Return Vandermonde Matrix
    from an array of x values.
    If possible, the type is conserved.

    >>> vandermatx([1,2,3])
    array([[1, 1, 1],
           [1, 2, 4],
           [1, 3, 9]])
    """

    x = np.asarray(x)
    N = x.size
    m_v = np.ones((N,N),x.dtype)
    m_v[:,1:] = x.view().reshape(N,1)
    m_v = m_v.cumprod(1)
    return m_v

def vanderinv(v):
    """Return the inverse matrix of Vandermonde matrix v
    Cf. Golub algorithm 4.6.1
    """

    N = v.shape[0]
    f = np.eye(N)
    x = v[:,1]
    n = N-1
    for k in range(0,n):
        for i in range(n,k,-1):
            f[i,:] = (f[i,:] - f[i-1])/(x[i] - x[i-k-1])
    for k in range(n-1,-1,-1):
        for i in range(k,n):
            f[i,:] = f[i,:] - f[i+1]*x[k]
    return f

def matrix_is_zero(a, eps):
    err = np.abs(a).max()
    if err <= eps:
        return True
    else:
        print "\nshould be zero ="
        print a
        return False

def matrix_are_equal(a, b, eps):
    err = matrix_is_zero(a-b, eps)
    if err:
        return True
    else:
        print "\na ="
        print a
        print "\nb ="
        print b
        return False

def matrix_is_unit(a, eps):
    s = a.shape
    n = s[0]
    print n
    if len(s) != 2 or s[1] != n :
        raise(TypeError("argument should be a square matrix"))
    return matrix_are_equal(a, np.eye(n, dtype = a.dtype), eps)

def test_vanderinv01():
    P = 2
    v = vandermat(P,1)
    vinv = vanderinv(v)
    return matrix_is_unit(np.dot(v,vinv), 1e-15)

def test_vanderinv02():
    v = vandermatx(np.array((-2,-0.5,1,3)))
    vinv = vanderinv(v)
    return matrix_is_unit(np.dot(v,vinv), 1e-14)

def matrix_product(ml):
    return reduce(lambda x, y: np.dot(x,y), ml)

def vanderinv_algorithm_as_matrix_product(x0,x1,x2):
    v = vandermatx(np.array((x0, x1, x2)))
    D0 = np.array(((  1,   0,   0),
                   ( -1,   1,   0),
                   (  0,  -1,   1)))
    N0 = np.diagflat((1,
                       1.0/(x1-x0),
                           1.0/(x2-x1)))
    D1 = np.array(((  1,   0,   0),
                   (  0,   1,   0),
                   (  0,  -1,   1)))
    N1 = np.diagflat((1,
                           1,
                            1.0/(x2-x0)))
    F2 = np.array(((  1,   0,   0),
                   (  0,   1, -x1),
                   (  0,   0,   1)))
    F1 = np.array(((  1, -x0,   0),
                   (  0,   1, -x0),
                   (  0,   0,   1)))
    U = np.dot(F1,F2)
    L = matrix_product((N1,D1,N0,D0))
    return (U,L)
    
def test_vanderinv_algorithm_as_matrix_product():
    (x0,x1,x2) = (-1, 0.5, 3)
    (U,L) = vanderinv_algorithm_as_matrix_product(x0,x1,x2)
    vinv = vanderinv(vandermatx((x0,x1,x2)))
    return matrix_are_equal(np.dot(U,L), vinv, 1e-15)

def vanderinvrprod(f,x):
    '''Produit d'un vecteur ligne par l'inverse de la matrice de Vardermonde
    définie par un vecteur d'abscisses'''
    N = np.size(x)
    for k in range(0,N-1):
        for i in range(N-1,k,-1):
            f[i] = f[i] - f[i-1]*x[k]
        #
    #
    for k in range(N-1,0,-1):
        f[k-1] = f[k-1] - f[k]/(x[k]-x[0])
        for i in range(k,N-1):
            f[i] = f[i]/(x[i] - x[i-k]) - f[i+1]/(x[i+1] - x[i-k+1])
        #
        f[N-1] = f[N-1]/(x[N-1] - x[N-k-1])
    #
    return f

def test_vanderinvrprod():
    x = np.array([-2,-0.5,1,3])
    v = vandermatx(x)
    vinv = vanderinv(v)
    xp = np.array([10,5,3,1.0])
    lvinv = np.dot(xp.view().reshape(1,4),vinv)
    lvinv2 = vanderinvrprod(xp,x)
    return matrix_are_equal(lvinv,lvinv2,1e-14)

# Algorithme centré version 1

def vanderinvrprodP2v1(ap,x):

    xm2,xm1,x0,x1,x2 = x

    P = np.array([
            [0,0,1,0,0],
            [0,1,0,0,0],
            [0,0,0,1,0],
            [1,0,0,0,0],
            [0,0,0,0,1]])
    X0 = np.array([
            [1,0,0,0,-x0],
            [0,1,0,-x0,0],
            [0,-x0,1,0,0],
            [-x0,0,0,1,0],
            [0,0,0,0,1]])
    Xm1 = np.array([
            [1,0,0,0,-xm1],
            [0,1,0,-xm1,0],
            [0,0,1,0,0],
            [-xm1,0,0,1,0],
            [0,0,0,0,1]])
    X1 = np.array([
            [1,0,0,0,-x1],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [-x1,0,0,1,0],
            [0,0,0,0,1]])
    Xm2 = np.array([
            [1,0,0,0,-xm2],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,1,0],
            [0,0,0,0,1]])
    D4 =  np.array([
            [1,0,0,0,0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,1,0],
            [-1/(x2-xm2),0,0,0,1/(x2-xm2)]])
    D3 =  np.array([
            [-1/(x1-xm2),0,0,1/(x1-xm2),0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,1,0],
            [0,0,0,-1/(x2-xm1),1/(x2-xm1)]])
    D2 =  np.array([
            [-1/(x0-xm2),1/(x0-xm2),0,0,0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,-1/(x1-xm1),0,1/(x1-xm1),0],
            [0,0,0,-1/(x2-x0),1/(x2-x0)]])
    D1 =  np.array([
            [-1/(xm1-xm2),1/(xm1-xm2),0,0,0],
            [0,-1/(x0-xm1),1/(x0-xm1),0,0],
            [0,0,1,0,0],
            [0,0,-1/(x1-x0),1/(x1-x0),0],
            [0,0,0,-1/(x2-x1),1/(x2-x1)]])
    return matrix_product((ap,P,X0,Xm1,X1,Xm2,D4,D3,D2,D1))


def test_vanderinvrprodP2v1():
    x = np.array((-10,-5,1,2,10),np.double)
    v = vandermatx(x)
    u = np.zeros((5,5),np.double)
    u[0,:] = vanderinvrprodP2v1(v[0,:],x)
    u[1,:] = vanderinvrprodP2v1(v[1,:],x)
    u[2,:] = vanderinvrprodP2v1(v[2,:],x)
    u[3,:] = vanderinvrprodP2v1(v[3,:],x)
    u[4,:] = vanderinvrprodP2v1(v[4,:],x)
    return matrix_is_unit(u, 1e-14)


'''version 2'''

def vanderinvrprodP2v2(ap,x):

    xm2,xm1,x0,x1,x2 = x

    X0 = np.array([
            [1,-x0,0,0,0],
            [0,1,-x0,0,0],
            [0,0,1,-x0,0],
            [0,0,0,1,-x0],
            [0,0,0,0,1]])
    Xm1 = np.array([
            [1,0,0,0,0],
            [0,1,-xm1,0,0],
            [0,0,1,-xm1,0],
            [0,0,0,1,-xm1],
            [0,0,0,0,1]])
    X1 = np.array([
            [1,0,0,0,0],
            [0,1,0,0,0],
            [0,0,1,-x1,0],
            [0,0,0,1,-x1],
            [0,0,0,0,1]])
    Xm2 = np.array([
            [1,0,0,0,0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,1,-xm2],
            [0,0,0,0,1]])
    D4 =  np.array([
            [1,0,0,0,0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,1,0],
            [0,0,0,-1/(x2-xm2),1/(x2-xm2)]])
    D3 =  np.array([
            [1,0,0,0,0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,1/(x1-xm2),-1/(x1-xm2),0],
            [0,0,-1/(x2-xm1),0,1/(x2-xm1)]])
    D2 =  np.array([
            [1,0,0,0,0],
            [0,1,0,0,0],
            [0,-1/(x1-xm1),1/(x1-xm1),0,0],
            [0,1/(x0-xm2),0,-1/(x0-xm2),0],
            [0,0,-1/(x2-x0),0,1/(x2-x0)]])
    D1 =  np.array([
            [1,0,0,0,0],
            [1/(x0-xm1),-1/(x0-xm1),0,0,0],
            [-1/(x1-x0),0,1/(x1-x0),0,0],
            [0,1/(xm1-xm2),0,-1/(xm1-xm2),0],
            [0,0,-1/(x2-x1),0,1/(x2-x1)]])
    PP = np.array([
            [0,0,1,0,0],
            [0,1,0,0,0],
            [0,0,0,1,0],
            [1,0,0,0,0],
            [0,0,0,0,1]])

    m = matrix_product((ap, X0, Xm1, X1, Xm2, D4, D3, D2, D1, PP))

    return m

def test_vanderinvrprodP2v2():
    x = np.array((-10,-5,1,2,10),np.double)
    v = vandermatx(x)
    u = np.zeros((5,5),np.double)
    u[0,:] = vanderinvrprodP2v2(v[0,:],x)
    u[1,:] = vanderinvrprodP2v2(v[1,:],x)
    u[2,:] = vanderinvrprodP2v2(v[2,:],x)
    u[3,:] = vanderinvrprodP2v2(v[3,:],x)
    u[4,:] = vanderinvrprodP2v2(v[4,:],x)
    return matrix_is_unit(u, 1e-14)


def vanderinvrprodcentered(fff,x):
    '''Produit d'un vecteur ligne par l'inverse de la matrice de Vardermonde'''
    N = np.size(x)
    if N%2 == 0:
        raise(ValueError)
    #
    P=N//2
    ii = np.arange(N)
    ip = ((ii+1)//2)*(1-2*(ii%2))
    ipn = ip+P
    ipn_inv = np.ndarray(N,np.integer)
    ipn_inv[ipn] = ii
    f = fff.copy()
    for k in range(0,N-1):
        xx = x[ipn[k]]
        for i in range(N-1,k,-1):
            f[i] = f[i] - f[i-1]*xx
        #
    #
    for k in range(N-1,0,-1):
        f[k] = f[k]/(x[ipn[k]]-x[ipn[k-1]])
        f[k-1] = f[k-1]-f[k]
        di = ipn[k] - ipn[k-1]
        for i in range(k+1,N):
            f[i] = f[i]/(x[ipn[i]] - x[ipn[i]+di])
            f[i-2] = f[i-2] - f[i]
            di = -di        
        #
    #
    return f[ipn_inv]

def test_vanderinvrprodcentered():
    x = np.array((-10,-5,1,2,10),np.double)
    v = vandermatx(x)
    u = np.zeros((5,5),np.double)
    u[0,:] = vanderinvrprodcentered(v[0,:],x)
    u[1,:] = vanderinvrprodcentered(v[1,:],x)
    u[2,:] = vanderinvrprodcentered(v[2,:],x)
    u[3,:] = vanderinvrprodcentered(v[3,:],x)
    u[4,:] = vanderinvrprodcentered(v[4,:],x)
    return matrix_is_unit(u, 1e-14)


scrouitch

def vanderinvrprodcenteredmulti(xp,x):
    '''Produit d'un ensemble de vecteurs ligne par l'inverse de la matrice de Vardermonde'''
    N = x.shape[1]
    if N%2 == 0:
        raise(ValueError)
    #
    P=N//2
    ii = np.arange(N)
    ip = ((ii+1)//2)*(1-2*(ii%2))
    ipn = ip+P
    ipn_inv = np.ndarray(N,np.integer)
    ipn_inv[ipn] = ii
    f = xp.copy()
    for k in range(0,N-1):
        xx = x[:,ipn[k]]
        for i in range(N-1,k,-1):
            f[:,i] = f[:,i] - f[:,i-1]*xx
        #
    #
    for k in range(N-1,0,-1):
        f[:,k] = f[:,k]/(x[:,ipn[k]]-x[:,ipn[k-1]])
        f[:,k-1] = f[:,k-1]-f[:,k]
        di = ipn[k] - ipn[k-1]
        for i in range(k+1,N):
            f[:,i] = f[:,i]/(x[:,ipn[i]] - x[:,ipn[i]+di])
            f[:,i-2] = f[:,i-2] - f[:,i]
            di = -di        
        #
    #
    return f[:,ipn_inv]


N=13
x = np.random.uniform(np.arange(N)-N//2-0.5,np.arange(N)-N//2+0.5)
#x = np.arange(N)-1.0*(N//2)
for i in range(N):
    xp = np.ones(N)
    xp[1:]=x[i]
    xp = np.cumprod(xp)
    r1 = vanderinvrprodcentered(xp,x)
    r2 = vanderinvrprod(xp,x)
    r1[i] = r1[i]-1.0
    r2[i] = r2[i]-1.0
    if i == N//2:
        print ""
    print([np.max(np.abs(r1)), np.max(np.abs(r2))])
    if i == N//2:
        print ""

x1 = np.array([-10,-5,1,2,10],np.double)
x2 = np.array([-2,-1,0,1,2],np.double)
x = np.vstack([x1,x2])
xp1 = np.array([1,-10,100,-1000,10000])
xp2 = np.array([1,2,4,8,16])
xp = np.vstack([xp1,xp2])

print(vanderinvrprodcenteredmulti(xp,x))

print 'suite'

def m1m2(dx,P):
    dx2l = dx[::2]
    dx2r = dx[1::2]
    dx1r = dx2l
    dx1l = np.roll(dx2r,1)
    dx1 = dx2l+dx2r
    dx2 = np.roll(dx1l+dx1r,-1)
    Npts = dx1.size
    N = 2*P+1
    dx1_ext = np.hstack([dx1[-P:],dx1,dx1[:P]])
    dx2_ext = np.hstack([dx2[-P:],dx2,dx2[:P]])
    
    x1 = np.zeros([Npts,N])
    x2 = np.zeros([Npts,N])
    for i in range(1,P+1):
        x1[:,P+i] = x1[:,P+i-1] + dx1_ext[P+i-1:P+i-1+Npts]
        x1[:,P-i] = x1[:,P-i+1] - dx1_ext[P-i:P-i+Npts]
        x2[:,P+i] = x2[:,P+i-1] + dx2_ext[P+i-1:P+i-1+Npts]
        x2[:,P-i] = x2[:,P-i+1] - dx2_ext[P-i:P-i+Npts]
    #
    ip = np.arange(1,N+1)
    xp1l = np.ndarray([Npts,N])
    xp1r = np.ndarray([Npts,N])
    xp2l = np.ndarray([Npts,N])
    xp2r = np.ndarray([Npts,N])
    dx1lc = dx1l.copy().reshape([Npts,1])
    dx1rc = dx1r.copy().reshape([Npts,1])
    dx2lc = dx2l.copy().reshape([Npts,1])
    dx2rc = dx2r.copy().reshape([Npts,1])
    xp1l[:,:] = -dx1lc
    xp1r[:,:] = dx1rc
    xp2l[:,:] = -dx2lc
    xp2r[:,:] = dx2rc
    xp1l = np.cumprod(xp1l,1)
    xp1r = np.cumprod(xp1r,1)
    xp2l = np.cumprod(xp2l,1)
    xp2r = np.cumprod(xp2r,1)
    xp1 = (xp1r - xp1l)/(np.double(ip)*(dx1lc+dx1rc))
    xp2 = (xp2r - xp2l)/(np.double(ip)*(dx2lc+dx2rc))
    
    m1 = vanderinvrprodcenteredmulti(xp1,x1)
    m2 = vanderinvrprodcenteredmulti(xp2,x2)
    return m1,m2

dx = np.array([0.1,0.2,0.4,0.8,1.6,1.6,0.8,0.4,0.2,0.1])
#dx = np.ones(10)*0.1


m1 = list()
m2 = list()
for i in range(0,4):
    t1,t2 = m1m2(dx,i)
    m1.append(t1)
    m2.append(t2)

class Pmatrix:
    def __init__(self, dx, P):
        self.m1 = list()
        self.m2 = list()
