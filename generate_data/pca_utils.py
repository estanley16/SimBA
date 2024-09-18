'''
author: matthias wilms
'''

import numpy as np
import scipy as sp
import kernels
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.sparse.csgraph as csgraph
from numpy import matlib

def mean_error_2d_contour(gt,pred):
    return np.mean(np.sqrt(np.square(gt[0::2,:]-pred[0::2,:])+np.square(gt[1::2,:]-pred[1::2,:])),axis=0)

def compute_simple_contour_distance_matrix(num_points):
    dist_matrix=np.zeros((num_points,num_points))

    pos_dist=np.array(range(0,num_points))
    neg_dist=np.array(range(num_points,0,-1))
    for i in range(0,num_points):
        dist_matrix[i,:]=np.min([np.abs(pos_dist),neg_dist],axis=0)
        pos_dist=pos_dist-1
        neg_dist=neg_dist+1

    temp=np.kron(dist_matrix,np.ones((2,2)))
    return np.min([temp.T,temp],axis=0)

def compute_euclidean_2d_point_distance_matrix(base_contour):
    dist_matrix=np.sqrt((base_contour[0::2]-np.matlib.repmat(base_contour[0::2],1,base_contour[0::2].shape[0]).T)**2+(base_contour[1::2]-np.matlib.repmat(base_contour[1::2],1,base_contour[1::2].shape[0]).T)**2)

    return np.kron(dist_matrix,np.ones((2,2)))

def compute_pseudo_geodesic_2d_point_distance_matrix(base_contour):
    euclidean_dist_matrix=np.sqrt((base_contour[0::2]-np.matlib.repmat(base_contour[0::2],1,base_contour[0::2].shape[0]).T)**2+(base_contour[1::2]-np.matlib.repmat(base_contour[1::2],1,base_contour[1::2].shape[0]).T)**2)

    succ=np.diag(euclidean_dist_matrix,1)
    succ=np.concatenate((succ,[euclidean_dist_matrix[0,-1]])) # complete +1 'diagonal'


    geo_dist_matrix=np.zeros(euclidean_dist_matrix.shape)

    for i in range(0,euclidean_dist_matrix.shape[1]):
        curr_succ=np.roll(succ,-i)
        curr_succ_a=np.concatenate(([0],np.cumsum(curr_succ[0:-1])))
        curr_succ_b=np.concatenate(([0],np.flip(np.cumsum(np.flip(curr_succ[1:],axis=0)),axis=0)))
        curr_dist=np.roll(np.minimum(curr_succ_a,curr_succ_b),+i)
        geo_dist_matrix[:,i]=curr_dist

    return np.kron(geo_dist_matrix,np.ones((2,2)))

def compute_multi_object_pseudo_euclidean_geodesic_shortest_path_2d_point_distance_matrix(base_contour,obj_indicator,eta,kappa):
    euclidean_dist_matrix=np.kron(np.sqrt((base_contour[0::2]-np.matlib.repmat(base_contour[0::2],1,base_contour[0::2].shape[0]).T)**2+(base_contour[1::2]-np.matlib.repmat(base_contour[1::2],1,base_contour[1::2].shape[0]).T)**2),np.ones((2,2)))


    geo_dist_matrix=np.ones(euclidean_dist_matrix.shape)*np.finfo(euclidean_dist_matrix.dtype).max

    unique_objects=np.unique(obj_indicator)
    for i in range(0,len(unique_objects)):
        curr_object=base_contour[obj_indicator==unique_objects[i]]

        curr_geo_dist=compute_pseudo_geodesic_2d_point_distance_matrix(curr_object)

        idcs=np.where(obj_indicator==unique_objects[i])
        idx_min=np.min(idcs)
        idx_max=np.max(idcs)
        geo_dist_matrix[idx_min:idx_max+1,idx_min:idx_max+1]=curr_geo_dist

    combined_dist=np.minimum(geo_dist_matrix,euclidean_dist_matrix*eta+kappa)
    dist_matrix=csgraph.shortest_path(combined_dist[0::2,0::2],directed=False)

    return np.kron(dist_matrix,np.ones((2,2)))


def corrcov(cov_matrix):
    sigma=np.sqrt(np.diag(cov_matrix))
    corr_matrix=cov_matrix/(np.asmatrix(sigma).T@np.asmatrix(sigma))
    corr_matrix=(corr_matrix.T+corr_matrix)/2
    np.fill_diagonal(corr_matrix,1)
    return corr_matrix,sigma

def covcorr(corr_matrix,sigmas):
    return np.diag(sigmas)@corr_matrix@np.diag(sigmas)

def higham_closest_corr_matrix(corr_matrix,max_iterations=1000,tol=1e-5):
    X=Y=corr_matrix
    correction_matrix=np.zeros(corr_matrix.shape)
    diffX=tol+1
    diffY=tol+1
    diffXY=tol+1

    i=1
    while np.max([diffX,diffY,diffXY]) > tol and i <= max_iterations:
        Xold=np.copy(X)
        R=Y-correction_matrix

        #projection onto space of psd matrices
        eig_vals,eig_vecs=np.linalg.eig(R)
        X=eig_vecs*np.diag(np.max([eig_vals,np.zeros(eig_vals.shape)],axis=0))*eig_vecs.T
        X=(X.T+X)/2
        correction_matrix=X-R

        Yold=np.copy(Y)
        #projection onto space of matrices with unit diagonal
        Y=np.copy(X)
        np.fill_diagonal(Y,1)

        #compute differences
        diffX=np.linalg.norm(X-Xold)/np.linalg.norm(X)
        diffY=np.linalg.norm(Y-Yold)/np.linalg.norm(Y)
        diffXY=np.linalg.norm(Y-X)/np.linalg.norm(Y)

        i=i+1

    #Higham usually returns X but here we use Y to ensure that the diagonal elements are 1 --> no correction performed after last step!
    return Y

def merge_subspace_models_closest_rotation(modelA, modelB,decorrelation=False,decorrelation_mode='full'):
    """
    Merges two subspaces of different dimensions (rank(A)<=rank(B)) by
    finding the closest (same dimension) subspace to B that fully contains A.

    Based on:
        KE YE AND LEK-HENG LIM: DISTANCE BETWEEN SUBSPACES OF
                    DIFFERENT DIMENSIONS, http://arxiv.org/abs/1407.0900v1
    """
    if modelA.basis.shape[1] >= modelB.basis.shape[1]:
        return modelA.translation_vector,modelA.basis,modelA.eigenvalues

    U,S,Vt=np.linalg.svd(modelA.basis.T@modelB.basis)
    V=Vt.T

    rotA=modelA.basis@U
    rotB=modelB.basis@V
    new_basis=np.zeros((modelA.basis.shape[0],modelB.basis.shape[1]))
    new_basis[:,0:modelA.basis.shape[1]]=rotA
    new_basis[:,modelA.basis.shape[1]:]=rotB[:,modelA.basis.shape[1]:]

    rotA_evs=U.T@np.diagflat(modelA.eigenvalues)@U
    rotB_evs=Vt@np.diagflat(modelB.eigenvalues)@V

    new_evs_old=np.zeros((modelB.basis.shape[1],modelB.basis.shape[1]))
    new_evs_old[0:rotA_evs.shape[0],0:rotA_evs.shape[1]]=rotA_evs
    new_evs_old[rotA_evs.shape[0]:,rotA_evs.shape[1]:]=rotB_evs[rotA_evs.shape[0]:,rotA_evs.shape[1]:]

    new_evs=np.zeros((modelB.basis.shape[1],modelB.basis.shape[1]))
    new_evs[0:rotA_evs.shape[0],0:rotA_evs.shape[1]]=rotA_evs*0.5
    b_evs_new_basis=new_basis.T@rotB@rotB_evs@rotB.T@new_basis
    new_evs=new_evs+(b_evs_new_basis*0.5)

    new_evs=new_evs_old


    if decorrelation:
        if decorrelation_mode == 'full':
            U,S,Vt=np.linalg.svd(new_basis@new_evs@new_basis.T)
            new_basis=U[:,0:new_basis.shape[1]]
            new_evs=S*(np.sum(modelA.eigenvalues)/np.sum(S))
            new_evs=new_evs[0:new_basis.shape[1]]
        elif decorrelation_mode == 'kernel':
            L=np.linalg.cholesky(new_evs)
            new_basis,new_evs=eig_fast_spsd_kernel(new_basis@L,[(kernels.CovKernel(1),None,'data',None,1)],new_evs.shape[0],sampling_factor=2)
            new_evs=new_evs*(np.sum(modelB.eigenvalues)/np.sum(new_evs))
            new_evs=new_evs[0:new_basis.shape[1]]
    else:
        new_evs=modelB.eigenvalues

    return modelA.translation_vector,new_basis,new_evs

def merge_subspace_models_closest_rotation_decorr(modelA, modelB):
    return merge_subspace_models_closest_rotation(modelA, modelB,True,'full')

def merge_subspace_models_closest_rotation_decorr_kernel(modelA, modelB):
    return merge_subspace_models_closest_rotation(modelA, modelB,True,'kernel')

def eig_fast_spsd_kernel(data,kernel_list,rank,sampling_factor=None):
    """
    Computes the eigendecomposition of a low-rank kernel matrix implicitly derived
    from the data vectors (column vectors!) and the list of kernels (order matters!; left to right!)
    using a randomized sampling approach

    Method is based on

    Towards More Efficient SPSD Matrix Approximation and CUR Matrix Decomposition by Wang et al.
    http://www.jmlr.org/papers/volume17/15-190/15-190.pdf

    Input:

    kernels --> list of kernels to be applied where each entry consists of 5 elements (kernel, concat_op, data_string, dist_func, weight)
        kernel --> instance of a kernel derived from KernelBase
        concat_op --> any operator that can be applied to numpy objects (np.add, np.mult,...)
        data_string --> indicates whether this kernel operates on the 'data' or the coord distances 'dist'
        dist_func --> distance function derived from DistanceBase. Only used when data_string='dist'
        weight --> multiplicative weight applied to the kernel result before concatenation; i.e. enables weighted linear combination of different kernels (concat_op = np.add)
    """

    n = data.shape[0]
    m = data.shape[1]
    if sampling_factor is None:
        sampling_factor=np.int32(np.ceil(np.sqrt(rank*n)))
    num_samples=rank*sampling_factor


    #first sketch --> nystroem approximation
    uni_sampling = np.sort(np.random.choice(n,rank,replace=False))# uniform sampling
    if kernel_list[0][2] == 'data':
        sketch = kernel_list[0][4]*kernel_list[0][0].apply(data,data[uni_sampling,:])
    else:
        sketch = kernel_list[0][4]*kernel_list[0][0].apply(kernel_list[0][3].dist(np.arange(0,data.shape[0]),uni_sampling))
    if len(kernel_list) > 1:
        for i in range(1,len(kernel_list)):
            if kernel_list[i][2] == 'data':
                sketch=kernel_list[i][1](sketch,kernel_list[i][4]*kernel_list[i][0].apply(data,data[uni_sampling,:]))
            else:
                sketch=kernel_list[i][1](sketch,kernel_list[i][4]*kernel_list[i][0].apply(kernel_list[i][3].dist(np.arange(0,data.shape[0]),uni_sampling)))

    Q,_temp,=np.linalg.qr(sketch)

    #second sketch
    sampling_prob=np.sum(np.square(Q),axis=1)
    sampling_prob=sampling_prob/np.sum(sampling_prob)

    lev_sampling=np.sort(np.random.choice(n,num_samples,replace=True,p=np.ravel(sampling_prob)))
    lev_sampling=np.unique(np.concatenate((lev_sampling,uni_sampling)))
    QInv=np.linalg.pinv(Q[lev_sampling,:])

    if kernel_list[0][2] == 'data':
        sketch = kernel_list[0][4]*kernel_list[0][0].apply(data[lev_sampling,:],data[lev_sampling,:])
    else:
        sketch = kernel_list[0][4]*kernel_list[0][0].apply(kernel_list[0][3].dist(lev_sampling,lev_sampling))

    if len(kernel_list) > 1:
        for i in range(1,len(kernel_list)):
            if kernel_list[i][2] == 'data':
                sketch=kernel_list[i][1](sketch,kernel_list[i][4]*kernel_list[i][0].apply(data[lev_sampling,:],data[lev_sampling,:]))
            else:
                sketch=kernel_list[i][1](sketch,kernel_list[i][4]*kernel_list[i][0].apply(kernel_list[i][3].dist(lev_sampling,lev_sampling)))

    U=QInv@sketch@QInv.T
    UC,SC,_=np.linalg.svd(U,full_matrices=False)

    idx=np.argsort(-SC)
    eigenvectors=(Q@UC)[:,idx]
    eigenvalues=SC[idx]

    return eigenvectors,eigenvalues

def eig_kernel(data,kernel_list,rank):

    if kernel_list[0][2] == 'data':
        kernel_matrix = kernel_list[0][4]*kernel_list[0][0].apply(data,data)
    else:
        kernel_matrix = kernel_list[0][4]*kernel_list[0][0].apply(kernel_list[0][3].dist(np.arange(0,data.shape[0]),np.arange(0,data.shape[0])))

    if len(kernel_list) > 1:
        for i in range(1,len(kernel_list)):
            if kernel_list[i][2] == 'data':
                kernel_matrix=kernel_list[i][1](kernel_matrix,kernel_list[i][4]*kernel_list[i][0].apply(data,data))
            else:
                kernel_matrix=kernel_list[i][1](kernel_matrix,kernel_list[i][4]*kernel_list[i][0].apply(kernel_list[i][3].dist(np.arange(0,data.shape[0]),np.arange(0,data.shape[0]))))

    eig_vals,eig_vecs=sp.linalg.eigh(kernel_matrix,eigvals=(kernel_matrix.shape[0]-rank,kernel_matrix.shape[0]-1))

    return eig_vecs,eig_vals
