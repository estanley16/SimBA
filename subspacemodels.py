import numpy as np
import pca_utils as utils
import scipy as sp
from scipy.stats import truncnorm

class SubspaceModel:

    def __init__(self, translation_vector, basis, eigenvalues, base_mesh=None):
        self.translation_vector=translation_vector
        self.basis=basis
        self.eigenvalues=eigenvalues
        self.base_mesh=base_mesh

    @property
    def translation_vector(self):
        return self._translation_vector

    @translation_vector.setter
    def translation_vector(self, value):
        if len(value.shape) > 1:
            self._translation_vector=np.asmatrix(value)
        else:
            self._translation_vector=np.asmatrix(value).T

    @property
    def base_mesh(self):
        return self._base_mesh

    @base_mesh.setter
    def base_mesh(self, value):
        self._base_mesh=value

    @property
    def basis(self):
        return self._basis

    @basis.setter
    def basis(self, value):
        self._basis=np.asmatrix(value)

    @property
    def eigenvalues(self):
        return self._eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, value):
        if len(value.shape) > 1:
            self._eigenvalues=np.asmatrix(value)
        else:
            self._eigenvalues=np.asmatrix(value).T
    
    def project_onto_subspace(self, data, num_components=-1):
        if len(data.shape) > 1:
            temp_data=np.asmatrix(data)
        else:
            temp_data=np.asmatrix(data).T

        if (num_components < 1) or (num_components > self.basis.shape[1]):
            num_components=self.basis.shape[1]

        return np.matmul(self.basis[:,0:num_components].T,(temp_data-self.translation_vector))

    def project_onto_original_space(self, data, num_components=-1):
        if len(data.shape) > 1:
            temp_data=np.asmatrix(data)
        else:
            temp_data=np.asmatrix(data).T

        if (num_components < 1) or (num_components > self.basis.shape[1]):
            num_components=self.basis.shape[1]

        return np.matmul(self.basis[:,0:num_components],temp_data)+self.translation_vector


    def approximate(self, data, error_func=None, num_components=-1):
        if len(data.shape) > 1:
            temp_data=np.asmatrix(data)
        else:
            temp_data=np.asmatrix(data).T
        if error_func is None:
            return self.project_onto_original_space(self.project_onto_subspace(temp_data,num_components),num_components)
        else:
            proj_data=self.project_onto_original_space(self.project_onto_subspace(temp_data,num_components),num_components)
            return proj_data,error_func(data,proj_data)

    def approximate_alpha_box_restriction(self, data, alpha=1, error_func=None, num_components=-1):
        if len(data.shape) > 1:
            temp_data=np.asmatrix(data)
        else:
            temp_data=np.asmatrix(data).T

        subspace_coords=self.project_onto_subspace(temp_data,num_components)            
        border_alpha_box=np.sqrt(self.eigenvalues[0:subspace_coords.shape[0],:])*alpha
        if error_func is None:
            return self.project_onto_original_space(np.copysign(np.minimum(np.abs(subspace_coords),border_alpha_box),subspace_coords),num_components)
        else:
            proj_data=self.project_onto_original_space(np.copysign(np.minimum(np.abs(subspace_coords),border_alpha_box),subspace_coords),num_components)
            return proj_data,error_func(data,proj_data)

    def compute_specificity(self, test_data, alpha, error_func, num_samples=1000,num_components=-1,mode='normal'):
        if (num_components < 0) or (num_components > self.basis.shape[1]):
            num_components=self.basis.shape[1]

        if mode=='normal':
            rand_samples=np.random.normal(0,1*alpha,(num_components,num_samples))
        elif mode=='uniform':
            rand_samples=np.random.uniform(-alpha,alpha,(num_components,num_samples))
        elif mode=='normaltrunc':
            rand_samples=truncnorm.rvs(-alpha,alpha,size=(num_components,num_samples))    
        elif mode=='normalbouned':
            rand_samples_needed=num_components*num_samples
            rand_samples=np.array([])
            while rand_samples.shape[0]<rand_samples_needed:
                new_samples=np.random.normal(0,1,num_components*num_samples)
                new_samples=new_samples[np.abs(new_samples)<=alpha]
                rand_samples=np.concatenate((rand_samples,new_samples),axis=None)

            rand_samples=np.reshape(rand_samples[0:num_components*num_samples],(num_components,num_samples))
            
        rand_shapes=self.translation_vector+np.multiply(self.basis[:,0:num_components],np.sqrt(self.eigenvalues[0:num_components]).T)@rand_samples

        min_dists=np.zeros(num_samples)

        #computationally inefficient but saves memory with large data (limited broadcasting)
        for i in range(0,num_samples):
            min_dists[i]=np.min(error_func(test_data,rand_shapes[:,i]))

        return min_dists

    def compute_specificity_other_model(self, other_model, alpha, error_func, num_samples=1000,num_components=-1,mode='normal'):
        if (num_components < 0) or (num_components > self.basis.shape[1]):
            num_components=self.basis.shape[1]

        if mode=='normal':
           rand_samples=np.random.normal(0,1*alpha,(num_components,num_samples))
           rand_shapes=self.translation_vector+np.multiply(self.basis[:,0:num_components],np.sqrt(self.eigenvalues[0:num_components]).T)@rand_samples

        _,min_dists=other_model.approximate_alpha_box_restriction(rand_shapes,3,error_func)

        return min_dists

    def compute_generalization_vs_specificity(self, test_data, alphas, error_func, num_components=-1, num_samples=1000):
        
        if (num_components < 0) or (num_components > self.basis.shape[1]):
            num_components=self.basis.shape[1]
        
        generalization_error=np.zeros(len(alphas[0]))
        specificity_error=np.zeros(len(alphas[0]))
    
        for i in range(0,len(alphas[0])):
            _,temp_result=self.approximate_alpha_box_restriction(test_data,alphas[0][i],error_func)
            generalization_error[i]=np.mean(temp_result)
            specificity_error[i]=np.mean(self.compute_specificity(test_data,alphas[1][i],error_func,num_samples,num_components,'normalbouned'))

        return generalization_error,specificity_error

    def compute_generalization_vs_specificity_components(self, test_data, components, error_func, num_samples=1000):
        
        generalization_error=np.zeros(len(components))
        specificity_error=np.zeros(len(components))
    
        for i in range(0,len(components)):
            curr_num=components[i]
            if curr_num>self.basis.shape[1]:
                curr_num=self.basis.shape[1]
            _,temp_result=self.approximate(test_data,error_func,num_components=curr_num)
            generalization_error[i]=np.mean(temp_result)
            specificity_error[i]=np.mean(self.compute_specificity(test_data,1,error_func,num_samples,curr_num,'normal'))

        return generalization_error,specificity_error      

    def compute_generalization_vs_specificity_other_model(self, other_model, test_data, alphas, error_func, num_components=-1, num_samples=1000):
        
        if (num_components < 0) or (num_components > self.basis.shape[1]):
            num_components=self.basis.shape[1]
        
        generalization_error=np.zeros(len(alphas))
        specificity_error=np.zeros(len(alphas))
    
        for i in range(0,len(alphas)):
            curr_alpha=alphas[i]
            _,temp_result=self.approximate_alpha_box_restriction(test_data,curr_alpha,error_func)
            generalization_error[i]=np.mean(temp_result)
            specificity_error[i]=np.mean(self.compute_specificity_other_model(other_model,curr_alpha,error_func,num_samples))

        return generalization_error,specificity_error   

    def sample_randn(self, num_samples):
        return self.translation_vector+(np.multiply(self.basis,np.sqrt(self.eigenvalues).T)@np.random.randn(self.basis.shape[1],num_samples))

class SubspaceModelGenerator:

    @staticmethod
    def compute_pca_subspace(data, variability_retained, debug=False):
        mean_vector=np.mean(data,axis=1)
        centered_data=data-mean_vector

        if centered_data.shape[0] > centered_data.shape[1]:
            #decompose inner-product matrix
            inner_prod=(1/(centered_data.shape[1]-1))*(centered_data.T@centered_data)
    
            eig_vals,eig_vecs=np.linalg.eig(inner_prod)
            eig_vals=np.real(eig_vals)
            eig_vecs=np.real(eig_vecs)            

            eig_vecs=centered_data@eig_vecs
            eig_vecs=eig_vecs/np.linalg.norm(eig_vecs,axis=0)
            
        else:
            #decompose covariance matrix
            cov_matrix=(1/(centered_data.shape[1]-1))*(centered_data@centered_data.T)
            
            eig_vals,eig_vecs=np.linalg.eig(cov_matrix)
            eig_vals=np.real(eig_vals)
            eig_vecs=np.real(eig_vecs)
        
        #sort eigenvalues/vectors
        idx=np.argsort(-eig_vals) #numerically slightly inaccurate because of negation
        eig_vals=eig_vals[idx]
        eig_vecs=eig_vecs[:,idx]

        if debug:
            print('   evs: ['+str(eig_vals[-1])+','+str(eig_vals[0])+'] (sum='+str(np.sum(eig_vals)) +')')

        #retain requested variability
        subspace_rank=1
        requested_variability=np.sum(eig_vals)*variability_retained
        for i in range(1,len(eig_vals)+1):
            if np.sum(eig_vals[0:i])>=requested_variability or i==centered_data.shape[1]-1:
                subspace_rank=i
                break

        eig_vals=eig_vals[0:subspace_rank]
        eig_vecs=eig_vecs[:,0:subspace_rank]
        
        return SubspaceModel(mean_vector,eig_vecs,eig_vals)

    @staticmethod
    def compute_localized_subspace_media(data, variability_retained, distance_matrix, distance_schedule,test_data=None,test_method=utils.mean_error_2d_contour,repair_method=utils.higham_closest_corr_matrix, merge_method=utils.merge_subspace_models_closest_rotation,debug=False):
        mean_vector=np.mean(data,axis=1)
        centered_data=data-mean_vector
        cov_matrix=(1/(centered_data.shape[1]-1))*(centered_data@centered_data.T)
        corr_matrix,cov_sigmas=utils.corrcov(cov_matrix)

        local_models=[]
        combined_models=[]

        for lvl in range(0,len(distance_schedule)):            
            if lvl == 0:
                if hasattr(variability_retained, "__getitem__"):
                    global_model=SubspaceModelGenerator.compute_pca_subspace(data, variability_retained[lvl])
                else:
                    global_model=SubspaceModelGenerator.compute_pca_subspace(data, variability_retained)
                local_models.append(global_model)
            else:
                curr_corr_matrix=np.multiply(np.float32(distance_matrix<=distance_schedule[lvl]),corr_matrix)
                   
                #use higham-like repair method
                if repair_method is not None:
                    curr_cov_matrix=utils.covcorr(repair_method(curr_corr_matrix),cov_sigmas)
                    curr_cov_matrix=(curr_cov_matrix.T+curr_cov_matrix)/2
                else:
                    curr_cov_matrix=utils.covcorr(curr_corr_matrix,cov_sigmas)
                    print('BE CAREFUL! No method specified to repair the cov. matrix!')

                #decompose corrected covariance matrix                
                eig_vals,eig_vecs=np.linalg.eig(curr_cov_matrix)
                eig_vals=np.real(eig_vals)
                eig_vecs=np.real(eig_vecs)

                #sort eigenvalues/vectors
                idx=np.argsort(-eig_vals) #numerically slightly inaccurate because of negation
                eig_vals=eig_vals[idx]
                eig_vecs=eig_vecs[:,idx]

                if debug:
                    print('   evs: ['+str(eig_vals[-1])+','+str(eig_vals[0])+'] (sum='+str(np.sum(eig_vals)) +')')

                #retain requested variability
                subspace_rank=1
                if hasattr(variability_retained, "__getitem__"):
                    requested_variability=np.sum(eig_vals)*variability_retained[lvl]
                else:
                    requested_variability=np.sum(eig_vals)*variability_retained
                for i in range(1,len(eig_vals)+1):
                    if np.sum(eig_vals[0:i])>=requested_variability:
                        subspace_rank=i
                        break

                eig_vals=eig_vals[0:subspace_rank]
                eig_vecs=eig_vecs[:,0:subspace_rank]                
                local_model=SubspaceModel(mean_vector,eig_vecs,eig_vals)
                local_models.append(local_model)

                #merge models

                temp_trans,temp_basis,temp_evs=merge_method(global_model,local_model)
                global_model=SubspaceModel(temp_trans,temp_basis,temp_evs)
            if debug:
                combined_models.append(global_model)
                if test_data is None:
                    print('  level '+str(lvl+1)+' (dist='+str(distance_schedule[lvl])+') --> rank='+str(global_model.basis.shape[1]))
                else:
                    approx,error=global_model.approximate(test_data,test_method)
                    print('  level '+str(lvl+1)+' (dist='+str(distance_schedule[lvl])+') --> rank='+str(global_model.basis.shape[1])+' & error='+str(np.mean(error)))

        if debug:
            return global_model,local_models,combined_models
        else:
            return global_model

    @staticmethod
    def compute_localized_subspace_kernel(data,variability_retained,distance_schedule,kernel_list,max_rank,test_data=None,test_method=utils.mean_error_2d_contour,merge_method=utils.merge_subspace_models_closest_rotation,eig_method=utils.eig_kernel,debug=False):
        mean_vector=np.mean(data,axis=1)
        centered_data=data-mean_vector
        
        local_models=[]
        combined_models=[]

        for lvl in range(0,len(distance_schedule)):            
            #print('kernel: '+str(kernel_list[lvl]))

            if hasattr(max_rank, "__getitem__"):
                eig_vecs,eig_vals=eig_method(centered_data,kernel_list[lvl],max_rank[lvl])
            else:
                eig_vecs,eig_vals=eig_method(centered_data,kernel_list[lvl],max_rank)

            #sort eigenvalues/vectors
            idx=np.argsort(-eig_vals) #numerically slightly inaccurate because of negation
            eig_vals=eig_vals[idx]
            eig_vecs=eig_vecs[:,idx]
            if debug:
                print('   evs: ['+str(eig_vals[-1])+','+str(eig_vals[0])+'] (sum='+str(np.sum(eig_vals)) +')')

            #retain requested variability
            subspace_rank=1
            if hasattr(variability_retained, "__getitem__"):
                requested_variability=np.sum(eig_vals)*variability_retained[lvl]
            else:
                requested_variability=np.sum(eig_vals)*variability_retained
            for i in range(1,len(eig_vals)+1):
                if np.sum(eig_vals[0:i])>=requested_variability:
                    subspace_rank=i
                    break

            eig_vals=eig_vals[0:subspace_rank]
            eig_vecs=eig_vecs[:,0:subspace_rank]                
            local_model=SubspaceModel(mean_vector,eig_vecs,eig_vals)
            local_models.append(local_model)

            #merge models
            if lvl==0:
                temp_trans=local_model.translation_vector
                temp_basis=local_model.basis
                temp_evs=local_model.eigenvalues
            else:
                temp_trans,temp_basis,temp_evs=merge_method(global_model,local_model)
            global_model=SubspaceModel(temp_trans,temp_basis,temp_evs)            
            if debug:
                combined_models.append(global_model)
                if test_data is None:
                    print('  level '+str(lvl+1)+' (dist='+str(distance_schedule[lvl])+') --> rank='+str(global_model.basis.shape[1]))
                else:
                    approx,error=global_model.approximate(test_data,test_method)
                    print('  level '+str(lvl+1)+' (dist='+str(distance_schedule[lvl])+') --> rank='+str(global_model.basis.shape[1])+' & error='+str(np.mean(error)))

        if debug:
            return global_model,local_models,combined_models
        else:
            return global_model
        

    
                
                    
                    

                
            
            
            
        







    





