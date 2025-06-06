from .cupy_pal import cp2np, np2cp, get_module_array, get_module_array_scipy, iscupy, np, sn, enable_cupy
import time
import copy
import bilby
import icarogw
from .wrappers import FlatLambdaCDM_wrap

# LVK Reviewed
class hierarchical_likelihood(bilby.Likelihood):
    def __init__(self, posterior_samples_dict, injections, rate_model, nparallel=None, neffPE=20,neffINJ=None,likelihood_variance_thr=None):
        '''
        Base class for an hierachical liklihood. It just saves all the input requirements for a general hierarchical analysis
        
        Parameters
        ----------
        posterior_samples_dict: dict
            Dictionary containing the posterior samples class
        injections: class
            Injection class from its module 
        rate_model: class
            Rate model to compute the CBC rate per year at the detector, taken from the wrapper module.
        nparallel: int
            Number of samples to use per event, if None it will use the maximum number of PE samples in common to all events
        neffPE: int
            Effective number of samples per event that must contribute the prior evaluation
        neffINJ: int
            Number of effective injections needed to evaluate the selection bias, if None we will assume 4* observed signals.
        likelihood_variance_thr: int
            Likelihood variance to use for the cut, the suggested value if 1 from (2304.06138).
        '''
        
        # Saves injections in a cupyfied format
        self.injections=injections
        self.neffPE=neffPE
        self.rate_model=rate_model
        self.posterior_samples_dict=posterior_samples_dict
        self.posterior_samples_dict.build_parallel_posterior(nparallel=nparallel)
        self.likelihood_variance_thr = likelihood_variance_thr

        if likelihood_variance_thr is not None:
            print('Using Likelihood variance as numerical stability estimator \n We will not consider neffPE or neffINJ')
            self.neffPE = -1.
            self.neffINJ = -1.
        else:    
            print('Using neffPE and neffINJ as numerical stability estimators')
            if neffINJ is None:
                print('Setting neffINJ as 4 times observed signals')
                self.neffINJ=4*self.posterior_samples_dict.n_ev
            else:
                self.neffINJ=neffINJ
        
        super().__init__(parameters={ll: None for ll in self.rate_model.population_parameters})
                
    def log_likelihood(self):
        '''
        Evaluates and return the log-likelihood
        '''

        # enable_cupy()
        # self.injections.cupyfy()
        # self.posterior_samples_dict.cupyfy()
        # self.rate_model.cw=FlatLambdaCDM_wrap(2.)
                
        #Update the rate model with the population parameters
        self.rate_model.update(**{key:self.parameters[key] for key in self.rate_model.population_parameters})
        # Update the sensitivity estimation with the new model
        self.injections.update_weights(self.rate_model)
        Neff=self.injections.effective_injections_number()
        # If the injections are not enough return 0, you cannot go to that point. This is done because the number of injections that you have
        # are not enough to calculate the selection effect
        
        xp = get_module_array(self.injections.log_weights)
        
        if (Neff<self.neffINJ) | (Neff==0.):
            return float(xp.nan_to_num(-xp.inf))
        
        # Update the weights on the PE
        self.posterior_samples_dict.update_weights(self.rate_model)
        neff_PE_ev = self.posterior_samples_dict.get_effective_number_of_PE()
      
        if xp.any(neff_PE_ev<self.neffPE):
            return float(xp.nan_to_num(-xp.inf))

        self.likelihood_variance = (xp.power(self.posterior_samples_dict.n_ev,2.)/Neff)*(1-Neff/self.injections.ntotal)+xp.sum(
            xp.power(neff_PE_ev,-1.)*(1-neff_PE_ev/self.posterior_samples_dict.Ns_array)
        )

        if self.likelihood_variance_thr is not None:
            if self.likelihood_variance > self.likelihood_variance_thr:
                return float(xp.nan_to_num(-xp.inf))
            
        # Combine all the terms  
        if self.rate_model.scale_free:
            # Log likelihood for scale free model, Eq. 1.3 on the document
            log_likeli = xp.sum(xp.log(self.posterior_samples_dict.sum_weights))-self.posterior_samples_dict.n_ev*xp.log(self.injections.pseudo_rate)
        else:
            Nexp=self.injections.expected_number_detections()
            # Log likelihood for  the model, Eq. 1.1 on the document
            log_likeli = -Nexp + self.posterior_samples_dict.n_ev*xp.log(self.injections.Tobs)+xp.sum(xp.log(self.posterior_samples_dict.sum_weights))
        
        # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
        # python valye 1e-309
        if log_likeli == xp.inf:
            raise ValueError('LOG-likelihood must be smaller than infinite')

        if xp.isnan(log_likeli):
            log_likeli = float(xp.nan_to_num(-xp.inf))
        else:
            log_likeli = float(xp.nan_to_num(log_likeli))
            
        return float(cp2np(log_likeli))


class hierarchical_likelihood_v1(bilby.Likelihood):
    def __init__(self, posterior_samples_dict, injections, rate_model, nparallel=None, neffPE=20,neffINJ=None,likelihood_variance_thr=None):
        '''
        Base class for an hierachical liklihood. It just saves all the input requirements for a general hierarchical analysis
        
        Parameters
        ----------
        posterior_samples_dict: dict
            Dictionary containing the posterior samples class
        injections: class
            Injection class from its module 
        rate_model: class
            Rate model to compute the CBC rate per year at the detector, taken from the wrapper module.
        nparallel: int
            Number of samples to use per event, if None it will use the maximum number of PE samples in common to all events
        neffPE: int
            Effective number of samples per event that must contribute the prior evaluation
        neffINJ: int
            Number of effective injections needed to evaluate the selection bias, if None we will assume 4* observed signals.
        likelihood_variance_thr: int
            Likelihood variance to use for the cut, the suggested value if 1 from (2304.06138).
        '''
        
        # Saves injections in a cupyfied format
        self.injections=injections
        self.neffPE=neffPE
        self.rate_model=rate_model
        self.posterior_samples_dict=posterior_samples_dict
        self.posterior_samples_dict.build_parallel_posterior(nparallel=nparallel)
        self.likelihood_variance_thr = likelihood_variance_thr

        if likelihood_variance_thr is not None:
            print('Using Likelihood variance as numerical stability estimator \n We will not consider neffPE or neffINJ')
            self.neffPE = -1.
            self.neffINJ = -1.
        else:    
            print('Using neffPE and neffINJ as numerical stability estimators')
            if neffINJ is None:
                print('Setting neffINJ as 4 times observed signals')
                self.neffINJ=4*self.posterior_samples_dict.n_ev
            else:
                self.neffINJ=neffINJ
        
        super().__init__(parameters={ll: None for ll in self.rate_model.population_parameters})
                
    def log_likelihood(self):
        '''
        Evaluates and return the log-likelihood
        '''

        # enable_cupy()
        # self.injections.cupyfy()
        # self.posterior_samples_dict.cupyfy()
        # self.rate_model.cw=FlatLambdaCDM_wrap(2.)
                
        #Update the rate model with the population parameters
        self.rate_model.update(**{key:self.parameters[key] for key in self.rate_model.population_parameters})
        # Update the sensitivity estimation with the new model
        self.injections.update_weights(self.rate_model)
        Neff=self.injections.effective_injections_number()
        # If the injections are not enough return 0, you cannot go to that point. This is done because the number of injections that you have
        # are not enough to calculate the selection effect
        
        xp = get_module_array(self.injections.log_weights)
        
        if (Neff<self.neffINJ) | (Neff==0.):
            return -xp.inf
        
        # Update the weights on the PE
        self.posterior_samples_dict.update_weights(self.rate_model)
        neff_PE_ev = self.posterior_samples_dict.get_effective_number_of_PE()
      
        if xp.any(neff_PE_ev<self.neffPE):
            return -xp.inf

        self.likelihood_variance = (xp.power(self.posterior_samples_dict.n_ev,2.)/Neff)*(1-Neff/self.injections.ntotal)+xp.sum(
            xp.power(neff_PE_ev,-1.)*(1-neff_PE_ev/self.posterior_samples_dict.Ns_array)
        )

        if self.likelihood_variance_thr is not None:
            if self.likelihood_variance > self.likelihood_variance_thr:
                return -xp.inf
            
        # Combine all the terms  
        if self.rate_model.scale_free:
            # Log likelihood for scale free model, Eq. 1.3 on the document
            log_likeli = xp.sum(xp.log(self.posterior_samples_dict.sum_weights))-self.posterior_samples_dict.n_ev*xp.log(self.injections.pseudo_rate)
        else:
            Nexp=self.injections.expected_number_detections()
            # Log likelihood for  the model, Eq. 1.1 on the document
            log_likeli = -Nexp + self.posterior_samples_dict.n_ev*xp.log(self.injections.Tobs)+xp.sum(xp.log(self.posterior_samples_dict.sum_weights))
        
        # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
        # python valye 1e-309
        if log_likeli == xp.inf:
            raise ValueError('LOG-likelihood must be smaller than infinite')

        if xp.isnan(log_likeli):
            log_likeli = -xp.inf
        else:
            log_likeli = log_likeli
            
        return cp2np(log_likeli)


class hierarchical_likelihood_noevents(bilby.Likelihood):
    def __init__(self, injections, rate_model):
        '''
        Base class for an hierachical liklihood. It just saves all the input requirements for a general hierarchical analysis
        
        Parameters
        ----------
        posterior_samples_dict: dict
            Dictionary containing the posterior samples class
        injections: class
            Injection class from its module 
        rate_model: class
            Rate model to compute the CBC rate per year at the detector, taken from the wrapper module.
        '''
        
        # Saves injections in a cupyfied format
        self.injections=injections
        self.rate_model=rate_model
        super().__init__(parameters={ll: None for ll in self.rate_model.population_parameters})
                
    def log_likelihood(self):
        '''
        Evaluates and return the log-likelihood
        '''
        #Update the rate model with the population parameters
        self.rate_model.update(**{key:self.parameters[key] for key in self.rate_model.population_parameters})
        # Update the sensitivity estimation with the new model
        self.injections.update_weights(self.rate_model)
        
        xp = get_module_array(self.injections.log_weights)

        Nexp=self.injections.expected_number_detections()
        # Log likelihood for  the model, Eq. 1.1 on the document
        log_likeli = -Nexp 
        
        # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
        # python valye 1e-309
        if log_likeli == xp.inf:
            raise ValueError('LOG-likelihood must be smaller than infinite')

        if xp.isnan(log_likeli):
            log_likeli = float(xp.nan_to_num(-xp.inf))
        else:
            log_likeli = float(xp.nan_to_num(log_likeli))
            
        return float(cp2np(log_likeli))
                
