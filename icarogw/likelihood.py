from .cupy_pal import cp2np, get_module_array
from .stochastic import spectral_siren_vanilla_omega_gw
import bilby
import numpy as np
import copy as cp
from scipy.interpolate import RegularGridInterpolator

class hierarchical_likelihood_v1_bright_and_dark(bilby.Likelihood):
    # Note the posterior_samples_dict_bright_list is a list correspnding
    # to the various EM counterparts detected
    def __init__(self, list_posterior_samples_dict_bright, list_detection_probabilities,
                 posterior_samples_dict_dark, pmiss_model,
                 injections,
                 rate_model,
                 nparallel_bright=None,nparallel_dark=None, 
                 neffPE=20, neffINJ=None, likelihood_variance_thr=None):

        # Saves injections in a cupyfied format
        self.injections=injections
        self.neffPE=neffPE

        # Rate models for bright and dark sirens
        self.rate_model=rate_model

        # Posterior samples for bright and dark sirens
        self.list_posterior_samples_dict_bright=list_posterior_samples_dict_bright
        self.posterior_samples_dict_dark=posterior_samples_dict_dark
        
        # Saves the detection probabilities models 
        self.list_detection_probabilities=list_detection_probabilities
        self.pmiss_model=pmiss_model

        self.list_distance_kdes_interpolant_bright = []

        for i in range(len(self.list_posterior_samples_dict_bright)):
            list_kde = []
            self.list_posterior_samples_dict_bright[i].build_parallel_posterior(nparallel=nparallel_bright)        
            for key in self.list_posterior_samples_dict_bright[i].posterior_samples_dict.keys():
                self.list_posterior_samples_dict_bright[i].posterior_samples_dict[key].fit_gaussian_kde(['luminosity_distance','cos_theta_jn'])
                
                kde = self.list_posterior_samples_dict_bright[i].posterior_samples_dict[key].kde
                
                dl_interpo = np.linspace(self.list_posterior_samples_dict_bright[i].posterior_samples_dict[key].kde.dataset[0,:].min(), 
                                         self.list_posterior_samples_dict_bright[i].posterior_samples_dict[key].kde.dataset[0,:].max(),
                                         100)
                
                cos_theta_jn_interpo = np.linspace(self.list_posterior_samples_dict_bright[i].posterior_samples_dict[key].kde.dataset[1,:].min(),
                                                 self.list_posterior_samples_dict_bright[i].posterior_samples_dict[key].kde.dataset[1,:].max(),
                                                 200)
                
                XX, YY = np.meshgrid(dl_interpo, cos_theta_jn_interpo)
                logpdfvals = kde.logpdf(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

                ##################
                #  Testing box
                ##################
                # import matplotlib.pyplot as plt
                # plt.pcolormesh(XX, YY, np.exp(logpdfvals))
                # plt.show(block=True)
                # plt.close()
                ##################
                #  Testing box
                ##################

                list_kde.append(RegularGridInterpolator((dl_interpo, cos_theta_jn_interpo), logpdfvals.T,bounds_error=False, fill_value=-np.inf))
                
            # Each element of the lists below corresponds to the interpolant of the events for an EM counterpart
            self.list_distance_kdes_interpolant_bright.append(cp.deepcopy(list_kde))

        self.posterior_samples_dict_dark.build_parallel_posterior(nparallel=nparallel_dark)

        self.likelihood_variance_thr = likelihood_variance_thr

        if likelihood_variance_thr is not None:
            print('Using Likelihood variance as numerical stability estimator \n We will not consider neffPE or neffINJ')
            self.neffPE = -1.
            self.neffINJ = -1.
        else:    
            print('Using neffPE and neffINJ as numerical stability estimators')
            if neffINJ is None:
                print('Setting neffINJ as 4 times observed signals')
                nev_total = np.sum([self.list_posterior_samples_dict_bright[i].n_ev for i in range(len(self.list_posterior_samples_dict_bright))]) + self.posterior_samples_dict_dark.n_ev
                self.neffINJ=4*nev_total
            else:
                self.neffINJ=neffINJ
        
        super().__init__(parameters={ll: None for ll in self.rate_model.population_parameters})
                
    def log_likelihood(self):
        '''
        Evaluates and return the log-likelihood
        '''          

        self.rate_model.update(**{key:self.parameters[key] for key in self.rate_model.population_parameters})
        self.injections.update_weights(self.rate_model)
        Neff=self.injections.effective_injections_number()
        # If the injections are not enough return 0, you cannot go to that point. This is done because the number of injections that you have
        # are not enough to calculate the selection effect
        
        xp = get_module_array(self.injections.log_weights)
        
        if (Neff<self.neffINJ) | (Neff==0.):
            return -xp.inf
        
        # Update the weights on the PE for BRIGHT sirens
        self.rate_model.PEs_parameters = self.rate_model.PEs_parameters_bright # Needed to identify the relavant PEs parameters
        self.rate_model.log_rate_PE = self.rate_model.log_rate_PE_bright
        neff_PE_ev_bright = []
        for i in range(len(self.list_posterior_samples_dict_bright)):
            self.list_detection_probabilities[i].update_cosmology(self.rate_model.cw) # Update the cosmologies in the detection prob models
            self.rate_model.pdet_EM = self.list_detection_probabilities[i] # Updates in the rate model the EM detection prob
            self.rate_model.kde_dl_fits = self.list_distance_kdes_interpolant_bright[i] # Updates in the rate model the distance KDE for the bright sirens
            self.list_posterior_samples_dict_bright[i].update_weights(self.rate_model)
            neff_PE_ev_bright.append(self.list_posterior_samples_dict_bright[i].get_effective_number_of_PE())
        neff_PE_ev_bright = xp.hstack(neff_PE_ev_bright)

        # Update the weights on the PE for DARK sirens
        self.rate_model.PEs_parameters = self.rate_model.PEs_parameters_dark # Needed to identify the relavant PEs parameters
        self.rate_model.log_rate_PE = self.rate_model.log_rate_PE_dark
        self.pmiss_model.update_cosmology(self.rate_model.cw) # Update the cosmologies in the pmiss model
        self.rate_model.pdet_EM = self.pmiss_model # Updates in the rate model the EM detection prob for dark sirens
        self.posterior_samples_dict_dark.update_weights(self.rate_model)
        neff_PE_ev_dark = self.posterior_samples_dict_dark.get_effective_number_of_PE()

        neff_PE_ev_total = xp.hstack([neff_PE_ev_bright,neff_PE_ev_dark])

        if xp.any(neff_PE_ev_total<self.neffPE):
            return -xp.inf
        
        nev_total = xp.sum([self.list_posterior_samples_dict_bright[i].n_ev for i in range(len(self.list_posterior_samples_dict_bright))]) + self.posterior_samples_dict_dark.n_ev
        Ns_array = xp.hstack([self.list_posterior_samples_dict_bright[i].Ns_array for i in range(len(self.list_posterior_samples_dict_bright))] + [self.posterior_samples_dict_dark.Ns_array])
        self.likelihood_variance = (xp.power(nev_total,2.)/Neff)*(1-Neff/self.injections.ntotal)+xp.sum(
            xp.power(neff_PE_ev_total,-1.)*(1-neff_PE_ev_total/Ns_array)
        )

        if self.likelihood_variance_thr is not None:
            if self.likelihood_variance > self.likelihood_variance_thr:
                return -xp.inf
        
        # Combine all the terms  
        if self.rate_model.scale_free:
            # Log likelihood for scale free model, Eq. 1.3 on the document
            log_likeli = xp.sum(xp.log(self.posterior_samples_dict_dark.sum_weights))-self.posterior_samples_dict_dark.n_ev*xp.log(self.injections.pseudo_rate) +\
            xp.sum([xp.sum(xp.log(self.list_posterior_samples_dict_bright[i].sum_weights))-
                    self.list_posterior_samples_dict_bright[i].n_ev*xp.log(self.injections.pseudo_rate) 
                    for i in range(len(self.list_posterior_samples_dict_bright))])
        else:
            Nexp=self.injections.expected_number_detections()
            # Log likelihood for  the model, Eq. 1.1 on the document
            log_likeli = -Nexp + \
                self.posterior_samples_dict_dark.n_ev*xp.log(self.injections.Tobs)+xp.sum(xp.log(self.posterior_samples_dict_dark.sum_weights)) + \
                xp.sum([self.list_posterior_samples_dict_bright[i].n_ev*xp.log(self.injections.Tobs)+
                        xp.sum(xp.log(self.list_posterior_samples_dict_bright[i].sum_weights)) 
                        for i in range(len(self.list_posterior_samples_dict_bright))]) 
      
        
        # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
        # python valye 1e-309
        if log_likeli == xp.inf:
            raise ValueError('LOG-likelihood must be smaller than infinite')

        if xp.isnan(log_likeli):
            log_likeli = -xp.inf
        else:
            log_likeli = log_likeli
            
        return cp2np(log_likeli)

# class hierarchical_likelihood_v1_bright_and_dark(bilby.Likelihood):
#     def __init__(self, posterior_samples_dict_bright, posterior_samples_dict_dark,
#                  injections, 
#                  rate_model, 
#                  nparallel=None, neffPE=20,neffINJ=None,likelihood_variance_thr=None):        

#         # Saves injections in a cupyfied format
#         self.injections=injections
#         self.neffPE=neffPE

#         # Rate models for bright and dark sirens
#         self.rate_model=rate_model

#         # Posterior samples for bright and dark sirens
#         self.posterior_samples_dict_bright=posterior_samples_dict_bright
#         self.posterior_samples_dict_dark=posterior_samples_dict_dark
        
#         self.posterior_samples_dict_bright.build_parallel_posterior(nparallel=nparallel)
#         self.posterior_samples_dict_dark.build_parallel_posterior(nparallel=nparallel)

#         self.likelihood_variance_thr = likelihood_variance_thr

#         if likelihood_variance_thr is not None:
#             print('Using Likelihood variance as numerical stability estimator \n We will not consider neffPE or neffINJ')
#             self.neffPE = -1.
#             self.neffINJ = -1.
#         else:    
#             print('Using neffPE and neffINJ as numerical stability estimators')
#             if neffINJ is None:
#                 print('Setting neffINJ as 4 times observed signals')
#                 self.neffINJ=4*(self.posterior_samples_dict_bright.n_ev + self.posterior_samples_dict_dark.n_ev) 
#             else:
#                 self.neffINJ=neffINJ
        
#         super().__init__(parameters={ll: None for ll in self.rate_model.population_parameters})
                
#     def log_likelihood(self):
#         '''
#         Evaluates and return the log-likelihood
#         '''          

#         self.rate_model.update(**{key:self.parameters[key] for key in self.rate_model.population_parameters})
#         self.injections.update_weights(self.rate_model)
#         Neff=self.injections.effective_injections_number()
#         # If the injections are not enough return 0, you cannot go to that point. This is done because the number of injections that you have
#         # are not enough to calculate the selection effect
        
#         xp = get_module_array(self.injections.log_weights)
        
#         if (Neff<self.neffINJ) | (Neff==0.):
#             return -xp.inf
        
#         # Update the weights on the PE for BRIGHT sirens
#         self.rate_model.PEs_parameters = self.rate_model.PEs_parameters_bright # Needed to identify the relavant PEs parameters
#         self.rate_model.log_rate_PE = self.rate_model.log_rate_PE_bright 
#         self.posterior_samples_dict_bright.update_weights(self.rate_model)
#         neff_PE_ev_bright = self.posterior_samples_dict_bright.get_effective_number_of_PE()

#         # Update the weights on the PE for DARK sirens
#         self.rate_model.PEs_parameters = self.rate_model.PEs_parameters_dark # Needed to identify the relavant PEs parameters
#         self.rate_model.log_rate_PE = self.rate_model.log_rate_PE_dark 
#         self.posterior_samples_dict_dark.update_weights(self.rate_model)
#         neff_PE_ev_dark = self.posterior_samples_dict_dark.get_effective_number_of_PE()

#         neff_PE_ev_total = xp.hstack([neff_PE_ev_bright,neff_PE_ev_dark])

#         if xp.any(neff_PE_ev_total<self.neffPE):
#             return -xp.inf
        
#         nev_total = self.posterior_samples_dict_bright.n_ev + self.posterior_samples_dict_dark.n_ev
#         Ns_array = xp.hstack([self.posterior_samples_dict_bright.Ns_array,self.posterior_samples_dict_dark.Ns_array])

#         self.likelihood_variance = (xp.power(nev_total,2.)/Neff)*(1-Neff/self.injections.ntotal)+xp.sum(
#             xp.power(neff_PE_ev_total,-1.)*(1-neff_PE_ev_total/Ns_array)
#         )

#         if self.likelihood_variance_thr is not None:
#             if self.likelihood_variance > self.likelihood_variance_thr:
#                 return -xp.inf
        
#         # Combine all the terms  
#         if self.rate_model.scale_free:
#             # Log likelihood for scale free model, Eq. 1.3 on the document
#             log_likeli = xp.sum(xp.log(self.posterior_samples_dict_bright.sum_weights))-self.posterior_samples_dict_bright.n_ev*xp.log(self.injections.pseudo_rate) + \
#             xp.sum(xp.log(self.posterior_samples_dict_dark.sum_weights))-self.posterior_samples_dict_dark.n_ev*xp.log(self.injections.pseudo_rate)
#         else:
#             Nexp=self.injections.expected_number_detections()
#             # Log likelihood for  the model, Eq. 1.1 on the document
#             log_likeli = -Nexp + \
#                 self.posterior_samples_dict_bright.n_ev*xp.log(self.injections.Tobs)+xp.sum(xp.log(self.posterior_samples_dict_bright.sum_weights)) + \
#                 self.posterior_samples_dict_dark.n_ev*xp.log(self.injections.Tobs)+xp.sum(xp.log(self.posterior_samples_dict_dark.sum_weights))      
        
#         # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
#         # python valye 1e-309
#         if log_likeli == xp.inf:
#             raise ValueError('LOG-likelihood must be smaller than infinite')

#         if xp.isnan(log_likeli):
#             log_likeli = -xp.inf
#         else:
#             log_likeli = log_likeli
            
#         return cp2np(log_likeli)



# LVK Reviewed
class hierarchical_likelihood(bilby.Likelihood):
    def __init__(self, posterior_samples_dict, injections, rate_model, nparallel=None, neffPE=20,neffINJ=None,likelihood_variance_thr=None):
        '''
        Base class for an hierachical likelihood. It just saves all the input requirements for a general hierarchical analysis
        
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
        #print('Nexp',-Nexp)  

        # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
        # python valye 1e-309
        if log_likeli == xp.inf:
            raise ValueError('LOG-likelihood must be smaller than infinite')

        if xp.isnan(log_likeli):
            log_likeli = float(xp.nan_to_num(-xp.inf))
        else:
            log_likeli = float(xp.nan_to_num(log_likeli))
            
        return float(cp2np(log_likeli))

#LVK reviewed
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
        Base class for an hierachical likelihood. It just saves all the input requirements for a general hierarchical analysis
        
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

class Poisson_times_Stochastic_CBC_likelihood(hierarchical_likelihood):
    def __init__(self, posterior_samples_dict, injections, rate_model, look_up_Om0, stochastic_data, nparallel=None, neffPE=20, neffINJ=None):
        '''
        Class for a joint likelihood: hierachical likelihood combined to stochastic likelihood
        
        Parameters
        ----------
        posterior_samples_dict: dict
            Dictionary containing the posterior samples class
        injections: class
            Injection class from its module 
        rate_model: class
            Rate model to compute the CBC rate per year at the detector, taken from the wrapper module.
            
        look_up_Om0: dict
            OmegaGW weights, computed using the stochastic module
        
        stochastic_data: dict
            Dictonary containing stochastic data: Cf (normalised at H0=100), sigmas2 (normalised at H0=100) and freqs. 
        
        nparallel: int
            Number of samples to use per event, if None it will use the maximum number of PE samples in common to all events
        neffPE: int
            Effective number of samples per event that must contribute the prior evaluation
        neffINJ: int
            Number of effective injections needed to evaluate the selection bias, if None we will assume 4* observed signals.
        '''
        super().__init__(posterior_samples_dict, injections, rate_model, nparallel, neffPE, neffINJ)
        self.freqs = freqs
        self.look_up_Om0 = look_up_Om0
        self.stochastic_data = stochastic_data
    def log_likelihood(self):
        '''
        Evaluates and return the hierarchical log-likelihood
        '''
        hierarchical_log_likelihood = super().log_likelihood()
        stochastic_log_likelihood = self.stochastic_log_likelihood()
        return hierarchical_log_likelihood + stochastic_log_likelihood
    def stochastic_log_likelihood(self):
        '''
        Evaluates and return the stochastic log-likelihood
        '''
        Cf = self.stochastic_data['Cf']*np.power(self.parameters['H0']/100,-2)
        sigma2s = self.stochastic_data['sigma2s']*np.power(self.parameters['H0']/100,-4)
        freqs=self.stochastic_data['freqs']
        # Rate model is updated from the call of the poisson likelihood

        omega_gw = spectral_siren_vanilla_omega_gw(freqs, self.look_up_Om0, self.rate_model)

        # likelihood in eq. 9 from https://arxiv.org/abs/2503.14686
        diff = np.abs(omega_gw - Cf)
        log_stoch = -0.5 * np.sum(((diff**2.) / sigma2s)) #-0.5*np.sum(np.log(2*np.pi*(sigma2s)))
        # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
        # python value 1e-309
        if log_stoch == np.inf:
            raise ValueError('LOG-likelihood must be smaller than infinite')
        if np.isnan(log_stoch):
            log_stoch = float(np.nan_to_num(-np.inf))
        else:
            log_stoch = float(np.nan_to_num(log_stoch))
        
        return float(cp2np(log_stoch))
        
class Stochastic_likelihood_only(bilby.Likelihood):
    def __init__(self, rate_model, look_up_Om0, stochastic_data):
        '''
        Class for a stochastic likelihood.
        
        Parameters
        ----------
        rate_model: class
            Rate model to compute the CBC rate per year at the detector, taken from the wrapper module.
            
        look_up_Om0:
             OmegaGW weights, computed using the stochastic module
        
        stochastic_data: dict
            Dictonary containing stochastic data: Cf (normalised at H0=100), sigmas2 (normalised at H0=100) and freqs. 
        '''
        self.rate_model=rate_model
        self.look_up_Om0 = look_up_Om0
        self.stochastic_data = stochastic_data

        super().__init__(parameters={ll: None for ll in self.rate_model.population_parameters})
        
    def log_likelihood(self):
        '''
        Evaluates and return the hierarchical log-likelihood
        '''
        #Update the rate model with the population parameters
        self.rate_model.update(**{key:self.parameters[key] for key in self.rate_model.population_parameters})
        stochastic_log_likelihood = self.stochastic_log_likelihood()
        return stochastic_log_likelihood
    def stochastic_log_likelihood(self):
        '''
        Evaluates and return the stochastic log-likelihood
        '''
        Cf = self.stochastic_data['Cf']*np.power(self.parameters['H0']/100,-2)
        sigma2s = self.stochastic_data['sigma2s']*np.power(self.parameters['H0']/100,-4)
        freqs=self.stochastic_data['freqs']
        # Rate model is updated from the call of the poisson likelihood

        omega_gw = spectral_siren_vanilla_omega_gw(freqs, self.look_up_Om0, self.rate_model)
        # likelihood in eq. 9 from https://arxiv.org/abs/2503.14686
        diff = np.abs(omega_gw - Cf)
        log_stoch = -0.5 * np.sum(((diff**2.) / sigma2s)) #-0.5*np.sum(np.log(2*np.pi*(sigma2s)))
        # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
        # python value 1e-309
        if log_stoch == np.inf:
            raise ValueError('LOG-likelihood must be smaller than infinite')
        if np.isnan(log_stoch):
            log_stoch = float(np.nan_to_num(-np.inf))
        else:
            log_stoch = float(np.nan_to_num(log_stoch))
        
        return float(cp2np(log_stoch))
