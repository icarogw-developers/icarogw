from .cupy_pal import get_module_array, get_module_array_scipy, np, _set_xp_and_dtype
from .cosmology import alphalog_astropycosmology, cM_astropycosmology, extraD_astropycosmology, Xi0_astropycosmology, astropycosmology, eps0_astropycosmology
from .cosmology import  md_rate, md_gamma_rate, powerlaw_rate, beta_rate, beta_rate_line
from .priors import LowpassSmoothedProb, LowpassSmoothedProbEvolving, PowerLaw, BetaDistribution, TruncatedBetaDistribution, TruncatedGaussian, Bivariate2DGaussian, SmoothedPlusDipProb, BrokenPowerLawMultiPeak
from .priors import PowerLawGaussian, BrokenPowerLaw, PowerLawTwoGaussians, conditional_2dimpdf, conditional_2dimz_pdf, piecewise_constant_2d_distribution_normalized,paired_2dimpdf
from .priors import PowerLawStationary, PowerLawLinear, GaussianStationary, GaussianLinear, _mixed_linear_function, _mixed_double_sigmoid_function
from .priors import BrokenPowerLawTripleMultiPeak
from .priors import TriplePowerLaw, QuadruplePowerLaw
import copy
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, Flatw0waCDM

modgravity_wrappers = ['eps0_mod_wrap','Xi0_mod_wrap','extraD_mod_wrap',
                      'cM_mod_wrap','alphalog_mod_wrap']


lcdm_wrappers = ['FlatLambdaCDM_wrap','FlatwCDM_wrap',
                'Flatw0waCDM_wrap']

    
# A parent class for the rate
# LVK Reviewed
class rate_default(object):
    def evaluate(self,z):
        return self.rate.evaluate(z)
    def log_evaluate(self,z):
        return self.rate.log_evaluate(z)
    
# LVK Reviewed
class rateevolution_PowerLaw(rate_default):
    def __init__(self):
        self.population_parameters=['gamma']
    def update(self,**kwargs):
        self.rate=powerlaw_rate(**kwargs)

# LVK Reviewed
class rateevolution_Madau(rate_default):
    def __init__(self):
        self.population_parameters=['gamma','kappa','zp']
    def update(self,**kwargs):
        self.rate=md_rate(**kwargs)

class rateevolution_Madau_gamma(rate_default):
    def __init__(self):
        self.population_parameters=['gamma','kappa','zp','a','b','c']
    def update(self,**kwargs):
        self.rate=md_gamma_rate(**kwargs)

class rateevolution_beta(rate_default):
    def __init__(self):
        self.population_parameters=['a','b','c']
    def update(self,**kwargs):
        self.rate=beta_rate(**kwargs)

class rateevolution_beta_line(rate_default):
    def __init__(self):
        self.population_parameters=['a','b','c','d']
    def update(self,**kwargs):
        self.rate=beta_rate_line(**kwargs)


# LVK Reviewed
class FlatLambdaCDM_wrap(object):
    def __init__(self,zmax):
        self.population_parameters=['H0','Om0']
        self.cosmology=astropycosmology(zmax)
        self.astropycosmo=FlatLambdaCDM
    def update(self,**kwargs):
        self.cosmology.build_cosmology(self.astropycosmo(**kwargs))

# LVK Reviewed
class FlatwCDM_wrap(object):
    def __init__(self,zmax):
        self.population_parameters=['H0','Om0','w0']
        self.cosmology=astropycosmology(zmax)
        self.astropycosmo=FlatwCDM
    def update(self,**kwargs):
        self.cosmology.build_cosmology(self.astropycosmo(**kwargs))


class Flatw0waCDM_wrap(object):
    def __init__(self,zmax):
        self.population_parameters=['H0','Om0','w0','wa']
        self.cosmology=astropycosmology(zmax)
        self.astropycosmo=Flatw0waCDM
    def update(self,**kwargs):
        self.cosmology.build_cosmology(self.astropycosmo(**kwargs))

class eps0_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['eps0']
        self.cosmology=eps0_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.bgwrap.update(**bgdict)
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),eps0=kwargs['eps0'])

# LVK Reviewed
class Xi0_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['Xi0','n']
        self.cosmology=Xi0_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.bgwrap.update(**bgdict)
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),Xi0=kwargs['Xi0'],n=kwargs['n'])

# LVK Reviewed
class extraD_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['D','n','Rc']
        self.cosmology=extraD_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.bgwrap.update(**bgdict)
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),D=kwargs['D'],n=kwargs['n'],Rc=kwargs['Rc'])

# LVK Reviewed
class cM_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['cM']
        self.cosmology=cM_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.bgwrap.update(**bgdict)
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),cM=kwargs['cM'])

# LVK Reviewed
class alphalog_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['alphalog_1','alphalog_2','alphalog_3']
        self.cosmology=alphalog_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.bgwrap.update(**bgdict)
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),alphalog_1=kwargs['alphalog_1']
                                       ,alphalog_2=kwargs['alphalog_2'],alphalog_3=kwargs['alphalog_3'])

# A parent class for the standard 1D mass probabilities
class pm_prob(object):
    def pdf(self,mass_1_source):
        return self.prior.pdf(mass_1_source)
    def log_pdf(self,mass_1_source):
        return self.prior.log_pdf(mass_1_source)

class mass_ratio_prior_Gaussian(pm_prob):
    def __init__(self):
        self.population_parameters=['mu_q','sigma_q']
    def update(self,**kwargs):
        p1=TruncatedGaussian(kwargs['mu_q'],kwargs['sigma_q'],0.,1.)
        self.prior=p1

class mass_ratio_prior_Powerlaw(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha_q']
    def update(self,**kwargs):
        self.prior=PowerLaw(0.,1.,kwargs['alpha_q'])

class lowSmoothedwrapper(pm_prob):
   def __init__(self, mw):
        self.population_parameters = ['delta_m'] + mw.population_parameters
        self.mw = mw
   def update(self,**kwargs):
        self.mw.update(**{key:kwargs[key] for key in self.mw.population_parameters})
        self.prior = LowpassSmoothedProb(self.mw.prior,kwargs['delta_m'])
 
# A parent class for the standard mass probabilities
# LVK Reviewed
class pm1m2_prob(object):
    def pdf(self,mass_1_source,mass_2_source):
        return self.prior.pdf(mass_1_source,mass_2_source)
    def log_pdf(self,mass_1_source,mass_2_source):
        return self.prior.log_pdf(mass_1_source,mass_2_source)
    
class pm1m2z_prob(object):
    def pdf(self,mass_1_source,mass_2_source,z):
        return self.prior.pdf(mass_1_source,mass_2_source,z)
    def log_pdf(self,mass_1_source,mass_2_source,z):
        return self.prior.log_pdf(mass_1_source,mass_2_source,z)

#LVK reviewed
class massprior_PowerLaw(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha','mmin','mmax']
    def update(self,**kwargs):
        self.prior=PowerLaw(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'])
        
#LVK reviewed
class massprior_PowerLawPeak(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha','mmin','mmax','mu_g','sigma_g','lambda_peak']
    def update(self,**kwargs):
        self.prior=PowerLawGaussian(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'],kwargs['lambda_peak'],kwargs['mu_g'],
                                         kwargs['sigma_g'],kwargs['mmin'],kwargs['mu_g']+5*kwargs['sigma_g'])
        
#LVK reviewed
class massprior_BrokenPowerLaw(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha_1','alpha_2','mmin','mmax','b']
    def update(self,**kwargs):
        self.prior=BrokenPowerLaw(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha_1'],-kwargs['alpha_2'],kwargs['b'])
        
#LVK reviewed
class massprior_MultiPeak(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha','mmin','mmax','mu_g_low','sigma_g_low','lambda_g_low','mu_g_high','sigma_g_high','lambda_g']
    def update(self,**kwargs):
        self.prior=PowerLawTwoGaussians(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'],
                                             kwargs['lambda_g'],kwargs['lambda_g_low'],kwargs['mu_g_low'],
                                             kwargs['sigma_g_low'],kwargs['mmin'],kwargs['mu_g_low']+5*kwargs['sigma_g_low'],
                                             kwargs['mu_g_high'],kwargs['sigma_g_high'],kwargs['mmin'],kwargs['mu_g_high']+5*kwargs['sigma_g_high'])


#LVK reviewed
class massprior_BrokenPowerLawMultiPeak(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha_1','alpha_2','mmin','mmax','b','mu_g_low','sigma_g_low','lambda_g_low','mu_g_high','sigma_g_high','lambda_g']
    def update(self,**kwargs):
        self.prior=BrokenPowerLawMultiPeak(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha_1'],-kwargs['alpha_2'],kwargs['b'],
                                             kwargs['lambda_g'],kwargs['lambda_g_low'],kwargs['mu_g_low'],
                                             kwargs['sigma_g_low'],kwargs['mmin'],kwargs['mu_g_low']+5*kwargs['sigma_g_low'],
                                             kwargs['mu_g_high'],kwargs['sigma_g_high'],kwargs['mmin'],kwargs['mu_g_high']+5*kwargs['sigma_g_high'])

# New class for TripleMultipeak \(multipopulation model\)
class massprior_BrokenPowerLawTripleMultiPeak(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha_1','alpha_2','mmin','mmax','b','mu_g_1','sigma_g_1','lambda_g','mu_g_2','sigma_g_2','lambda_1','mu_g_3','sigma_g_3','lambda_2']
    def update(self,**kwargs):
        self.prior=BrokenPowerLawTripleMultiPeak(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha_1'],-kwargs['alpha_2'],kwargs['b'],
                                                 kwargs['lambda_g'],kwargs['lambda_1'],kwargs['lambda_2'],
                                                 kwargs['mu_g_1'],kwargs['sigma_g_1'],kwargs['mmin'],kwargs['mu_g_1']+5*kwargs['sigma_g_1'],
                                                 kwargs['mu_g_2'],kwargs['sigma_g_2'],kwargs['mmin'],kwargs['mu_g_2']+5*kwargs['sigma_g_2'],
                                                 kwargs['mu_g_3'],kwargs['sigma_g_3'],kwargs['mmin'],kwargs['mu_g_3']+5*kwargs['sigma_g_3'])


#LVK reviewed
class m1m2_conditioned(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters+['beta']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p1 = self.wrapper_m.prior
        p2 = PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta'])
        self.prior=conditional_2dimpdf(p1,p2)

#LVK reviewed
# class m1m2_conditioned_lowpass_m2(pm1m2z_prob):
class m1m2_conditioned_lowpass_m2(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters+['beta']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p1 = self.wrapper_m.prior
        p2 = LowpassSmoothedProb(PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta']),kwargs['delta_m'])
        # self.prior=conditional_2dimz_pdf(p1,p2)
        self.prior=conditional_2dimpdf(p1,p2)

#LVK reviewed
class m1m2_conditioned_lowpass(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters+['beta','delta_m']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p1 = LowpassSmoothedProb(self.wrapper_m.prior,kwargs['delta_m'])
        p2 = LowpassSmoothedProb(PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta']),kwargs['delta_m'])
        self.prior=conditional_2dimpdf(p1,p2)


#LVK reviewed
class m1m2_paired_massratio_dip(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters + ['beta','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        def pairing_function(m1,m2,beta=kwargs['beta']):
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.power(q,beta)
            toret[q>1] = 0.
            return toret
        
        self.prior=paired_2dimpdf(p,pairing_function)


#LVK reviewed
class m1m2_paired_massratio_dip_general(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters + ['beta_bottom','beta_top','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        
        def pairing_function(m1,m2,beta_bottom=kwargs['beta_bottom'],beta_top=kwargs['beta_top'],
                            rightdip=kwargs['rightdip']):
            
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.ones_like(q)
            idx = m2<=rightdip
            toret[idx] = xp.power(q[idx],beta_bottom)
            idx = m2>rightdip
            toret[idx] = xp.power(q[idx],beta_top)
            toret[q>1] = 0.
            return toret
        
        self.prior=paired_2dimpdf(p,pairing_function)


class m1m2_paired_massratio_bpl_dip_farah_2022(pm1m2_prob):
    def __init__(self):
        wrapper_m = massprior_BrokenPowerLaw()
        wrapper_m.population_parameters.remove('b')
        self.population_parameters = wrapper_m.population_parameters + ['beta_bottom','beta_top','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        kwargs['b'] = (kwargs['leftdip']-kwargs['mmin'])/(kwargs['mmax']-kwargs['mmin'])
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters+['b']})
        p = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        
        def pairing_function(m1,m2,beta_bottom=kwargs['beta_bottom'],beta_top=kwargs['beta_top']):
            
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.ones_like(q)
            idx = m2<=5.
            toret[idx] = xp.power(q[idx],beta_bottom)
            idx = m2>5.
            toret[idx] = xp.power(q[idx],beta_top)
            toret[q>1] = 0.
            return toret
        
        self.prior=paired_2dimpdf(p,pairing_function)

# Class for the BPLMultipopTriplePeak_dip
class m1m2_paired_bpl_triplepeak_dip(pm1m2_prob):
    def __init__(self):
        wrapper_m = massprior_BrokenPowerLawTripleMultiPeak()
        wrapper_m.population_parameters.remove('b')
        self.population_parameters = wrapper_m.population_parameters + ['beta_bottom','beta_top','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    
    def update(self,**kwargs):
        mbreak_NS = kwargs['leftdip'] + kwargs['leftdipsmooth']
        mbreak_BH = kwargs['rightdip'] - kwargs['rightdipsmooth']
        mbreak = 0.5*(mbreak_NS+mbreak_BH)
        kwargs['b'] = (mbreak-kwargs['mmin'])/(kwargs['mmax']-kwargs['mmin'])
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters+['b']})
        p = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        
        def pairing_function(m1,m2,beta_bottom=kwargs['beta_bottom'],beta_top=kwargs['beta_top'],mbreak=mbreak):
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.ones_like(q)
            idx = m2<=mbreak
            toret[idx] = xp.power(q[idx],beta_bottom)
            idx = m2>mbreak
            toret[idx] = xp.power(q[idx],beta_top)
            toret[q>1] = 0.
            return toret
        
        self.prior=paired_2dimpdf(p,pairing_function)

#LVK reviewed
class m1m2_paired_massratio_bplmulti_dip(pm1m2_prob):
    def __init__(self):
        wrapper_m = massprior_BrokenPowerLawMultiPeak()
        wrapper_m.population_parameters.remove('b')
        self.population_parameters = wrapper_m.population_parameters + ['beta_bottom','beta_top','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        mbreak_NS = kwargs['leftdip'] + kwargs['leftdipsmooth']
        mbreak_BH = kwargs['rightdip'] - kwargs['rightdipsmooth']
        mbreak = 0.5*(mbreak_NS+mbreak_BH)
        kwargs['b'] = (mbreak-kwargs['mmin'])/(kwargs['mmax']-kwargs['mmin'])
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters+['b']})
        p = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        
        def pairing_function(m1,m2,beta_bottom=kwargs['beta_bottom'],beta_top=kwargs['beta_top'],mbreak=mbreak):
            # The motivation for using only m2 for beta top and bottom is that if m2 is a NS for sure 
            # it is more probable that the binary comes from isolated stellar binaries.
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.ones_like(q)
            idx = m2<=mbreak
            toret[idx] = xp.power(q[idx],beta_bottom)
            idx = m2>mbreak
            toret[idx] = xp.power(q[idx],beta_top)
            toret[q>1] = 0.
            return toret
        
        self.prior=paired_2dimpdf(p,pairing_function)



#LVK reviewed
class m1m2_paired_massratio_bplmulti_dip_conditioned(pm1m2_prob):
    def __init__(self):
        wrapper_m = massprior_BrokenPowerLawMultiPeak()
        wrapper_m.population_parameters.remove('b')
        self.population_parameters = wrapper_m.population_parameters + ['beta_bottom','beta_top','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        mbreak_NS = kwargs['leftdip'] + kwargs['leftdipsmooth']
        mbreak_BH = kwargs['rightdip'] - kwargs['rightdipsmooth']
        mbreak = 0.5*(mbreak_NS+mbreak_BH)
        kwargs['b'] = (mbreak-kwargs['mmin'])/(kwargs['mmax']-kwargs['mmin'])
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters+['b']})
        p1 = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        
        # Equivalent to a broken power law distribution in q = m2/m1
        bpl = BrokenPowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta_bottom'],kwargs['beta_top'],kwargs['b'])
        p2 = LowpassSmoothedProb(bpl,kwargs['bottomsmooth'])
        
        self.prior=conditional_2dimpdf(p1,p2)

#LVK reviewed
class m1m2_paired(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters + ['beta']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
    
        def pairing_function(m1,m2,beta=kwargs['beta']):
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.power(q,beta)
            toret[q>1] = 0.
            return toret
        self.prior=paired_2dimpdf(self.wrapper_m.prior,pairing_function)


class massprior_BinModel2d(pm1m2_prob):
    def __init__(self, n_bins_1d):
        self.population_parameters=['mmin','mmax']
        n_bins_total = int(n_bins_1d * (n_bins_1d + 1) / 2)
        self.bin_parameter_list = ['bin_' + str(i) for i in range(n_bins_total)]
        self.population_parameters += self.bin_parameter_list
    def update(self,**kwargs):
        kwargs_bin_parameters = np.array([kwargs[key] for key in self.bin_parameter_list])
        
        pdf_dist = piecewise_constant_2d_distribution_normalized(
            kwargs['mmin'], 
            kwargs['mmax'],
            kwargs_bin_parameters
        )
        
        self.prior=pdf_dist


# ----------- #
# Spin models #
# ----------- #

class spinprior_default_evolving_gaussian(object):
    def __init__(self):
        self.population_parameters=['mu_chi','sigma_chi','mu_dot','sigma_dot'
                                    ,'sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']

    def update(self,**kwargs):
        self.mu_chi = kwargs['mu_chi']
        self.sigma_chi = kwargs['sigma_chi']
        self.mu_dot = kwargs['mu_dot']
        self.sigma_dot = kwargs['sigma_dot']     
        self.csi_spin = kwargs['csi_spin']
        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)

    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):

        xp = get_module_array(chi_1)
        sx = get_module_array_scipy(chi_1)
 
        mu_chi_1 = self.mu_chi + self.mu_dot*mass_1_source
        sigma_chi_1 = self.sigma_chi + self.sigma_dot*mass_1_source
        mu_chi_2 = self.mu_chi + self.mu_dot*mass_2_source
        sigma_chi_2 = self.sigma_chi + self.sigma_dot*mass_2_source

        a, b = (0. - mu_chi_1) / sigma_chi_1, (1. - mu_chi_1) / sigma_chi_1 
        g1 = sx.stats.truncnorm.pdf(chi_1,a,b,loc=mu_chi_1,scale=sigma_chi_1)

        a, b = (0. - mu_chi_2) / sigma_chi_2, (1. - mu_chi_2) / sigma_chi_2 
        g2 = sx.stats.truncnorm.pdf(chi_2,a,b,loc=mu_chi_2,scale=sigma_chi_2)

        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))

        out = xp.log(g1)+xp.log(g2)+log_angular_part
        
        return out
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source))

class spinprior_default_beta_window_gaussian(object):
    def __init__(self):
        self.population_parameters= ['mt', 
                                     'delta_mt','mix_f',
                                     'alpha_chi','beta_chi',
                                     'mu_chi','sigma_chi',
                                     'sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']
    

    def update(self,**kwargs):
        
        self.alpha_chi = kwargs['alpha_chi']
        self.beta_chi = kwargs['beta_chi']
        if (self.alpha_chi <= 1) | (self.beta_chi <= 1) :
            raise ValueError('Alpha and Beta must be > 1') 
        self.beta_pdf_chi = BetaDistribution(self.alpha_chi,self.beta_chi)
        
        self.mu_chi = kwargs['mu_chi']
        self.sigma_chi = kwargs['sigma_chi']
        self.csi_spin = kwargs['csi_spin']
        self.gaussian_pdf_chi = TruncatedGaussian(kwargs['mu_chi'],kwargs['sigma_chi'],0.,1.)

        self.mt, self.delta_mt, self.mix_f = kwargs['mt'], kwargs['delta_mt'], kwargs['mix_f']

        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)

    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        
        xp = get_module_array(chi_1)
        # FIXME: The sigmoid function implementation has been changed. Check it is correct.
        wz_1 = _mixed_double_sigmoid_function(mass_1_source, self.mix_f, 0., self.mt, self.delta_mt)
        wz_2 = _mixed_double_sigmoid_function(mass_2_source, self.mix_f, 0., self.mt, self.delta_mt)

        pdf_1 = wz_1*self.beta_pdf_chi.pdf(chi_1)+(1-wz_1)*self.gaussian_pdf_chi.pdf(chi_1)
        pdf_2 = wz_2*self.beta_pdf_chi.pdf(chi_2)+(1-wz_2)*self.gaussian_pdf_chi.pdf(chi_2)

        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))
        
        out = xp.log(pdf_1)+xp.log(pdf_2)+log_angular_part
        
        return out
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source))


class spinprior_default_beta_window_beta(object):
    def __init__(self):
        self.population_parameters= ['mt', 
                                     'delta_mt','mix_f',
                                     'alpha_chi_low','beta_chi_low',
                                     'alpha_chi_high','beta_chi_high',
                                     'sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']
    

    def update(self,**kwargs):
        
        self.alpha_chi_low = kwargs['alpha_chi_low']
        self.beta_chi_low = kwargs['beta_chi_low']
        self.alpha_chi_high = kwargs['alpha_chi_high']
        self.beta_chi_high = kwargs['beta_chi_high']
        self.csi_spin = kwargs['csi_spin']
        
        if (self.alpha_chi_low <= 1) | (self.beta_chi_low <= 1) | (self.alpha_chi_high <= 1) | (self.beta_chi_high <= 1):
            raise ValueError('Alpha and Beta must be > 1') 
        
        self.beta_pdf_chi_low = BetaDistribution(self.alpha_chi_low,self.beta_chi_low)
        self.beta_pdf_chi_high = BetaDistribution(self.alpha_chi_high,self.beta_chi_high)

        self.mt, self.delta_mt, self.mix_f = kwargs['mt'], kwargs['delta_mt'], kwargs['mix_f']

        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)

    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        
        xp = get_module_array(chi_1)
        # FIXME: The sigmoid function implementation has been changed. Check it is correct.
        wz_1 = _mixed_double_sigmoid_function(mass_1_source, self.mix_f, 0., self.mt, self.delta_mt)
        wz_2 = _mixed_double_sigmoid_function(mass_2_source, self.mix_f, 0., self.mt, self.delta_mt)

        pdf_1 = wz_1*self.beta_pdf_chi_low.pdf(chi_1)+(1-wz_1)*self.beta_pdf_chi_high.pdf(chi_1)
        pdf_2 = wz_2*self.beta_pdf_chi_low.pdf(chi_2)+(1-wz_2)*self.beta_pdf_chi_high.pdf(chi_2)

        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))
        
        out = xp.log(pdf_1)+xp.log(pdf_2)+log_angular_part
        
        return out
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source))

        
#LVK reviewed
class spinprior_default(object):
    def __init__(self):
        self.population_parameters=['alpha_chi','beta_chi','sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']
        self.name='DEFAULT'

    def update(self,**kwargs):
        self.alpha_chi = kwargs['alpha_chi']
        self.beta_chi = kwargs['beta_chi']
        self.csi_spin = kwargs['csi_spin']
        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)
        if (self.alpha_chi <= 1) | (self.beta_chi <= 1) :
            raise ValueError('Alpha and Beta must be > 1') 
        self.beta_pdf = BetaDistribution(self.alpha_chi,self.beta_chi)
    
    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2):
        xp = get_module_array(chi_1)
        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))
        return self.beta_pdf.log_pdf(chi_1)+self.beta_pdf.log_pdf(chi_2)+log_angular_part
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2))


#LVK reviewed
class spinprior_default_gaussian(object):
    def __init__(self):
        self.population_parameters=['mu_chi_1','mu_chi_2','sigma_chi_1','sigma_chi_2','sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']

    def update(self,**kwargs):        
        self.csi_spin = kwargs['csi_spin']
        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)
        self.g1 = TruncatedGaussian(kwargs['mu_chi_1'],kwargs['sigma_chi_1'],0.,1.)
        self.g2 = TruncatedGaussian(kwargs['mu_chi_2'],kwargs['sigma_chi_2'],0.,1.)
    
    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2):
        xp = get_module_array(chi_1)
        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))
        return self.g1.log_pdf(chi_1)+self.g2.log_pdf(chi_2)+log_angular_part
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2))



########### TGR Implementation ######################
class pseobprior_gaussian(object):
    def __init__(self):
        self.population_parameters=['mu_domega220','sigma_domega220','mu_dtau220','sigma_dtau220','rho_pseob']
    def update(self,**kwargs):
        # Note, the bounds below are set to be consistent with Lorenzo's injections
        self.pdf_evaluator=Bivariate2DGaussian(x1min=-10,x1max=10.,x1mean=kwargs['mu_domega220'],
                                               x2min=-10,x2max=10.,x2mean=kwargs['mu_dtau220'],
                                               x1variance=kwargs['sigma_domega220']**2.,x12covariance=kwargs['rho_pseob']*kwargs['sigma_domega220']*kwargs['sigma_dtau220'],
                                               x2variance=kwargs['sigma_dtau220']**2.)
    def log_pdf(self,domega220,dtau220):
        return self.pdf_evaluator.log_pdf(domega220,dtau220)
    def pdf(self,domega220,dtau220):
        xp = get_module_array(domega220)
        return xp.exp(self.log_pdf(domega220,dtau220))
#####################################################


# LVK Reviewed
class spinprior_gaussian(object):
    def __init__(self):
        self.population_parameters=['mu_chi_eff','sigma_chi_eff','mu_chi_p','sigma_chi_p','rho']
        self.event_parameters=['chi_eff','chi_p']
        self.name='GAUSSIAN'
    def update(self,**kwargs):
        self.pdf_evaluator=Bivariate2DGaussian(x1min=-1.,x1max=1.,x1mean=kwargs['mu_chi_eff'],
                                               x2min=0.,x2max=1.,x2mean=kwargs['mu_chi_p'],
                                               x1variance=kwargs['sigma_chi_eff']**2.,x12covariance=kwargs['rho']*kwargs['sigma_chi_eff']*kwargs['sigma_chi_p'],
                                               x2variance=kwargs['sigma_chi_p']**2.)
    def log_pdf(self,chi_eff,chi_p):
        return self.pdf_evaluator.log_pdf(chi_eff,chi_p)
    def pdf(self,chi_eff,chi_p):
        xp = get_module_array(chi_eff)
        return xp.exp(self.log_pdf(chi_eff,chi_p))
      
class spinprior_ECOs_totally_reflective(object):
    def __init__(self,q=1.):
        # q=1 is the polar case, q = 2 is the axial case, m=2 fixed
        self.q=q
        self.population_parameters=['alpha_chi','beta_chi','eps', 'f_eco', 'sigma_chi_ECO']
        self.event_parameters=['chi_1','chi_2'] 
        self.name='DEFAULT'
        
    def get_chi_crit(self, eps):
        xp = get_module_array(eps)   
        return xp.pi*(1.+self.q)/(2*xp.abs(xp.log(eps)))

    def update(self,**kwargs):
        self.alpha_chi = kwargs['alpha_chi']
        self.beta_chi = kwargs['beta_chi']
        self.eps = kwargs['eps']
        self.f_eco = kwargs['f_eco']
        self.sigma = kwargs['sigma_chi_ECO']
        self.chi_crit = self.get_chi_crit(self.eps)
        if (self.alpha_chi <= 1) | (self.beta_chi <= 1) :
            raise ValueError('Alpha and Beta must be > 1') 
            
        self.beta_pdf = BetaDistribution(self.alpha_chi,self.beta_chi)
        self.truncatedbeta_pdf = TruncatedBetaDistribution(self.alpha_chi,self.beta_chi,self.chi_crit)
        self.truncatedgaussian_pdf = TruncatedGaussian(self.chi_crit, self.sigma, 0., self.chi_crit)
        self.lambda_eco = 1-self.beta_pdf.cdf(np.array([self.get_chi_crit(self.eps)]))[0]
        
        
    def pdf(self,chi_1,chi_2):
        p_chi_1 = self.f_eco*((1-self.lambda_eco)*self.truncatedbeta_pdf.pdf(chi_1) + self.lambda_eco*self.truncatedgaussian_pdf.pdf(chi_1)) + (1-self.f_eco)*self.beta_pdf.pdf(chi_1) 
        p_chi_2 = self.f_eco*((1-self.lambda_eco)*self.truncatedbeta_pdf.pdf(chi_2) + self.lambda_eco*self.truncatedgaussian_pdf.pdf(chi_2)) + (1-self.f_eco)*self.beta_pdf.pdf(chi_2) 
        return p_chi_1*p_chi_2
        
        
    def log_pdf(self,chi_1,chi_2):
        xp = get_module_array(chi_1)
        return xp.log(self.pdf(chi_1,chi_2))
    

# ------------------------ #
# Redshift evolving models #
# ------------------------ #

class PowerLaw_PowerLaw():
    '''
        Class implementing the mass function model for two stationary PowerLaws.

        Some options are available:
            - flag_powerlaw_smoothing applies a left window function to the PowerLaws.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, flag_powerlaw_smoothing = 1):
        
        self.population_parameters   = ['alpha_a', 'mmin_a', 'mmax_a', 'alpha_b', 'mmin_b', 'mmax_b', 'mix']
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing

        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m_a', 'delta_m_b']

    def update(self,**kwargs):

        self.alpha_a = kwargs['alpha_a']
        self.mmin_a  = kwargs['mmin_a']
        self.mmax_a  = kwargs['mmax_a']
        self.alpha_b = kwargs['alpha_b']
        self.mmin_b  = kwargs['mmin_b']
        self.mmax_b  = kwargs['mmax_b']
        self.mix     = kwargs['mix']

        if self.flag_powerlaw_smoothing:
            self.delta_m_a = kwargs['delta_m_a']
            self.delta_m_b = kwargs['delta_m_b']

    def pdf(self,m):

        xp = get_module_array(m)
        powerlaw_class_a = PowerLawStationary(self.alpha_a, self.mmin_a, self.mmax_a)
        powerlaw_class_b = PowerLawStationary(self.alpha_b, self.mmin_b, self.mmax_b)
        # Add left smoothing to the evolving PowerLaw.
        if self.flag_powerlaw_smoothing:
            powerlaw_class_a = LowpassSmoothedProb(powerlaw_class_a, self.delta_m_a)
            powerlaw_class_b = LowpassSmoothedProb(powerlaw_class_b, self.delta_m_b)
        powerlaw_part_a = powerlaw_class_a.pdf(m)
        powerlaw_part_b = powerlaw_class_b.pdf(m)

        # Impose the rate to be between [0,1].
        if (self.mix > 1) or (self.mix < 0):
            return xp.nan
        else:
            return self.mix * powerlaw_part_a + (1-self.mix) * powerlaw_part_b
    
    def log_pdf(self,m):
        xp = get_module_array(m)
        return xp.log(self.pdf(m))


class PowerLaw_PowerLaw_PowerLaw():
    '''
        Class implementing the mass function model for three stationary PowerLaws.

        Some options are available:
            - flag_powerlaw_smoothing applies a left window function to the PowerLaws.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, flag_powerlaw_smoothing = 1):
        
        self.population_parameters   = ['alpha_a', 'mmin_a', 'mmax_a', 'alpha_b', 'mmin_b', 'mmax_b', 'alpha_c', 'mmin_c', 'mmax_c', 'mix_alpha', 'mix_beta']
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing

        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m_a', 'delta_m_b', 'delta_m_c']

    def update(self,**kwargs):

        self.alpha_a   = kwargs['alpha_a']
        self.mmin_a    = kwargs['mmin_a']
        self.mmax_a    = kwargs['mmax_a']
        self.alpha_b   = kwargs['alpha_b']
        self.mmin_b    = kwargs['mmin_b']
        self.mmax_b    = kwargs['mmax_b']
        self.alpha_c   = kwargs['alpha_c']
        self.mmin_c    = kwargs['mmin_c']
        self.mmax_c    = kwargs['mmax_c']
        self.mix_alpha = kwargs['mix_alpha']
        self.mix_beta  = kwargs['mix_beta']

        if self.flag_powerlaw_smoothing:
            self.delta_m_a = kwargs['delta_m_a']
            self.delta_m_b = kwargs['delta_m_b']
            self.delta_m_c = kwargs['delta_m_c']

    def pdf(self,m):

        xp = get_module_array(m)
        powerlaw_class_a = PowerLawStationary(self.alpha_a, self.mmin_a, self.mmax_a)
        powerlaw_class_b = PowerLawStationary(self.alpha_b, self.mmin_b, self.mmax_b)
        powerlaw_class_c = PowerLawStationary(self.alpha_c, self.mmin_c, self.mmax_c)
        # Add left smoothing to the evolving PowerLaw.
        if self.flag_powerlaw_smoothing:
            powerlaw_class_a = LowpassSmoothedProb(powerlaw_class_a, self.delta_m_a)
            powerlaw_class_b = LowpassSmoothedProb(powerlaw_class_b, self.delta_m_b)
            powerlaw_class_c = LowpassSmoothedProb(powerlaw_class_c, self.delta_m_c)
        powerlaw_part_a = powerlaw_class_a.pdf(m)
        powerlaw_part_b = powerlaw_class_b.pdf(m)
        powerlaw_part_c = powerlaw_class_c.pdf(m)

        # Impose the rate to be between [0,1].
        if (self.mix_alpha > 1) or (self.mix_alpha < 0) or (self.mix_beta > 1) or (self.mix_beta < 0) or (self.mix_alpha + self.mix_beta > 1):
            return xp.nan
        else:
            return self.mix_alpha * powerlaw_part_a + self.mix_beta * powerlaw_part_b + (1 - self.mix_beta - self.mix_alpha) * powerlaw_part_c

    def log_pdf(self,m):
        xp = get_module_array(m)
        return xp.log(self.pdf(m))


class PowerLaw_PowerLaw_PowerLaw_PowerLaw():
    '''
        Class implementing the mass function model for four stationary PowerLaws.

        Some options are available:
            - flag_powerlaw_smoothing applies a left window function to the PowerLaws.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, flag_powerlaw_smoothing = 1):
        
        self.population_parameters   = [
            'alpha_a', 'mmin_a', 'mmax_a',
            'alpha_b', 'mmin_b', 'mmax_b',
            'alpha_c', 'mmin_c', 'mmax_c',
            'alpha_d', 'mmin_d', 'mmax_d',
            'mix_alpha', 'mix_beta', 'mix_gamma'
        ]
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing

        if self.flag_powerlaw_smoothing: self.population_parameters += [
            'delta_m_a', 'delta_m_b', 'delta_m_c', 'delta_m_d'
        ]

    def update(self,**kwargs):

        self.alpha_a   = kwargs['alpha_a']
        self.mmin_a    = kwargs['mmin_a']
        self.mmax_a    = kwargs['mmax_a']
        self.alpha_b   = kwargs['alpha_b']
        self.mmin_b    = kwargs['mmin_b']
        self.mmax_b    = kwargs['mmax_b']
        self.alpha_c   = kwargs['alpha_c']
        self.mmin_c    = kwargs['mmin_c']
        self.mmax_c    = kwargs['mmax_c']
        self.alpha_d   = kwargs['alpha_d']
        self.mmin_d    = kwargs['mmin_d']
        self.mmax_d    = kwargs['mmax_d']
        self.mix_alpha = kwargs['mix_alpha']
        self.mix_beta  = kwargs['mix_beta']
        self.mix_gamma = kwargs['mix_gamma']

        if self.flag_powerlaw_smoothing:
            self.delta_m_a = kwargs['delta_m_a']
            self.delta_m_b = kwargs['delta_m_b']
            self.delta_m_c = kwargs['delta_m_c']
            self.delta_m_d = kwargs['delta_m_d']

    def pdf(self,m):

        xp = get_module_array(m)
        powerlaw_class_a = PowerLawStationary(self.alpha_a, self.mmin_a, self.mmax_a)
        powerlaw_class_b = PowerLawStationary(self.alpha_b, self.mmin_b, self.mmax_b)
        powerlaw_class_c = PowerLawStationary(self.alpha_c, self.mmin_c, self.mmax_c)
        powerlaw_class_d = PowerLawStationary(self.alpha_d, self.mmin_d, self.mmax_d)
        # Add left smoothing to the evolving PowerLaw.
        if self.flag_powerlaw_smoothing:
            powerlaw_class_a = LowpassSmoothedProb(powerlaw_class_a, self.delta_m_a)
            powerlaw_class_b = LowpassSmoothedProb(powerlaw_class_b, self.delta_m_b)
            powerlaw_class_c = LowpassSmoothedProb(powerlaw_class_c, self.delta_m_c)
            powerlaw_class_d = LowpassSmoothedProb(powerlaw_class_d, self.delta_m_d)
        powerlaw_part_a = powerlaw_class_a.pdf(m)
        powerlaw_part_b = powerlaw_class_b.pdf(m)
        powerlaw_part_c = powerlaw_class_c.pdf(m)
        powerlaw_part_d = powerlaw_class_d.pdf(m)

        if (# Impose the rate to be between [0,1].
            self.mix_alpha  < 0 or self.mix_alpha  > 1 or
            self.mix_beta   < 0 or self.mix_beta   > 1 or
            self.mix_gamma  < 0 or self.mix_gamma  > 1 or
            (self.mix_alpha + self.mix_beta + self.mix_gamma > 1)
        ):
            return xp.nan
        else:
            return (
                self.mix_alpha * powerlaw_part_a +
                self.mix_beta  * powerlaw_part_b +
                self.mix_gamma * powerlaw_part_c +
                (1 - self.mix_alpha - self.mix_beta - self.mix_gamma) * powerlaw_part_d
            )

    def log_pdf(self,m):
        xp = get_module_array(m)
        return xp.log(self.pdf(m))


class PowerLaw_PowerLaw_Gaussian():
    '''
        Class implementing the mass function model for two stationary PowerLaws
        and a Gaussian peak.

        Some options are available:
            - flag_powerlaw_smoothing applies a left window function to the PowerLaws.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, flag_powerlaw_smoothing = 1):
        
        self.population_parameters   = ['alpha_a', 'mmin_a', 'mmax_a', 'alpha_b', 'mmin_b', 'mmax_b', 'mu_g', 'sigma_g', 'mix_alpha', 'mix_beta']
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing

        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m_a', 'delta_m_b']

    def update(self,**kwargs):

        self.alpha_a   = kwargs['alpha_a']
        self.mmin_a    = kwargs['mmin_a']
        self.mmax_a    = kwargs['mmax_a']
        self.alpha_b   = kwargs['alpha_b']
        self.mmin_b    = kwargs['mmin_b']
        self.mmax_b    = kwargs['mmax_b']
        self.mu_g      = kwargs['mu_g']
        self.sigma_g   = kwargs['sigma_g']
        self.mix_alpha = kwargs['mix_alpha']
        self.mix_beta  = kwargs['mix_beta']

        if self.flag_powerlaw_smoothing:
            self.delta_m_a = kwargs['delta_m_a']
            self.delta_m_b = kwargs['delta_m_b']

    def pdf(self,m):

        xp = get_module_array(m)
        powerlaw_class_a = PowerLawStationary(self.alpha_a, self.mmin_a,  self.mmax_a)
        powerlaw_class_b = PowerLawStationary(self.alpha_b, self.mmin_b,  self.mmax_b)
        gaussian_class   = GaussianStationary(self.mu_g,    self.sigma_g, self.mmin_a)
        # Add left smoothing to the evolving PowerLaw.
        if self.flag_powerlaw_smoothing:
            powerlaw_class_a = LowpassSmoothedProb(powerlaw_class_a, self.delta_m_a)
            powerlaw_class_b = LowpassSmoothedProb(powerlaw_class_b, self.delta_m_b)
        powerlaw_part_a = powerlaw_class_a.pdf(m)
        powerlaw_part_b = powerlaw_class_b.pdf(m)
        gaussian_part   = gaussian_class.pdf(m)

        # Impose the rate to be between [0,1].
        if (self.mix_alpha > 1) or (self.mix_alpha < 0) or (self.mix_beta > 1) or (self.mix_beta < 0) or (self.mix_alpha + self.mix_beta > 1):
            return xp.nan
        else:
            return self.mix_alpha * powerlaw_part_a + self.mix_beta * powerlaw_part_b + (1 - self.mix_beta - self.mix_alpha) * gaussian_part
    
    def log_pdf(self,m):
        xp = get_module_array(m)
        return xp.log(self.pdf(m))


class PowerLaw_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for a stationary PowerLaw and a redshift linearly-dependent Gaussian peak.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the PowerLaw and the Gaussian.
            - flag_powerlaw_smoothing applies a left window function to the PowerLaw.
            - flag_redshift_mixture allows for the transition function to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_powerlaw_smoothing = 1, flag_redshift_mixture = 1):
        
        self.population_parameters   = ['alpha', 'mmin', 'mmax', 'mu_z0', 'mu_z1', 'sigma_z0', 'sigma_z1', 'mix_z0']
        self.redshift_transition     = redshift_transition
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.flag_redshift_mixture   = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_z1']
            if self.redshift_transition == 'sigmoid': self.population_parameters += ['zt', 'delta_zt']
        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m']

    def update(self,**kwargs):

        self.alpha    = kwargs['alpha']
        self.mmin     = kwargs['mmin']
        self.mmax     = kwargs['mmax']
        self.mu_z0    = kwargs['mu_z0']
        self.mu_z1    = kwargs['mu_z1']
        self.sigma_z0 = kwargs['sigma_z0']
        self.sigma_z1 = kwargs['sigma_z1']
        self.mix_z0   = kwargs['mix_z0']

        if self.flag_redshift_mixture:
            self.mix_z1 = kwargs['mix_z1']
            if self.redshift_transition == 'sigmoid': self.zt, self.delta_zt = kwargs['zt'], kwargs['delta_zt']
        if self.flag_powerlaw_smoothing: self.delta_m = kwargs['delta_m']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz = _mixed_linear_function(         z, self.mix_z0, self.mix_z1)
            elif self.redshift_transition == 'sigmoid':
                wz = _mixed_double_sigmoid_function( z, self.mix_z0, self.mix_z1, self.zt, self.delta_zt)
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz = xp.array(self.mix_z0)

        powerlaw_class = PowerLawStationary(self.alpha, self.mmin, self.mmax)
        # Add left smoothing to the evolving PowerLaw.
        if self.flag_powerlaw_smoothing: powerlaw_class = LowpassSmoothedProb(powerlaw_class, self.delta_m)
        gaussian_class = GaussianLinear(z, self.mu_z0, self.mu_z1, self.sigma_z0, self.sigma_z1, self.mmin)
        powerlaw_part  = powerlaw_class.pdf(m)
        gaussian_part  = gaussian_class.pdf(m)

        # Impose the rate to be between [0,1].
        if (xp.any(wz > 1)) or (xp.any(wz < 0)):
            return xp.nan
        else:
            return wz * powerlaw_part + (1-wz) * gaussian_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))
    

class PowerLaw_GaussianRedshiftLinear_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for a stationary PowerLaw and two redshift linearly-dependent Gaussian peaks.

        Some options are available:
            - redshift_transition sets the functions for the redshift transition
            between the PowerLaw and the two Gaussian peaks. The transition is
            the same between the PowerLaw and the first Gaussian a, and the two
            Gaussians a and b.
            - flag_powerlaw_smoothing applies a left window function to the PowerLaw.
            - flag_redshift_mixture allows for the transition functions to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_powerlaw_smoothing = 1, flag_redshift_mixture = 1):
        
        self.population_parameters   = ['alpha', 'mmin', 'mmax', 'mu_a_z0', 'mu_a_z1', 'sigma_a_z0', 'sigma_a_z1', 'mu_b_z0', 'mu_b_z1', 'sigma_b_z0', 'sigma_b_z1', 'mix_alpha_z0', 'mix_beta_z0']
        self.redshift_transition     = redshift_transition
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.flag_redshift_mixture   = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_alpha_z1', 'mix_beta_z1']
            if self.redshift_transition == 'sigmoid': self.population_parameters += ['zt', 'delta_zt']
        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m']

    def update(self,**kwargs):

        self.alpha        = kwargs['alpha']
        self.mmin         = kwargs['mmin']
        self.mmax         = kwargs['mmax']
        self.mu_a_z0      = kwargs['mu_a_z0']
        self.mu_a_z1      = kwargs['mu_a_z1']
        self.sigma_a_z0   = kwargs['sigma_a_z0']
        self.sigma_a_z1   = kwargs['sigma_a_z1']
        self.mu_b_z0      = kwargs['mu_b_z0']
        self.mu_b_z1      = kwargs['mu_b_z1']
        self.sigma_b_z0   = kwargs['sigma_b_z0']
        self.sigma_b_z1   = kwargs['sigma_b_z1']
        self.mix_alpha_z0 = kwargs['mix_alpha_z0']
        self.mix_beta_z0  = kwargs['mix_beta_z0']

        if self.flag_redshift_mixture:
            self.mix_alpha_z1 = kwargs['mix_alpha_z1']
            self.mix_beta_z1  = kwargs['mix_beta_z1']
            if self.redshift_transition == 'sigmoid': self.zt, self.delta_zt = kwargs['zt'], kwargs['delta_zt']
        if self.flag_powerlaw_smoothing: self.delta_m = kwargs['delta_m']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz_alpha = _mixed_linear_function(         z, self.mix_alpha_z0, self.mix_alpha_z1)
                wz_beta  = _mixed_linear_function(         z, self.mix_beta_z0 , self.mix_beta_z1 )
            elif self.redshift_transition == 'sigmoid':
                wz_alpha = _mixed_double_sigmoid_function( z, self.mix_alpha_z0, self.mix_alpha_z1, self.zt, self.delta_zt)
                wz_beta  = _mixed_double_sigmoid_function( z, self.mix_beta_z0 , self.mix_beta_z1 , self.zt, self.delta_zt)
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            # Need to convert to arrays to ensure CuPy compatibility.
            wz_alpha = xp.array(self.mix_alpha_z0)
            wz_beta  = xp.array(self.mix_beta_z0)

        powerlaw_class = PowerLawStationary(self.alpha, self.mmin, self.mmax)
        # Add left smoothing to the evolving PowerLaw.
        if self.flag_powerlaw_smoothing: powerlaw_class = LowpassSmoothedProb(powerlaw_class, self.delta_m)
        gaussian_a_class = GaussianLinear(z, self.mu_a_z0, self.mu_a_z1, self.sigma_a_z0, self.sigma_a_z1, self.mmin)
        gaussian_b_class = GaussianLinear(z, self.mu_b_z0, self.mu_b_z1, self.sigma_b_z0, self.sigma_b_z1, self.mmin)
        powerlaw_part    = powerlaw_class.pdf(m)
        gaussian_a_part  = gaussian_a_class.pdf(m)
        gaussian_b_part  = gaussian_b_class.pdf(m)

        # Impose the rate to be between [0,1].
        if (xp.any(wz_alpha > 1)) or (xp.any(wz_alpha < 0)) or (xp.any(wz_beta > 1)) or (xp.any(wz_beta < 0)) or (xp.any((wz_alpha + wz_beta) > 1)):
            return xp.nan
        else:
            return wz_alpha * powerlaw_part + wz_beta * gaussian_a_part + (1 - wz_beta - wz_alpha) * gaussian_b_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


class PowerLawRedshiftLinear_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for both a redshift linearly-dependent PowerLaw and Gaussian peak.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the PowerLaw and the Gaussian.
            - flag_powerlaw_smoothing applies a left window function to the PowerLaw.
            The smoothing slows heavily down the model evaluation.
            - flag_redshift_mixture allows for the transition function to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_powerlaw_smoothing = 0, flag_redshift_mixture = 1):
        
        self.population_parameters   = ['alpha_z0', 'alpha_z1', 'mmin_z0', 'mmin_z1', 'mmax_z0', 'mmax_z1', 'mu_z0', 'mu_z1', 'sigma_z0', 'sigma_z1', 'mix_z0']
        self.redshift_transition     = redshift_transition
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.flag_redshift_mixture   = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_z1']
            if self.redshift_transition == 'sigmoid': self.population_parameters += ['zt', 'delta_zt']
        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m']

    def update(self,**kwargs):

        self.alpha_z0 = kwargs['alpha_z0']
        self.alpha_z1 = kwargs['alpha_z1']
        self.mmin_z0  = kwargs['mmin_z0']
        self.mmin_z1  = kwargs['mmin_z1']
        self.mmax_z0  = kwargs['mmax_z0']
        self.mmax_z1  = kwargs['mmax_z1']
        self.mu_z0    = kwargs['mu_z0']
        self.mu_z1    = kwargs['mu_z1']
        self.sigma_z0 = kwargs['sigma_z0']
        self.sigma_z1 = kwargs['sigma_z1']
        self.mix_z0   = kwargs['mix_z0']

        if self.flag_redshift_mixture:
            self.mix_z1 = kwargs['mix_z1']
            if self.redshift_transition == 'sigmoid': self.zt, self.delta_zt = kwargs['zt'], kwargs['delta_zt']
        if self.flag_powerlaw_smoothing: self.delta_m = kwargs['delta_m']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz = _mixed_linear_function(         z, self.mix_z0, self.mix_z1)
            elif self.redshift_transition == 'sigmoid':
                wz = _mixed_double_sigmoid_function( z, self.mix_z0, self.mix_z1, self.zt, self.delta_zt)
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz = xp.array(self.mix_z0)

        powerlaw_class = PowerLawLinear(z, self.alpha_z0, self.alpha_z1, self.mmin_z0, self.mmin_z1, self.mmax_z0, self.mmax_z1)
        # Add left smoothing to the evolving PowerLaw.
        # WARNING: The implementation is very slow, because the integral to normalise the windowed
        # distribution p(m1|z) needs to be computed at all redshifts corresponding to the PE samples and injections.
        if self.flag_powerlaw_smoothing: powerlaw_class = LowpassSmoothedProbEvolving(powerlaw_class, self.delta_m)
        gaussian_class = GaussianLinear(z, self.mu_z0, self.mu_z1, self.sigma_z0, self.sigma_z1, self.mmin_z0)
        powerlaw_part  = powerlaw_class.pdf(m)
        gaussian_part  = gaussian_class.pdf(m)

        # Impose the rate to be between [0,1].
        if (xp.any(wz > 1)) or (xp.any(wz < 0)):
            return xp.nan
        else:
            return wz * powerlaw_part + (1-wz) * gaussian_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))
    

class PowerLawRedshiftLinear_PowerLawRedshiftLinear_PowerLawRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for three redshift linearly-dependent PowerLaws.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the PowerLaws (a, b and c).
            - flag_powerlaw_smoothing applies a left window function to the PowerLaws.
            The smoothing slows heavily down the model evaluation.
            - flag_redshift_mixture allows for the transition functions to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_powerlaw_smoothing = 0, flag_redshift_mixture = 1):
        
        self.population_parameters   = ['alpha_a_z0', 'alpha_a_z1', 'mmin_a_z0', 'mmin_a_z1', 'mmax_a_z0', 'mmax_a_z1', 'alpha_b_z0', 'alpha_b_z1', 'mmin_b_z0', 'mmin_b_z1', 'mmax_b_z0', 'mmax_b_z1', 'alpha_c_z0', 'alpha_c_z1', 'mmin_c_z0', 'mmin_c_z1', 'mmax_c_z0', 'mmax_c_z1', 'mix_alpha_z0', 'mix_beta_z0']
        self.redshift_transition     = redshift_transition
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.flag_redshift_mixture   = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_alpha_z1', 'mix_beta_z1']
        if self.flag_powerlaw_smoothing:
            self.population_parameters += ['delta_m_a', 'delta_m_b', 'delta_m_c']

    def update(self,**kwargs):

        self.alpha_a_z0   = kwargs['alpha_a_z0']
        self.alpha_a_z1   = kwargs['alpha_a_z1']
        self.mmin_a_z0    = kwargs['mmin_a_z0']
        self.mmin_a_z1    = kwargs['mmin_a_z1']
        self.mmax_a_z0    = kwargs['mmax_a_z0']
        self.mmax_a_z1    = kwargs['mmax_a_z1']
        self.alpha_b_z0   = kwargs['alpha_b_z0']
        self.alpha_b_z1   = kwargs['alpha_b_z1']
        self.mmin_b_z0    = kwargs['mmin_b_z0']
        self.mmin_b_z1    = kwargs['mmin_b_z1']
        self.mmax_b_z0    = kwargs['mmax_b_z0']
        self.mmax_b_z1    = kwargs['mmax_b_z1']
        self.alpha_c_z0   = kwargs['alpha_c_z0']
        self.alpha_c_z1   = kwargs['alpha_c_z1']
        self.mmin_c_z0    = kwargs['mmin_c_z0']
        self.mmin_c_z1    = kwargs['mmin_c_z1']
        self.mmax_c_z0    = kwargs['mmax_c_z0']
        self.mmax_c_z1    = kwargs['mmax_c_z1']
        self.mix_alpha_z0 = kwargs['mix_alpha_z0']
        self.mix_beta_z0  = kwargs['mix_beta_z0']

        if self.flag_redshift_mixture:
            self.mix_alpha_z1 = kwargs['mix_alpha_z1']
            self.mix_beta_z1  = kwargs['mix_beta_z1']
        if self.flag_powerlaw_smoothing:
            self.delta_m_a    = kwargs['delta_m_a']
            self.delta_m_b    = kwargs['delta_m_b']
            self.delta_m_c    = kwargs['delta_m_c']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz_alpha = _mixed_linear_function(z, self.mix_alpha_z0, self.mix_alpha_z1)
                wz_beta  = _mixed_linear_function(z, self.mix_beta_z0,  self.mix_beta_z1 )
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz_alpha = xp.array(self.mix_alpha_z0)
            wz_beta  = xp.array(self.mix_beta_z0 )

        powerlaw_class_a = PowerLawLinear(z, self.alpha_a_z0, self.alpha_a_z1, self.mmin_a_z0, self.mmin_a_z1, self.mmax_a_z0, self.mmax_a_z1)
        powerlaw_class_b = PowerLawLinear(z, self.alpha_b_z0, self.alpha_b_z1, self.mmin_b_z0, self.mmin_b_z1, self.mmax_b_z0, self.mmax_b_z1)
        powerlaw_class_c = PowerLawLinear(z, self.alpha_c_z0, self.alpha_c_z1, self.mmin_c_z0, self.mmin_c_z1, self.mmax_c_z0, self.mmax_c_z1)
        # Add left smoothing to the evolving PowerLaw.
        # WARNING: The implementation is very slow, because the integral to normalise the windowed
        # distribution p(m1|z) needs to be computed at all redshifts corresponding to the PE samples and injections.
        if self.flag_powerlaw_smoothing:
            powerlaw_class_a = LowpassSmoothedProbEvolving(powerlaw_class_a, self.delta_m_a)
            powerlaw_class_b = LowpassSmoothedProbEvolving(powerlaw_class_b, self.delta_m_b)
            powerlaw_class_c = LowpassSmoothedProbEvolving(powerlaw_class_c, self.delta_m_c)
        powerlaw_part_a  = powerlaw_class_a.pdf(m)
        powerlaw_part_b  = powerlaw_class_b.pdf(m)
        powerlaw_part_c  = powerlaw_class_c.pdf(m)

        # Impose the rate to be between [0,1].
        if (xp.any(wz_alpha > 1)) or (xp.any(wz_alpha < 0)) or (xp.any(wz_beta > 1)) or (xp.any(wz_beta < 0)) or (xp.any(wz_alpha + wz_beta > 1)):
            return xp.nan
        else:
            return wz_alpha * powerlaw_part_a + wz_beta * powerlaw_part_b + (1 - wz_beta - wz_alpha) * powerlaw_part_c
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


class PowerLawRedshiftLinear_PowerLawRedshiftLinear_PowerLawRedshiftLinear_PowerLawRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for four redshift linearly-dependent PowerLaws.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the PowerLaws (a, b, c and d).
            - flag_powerlaw_smoothing applies a left window function to the PowerLaws.
            The smoothing slows heavily down the model evaluation.
            - flag_redshift_mixture allows for the transition functions to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_powerlaw_smoothing = 0, flag_redshift_mixture = 1):
        
        self.population_parameters   = ['alpha_a_z0', 'alpha_a_z1', 'mmin_a_z0', 'mmin_a_z1', 'mmax_a_z0', 'mmax_a_z1', 'alpha_b_z0', 'alpha_b_z1', 'mmin_b_z0', 'mmin_b_z1', 'mmax_b_z0', 'mmax_b_z1', 'alpha_c_z0', 'alpha_c_z1', 'mmin_c_z0', 'mmin_c_z1', 'mmax_c_z0', 'mmax_c_z1', 'alpha_d_z0', 'alpha_d_z1', 'mmin_d_z0', 'mmin_d_z1', 'mmax_d_z0', 'mmax_d_z1', 'mix_alpha_z0', 'mix_beta_z0', 'mix_gamma_z0']
        self.redshift_transition     = redshift_transition
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.flag_redshift_mixture   = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_alpha_z1', 'mix_beta_z1', 'mix_gamma_z1']
        if self.flag_powerlaw_smoothing:
            self.population_parameters += ['delta_m_a', 'delta_m_b', 'delta_m_c', 'delta_m_d']

    def update(self,**kwargs):

        self.alpha_a_z0   = kwargs['alpha_a_z0']
        self.alpha_a_z1   = kwargs['alpha_a_z1']
        self.mmin_a_z0    = kwargs['mmin_a_z0']
        self.mmin_a_z1    = kwargs['mmin_a_z1']
        self.mmax_a_z0    = kwargs['mmax_a_z0']
        self.mmax_a_z1    = kwargs['mmax_a_z1']
        self.alpha_b_z0   = kwargs['alpha_b_z0']
        self.alpha_b_z1   = kwargs['alpha_b_z1']
        self.mmin_b_z0    = kwargs['mmin_b_z0']
        self.mmin_b_z1    = kwargs['mmin_b_z1']
        self.mmax_b_z0    = kwargs['mmax_b_z0']
        self.mmax_b_z1    = kwargs['mmax_b_z1']
        self.alpha_c_z0   = kwargs['alpha_c_z0']
        self.alpha_c_z1   = kwargs['alpha_c_z1']
        self.mmin_c_z0    = kwargs['mmin_c_z0']
        self.mmin_c_z1    = kwargs['mmin_c_z1']
        self.mmax_c_z0    = kwargs['mmax_c_z0']
        self.mmax_c_z1    = kwargs['mmax_c_z1']
        self.alpha_d_z0   = kwargs['alpha_d_z0']
        self.alpha_d_z1   = kwargs['alpha_d_z1']
        self.mmin_d_z0    = kwargs['mmin_d_z0']
        self.mmin_d_z1    = kwargs['mmin_d_z1']
        self.mmax_d_z0    = kwargs['mmax_d_z0']
        self.mmax_d_z1    = kwargs['mmax_d_z1']
        self.mix_alpha_z0 = kwargs['mix_alpha_z0']
        self.mix_beta_z0  = kwargs['mix_beta_z0']
        self.mix_gamma_z0 = kwargs['mix_gamma_z0']

        if self.flag_redshift_mixture:
            self.mix_alpha_z1 = kwargs['mix_alpha_z1']
            self.mix_beta_z1  = kwargs['mix_beta_z1']
            self.mix_gamma_z1 = kwargs['mix_gamma_z1']
        if self.flag_powerlaw_smoothing:
            self.delta_m_a    = kwargs['delta_m_a']
            self.delta_m_b    = kwargs['delta_m_b']
            self.delta_m_c    = kwargs['delta_m_c']
            self.delta_m_d    = kwargs['delta_m_d']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz_alpha = _mixed_linear_function(z, self.mix_alpha_z0, self.mix_alpha_z1)
                wz_beta  = _mixed_linear_function(z, self.mix_beta_z0,  self.mix_beta_z1 )
                wz_gamma = _mixed_linear_function(z, self.mix_gamma_z0, self.mix_gamma_z1)
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz_alpha = xp.array(self.mix_alpha_z0)
            wz_beta  = xp.array(self.mix_beta_z0 )
            wz_gamma = xp.array(self.mix_gamma_z0)

        powerlaw_class_a = PowerLawLinear(z, self.alpha_a_z0, self.alpha_a_z1, self.mmin_a_z0, self.mmin_a_z1, self.mmax_a_z0, self.mmax_a_z1)
        powerlaw_class_b = PowerLawLinear(z, self.alpha_b_z0, self.alpha_b_z1, self.mmin_b_z0, self.mmin_b_z1, self.mmax_b_z0, self.mmax_b_z1)
        powerlaw_class_c = PowerLawLinear(z, self.alpha_c_z0, self.alpha_c_z1, self.mmin_c_z0, self.mmin_c_z1, self.mmax_c_z0, self.mmax_c_z1)
        powerlaw_class_d = PowerLawLinear(z, self.alpha_d_z0, self.alpha_d_z1, self.mmin_d_z0, self.mmin_d_z1, self.mmax_d_z0, self.mmax_d_z1)
        # Add left smoothing to the evolving PowerLaw.
        # WARNING: The implementation is very slow, because the integral to normalise the windowed
        # distribution p(m1|z) needs to be computed at all redshifts corresponding to the PE samples and injections.
        if self.flag_powerlaw_smoothing:
            powerlaw_class_a = LowpassSmoothedProbEvolving(powerlaw_class_a, self.delta_m_a)
            powerlaw_class_b = LowpassSmoothedProbEvolving(powerlaw_class_b, self.delta_m_b)
            powerlaw_class_c = LowpassSmoothedProbEvolving(powerlaw_class_c, self.delta_m_c)
            powerlaw_class_d = LowpassSmoothedProbEvolving(powerlaw_class_d, self.delta_m_d)
        powerlaw_part_a  = powerlaw_class_a.pdf(m)
        powerlaw_part_b  = powerlaw_class_b.pdf(m)
        powerlaw_part_c  = powerlaw_class_c.pdf(m)
        powerlaw_part_d  = powerlaw_class_d.pdf(m)

        # Impose the rate to be between [0,1].
        if (xp.any(wz_alpha > 1)) or (xp.any(wz_alpha < 0)) or (xp.any(wz_beta > 1)) or (xp.any(wz_beta < 0)) or (xp.any(wz_gamma > 1)) or (xp.any(wz_gamma < 0)) or (xp.any(wz_alpha + wz_beta + wz_gamma > 1)):
            return xp.nan
        else:
            return wz_alpha * powerlaw_part_a + wz_beta * powerlaw_part_b + wz_gamma * powerlaw_part_c + (1 - wz_beta - wz_alpha - wz_gamma) * powerlaw_part_d
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


class PowerLawRedshiftLinear_PowerLawRedshiftLinear_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for both two redshift linearly-dependent PowerLaws and a Gaussian peak.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the PowerLaws (a and b) and the Gaussian.
            - flag_powerlaw_smoothing applies a left window function to the PowerLaws.
            The smoothing slows heavily down the model evaluation.
            - flag_redshift_mixture allows for the transition functions to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_powerlaw_smoothing = 0, flag_redshift_mixture = 1):
        
        self.population_parameters   = ['alpha_a_z0', 'alpha_a_z1', 'mmin_a_z0', 'mmin_a_z1', 'mmax_a_z0', 'mmax_a_z1', 'alpha_b_z0', 'alpha_b_z1', 'mmin_b_z0', 'mmin_b_z1', 'mmax_b_z0', 'mmax_b_z1', 'mu_z0', 'mu_z1', 'sigma_z0', 'sigma_z1', 'mix_alpha_z0', 'mix_beta_z0']
        self.redshift_transition     = redshift_transition
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.flag_redshift_mixture   = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_alpha_z1', 'mix_beta_z1']
        if self.flag_powerlaw_smoothing:
            self.population_parameters += ['delta_m_a', 'delta_m_b']

    def update(self,**kwargs):

        self.alpha_a_z0   = kwargs['alpha_a_z0']
        self.alpha_a_z1   = kwargs['alpha_a_z1']
        self.mmin_a_z0    = kwargs['mmin_a_z0']
        self.mmin_a_z1    = kwargs['mmin_a_z1']
        self.mmax_a_z0    = kwargs['mmax_a_z0']
        self.mmax_a_z1    = kwargs['mmax_a_z1']
        self.alpha_b_z0   = kwargs['alpha_b_z0']
        self.alpha_b_z1   = kwargs['alpha_b_z1']
        self.mmin_b_z0    = kwargs['mmin_b_z0']
        self.mmin_b_z1    = kwargs['mmin_b_z1']
        self.mmax_b_z0    = kwargs['mmax_b_z0']
        self.mmax_b_z1    = kwargs['mmax_b_z1']
        self.mu_z0        = kwargs['mu_z0']
        self.mu_z1        = kwargs['mu_z1']
        self.sigma_z0     = kwargs['sigma_z0']
        self.sigma_z1     = kwargs['sigma_z1']
        self.mix_alpha_z0 = kwargs['mix_alpha_z0']
        self.mix_beta_z0  = kwargs['mix_beta_z0']

        if self.flag_redshift_mixture:
            self.mix_alpha_z1 = kwargs['mix_alpha_z1']
            self.mix_beta_z1  = kwargs['mix_beta_z1']
        if self.flag_powerlaw_smoothing:
            self.delta_m_a    = kwargs['delta_m_a']
            self.delta_m_b    = kwargs['delta_m_b']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz_alpha = _mixed_linear_function(z, self.mix_alpha_z0, self.mix_alpha_z1)
                wz_beta  = _mixed_linear_function(z, self.mix_beta_z0,  self.mix_beta_z1 )
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz_alpha = xp.array(self.mix_alpha_z0)
            wz_beta  = xp.array(self.mix_beta_z0)

        powerlaw_class_a = PowerLawLinear(z, self.alpha_a_z0, self.alpha_a_z1, self.mmin_a_z0, self.mmin_a_z1, self.mmax_a_z0, self.mmax_a_z1)
        powerlaw_class_b = PowerLawLinear(z, self.alpha_b_z0, self.alpha_b_z1, self.mmin_b_z0, self.mmin_b_z1, self.mmax_b_z0, self.mmax_b_z1)
        gaussian_class   = GaussianLinear(z, self.mu_z0, self.mu_z1, self.sigma_z0, self.sigma_z1, self.mmin_a_z0)
        # Add left smoothing to the evolving PowerLaw.
        # WARNING: The implementation is very slow, because the integral to normalise the windowed
        # distribution p(m1|z) needs to be computed at all redshifts corresponding to the PE samples and injections.
        if self.flag_powerlaw_smoothing:
            powerlaw_class_a = LowpassSmoothedProbEvolving(powerlaw_class_a, self.delta_m_a)
            powerlaw_class_b = LowpassSmoothedProbEvolving(powerlaw_class_b, self.delta_m_b)
        powerlaw_part_a  = powerlaw_class_a.pdf(m)
        powerlaw_part_b  = powerlaw_class_b.pdf(m)
        gaussian_part    = gaussian_class.pdf(m)
    
        # Impose the rate to be between [0,1].
        if (xp.any(wz_alpha > 1)) or (xp.any(wz_alpha < 0)) or (xp.any(wz_beta > 1)) or (xp.any(wz_beta < 0)) or (xp.any(wz_alpha + wz_beta > 1)):
            return xp.nan
        else:
            return wz_alpha * powerlaw_part_a + wz_beta * powerlaw_part_b + (1 - wz_beta - wz_alpha) * gaussian_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


class GaussianRedshiftLinear_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for two redshift linearly-dependent Gaussian peaks.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the PowerLaw and the Gaussian.
            - flag_redshift_mixture allows for the transition function to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_redshift_mixture = 1):
        
        self.population_parameters = ['mu_a_z0', 'mu_a_z1', 'sigma_a_z0', 'sigma_a_z1', 'mu_b_z0', 'mu_b_z1', 'sigma_b_z0', 'sigma_b_z1', 'mix_z0', 'mmin_g']
        self.redshift_transition   = redshift_transition
        self.flag_redshift_mixture = flag_redshift_mixture
        
        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_z1']
            if self.redshift_transition == 'sigmoid': self.population_parameters += ['zt', 'delta_zt']

    def update(self,**kwargs):

        self.mu_a_z0    = kwargs['mu_a_z0']
        self.mu_a_z1    = kwargs['mu_a_z1']
        self.sigma_a_z0 = kwargs['sigma_a_z0']
        self.sigma_a_z1 = kwargs['sigma_a_z1']
        self.mu_b_z0    = kwargs['mu_b_z0']
        self.mu_b_z1    = kwargs['mu_b_z1']
        self.sigma_b_z0 = kwargs['sigma_b_z0']
        self.sigma_b_z1 = kwargs['sigma_b_z1']
        self.mix_z0     = kwargs['mix_z0']
        self.mmin_g     = kwargs['mmin_g']

        if self.flag_redshift_mixture:
            self.mix_z1 = kwargs['mix_z1']
            if self.redshift_transition == 'sigmoid': self.zt, self.delta_zt = kwargs['zt'], kwargs['delta_zt']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz = _mixed_linear_function(         z, self.mix_z0, self.mix_z1)
            elif self.redshift_transition == 'sigmoid':
                wz = _mixed_double_sigmoid_function( z, self.mix_z0, self.mix_z1, self.zt, self.delta_zt)
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz = xp.array(self.mix_z0)

        gaussian_a_class = GaussianLinear(z, self.mu_a_z0, self.mu_a_z1, self.sigma_a_z0, self.sigma_a_z1, self.mmin_g)
        gaussian_b_class = GaussianLinear(z, self.mu_b_z0, self.mu_b_z1, self.sigma_b_z0, self.sigma_b_z1, self.mmin_g)
        gaussian_a_part  = gaussian_a_class.pdf(m)
        gaussian_b_part  = gaussian_b_class.pdf(m)

        # Impose the rate to be between [0,1].
        if (xp.any(wz > 1)) or (xp.any(wz < 0)):
            return xp.nan
        else:
            return wz * gaussian_a_part + (1-wz) * gaussian_b_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


class GaussianRedshiftLinear_GaussianRedshiftLinear_GaussianRedshiftLinear():
    '''
        Class implementing the mass function model conditioned on redshift p(m1|z),
        for three redshift linearly-dependent Gaussian peaks.

        Some options are available:
            - redshift_transition sets the function for the redshift transition
            between the three Gaussian peaks. The transition is the same between the
            first two Gaussians a and b, and the other two Gaussians b and c.
            - flag_redshift_mixture allows for the transition functions to evolve with redshift.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, redshift_transition = 'linear', flag_redshift_mixture = 1):

        self.population_parameters = ['mu_a_z0', 'mu_a_z1', 'sigma_a_z0', 'sigma_a_z1', 'mu_b_z0', 'mu_b_z1', 'sigma_b_z0', 'sigma_b_z1', 'mu_c_z0', 'mu_c_z1', 'sigma_c_z0', 'sigma_c_z1', 'mix_alpha_z0', 'mix_beta_z0', 'mmin_g']
        self.redshift_transition   = redshift_transition
        self.flag_redshift_mixture = flag_redshift_mixture

        if self.flag_redshift_mixture:
            self.population_parameters += ['mix_alpha_z1', 'mix_beta_z1']
            if self.redshift_transition == 'sigmoid': self.population_parameters += ['zt', 'delta_zt']

    def update(self,**kwargs):

        self.mu_a_z0      = kwargs['mu_a_z0']
        self.mu_a_z1      = kwargs['mu_a_z1']
        self.sigma_a_z0   = kwargs['sigma_a_z0']
        self.sigma_a_z1   = kwargs['sigma_a_z1']
        self.mu_b_z0      = kwargs['mu_b_z0']
        self.mu_b_z1      = kwargs['mu_b_z1']
        self.sigma_b_z0   = kwargs['sigma_b_z0']
        self.sigma_b_z1   = kwargs['sigma_b_z1']
        self.mu_c_z0      = kwargs['mu_c_z0']
        self.mu_c_z1      = kwargs['mu_c_z1']
        self.sigma_c_z0   = kwargs['sigma_c_z0']
        self.sigma_c_z1   = kwargs['sigma_c_z1']
        self.mix_alpha_z0 = kwargs['mix_alpha_z0']
        self.mix_beta_z0  = kwargs['mix_beta_z0']
        self.mmin_g       = kwargs['mmin_g']

        if self.flag_redshift_mixture:
            self.mix_alpha_z1 = kwargs['mix_alpha_z1']
            self.mix_beta_z1  = kwargs['mix_beta_z1']
            if self.redshift_transition == 'sigmoid': self.zt, self.delta_zt = kwargs['zt'], kwargs['delta_zt']

    def pdf(self,m,z):

        xp = get_module_array(m)
        if self.flag_redshift_mixture:
            if   self.redshift_transition == 'linear':
                wz_alpha = _mixed_linear_function(         z, self.mix_alpha_z0, self.mix_alpha_z1)
                wz_beta  = _mixed_linear_function(         z, self.mix_beta_z0 , self.mix_beta_z1 )
            elif self.redshift_transition == 'sigmoid':
                wz_alpha = _mixed_double_sigmoid_function( z, self.mix_alpha_z0, self.mix_alpha_z1, self.zt, self.delta_zt)
                wz_beta  = _mixed_double_sigmoid_function( z, self.mix_beta_z0 , self.mix_beta_z1 , self.zt, self.delta_zt)
            else:
                raise ValueError('The slected redshift transition model {} does not exist. Exiting.'.format(self.redshift_transition))
        else:
            wz_alpha = xp.array(self.mix_alpha_z0)
            wz_beta  = xp.array(self.mix_beta_z0)

        gaussian_a_class = GaussianLinear(z, self.mu_a_z0, self.mu_a_z1, self.sigma_a_z0, self.sigma_a_z1, self.mmin_g)
        gaussian_b_class = GaussianLinear(z, self.mu_b_z0, self.mu_b_z1, self.sigma_b_z0, self.sigma_b_z1, self.mmin_g)
        gaussian_c_class = GaussianLinear(z, self.mu_c_z0, self.mu_c_z1, self.sigma_c_z0, self.sigma_c_z1, self.mmin_g)
        gaussian_a_part  = gaussian_a_class.pdf(m)
        gaussian_b_part  = gaussian_b_class.pdf(m)
        gaussian_c_part  = gaussian_c_class.pdf(m)

        # Impose the rate to be between [0,1].
        if   (xp.any(wz_alpha > 1)) or (xp.any(wz_alpha < 0)) or (xp.any(wz_beta > 1)) or (xp.any(wz_beta < 0)) or (xp.any(wz_alpha + wz_beta > 1)):
            return xp.nan
        else:
            return wz_alpha * gaussian_a_part + wz_beta * gaussian_b_part + (1 - wz_beta - wz_alpha) * gaussian_c_part

    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


class GaussianEvolving():
    '''
        Class implementing the mass function model conditioned on redshift p(m|z), for
        one redshift evolving Gaussian peak with arbitrary polynomial redshift expansion.

        Some options are available:
            - order sets the order of the redshift expansion. The population
            parameters [mu_zx, sigma_zx] are automatically defined from the
            selected order.
            Ex. order = 2 gives mu_z = mu_z0 + mu_z1*z + mu_z2*z^2

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, order = 1):

        self.order = order
        mu_list    = ['mu_z{}'.format(   i) for i in range(order + 1)]
        sigma_list = ['sigma_z{}'.format(i) for i in range(order + 1)]
        self.population_parameters = mu_list + sigma_list

    def update(self, **kwargs):

        for par in kwargs.keys():
            globals()['self.%s' % par] = kwargs[par]

    def polynomial(self, expansion_order, x, variable):
        
        pol = 0
        for i in range(expansion_order + 1):
            par = '{}_z{}'.format(variable, i)
            pol += globals()['self.%s' % par] * x ** i
        return pol

    def log_pdf(self, m, z):

        xp = get_module_array(m)
        sx = get_module_array_scipy(m)
        self.muz    = self.polynomial(self.order, z, 'mu')
        self.sigmaz = self.polynomial(self.order, z, 'sigma')
        a, b = (0. - self.muz) / self.sigmaz, (self.muz + 6*self.muz - self.muz) / self.sigmaz  # Truncte the Gaussian at zero and mu+6*sigma. This improve the numerical stability.
        gaussian = xp.log( sx.stats.truncnorm.pdf(m, a, b, loc = self.muz, scale = self.sigmaz) )
        return gaussian

    def pdf(self, m, z):
        xp = get_module_array(m)
        return xp.exp(self.log_pdf(m, z))
    
    def return_mu_sigma(self):
        return self.muz, self.sigmaz


###########
# Splines #
###########

def _safe_divide(xp, numer, denom, dtype = None):
    """
    Safe elementwise divide: out = numer / denom but skip positions where denom == 0.

    - xp: numpy or cupy module
    - numer, denom: scalars or arrays broadcastable to a common shape
    - dtype: optional output dtype (prefer self.dtype)
    """
    denom_b = xp.asarray(denom)
    numer_b = xp.asarray(numer)
    try:
        numer_b, denom_b = xp.broadcast_arrays(numer_b, denom_b)
    except Exception:
        numer_b = xp.full_like(denom_b, numer_b, dtype=denom_b.dtype)

    out_dtype = dtype if dtype is not None else denom_b.dtype
    out = xp.zeros_like(denom_b, dtype=out_dtype)

    # fast path: ufunc with where if supported
    try:
        xp.divide(numer_b, denom_b, out=out, where=(denom_b != 0))
        return out
    except TypeError:
        # fallback: masked assignment (works on older CuPy)
        mask = denom_b != 0
        out[mask] = numer_b[mask] / denom_b[mask]
        return out

def _safe_log(xp, p):
    out = xp.full_like(p, -xp.inf)
    try:
        xp.log(p, out=out, where=(p > 0))
    except TypeError:
        # older CuPy may not accept `where`; do masked assignment instead
        mask = p > 0
        if mask.size:  # no heavy sync; .size is metadata
            out[mask] = xp.log(p[mask])
    return out


class QuadraticSpline:
    """
    Fully analytic quadratic B-spline model (degree 2) of pdf using closed-form expressions from the
    Cox-De Boor algorithm. Each basis element in each span is converted into a Bernstein polynomial,
    allowing for fast evaluation and analytic integral for normalisation.

    The implementation is vectorised and GPU-compatible.

    Monomial coefficient ordering: [a0, a1, a2] with
      P(x) = a0 + a1 * x + a2 * x^2

    Behaviour:
    - Interior coefficients are normalised to one. They follow a Dirichlet distribution if
      independently drawn from a Gamma prior (i.e. uniform weights on the simplex).
    - First and last coefficients remain fixed to zero.
    """

    def __init__(self, n_basis: int = 6, spacing: str = "uniform"):
        # Compatibility: user supplies interior_count and we add +2
        self.n_basis = int(n_basis) + 2
        self.degree = 2
        self.spacing = spacing

        if spacing not in ("uniform", "log"):
            raise ValueError("spacing must be 'uniform' or 'log'")

        self.population_parameters = (['mmin', 'mmax'] + [f'c{i}' for i in range(1, self.n_basis - 1)])

        # Placeholders
        self.xmin = None
        self.xmax = None
        self.weights = None # Shape (n_basis,)
        self.knots = None # Shape (n_knots,)
        self.Q = None # Shape (n_spans, 3)
        self.norm = None

        # Backend (numpy or cupy) and dtype (float32 on GPU by default)
        self.xp, self.dtype = _set_xp_and_dtype()

    def _make_knots(self):
        k = self.degree
        n = self.n_basis
        xp = self.xp

        if self.spacing == "uniform":
            interior = xp.linspace(self.xmin, self.xmax, n - k + 1, dtype=self.dtype)
        else:
            if self.xmin <= 0:
                raise ValueError("log spacing requires mmin > 0")
            interior = xp.exp(xp.linspace(xp.log(self.xmin), xp.log(self.xmax), n - k + 1, dtype=self.dtype))

        return xp.concatenate([xp.full(k + 1, interior[0], dtype=self.dtype), interior[1:-1], xp.full(k + 1, interior[-1], dtype=self.dtype)])

    def _mono_to_bern_quadratic_vec(self, mono, t0, t1):
        """
        Vectorized conversion mono -> Bernstein for quadratics.
        mono: (...,3) representing [a0, a1, a2] -> P(x) = a0 + a1 x + a2 x^2
        t0, t1: arrays broadcastable to mono[...,0]
        returns: (...,3) Bernstein control points [C0,C1,C2]
        """
        xp = self.xp
        # Convert to polynomial coefficient order used in earlier formulas:
        # a (x^2) = mono[...,2], b (x) = mono[...,1], c (const) = mono[...,0]
        a = mono[..., 2]
        b = mono[..., 1]
        c = mono[..., 0]
        h = t1 - t0

        Q0 = a * (t0 * t0) + b * t0 + c
        Q2 = a * (t1 * t1) + b * t1 + c
        Q1 = Q0 + 0.5 * h * (2.0 * a * t0 + b)

        return xp.stack([Q0, Q1, Q2], axis=-1)

    def _build_bernstein_representation(self):
        """
        Fully vectorized quadratic builder:
        - compute contributions for local offsets 0,1,2 in bulk
        - scatter-accumulate into Q (n_spans, 3)
        """
        xp = self.xp
        t = self.knots
        w = self.weights
        n = self.n_basis
        n_spans = int(t.size - 1)

        Q = xp.zeros((n_spans, 3), dtype=self.dtype)
        idx = xp.arange(n, dtype=int)

        # Gather per-basis knot arrays
        t0 = t.take(idx)
        t1 = t.take(idx + 1)
        t2 = t.take(idx + 2)
        t3 = t.take(idx + 3)

        # --- span i (local offset 0) ---
        d0 = (t2 - t0) * (t1 - t0)

        # compute coefficients safely (no divide-by-zero warnings)
        a0 = _safe_divide(xp, 1.0, d0, dtype=self.dtype)
        b0 = _safe_divide(xp, -2.0 * t0, d0, dtype=self.dtype)
        c0 = _safe_divide(xp, t0 * t0, d0, dtype=self.dtype)

        # assemble monos and bernstein controls for all bases (n is small)
        mono0 = xp.stack([c0, b0, a0], axis=-1)              # shape (n,3)
        ctrl0 = self._mono_to_bern_quadratic_vec(mono0, t0, t1)  # shape (n,3)

        # find active basis indices on device and scatter-add into Q
        sel0 = xp.nonzero(d0 > 0)[0]
        if sel0.size:
            sel_i = idx[sel0]   # basis indices
            sel_s = sel_i       # global span indices (i + 0)
            xp.add.at(Q, sel_s, ctrl0[sel0] * w[sel0, None])

        # --- span i+1 (local offset 1) ---
        d1 = (t2 - t0) * (t2 - t1)
        d2 = (t3 - t1) * (t2 - t1)

        # compute coefficients safely (no divide-by-zero warnings)
        term1 = _safe_divide(xp, 1.0, d1, dtype=self.dtype)
        term2 = _safe_divide(xp, 1.0, d2, dtype=self.dtype)

        a1 = - (term1 + term2)
        b1 = _safe_divide(xp, (t0 + t2), d1, dtype=self.dtype) + _safe_divide(xp, (t1 + t3), d2, dtype=self.dtype)
        c1 = - (_safe_divide(xp, t0 * t2, d1, dtype=self.dtype) + _safe_divide(xp, t1 * t3, d2, dtype=self.dtype))

        # assemble monos and bernstein controls for all bases
        mono1 = xp.stack([c1, b1, a1], axis=-1)          # shape (n,3)
        ctrl1 = self._mono_to_bern_quadratic_vec(mono1, t1, t2)  # shape (n,3)

        # find active indices and scatter-add into Q at span = i + 1
        sel1 = xp.nonzero((d1 > 0) & (d2 > 0))[0]
        if sel1.size:
            sel_i = idx[sel1]        # basis indices
            sel_s = sel_i + 1        # global span indices (i + 1)
            xp.add.at(Q, sel_s, ctrl1[sel1] * w[sel1, None])

        # --- span i+2 (local offset 2) ---
        d3 = (t3 - t1) * (t3 - t2)

        # compute coefficients safely (no divide-by-zero warnings)
        a2 = _safe_divide(xp, 1.0, d3, dtype=self.dtype)
        b2 = _safe_divide(xp, -2.0 * t3, d3, dtype=self.dtype)
        c2 = _safe_divide(xp, t3 * t3, d3, dtype=self.dtype)

        # assemble monos and bernstein controls for all bases
        mono2 = xp.stack([c2, b2, a2], axis=-1)               # shape (n,3)
        ctrl2 = self._mono_to_bern_quadratic_vec(mono2, t2, t3)  # shape (n,3)

        # find active indices and scatter-add into Q at span = i + 2
        sel2 = xp.nonzero(d3 > 0)[0]
        if sel2.size:
            sel_i = idx[sel2]         # basis indices
            sel_s = sel_i + 2         # global span indices (i + 2)
            xp.add.at(Q, sel_s, ctrl2[sel2] * w[sel2, None])

        self.Q = Q
        h = xp.diff(t)
        self.norm = xp.sum(xp.where(h > 0, h * xp.sum(Q, axis=1) / 3.0, 0.0))

        if float(self.norm) <= 0.0:
            raise ValueError("Spline has zero integral")

    # ------------------
    # Update / evaluate
    # ------------------
    def update(self, **kwargs):
        self.xmin = kwargs['mmin']
        self.xmax = kwargs['mmax']
        xp = self.xp

        # Assemble weights (interior coefficients only)
        w = xp.zeros(self.n_basis, dtype=self.dtype)
        for i in range(1, self.n_basis - 1):
            w[i] = kwargs[f'c{i}']

        s = float(xp.sum(w))
        if s <= 0:
            raise ValueError("sum of weights must be > 0")
        
        self.weights = (w / s).astype(self.dtype)
        self.knots = self._make_knots()
        self._build_bernstein_representation()

    def _bernstein_quadratic(self, u):
        xp = self.xp
        u = xp.asarray(u, dtype=self.dtype)
        return xp.stack([(1 - u) ** 2, 2 * u * (1 - u), u ** 2], axis=-1)

    def basis(self, i, x):
        """
        Evaluate basis i by rebuilding Q with only that basis active (cheap for n_basis small).
        """
        xp = self.xp
        if i < 0 or i >= self.n_basis:
            raise IndexError("Invalid basis index")
        x = xp.asarray(x, dtype=self.dtype)

        # Temporarily set weights to select basis i
        w_saved = self.weights
        Q_saved = self.Q

        try:
            w_sel = xp.zeros_like(self.weights)
            w_sel[i] = 1.0
            self.weights = w_sel
            self._build_bernstein_representation()
            Q_i = self.Q.copy()
        finally:
            # Restore weights and Q without rebuilding full Q
            self.weights = w_saved
            self.Q = Q_saved

        spans = xp.searchsorted(self.knots, x, side="right") - 1
        spans = xp.clip(spans, 0, len(Q_i) - 1)
        t0s = self.knots[spans]
        t1s = self.knots[spans + 1]
        h = t1s - t0s
        u = xp.zeros_like(x, dtype=self.dtype)
        mask = h > 0
        u[mask] = (x[mask] - t0s[mask]) / h[mask]

        B = self._bernstein_quadratic(u)
        return xp.sum(B * Q_i[spans], axis=-1)

    def evaluate(self, x):
        xp = self.xp
        x = xp.asarray(x, dtype=self.dtype)

        spans = xp.searchsorted(self.knots, x, side="right") - 1
        spans = xp.clip(spans, 0, len(self.Q) - 1)
        t0 = self.knots[spans]
        t1 = self.knots[spans + 1]
        h = t1 - t0
        u = xp.zeros_like(x, dtype=self.dtype)
        mask = h > 0
        u[mask] = (x[mask] - t0[mask]) / h[mask]

        B = self._bernstein_quadratic(u)
        return xp.sum(B * self.Q[spans], axis=-1)

    def pdf(self, x):
        return self.evaluate(x) / self.norm

    def log_pdf(self, x):
        xp = self.xp
        p = self.pdf(x)
        return _safe_log(xp, p)


class CubicSpline:
    """
    Fully analytic cubic B-spline model (degree 3) of pdf using closed-form expressions from the
    Cox-De Boor algorithm. Each basis element in each span is converted into a Bernstein polynomial,
    allowing for fast evaluation and analytic integral for normalisation.

    The implementation is vectorised and GPU-compatible.

    Monomial coefficient ordering: [a0, a1, a2, a3] with
      P(x) = a0 + a1 * x + a2 * x^2 + a3 * x^3

    Behaviour:
    - Interior coefficients are normalised to one. They follow a Dirichlet distribution if
      independently drawn from a Gamma prior (i.e. uniform weights on the simplex).
    - First and last coefficients remain fixed to zero.
    """

    def __init__(self, n_basis: int = 6, spacing: str = "uniform"):
        self.n_basis = int(n_basis) + 2
        self.degree = 3
        self.spacing = spacing

        if spacing not in ("uniform", "log"):
            raise ValueError("spacing must be 'uniform' or 'log'")
        
        self.population_parameters = (['mmin', 'mmax'] + [f'c{i}' for i in range(1, self.n_basis - 1)])

        self.xmin = None
        self.xmax = None
        self.weights = None # Shape (n_basis,)
        self.knots = None # Shape (n_knots,)
        self.Q = None # Shape (n_spans, 4)
        self.norm = None

        # Backend (numpy or cupy) and dtype (float32 on GPU by default)
        self.xp, self.dtype = _set_xp_and_dtype()

    def _make_knots(self):
        k = self.degree
        n = self.n_basis
        xp = self.xp

        if self.spacing == "uniform":
            interior = xp.linspace(self.xmin, self.xmax, n - k + 1, dtype=self.dtype)
        else:
            if self.xmin <= 0:
                raise ValueError("log spacing requires mmin > 0")
            interior = xp.exp(xp.linspace(xp.log(self.xmin), xp.log(self.xmax), n - k + 1, dtype=self.dtype))

        return xp.concatenate([xp.full(k + 1, interior[0], dtype=self.dtype), interior[1:-1], xp.full(k + 1, interior[-1], dtype=self.dtype)])

    def _mono_to_bern_cubic_vec(self, mono, t0, t1):
        """
        Vectorized mono->Bernstein for cubics.
        mono: (m,4) with columns [a0,a1,a2,a3] (increasing order)
        t0, t1: arrays shape (m,)
        returns: (m,4) Bernstein control points [C0..C3]
        """
        xp = self.xp

        a0 = mono[:, 0]
        a1 = mono[:, 1]
        a2 = mono[:, 2]
        a3 = mono[:, 3]
        h = t1 - t0
        t0b = t0
        b0 = a0 + a1 * t0b + a2 * (t0b ** 2) + a3 * (t0b ** 3)
        b1 = h * (a1 + 2.0 * t0b * a2 + 3.0 * (t0b ** 2) * a3)
        b2 = (h ** 2) * (a2 + 3.0 * t0b * a3)
        b3 = (h ** 3) * a3

        C0 = b0
        C1 = b0 + (1.0 / 3.0) * b1
        C2 = b0 + (2.0 / 3.0) * b1 + (1.0 / 3.0) * b2
        C3 = b0 + b1 + b2 + b3

        return xp.stack([C0, C1, C2, C3], axis=-1)

    def _build_bernstein_representation(self):
        """
        Fully vectorized cubic builder using safe divides.
        """
        xp = self.xp
        t = self.knots
        w = self.weights
        n = self.n_basis
        n_spans = int(t.size - 1)

        Q = xp.zeros((n_spans, 4), dtype=self.dtype)
        idx = xp.arange(n, dtype=int)

        # Gather per-basis knots vectors (length n)
        k0 = t.take(idx)
        k1 = t.take(idx + 1)
        k2 = t.take(idx + 2)
        k3 = t.take(idx + 3)
        k4 = t.take(idx + 4)

        # Quadratic coefficients for N_{i,2} on spans i,i+1,i+2 (vectorized) using safe divides
        d0 = (k2 - k0) * (k1 - k0)
        aq0 = _safe_divide(xp, 1.0, d0, dtype=self.dtype)
        bq0 = _safe_divide(xp, -2.0 * k0, d0, dtype=self.dtype)
        cq0 = _safe_divide(xp, k0 * k0, d0, dtype=self.dtype)

        d1 = (k2 - k0) * (k2 - k1)
        d2 = (k3 - k1) * (k2 - k1)
        term1 = _safe_divide(xp, 1.0, d1, dtype=self.dtype)
        term2 = _safe_divide(xp, 1.0, d2, dtype=self.dtype)
        aq1 = - (term1 + term2)
        bq1 = _safe_divide(xp, (k0 + k2), d1, dtype=self.dtype) + _safe_divide(xp, (k1 + k3), d2, dtype=self.dtype)
        cq1 = - (_safe_divide(xp, k0 * k2, d1, dtype=self.dtype) + _safe_divide(xp, k1 * k3, d2, dtype=self.dtype))

        d3 = (k3 - k1) * (k3 - k2)
        aq2 = _safe_divide(xp, 1.0, d3, dtype=self.dtype)
        bq2 = _safe_divide(xp, -2.0 * k3, d3, dtype=self.dtype)
        cq2 = _safe_divide(xp, k3 * k3, d3, dtype=self.dtype)

        # Quadratic coefficients for N_{i+1,2} on spans i+1,i+2,i+3 (shifted) using safe divides
        d0p = (k3 - k1) * (k2 - k1)
        ap0 = _safe_divide(xp, 1.0, d0p, dtype=self.dtype)
        bp0 = _safe_divide(xp, -2.0 * k1, d0p, dtype=self.dtype)
        cp0 = _safe_divide(xp, k1 * k1, d0p, dtype=self.dtype)

        d1p = (k3 - k1) * (k3 - k2)
        d2p = (k4 - k2) * (k3 - k2)
        term1p = _safe_divide(xp, 1.0, d1p, dtype=self.dtype)
        term2p = _safe_divide(xp, 1.0, d2p, dtype=self.dtype)
        ap1 = - (term1p + term2p)
        bp1 = _safe_divide(xp, (k1 + k3), d1p, dtype=self.dtype) + _safe_divide(xp, (k2 + k4), d2p, dtype=self.dtype)
        cp1 = - (_safe_divide(xp, k1 * k3, d1p, dtype=self.dtype) + _safe_divide(xp, k2 * k4, d2p, dtype=self.dtype))

        d3p = (k4 - k2) * (k4 - k3)
        ap2 = _safe_divide(xp, 1.0, d3p, dtype=self.dtype)
        bp2 = _safe_divide(xp, -2.0 * k4, d3p, dtype=self.dtype)
        cp2 = _safe_divide(xp, k4 * k4, d3p, dtype=self.dtype)

        # Denominators in Cox-de Boor cubic recurrence
        L = k3 - k0
        R = k4 - k1

        # Compute per-basis cubic coefficients for each local span (i..i+3) using safe divides
        s0_A3 = _safe_divide(xp, aq0, L, dtype=self.dtype)
        s0_A2 = _safe_divide(xp, (bq0 - aq0 * k0), L, dtype=self.dtype)
        s0_A1 = _safe_divide(xp, (cq0 - bq0 * k0), L, dtype=self.dtype)
        s0_A0 = _safe_divide(xp, (-cq0 * k0), L, dtype=self.dtype)

        # Span i+1: left (aq1...) and right (ap0...)
        L3 = _safe_divide(xp, aq1, L, dtype=self.dtype)
        L2 = _safe_divide(xp, (bq1 - aq1 * k0), L, dtype=self.dtype)
        L1 = _safe_divide(xp, (cq1 - bq1 * k0), L, dtype=self.dtype)
        L0 = _safe_divide(xp, (-cq1 * k0), L, dtype=self.dtype)

        R3 = _safe_divide(xp, -ap0, R, dtype=self.dtype)
        R2 = _safe_divide(xp, (ap0 * k4 - bp0), R, dtype=self.dtype)
        R1 = _safe_divide(xp, (bp0 * k4 - cp0), R, dtype=self.dtype)
        R0 = _safe_divide(xp, (cp0 * k4), R, dtype=self.dtype)

        s1_A3 = L3 + R3
        s1_A2 = L2 + R2
        s1_A1 = L1 + R1
        s1_A0 = L0 + R0

        # Span i+2: left (aq2...) and right (ap1...)
        L3 = _safe_divide(xp, aq2, L, dtype=self.dtype)
        L2 = _safe_divide(xp, (bq2 - aq2 * k0), L, dtype=self.dtype)
        L1 = _safe_divide(xp, (cq2 - bq2 * k0), L, dtype=self.dtype)
        L0 = _safe_divide(xp, (-cq2 * k0), L, dtype=self.dtype)

        R3 = _safe_divide(xp, -ap1, R, dtype=self.dtype)
        R2 = _safe_divide(xp, (ap1 * k4 - bp1), R, dtype=self.dtype)
        R1 = _safe_divide(xp, (bp1 * k4 - cp1), R, dtype=self.dtype)
        R0 = _safe_divide(xp, (cp1 * k4), R, dtype=self.dtype)

        s2_A3 = L3 + R3
        s2_A2 = L2 + R2
        s2_A1 = L1 + R1
        s2_A0 = L0 + R0

        # Span i+3: only right (ap2...)
        s3_A3 = _safe_divide(xp, -ap2, R, dtype=self.dtype)
        s3_A2 = _safe_divide(xp, (ap2 * k4 - bp2), R, dtype=self.dtype)
        s3_A1 = _safe_divide(xp, (bp2 * k4 - cp2), R, dtype=self.dtype)
        s3_A0 = _safe_divide(xp, (cp2 * k4), R, dtype=self.dtype)

        # For each local offset j = 0..3, gather mono coeffs for all bases and scatter to Q at s = i + j.
        for local_j, (A0_arr, A1_arr, A2_arr, A3_arr) in enumerate([
            (s0_A0, s0_A1, s0_A2, s0_A3),
            (s1_A0, s1_A1, s1_A2, s1_A3),
            (s2_A0, s2_A1, s2_A2, s2_A3),
            (s3_A0, s3_A1, s3_A2, s3_A3),
        ]):
            s_indices = idx + local_j
            # Mask valid spans
            valid_mask = (s_indices >= 0) & (s_indices < n_spans)

            if not xp.any(valid_mask):
                continue

            # Select masked entries
            sel = valid_mask
            sel_i = idx[sel] # Basis indices selected
            sel_s = s_indices[sel] # Corresponding span indices

            # Prepare monomial array for these selected basis-span pairs, shape (m,4)
            mono_sel = xp.stack([A0_arr[sel], A1_arr[sel], A2_arr[sel], A3_arr[sel]], axis=-1)
            # Get span endpoints for these selected spans
            t0_sel = t.take(sel_s)
            t1_sel = t.take(sel_s + 1)
            # Compute Bernstein control points for all selected entries
            ctrl = self._mono_to_bern_cubic_vec(mono_sel, t0_sel, t1_sel) # (m,4)

            # Accumulate weighted contributions into Q at rows sel_s
            Q[sel_s] += (w[sel_i, None] * ctrl)

        self.Q = Q
        h = xp.diff(t)
        self.norm = xp.sum(xp.where(h > 0, h * xp.sum(Q, axis=1) / 4.0, 0.0))

        if float(self.norm) <= 0.0:
            raise ValueError("Spline has zero integral")

    # ------------------
    # Update / evaluate
    # ------------------
    def update(self, **kwargs):
        self.xmin = kwargs['mmin']
        self.xmax = kwargs['mmax']
        xp = self.xp

        # Assemble weights (interior coefficients only)
        w = xp.zeros(self.n_basis, dtype=self.dtype)
        for i in range(1, self.n_basis - 1):
            w[i] = kwargs[f'c{i}']

        s = float(xp.sum(w))
        if s <= 0:
            raise ValueError("sum of weights must be > 0")
        
        self.weights = (w / s).astype(self.dtype)
        self.knots = self._make_knots()
        self._build_bernstein_representation()

    def _bernstein_cubic(self, u):
        xp = self.xp
        u = xp.asarray(u, dtype=self.dtype)
        one_minus = (1 - u)
        return xp.stack([one_minus ** 3, 3 * u * (one_minus ** 2), 3 * (u ** 2) * one_minus, u ** 3], axis=-1)

    def basis(self, i, x):
        xp = self.xp
        if i < 0 or i >= self.n_basis:
            raise IndexError("Invalid basis index")
        x = xp.asarray(x, dtype=self.dtype)

        # Temporarily set weights to select basis i
        w_saved = self.weights
        Q_saved = self.Q

        try:
            w_sel = xp.zeros_like(self.weights)
            w_sel[i] = 1.0
            self.weights = w_sel
            self._build_bernstein_representation()
            Q_i = self.Q.copy()
        finally:
            # Restore weights and Q without rebuilding full Q
            self.weights = w_saved
            self.Q = Q_saved

        spans = xp.searchsorted(self.knots, x, side="right") - 1
        spans = xp.clip(spans, 0, len(Q_i) - 1)
        t0s = self.knots[spans]
        t1s = self.knots[spans + 1]
        h = t1s - t0s
        u = xp.zeros_like(x, dtype=self.dtype)
        mask = h > 0
        u[mask] = (x[mask] - t0s[mask]) / h[mask]

        B = self._bernstein_cubic(u)
        return xp.sum(B * Q_i[spans], axis=-1)

    def evaluate(self, x):
        xp = self.xp
        x = xp.asarray(x, dtype=self.dtype)

        spans = xp.searchsorted(self.knots, x, side="right") - 1
        spans = xp.clip(spans, 0, len(self.Q) - 1)
        t0 = self.knots[spans]
        t1 = self.knots[spans + 1]
        h = t1 - t0
        u = xp.zeros_like(x, dtype=self.dtype)
        mask = h > 0
        u[mask] = (x[mask] - t0[mask]) / h[mask]
        
        B = self._bernstein_cubic(u)
        return xp.sum(B * self.Q[spans], axis=-1)

    def pdf(self, x):
        return self.evaluate(x) / self.norm

    def log_pdf(self, x):
        xp = self.xp
        p = self.pdf(x)
        return _safe_log(xp, p)


class LogSplineCoxDeBoor:
    """
    B-spline model of log(pdf) for arbitrary degree using the Cox-De Boor algorithm.
    The spline is defined by n_basis basis functions of given degree over [mmin, mmax],
    with either uniform or logarithmic knot spacing. Each basis has an associated coefficient
    that defines the log(pdf) as a linear combination of the basis functions.

    The normalization is computed numerically by evaluating the spline on a grid.
    The implementation is vectorised and GPU-compatible.

    Behaviour:
    - Interior coefficients are used exactly as provided (no mean subtraction).
    - First and last coefficients remain fixed to zero. This fixes the gauge invariance from adding
      a global constant to all coefficients.
    - Coefficients can be negative as the log(pdf), but still ensuring that the pdf is always positive.
    """

    def __init__(self, n_basis: int = 6, degree: int = 2, spacing: str = "uniform"):
        """
        n_basis: number of interior (free) coefficients. The full basis count = n_basis + 2.
        degree: polynomial degree of the spline (typically 1..4).
        spacing: "uniform" or "log" for knot spacing on [mmin, mmax]
        """
        self.n_basis_user = int(n_basis)
        if self.n_basis_user < 1:
            raise ValueError("n_basis must be >= 1 (at least one interior basis).")
        self.n_basis = self.n_basis_user + 2  # include first and last fixed bases
        self.degree = int(degree)
        if self.degree < 0:
            raise ValueError("degree must be non-negative")
        self.spacing = spacing
        if spacing not in ("uniform", "log"):
            raise ValueError("spacing must be 'uniform' or 'log'")

        # population parameters: mmin, mmax, c1..cN (interior coefficients)
        self.population_parameters = (['mmin', 'mmax'] +
                                      [f'c{i}' for i in range(1, self.n_basis_user + 1)])

        # placeholders
        self.xmin = None
        self.xmax = None
        self.knots = None  # knot vector
        self.weights = None  # full weights array length n_basis (first/last fixed)
        self.norm = None  # log-normalization constant (log integral)
        self._grid_size = 1000  # number of points for numerical normalization

        # xp and dtype
        self.xp, self.dtype = _set_xp_and_dtype()

    # -----------------
    # Knot construction
    # -----------------
    def _make_knots(self):
        xp = self.xp
        k = self.degree
        n = self.n_basis

        # number of interior knots (excluding clamped repeats)
        # For n basis functions and degree k, knot vector length = n + k + 1
        # The number of unique internal knots (excluding clamped endpoint repeats) is:
        n_int = n - k - 1
        if n_int < 0:
            raise ValueError("n_basis (including the two fixed ends) must be >= degree + 1")

        if self.spacing == "uniform":
            # we want n_int+2 endpoints in the linspace (including x_min & x_max),
            # then drop the endpoints to get interior unique positions
            interior = xp.linspace(self.xmin, self.xmax, n_int + 2, dtype=self.dtype)
        else:
            if float(self.xmin) <= 0:
                raise ValueError("log spacing requires mmin > 0")
            interior = xp.exp(
                xp.linspace(xp.log(self.xmin), xp.log(self.xmax), n_int + 2, dtype=self.dtype)
            )

        # remove endpoints (they will be added by clamping)
        if interior.size > 2:
            interior = interior[1:-1]
        else:
            interior = xp.asarray([], dtype=self.dtype)

        # clamped ends: repeat endpoints k+1 times
        t_start = xp.full(k + 1, self.xmin, dtype=self.dtype)
        t_end = xp.full(k + 1, self.xmax, dtype=self.dtype)

        knots = xp.concatenate([t_start, interior, t_end])
        # final length should be n + k + 1
        expected = n + k + 1
        if knots.size != expected:
            # If something went wrong, raise a helpful error
            raise RuntimeError(f"Unexpected knot vector length {knots.size}, expected {expected}")
        return knots

    # ------------------------
    # Cox-de Boor (vectorized)
    # ------------------------
    def basis(self, x):
        """
        Compute all B-spline basis functions at positions x.
        Returns array shape (..., n_basis) matching x's leading dimensions.
        """
        xp = self.xp
        dtype = self.dtype

        scalar_input = False
        x_arr = xp.asarray(x, dtype=dtype)
        orig_shape = x_arr.shape
        # flatten to 1D for evaluation
        if x_arr.ndim == 0:
            x_arr = x_arr[None]
            scalar_input = True
        else:
            x_arr = x_arr.ravel()
        N = x_arr.size

        t = self.knots
        if t is None:
            raise RuntimeError("knots have not been constructed; call update(...) first.")
        k = self.degree
        n_basis = self.n_basis

        # Degree 0 basis:
        B = xp.zeros((N, n_basis), dtype=dtype)
        t0 = t[:n_basis]
        t1 = t[1:n_basis + 1]

        # include left-closed intervals [t_i, t_{i+1}) except the last interval which is closed on right
        mask = (x_arr[:, None] >= t0[None, :]) & (x_arr[:, None] < t1[None, :])
        B[mask] = 1.0

        # left boundary exact equality (x == t[0]) -> first basis = 1
        left = (x_arr == t[0])
        if xp.any(left):
            B[left, 0] = 1.0

        # right boundary exact equality (x == t[-1]) -> last basis = 1
        right = (x_arr == t[-1])
        if xp.any(right):
            B[right, -1] = 1.0

        # Recursion for d = 1..k
        for d in range(1, k + 1):
            # slices for denominators (length n_basis)
            t_i = t[:n_basis]                   # t[i]
            t_id = t[d:n_basis + d]             # t[i + d]
            t_ip1 = t[1:n_basis + 1]            # t[i+1]
            t_ip1d1 = t[d + 1:n_basis + d + 1]  # t[i + d + 1]

            denom1 = t_id - t_i         # shape (n_basis,)
            denom2 = t_ip1d1 - t_ip1    # shape (n_basis,)

            # term1: (x - t_i) / (t_{i+d} - t_i) * B^{d-1}_i
            numer1 = (x_arr[:, None] - t_i[None, :])
            term1 = _safe_divide(xp, numer1, denom1[None, :], dtype) * B

            # term2: (t_{i+d+1} - x) / (t_{i+d+1} - t_{i+1}) * B^{d-1}_{i+1}
            term2 = xp.zeros_like(B)
            if n_basis > 1:
                numer2 = (t_ip1d1[None, :-1] - x_arr[:, None])  # aligns with i = 0..n_basis-2
                denom2_slice = denom2[None, :-1]
                # B[:, 1:] corresponds to B^{d-1}_{i+1}
                term2[:, :-1] = _safe_divide(xp, numer2, denom2_slice, dtype) * B[:, 1:]

            B = term1 + term2

        # reshape back to original shape + n_basis
        out_shape = orig_shape + (n_basis,)
        B = B.reshape(out_shape)
        if scalar_input:
            return B[0]  # return shape (n_basis,)
        return B  # shape (..., n_basis)

    # ----------------------
    # Update / Normalization
    # ----------------------
    def update(self, **kwargs):
        """
        Update spline parameters.

        Required kwargs:
        - mmin: left endpoint
        - mmax: right endpoint
        - c1..cN: interior coefficients (N == n_basis_user)

        Interior coefficients are used exactly as provided (first and last coefficients remain 0).
        After updating weights, the numeric normalization constant is computed.
        """
        xp = self.xp
        dtype = self.dtype

        if 'mmin' not in kwargs or 'mmax' not in kwargs:
            raise ValueError("update requires 'mmin' and 'mmax'")

        self.xmin = xp.asarray(kwargs['mmin'], dtype=dtype)
        self.xmax = xp.asarray(kwargs['mmax'], dtype=dtype)
        if float(self.xmax) <= float(self.xmin):
            raise ValueError("mmax must be > mmin")

        if self.spacing == "log" and float(self.xmin) <= 0:
            raise ValueError("log spacing requires mmin > 0")

        # Build knots
        self.knots = self._make_knots()

        # Assemble weights (first and last are fixed zero)
        w = xp.zeros(self.n_basis, dtype=dtype)
        # interior coefficients from kwargs: c1..cN
        interior = xp.zeros(self.n_basis_user, dtype=dtype)
        for i in range(1, self.n_basis_user + 1):
            interior[i - 1] = xp.asarray(kwargs.get(f'c{i}', 0.0), dtype=dtype)

        w[1:-1] = interior
        self.weights = w

        # Compute normalization constant
        self._compute_norm()

    def _compute_norm(self):
        xp = self.xp
        dtype = self.dtype

        # Build integration grid
        if self.spacing == "log":
            grid = xp.exp(
                xp.linspace(xp.log(self.xmin), xp.log(self.xmax), self._grid_size, dtype=dtype)
            )
        else:
            grid = xp.linspace(self.xmin, self.xmax, self._grid_size, dtype=dtype)

        B = self.basis(grid)  # shape (grid_size, n_basis)
        # dot each row of B with weights -> logp at grid
        logp = xp.dot(B, self.weights)

        # stabilize with log-sum-exp style
        m = float(xp.max(logp))
        shifted = xp.exp(logp - m)
        Z = float(xp.trapz(shifted, grid))
        if Z <= 0 or (not xp.isfinite(Z)):
            raise ValueError("Spline has zero or invalid integral during normalization.")

        self.norm = m + xp.log(Z)

    # ----------
    # Evaluation
    # ----------
    def log_pdf(self, x):
        """
        Evaluate log(pdf) at x (scalar or array). Values outside [xmin, xmax] return -inf.
        """
        xp = self.xp
        dtype = self.dtype

        x_arr = xp.asarray(x, dtype=dtype)
        scalar_input = False
        if x_arr.ndim == 0:
            x_arr = x_arr[None]
            scalar_input = True

        out = xp.full(x_arr.shape, -xp.inf, dtype=dtype)
        # mask in domain (closed interval)
        mask = (x_arr >= self.xmin) & (x_arr <= self.xmax)
        if not xp.any(mask):
            return out[0] if scalar_input else out

        B = self.basis(x_arr[mask])  # shape (M, n_basis)
        vals = xp.dot(B, self.weights) - self.norm
        out[mask] = vals

        return out[0] if scalar_input else out
    
    def pdf(self, x):
        xp = self.xp
        lp = self.log_pdf(x)
        return xp.exp(lp)

    # ---------
    # Utilities
    # ---------
    def get_weights(self):
        if self.weights is None:
            return None
        try:
            return self.weights.get()
        except Exception:
            return self.weights

    def get_knots(self):
        if self.knots is None:
            return None
        try:
            return self.knots.get()
        except Exception:
            return self.knots


class massprior_3PL_globmax(pm_prob):
    """
    3 Power-Laws model with a global mmax parameter shared by all PL components.
    Optional smoothing of low end of PL components.
    
    Dummy 'mmin' and 'delta_m' parameters to make it compatible with
    conditional wrappers without entangling p(m1) and p(m2).
    It is also usable outside conditional wrappers 
    e.g with p(q) parametrisation or pairing functions, 
    but 'mmin' and 'delta_m' need to be fixed.

    Note that PL components are parametrised as follows:
    $PL(m) \propto m^{- \alpha}$. 
    Consequently, the model is mostly relevant for positive alpha parameters.
    """

    def __init__(self, flag_powerlaw_smoothing=False):
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.population_parameters = [
            'alpha_a', 
            'mmin_a', 
            'alpha_b', 
            'mmin_b', 
            'alpha_c', 
            'mmin_c', 
            'm1max', 
            'mix_alpha', 
            'mix_beta', 
        ] + self.flag_powerlaw_smoothing*[
            'delta_m_a', 
            'delta_m_b', 
            'delta_m_c'
        ] + [
            'mmin', # dummy parameter not used by self.prior
            'delta_m', # dummy parameter not used by self.prior
            'mmax', # dummy parameter not used by self.prior
        ]

    def update(self,**kwargs):
        self.prior = TriplePowerLaw(
            alpha_a   = -kwargs['alpha_a'], 
            mmin_a    = kwargs['mmin_a'], 
            mmax_a    = kwargs['m1max'], 
            alpha_b   = -kwargs['alpha_b'], 
            mmin_b    = kwargs['mmin_b'], 
            mmax_b    = kwargs['m1max'], 
            alpha_c   = -kwargs['alpha_c'], 
            mmin_c    = kwargs['mmin_c'], 
            mmax_c    = kwargs['m1max'], 
            mix_a     = kwargs['mix_alpha'], 
            mix_b     = kwargs['mix_beta'], 
            smooth    = self.flag_powerlaw_smoothing, 
            delta_m_a = kwargs.get('delta_m_a', 1.0), # if no smoothing, default 1. value
            delta_m_b = kwargs.get('delta_m_b', 1.0), # if no smoothing, default 1. value
            delta_m_c = kwargs.get('delta_m_c', 1.0), # if no smoothing, default 1. value
        )


class massprior_3PL_globmax_jointmin(pm_prob):
    """
    3 Power-Laws model with a global mmax parameter shared by all PL components.
    Optional smoothing of low end of PL components.
    
    Dummy 'mmin' and 'delta_m' parameters to make it compatible with
    conditional wrappers without entangling p(m1) and p(m2).
    It is also usable outside conditional wrappers 
    e.g with p(q) parametrisation or pairing functions, 
    but 'mmin' and 'delta_m' need to be fixed.

    Note that PL components are parametrised as follows:
    $PL(m) \propto m^{- \alpha}$. 
    Consequently, the model is mostly relevant for positive alpha parameters.
    """

    def __init__(self, flag_powerlaw_smoothing=False):
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.population_parameters = [
            'alpha_a', 
            'mmin', 
            'alpha_b', 
            'mmin_b', 
            'alpha_c', 
            'mmin_c', 
            'm1max', 
            'mix_alpha', 
            'mix_beta', 
        ] + self.flag_powerlaw_smoothing*[
            'delta_m_a', 
            'delta_m_b', 
            'delta_m_c'
        ] + [
            'delta_m', # dummy parameter not used by self.prior
            'mmax', # dummy parameter not used by self.prior
        ]

    def update(self,**kwargs):
        self.prior = TriplePowerLaw(
            alpha_a   = -kwargs['alpha_a'], 
            mmin_a    = kwargs['mmin'], 
            mmax_a    = kwargs['m1max'], 
            alpha_b   = -kwargs['alpha_b'], 
            mmin_b    = kwargs['mmin_b'], 
            mmax_b    = kwargs['m1max'], 
            alpha_c   = -kwargs['alpha_c'], 
            mmin_c    = kwargs['mmin_c'], 
            mmax_c    = kwargs['m1max'], 
            mix_a     = kwargs['mix_alpha'], 
            mix_b     = kwargs['mix_beta'], 
            smooth    = self.flag_powerlaw_smoothing, 
            delta_m_a = kwargs.get('delta_m_a', 1.0), # if no smoothing, default 1. value
            delta_m_b = kwargs.get('delta_m_b', 1.0), # if no smoothing, default 1. value
            delta_m_c = kwargs.get('delta_m_c', 1.0), # if no smoothing, default 1. value
        )


class massprior_3PL_globmax_jointsmooth(pm_prob):
    """
    3 Power-Laws model with a global mmax parameter shared by all PL components.
    Optional smoothing of low end of PL components.
    
    Dummy 'mmin' and 'delta_m' parameters to make it compatible with
    conditional wrappers without entangling p(m1) and p(m2).
    It is also usable outside conditional wrappers 
    e.g with p(q) parametrisation or pairing functions, 
    but 'mmin' and 'delta_m' need to be fixed.

    Note that PL components are parametrised as follows:
    $PL(m) \propto m^{- \alpha}$. 
    Consequently, the model is mostly relevant for positive alpha parameters.
    """

    def __init__(self, flag_powerlaw_smoothing=False):
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.population_parameters = [
            'alpha_a', 
            'mmin_a', 
            'alpha_b', 
            'mmin_b', 
            'alpha_c', 
            'mmin_c', 
            'm1max', 
            'mix_alpha', 
            'mix_beta', 
        ] + self.flag_powerlaw_smoothing*[
            'delta_m', 
            'delta_m_b', 
            'delta_m_c'
        ] + [
            'mmin', # dummy parameter not used by self.prior
            'mmax', # dummy parameter not used by self.prior
        ]

    def update(self,**kwargs):
        self.prior = TriplePowerLaw(
            alpha_a   = -kwargs['alpha_a'], 
            mmin_a    = kwargs['mmin_a'], 
            mmax_a    = kwargs['m1max'], 
            alpha_b   = -kwargs['alpha_b'], 
            mmin_b    = kwargs['mmin_b'], 
            mmax_b    = kwargs['m1max'], 
            alpha_c   = -kwargs['alpha_c'], 
            mmin_c    = kwargs['mmin_c'], 
            mmax_c    = kwargs['m1max'], 
            mix_a     = kwargs['mix_alpha'], 
            mix_b     = kwargs['mix_beta'], 
            smooth    = self.flag_powerlaw_smoothing, 
            delta_m_a = kwargs.get('delta_m', 1.0), # if no smoothing, default 1. value
            delta_m_b = kwargs.get('delta_m_b', 1.0), # if no smoothing, default 1. value
            delta_m_c = kwargs.get('delta_m_c', 1.0), # if no smoothing, default 1. value
        )


class massprior_3PL_globmax_jointmax(pm_prob):
    """
    3 Power-Laws model with a global mmax parameter shared by all PL components.
    Optional smoothing of low end of PL components.
    
    Dummy 'mmin' and 'delta_m' parameters to make it compatible with
    conditional wrappers without entangling p(m1) and p(m2).
    It is also usable outside conditional wrappers 
    e.g with p(q) parametrisation or pairing functions, 
    but 'mmin' and 'delta_m' need to be fixed.

    Note that PL components are parametrised as follows:
    $PL(m) \propto m^{- \alpha}$. 
    Consequently, the model is mostly relevant for positive alpha parameters.
    """

    def __init__(self, flag_powerlaw_smoothing=False):
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.population_parameters = [
            'alpha_a', 
            'mmin_a', 
            'alpha_b', 
            'mmin_b', 
            'alpha_c', 
            'mmin_c', 
            'mmax', 
            'mix_alpha', 
            'mix_beta', 
        ] + self.flag_powerlaw_smoothing*[
            'delta_m_a', 
            'delta_m_b', 
            'delta_m_c'
        ] + [
            'mmin', # dummy parameter not used by self.prior
            'delta_m', # dummy parameter not used by self.prior
        ]

    def update(self,**kwargs):
        self.prior = TriplePowerLaw(
            alpha_a   = -kwargs['alpha_a'], 
            mmin_a    = kwargs['mmin_a'], 
            mmax_a    = kwargs['mmax'], 
            alpha_b   = -kwargs['alpha_b'], 
            mmin_b    = kwargs['mmin_b'], 
            mmax_b    = kwargs['mmax'], 
            alpha_c   = -kwargs['alpha_c'], 
            mmin_c    = kwargs['mmin_c'], 
            mmax_c    = kwargs['mmax'], 
            mix_a     = kwargs['mix_alpha'], 
            mix_b     = kwargs['mix_beta'], 
            smooth    = self.flag_powerlaw_smoothing, 
            delta_m_a = kwargs.get('delta_m_a', 1.0), # if no smoothing, default 1. value
            delta_m_b = kwargs.get('delta_m_b', 1.0), # if no smoothing, default 1. value
            delta_m_c = kwargs.get('delta_m_c', 1.0), # if no smoothing, default 1. value
        )


class massprior_3PL_globmax_jointminsmooth(pm_prob):
    """
    3 Power-Laws model with a global mmax parameter shared by all PL components.
    Optional smoothing of low end of PL components.
    
    Dummy 'mmin' and 'delta_m' parameters to make it compatible with
    conditional wrappers without entangling p(m1) and p(m2).
    It is also usable outside conditional wrappers 
    e.g with p(q) parametrisation or pairing functions, 
    but 'mmin' and 'delta_m' need to be fixed.

    Note that PL components are parametrised as follows:
    $PL(m) \propto m^{- \alpha}$. 
    Consequently, the model is mostly relevant for positive alpha parameters.
    """

    def __init__(self, flag_powerlaw_smoothing=False):
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.population_parameters = [
            'alpha_a', 
            'mmin', 
            'alpha_b', 
            'mmin_b', 
            'alpha_c', 
            'mmin_c', 
            'm1max', 
            'mix_alpha', 
            'mix_beta', 
        ] + self.flag_powerlaw_smoothing*[
            'delta_m', 
            'delta_m_b', 
            'delta_m_c'
        ] + [
            'mmax', # dummy parameter not used by self.prior
        ]

    def update(self,**kwargs):
        self.prior = TriplePowerLaw(
            alpha_a   = -kwargs['alpha_a'], 
            mmin_a    = kwargs['mmin'], 
            mmax_a    = kwargs['m1max'], 
            alpha_b   = -kwargs['alpha_b'], 
            mmin_b    = kwargs['mmin_b'], 
            mmax_b    = kwargs['m1max'], 
            alpha_c   = -kwargs['alpha_c'], 
            mmin_c    = kwargs['mmin_c'], 
            mmax_c    = kwargs['m1max'], 
            mix_a     = kwargs['mix_alpha'], 
            mix_b     = kwargs['mix_beta'], 
            smooth    = self.flag_powerlaw_smoothing, 
            delta_m_a = kwargs.get('delta_m', 1.0), # if no smoothing, default 1. value
            delta_m_b = kwargs.get('delta_m_b', 1.0), # if no smoothing, default 1. value
            delta_m_c = kwargs.get('delta_m_c', 1.0), # if no smoothing, default 1. value
        )


class massprior_3PL_globmax_jointminmax(pm_prob):
    """
    3 Power-Laws model with a global mmax parameter shared by all PL components.
    Optional smoothing of low end of PL components.
    
    Dummy 'mmin' and 'delta_m' parameters to make it compatible with
    conditional wrappers without entangling p(m1) and p(m2).
    It is also usable outside conditional wrappers 
    e.g with p(q) parametrisation or pairing functions, 
    but 'mmin' and 'delta_m' need to be fixed.

    Note that PL components are parametrised as follows:
    $PL(m) \propto m^{- \alpha}$. 
    Consequently, the model is mostly relevant for positive alpha parameters.
    """

    def __init__(self, flag_powerlaw_smoothing=False):
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.population_parameters = [
            'alpha_a', 
            'mmin', 
            'alpha_b', 
            'mmin_b', 
            'alpha_c', 
            'mmin_c', 
            'mmax', 
            'mix_alpha', 
            'mix_beta', 
        ] + self.flag_powerlaw_smoothing*[
            'delta_m_a', 
            'delta_m_b', 
            'delta_m_c'
        ] + [
            'delta_m', # dummy parameter not used by self.prior
        ]

    def update(self,**kwargs):
        self.prior = TriplePowerLaw(
            alpha_a   = -kwargs['alpha_a'], 
            mmin_a    = kwargs['mmin'], 
            mmax_a    = kwargs['mmax'], 
            alpha_b   = -kwargs['alpha_b'], 
            mmin_b    = kwargs['mmin_b'], 
            mmax_b    = kwargs['mmax'], 
            alpha_c   = -kwargs['alpha_c'], 
            mmin_c    = kwargs['mmin_c'], 
            mmax_c    = kwargs['mmax'], 
            mix_a     = kwargs['mix_alpha'], 
            mix_b     = kwargs['mix_beta'], 
            smooth    = self.flag_powerlaw_smoothing, 
            delta_m_a = kwargs.get('delta_m_a', 1.0), # if no smoothing, default 1. value
            delta_m_b = kwargs.get('delta_m_b', 1.0), # if no smoothing, default 1. value
            delta_m_c = kwargs.get('delta_m_c', 1.0), # if no smoothing, default 1. value
        )


class massprior_3PL_globmax_jointsmoothmax(pm_prob):
    """
    3 Power-Laws model with a global mmax parameter shared by all PL components.
    Optional smoothing of low end of PL components.
    
    Dummy 'mmin' and 'delta_m' parameters to make it compatible with
    conditional wrappers without entangling p(m1) and p(m2).
    It is also usable outside conditional wrappers 
    e.g with p(q) parametrisation or pairing functions, 
    but 'mmin' and 'delta_m' need to be fixed.

    Note that PL components are parametrised as follows:
    $PL(m) \propto m^{- \alpha}$. 
    Consequently, the model is mostly relevant for positive alpha parameters.
    """

    def __init__(self, flag_powerlaw_smoothing=False):
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.population_parameters = [
            'alpha_a', 
            'mmin_a', 
            'alpha_b', 
            'mmin_b', 
            'alpha_c', 
            'mmin_c', 
            'mmax', 
            'mix_alpha', 
            'mix_beta', 
        ] + self.flag_powerlaw_smoothing*[
            'delta_m', 
            'delta_m_b', 
            'delta_m_c'
        ] + [
            'mmin', # dummy parameter not used by self.prior
        ]

    def update(self,**kwargs):
        self.prior = TriplePowerLaw(
            alpha_a   = -kwargs['alpha_a'], 
            mmin_a    = kwargs['mmin_a'], 
            mmax_a    = kwargs['mmax'], 
            alpha_b   = -kwargs['alpha_b'], 
            mmin_b    = kwargs['mmin_b'], 
            mmax_b    = kwargs['mmax'], 
            alpha_c   = -kwargs['alpha_c'], 
            mmin_c    = kwargs['mmin_c'], 
            mmax_c    = kwargs['mmax'], 
            mix_a     = kwargs['mix_alpha'], 
            mix_b     = kwargs['mix_beta'], 
            smooth    = self.flag_powerlaw_smoothing, 
            delta_m_a = kwargs.get('delta_m', 1.0), # if no smoothing, default 1. value
            delta_m_b = kwargs.get('delta_m_b', 1.0), # if no smoothing, default 1. value
            delta_m_c = kwargs.get('delta_m_c', 1.0), # if no smoothing, default 1. value
        )


class massprior_3PL_globmax_jointminsmoothmax(pm_prob):
    """
    3 Power-Laws model with a global mmax parameter shared by all PL components.
    Optional smoothing of low end of PL components.
    
    Dummy 'mmin' and 'delta_m' parameters to make it compatible with
    conditional wrappers without entangling p(m1) and p(m2).
    It is also usable outside conditional wrappers 
    e.g with p(q) parametrisation or pairing functions, 
    but 'mmin' and 'delta_m' need to be fixed.

    Note that PL components are parametrised as follows:
    $PL(m) \propto m^{- \alpha}$. 
    Consequently, the model is mostly relevant for positive alpha parameters.
    """

    def __init__(self, flag_powerlaw_smoothing=False):
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.population_parameters = [
            'alpha_a', 
            'mmin', 
            'alpha_b', 
            'mmin_b', 
            'alpha_c', 
            'mmin_c', 
            'mmax', 
            'mix_alpha', 
            'mix_beta', 
        ] + self.flag_powerlaw_smoothing*[
            'delta_m', 
            'delta_m_b', 
            'delta_m_c'
        ]

    def update(self,**kwargs):
        self.prior = TriplePowerLaw(
            alpha_a   = -kwargs['alpha_a'], 
            mmin_a    = kwargs['mmin'], 
            mmax_a    = kwargs['mmax'], 
            alpha_b   = -kwargs['alpha_b'], 
            mmin_b    = kwargs['mmin_b'], 
            mmax_b    = kwargs['mmax'], 
            alpha_c   = -kwargs['alpha_c'], 
            mmin_c    = kwargs['mmin_c'], 
            mmax_c    = kwargs['mmax'], 
            mix_a     = kwargs['mix_alpha'], 
            mix_b     = kwargs['mix_beta'], 
            smooth    = self.flag_powerlaw_smoothing, 
            delta_m_a = kwargs.get('delta_m', 1.0), # if no smoothing, default 1. value
            delta_m_b = kwargs.get('delta_m_b', 1.0), # if no smoothing, default 1. value
            delta_m_c = kwargs.get('delta_m_c', 1.0), # if no smoothing, default 1. value
        )


class massprior_4PL_global_mmax(pm_prob):
    """
    4 Power-Laws model with a global mmax parameter shared by all PL components.
    Optional smoothing of low end of PL components.

    Note that PL components are parametrised as follows:
    $PL(m) \propto m^{- \alpha}$. 
    Consequently, the model is mostly relevant for positive alpha parameters.
    """

    def __init__(self, flag_powerlaw_smoothing=False):
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.population_parameters = [
            'alpha_a', 
            'mmin', 
            'alpha_b', 
            'mmin_b', 
            'alpha_c', 
            'mmin_c', 
            'alpha_d', 
            'mmin_d', 
            'mmax', 
            'mix_alpha', 
            'mix_beta', 
            'mix_gamma', 
        ] + self.flag_powerlaw_smoothing*[
            'delta_m', 
            'delta_m_b', 
            'delta_m_c', 
            'delta_m_d'
        ]

    def update(self,**kwargs):
        self.prior = QuadruplePowerLaw(
            alpha_a   = -kwargs['alpha_a'], 
            mmin_a    = kwargs['mmin'], 
            mmax_a    = kwargs['mmax'], 
            alpha_b   = -kwargs['alpha_b'], 
            mmin_b    = kwargs['mmin_b'], 
            mmax_b    = kwargs['mmax'], 
            alpha_c   = -kwargs['alpha_c'], 
            mmin_c    = kwargs['mmin_c'], 
            mmax_c    = kwargs['mmax'], 
            alpha_d   = -kwargs['alpha_d'], 
            mmin_d    = kwargs['mmin_d'], 
            mmax_d    = kwargs['mmax'], 
            mix_a     = kwargs['mix_alpha'], 
            mix_b     = kwargs['mix_beta'], 
            mix_c     = kwargs['mix_gamma'], 
            smooth    = self.flag_powerlaw_smoothing, 
            delta_m_a = kwargs.get('delta_m', 1.0), # if no smoothing, default 1. value
            delta_m_b = kwargs.get('delta_m_b', 1.0), # if no smoothing, default 1. value
            delta_m_c = kwargs.get('delta_m_c', 1.0), # if no smoothing, default 1. value
            delta_m_d = kwargs.get('delta_m_c', 1.0) # if no smoothing, default 1. value
        )