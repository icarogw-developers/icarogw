from .cupy_pal import get_module_array, get_module_array_scipy, np, _set_xp_and_dtype
from .cosmology import alphalog_astropycosmology, cM_astropycosmology, extraD_astropycosmology, Xi0_astropycosmology, astropycosmology, eps0_astropycosmology
from .cosmology import  md_rate, md_gamma_rate, powerlaw_rate, beta_rate, beta_rate_line
from .priors import LowpassSmoothedProb, LowpassSmoothedProbEvolving, PowerLaw, BetaDistribution, TruncatedBetaDistribution, TruncatedGaussian, Bivariate2DGaussian, SmoothedPlusDipProb, BrokenPowerLawMultiPeak
from .priors import PowerLawGaussian, BrokenPowerLaw, PowerLawTwoGaussians, conditional_2dimpdf, conditional_2dimz_pdf, piecewise_constant_2d_distribution_normalized,paired_2dimpdf
from .priors import PowerLawStationary, PowerLawLinear, GaussianStationary, GaussianLinear, _mixed_linear_function, _mixed_double_sigmoid_function
from .priors import BrokenPowerLawTripleMultiPeak
from .priors import TriplePowerLaw, QuadruplePowerLaw
from .priors import logBspline, PowerLaw_logBspline
import copy
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, Flatw0waCDM
from scipy.special import expit

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


######################################################################

class massratio_PowerlawSmooth(object):
    '''
    Conditional mass-ratio distribution
    p(q | m1) propto q^{alpha_q}
    for q in [mmin / m1, 1], with smoothing in m2 = q m1
    '''
    def __init__(self, mw):
        self.population_parameters = ['alpha_q','delta_m']
        self.mw = mw

    def update(self, **kwargs):
        self.alpha_q = kwargs['alpha_q']
        self.delta_m = kwargs['delta_m']
        
    def pdf(self, mass_ratio, mass_1):
        mmin = self.mw.prior.minval
        qmin = mmin / mass_1 
        p_q = PowerLaw(qmin, 1., self.alpha_q)
        p_s = onesided_taperwindow_smoothing(mass=mass_1*mass_ratio,
                                        mmin = mmin,
                                        mmax = mass_1,
                                        delta_m = self.delta_m)
        # Not normalized anymore but should be ok with scale-free inferences.
        return p_q.pdf(mass_ratio)*p_s
        
    def log_pdf(self, mass_ratio, mass_1):
        return np.log(self.pdf(mass_ratio, mass_1))

def onesided_taperwindow_smoothing(mass, mmin, mmax, delta_m):
    '''
    Apply a one-sided Planck-taper window between mmin and mmin+delta_m.
    S = (1 + exp[ 1/x - 1/(1-x) ])^{-1}
    with x = (m - mmin) / delta_m
    '''
    if delta_m <= 0:
        return np.ones_like(mass)

    x = (mass - mmin) / delta_m
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    
    exponent = 1.0 / x - 1.0 / (1.0 - x)
    window = expit(-exponent)
    window *= (mass >= mmin) & (mass <= mmax)
    return window
    
######################################################################


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

# Need to be reviewed for O4b
class m1m2_paired_massratio_bpl_2peaks(pm1m2_prob):
    '''
    BPL + 2peaks for BBHs
    '''
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

class spinprior_default_gaussian_bis(object):
    '''
    Same as spinprior_default_gaussian(), but here chi_1 and chi_2 share the same Gaussian distribution.
    '''
    def __init__(self):
        self.population_parameters=['mu_chi','sigma_chi','sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']

    def update(self,**kwargs):        
        self.csi_spin = kwargs['csi_spin']
        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)
        self.g = TruncatedGaussian(kwargs['mu_chi'],kwargs['sigma_chi'],0.,1.)
    
    def log_pdf(self, chi_1, chi_2, cos_t_1, cos_t_2, **kwargs):
        xp = get_module_array(chi_1)
        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))
        return self.g.log_pdf(chi_1)+self.g.log_pdf(chi_2)+log_angular_part
        
    def pdf(self, chi_1, chi_2, cos_t_1, cos_t_2, **kwargs):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2))

# TO BE REVIEWED O4b
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

# TO BE REVIEWED O4b (priority 1)
class spinprior_default_gaussian_window_gaussian(object):
    '''
    TO BE REVIEWED FOR O4b
    Gaussian to Gaussian spin mass evolution
    '''
    def __init__(self):
        self.population_parameters= ['mt', 
                                     'delta_mt','mix_f',
                                     'mu_chi_1','sigma_chi_1',
                                     'mu_chi_2','sigma_chi_2',
                                     'sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']
    

    def update(self,**kwargs):
        
        self.mu_chi_1 = kwargs['mu_chi_1']
        self.sigma_chi_1 = kwargs['sigma_chi_1']
        self.mu_chi_2 = kwargs['mu_chi_2']
        self.sigma_chi_2 = kwargs['sigma_chi_2']
        self.csi_spin = kwargs['csi_spin']
        self.gaussian_pdf_chi_1 = TruncatedGaussian(kwargs['mu_chi_1'],kwargs['sigma_chi_1'],0.,1.)
        self.gaussian_pdf_chi_2 = TruncatedGaussian(kwargs['mu_chi_2'],kwargs['sigma_chi_2'],0.,1.)

        self.mt, self.delta_mt, self.mix_f = kwargs['mt'], kwargs['delta_mt'], kwargs['mix_f']

        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)

    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        
        xp = get_module_array(chi_1)
        # self.mix_f is \lambda_f in the Eqs in the overleaf (defines the value of the window function at m=0)
        wz_1 = _mixed_double_sigmoid_function(x=mass_1_source, xt=self.mt, delta_xt=self.delta_mt, mix_x0=self.mix_f, mix_x1=0.)
        wz_2 = _mixed_double_sigmoid_function(x=mass_2_source, xt=self.mt, delta_xt=self.delta_mt, mix_x0=self.mix_f, mix_x1=0.)

        pdf_1 = wz_1*self.gaussian_pdf_chi_1.pdf(chi_1)+(1-wz_1)*self.gaussian_pdf_chi_2.pdf(chi_1)
        pdf_2 = wz_2*self.gaussian_pdf_chi_1.pdf(chi_2)+(1-wz_2)*self.gaussian_pdf_chi_2.pdf(chi_2)

        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))
        
        out = xp.log(pdf_1)+xp.log(pdf_2)+log_angular_part
        
        return out
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source))

# TO BE REVIEWED O4b
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
        
        wz_1 = _mixed_double_sigmoid_function(x=mass_1_source, xt=self.mt, delta_xt=self.delta_mt, mix_x0=self.mix_f, mix_x1=0.)
        wz_2 = _mixed_double_sigmoid_function(x=mass_2_source, xt=self.mt, delta_xt=self.delta_mt, mix_x0=self.mix_f, mix_x1=0.)

        pdf_1 = wz_1*self.beta_pdf_chi.pdf(chi_1)+(1-wz_1)*self.gaussian_pdf_chi.pdf(chi_1)
        pdf_2 = wz_2*self.beta_pdf_chi.pdf(chi_2)+(1-wz_2)*self.gaussian_pdf_chi.pdf(chi_2)

        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))
        
        out = xp.log(pdf_1)+xp.log(pdf_2)+log_angular_part
        
        return out
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,mass_2_source))

# TO BE REVIEWED O4b
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
        
        wz_1 = _mixed_double_sigmoid_function(x=mass_1_source, xt=self.mt, delta_xt=self.delta_mt, mix_x0=self.mix_f, mix_x1=0.)
        wz_2 = _mixed_double_sigmoid_function(x=mass_2_source, xt=self.mt, delta_xt=self.delta_mt, mix_x0=self.mix_f, mix_x1=0.)

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
    
    def log_pdf(self, chi_1, chi_2, cos_t_1, cos_t_2, **kwargs):
        xp = get_module_array(chi_1)
        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))
        return self.g1.log_pdf(chi_1)+self.g2.log_pdf(chi_2)+log_angular_part
        
    def pdf(self, chi_1, chi_2, cos_t_1, cos_t_2, **kwargs):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2))



import numpy as np
from scipy.stats import truncnorm

def log_truncnorm_pdf(x, mean, std, lower, upper):
    # Standardize bounds
    a = (lower - mean) / std
    b = (upper - mean) / std
    # Standardize x
    x_std = (x - mean) / std

    # Get the log PDF from the truncated normal
    log_pdf = truncnorm.logpdf(x_std, a, b, loc=0, scale=1) - np.log(std)
    return log_pdf

class spinprior_default_gaussian_zeroed(object):
    def __init__(self):
        self.population_parameters=['mu_chi_1','mu_chi_2','sigma_chi_1','sigma_chi_2','sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']

    def update(self,**kwargs):        
        self.csi_spin = kwargs['csi_spin']
        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)
        self.mu_chi_1 = kwargs['mu_chi_1']
        self.mu_chi_2 = kwargs['mu_chi_2']
        self.sigma_chi_1 = kwargs['sigma_chi_1']
        self.sigma_chi_2 = kwargs['sigma_chi_2']
        
    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source, mass_2_source):
        xp = get_module_array(chi_1)
        amax_1 = xp.where(mass_1_source<2.0, 0.4, 1.0) # To be consistent with injections
        amax_2 = xp.where(mass_2_source<2.0, 0.4, 1.0)  # To be consistent with injections
        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))

        log_g1 = log_truncnorm_pdf(chi_1,self.mu_chi_1,self.sigma_chi_1,0.,amax_1)
        log_g2 = log_truncnorm_pdf(chi_2,self.mu_chi_2,self.sigma_chi_2,0.,amax_2)
        
        return log_g1+log_g2+log_angular_part
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source, mass_2_source):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source, mass_2_source))


class spinprior_default_gaussian_zeroed_spin_bis(object):
    def __init__(self):
        self.population_parameters=['mu_chi','sigma_chi','sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']

    def update(self,**kwargs):        
        self.csi_spin = kwargs['csi_spin']
        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)
        self.mu_chi_1 = kwargs['mu_chi']
        self.mu_chi_2 = kwargs['mu_chi']
        self.sigma_chi_1 = kwargs['sigma_chi']
        self.sigma_chi_2 = kwargs['sigma_chi']
        
    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source, mass_2_source):
        xp = get_module_array(chi_1)
        amax_1 = xp.where(mass_1_source<2.0, 0.4, 1.0) # To be consistent with injections
        amax_2 = xp.where(mass_2_source<2.0, 0.4, 1.0)  # To be consistent with injections
        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))

        log_g1 = log_truncnorm_pdf(chi_1,self.mu_chi_1,self.sigma_chi_1,0.,amax_1)
        log_g2 = log_truncnorm_pdf(chi_2,self.mu_chi_2,self.sigma_chi_2,0.,amax_2)
        
        return log_g1+log_g2+log_angular_part
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source, mass_2_source):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source, mass_2_source))



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

class PowerLaw():
    '''
        Class implementing the mass function model for one stationary PowerLaw.

        Some options are available:
            - flag_powerlaw_smoothing applies a left window function to the PowerLaw.

        The module is stand alone and not compatible with other wrappers.
    '''

    def __init__(self, flag_powerlaw_smoothing = 1):
        
        self.population_parameters   = ['alpha', 'mmin', 'mmax']
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing

        if self.flag_powerlaw_smoothing: self.population_parameters += ['delta_m']

    def update(self,**kwargs):

        self.alpha = kwargs['alpha']
        self.mmin  = kwargs['mmin']
        self.mmax  = kwargs['mmax']

        if self.flag_powerlaw_smoothing:
            self.delta_m = kwargs['delta_m']

    def pdf(self,m):

        powerlaw_class = PowerLawStationary(self.alpha, self.mmin, self.mmax)
        # Add left smoothing to the PowerLaw.
        if self.flag_powerlaw_smoothing:
            powerlaw_class = LowpassSmoothedProb(powerlaw_class, self.delta_m)
        powerlaw_part = powerlaw_class.pdf(m)

        return powerlaw_part
    
    def log_pdf(self,m):
        xp = get_module_array(m)
        return xp.log(self.pdf(m))


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


class Gaussian():

    def __init__(self):
        self.population_parameters = ['mu', 'sigma', 'mmin', 'mmax']

    def update(self,**kwargs):
        self.mu    = kwargs['mu']
        self.sigma = kwargs['sigma']
        self.mmin  = kwargs['mmin']
        self.mmax  = kwargs['mmax']

    def pdf(self,m):
        tmp = TruncatedGaussian(self.mu, self.sigma, self.mmin, self.mmax)
        return tmp.pdf(m)

    def log_pdf(self,m):
        xp = get_module_array(m)
        return xp.log(self.pdf(m))


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


class Gaussian_Gaussian_Gaussian_Gaussian():
    """
    Properly normalized exponential-of-Gaussians density:

        log p(m) = sum_i w_i * N(m | mu_i, sigma_i) - log Z

    Normalization is computed numerically on a fixed grid:
        - 1000 points
        - m in [mmin, 120]
    """

    def __init__(self):
        self.population_parameters = [
            'mu_a', 'sigma_a',
            'mu_b', 'sigma_b',
            'mu_c', 'sigma_c',
            'mu_d', 'sigma_d',
            'mmin',
            'mix_alpha', 'mix_beta', 'mix_gamma', 'mix_delta'
        ]

        # Set backend (numpy or cupy) and dtype once
        self.xp, self.dtype = _set_xp_and_dtype()

    def update(self, **kwargs):
        xp = self.xp

        # Gaussian parameters
        self.mu_a    = kwargs['mu_a']
        self.sigma_a = kwargs['sigma_a']
        self.mu_b    = kwargs['mu_b']
        self.sigma_b = kwargs['sigma_b']
        self.mu_c    = kwargs['mu_c']
        self.sigma_c = kwargs['sigma_c']
        self.mu_d    = kwargs['mu_d']
        self.sigma_d = kwargs['sigma_d']

        self.mmin = kwargs['mmin']

        # Cast weights onto the correct backend and dtype
        self.mix_alpha = xp.asarray(kwargs['mix_alpha'], dtype=self.dtype)
        self.mix_beta  = xp.asarray(kwargs['mix_beta'],  dtype=self.dtype)
        self.mix_gamma = xp.asarray(kwargs['mix_gamma'], dtype=self.dtype)
        self.mix_delta = xp.asarray(kwargs['mix_delta'], dtype=self.dtype)

        self._compute_log_norm()

    def _energy(self, m):
        """
        Energy function:
            E(m) = sum_i w_i * N_i(m)
        """
        xp = self.xp

        ga = xp.asarray(
            GaussianStationary(self.mu_a, self.sigma_a, self.mmin).pdf(m),
            dtype=self.dtype,
        )
        gb = xp.asarray(
            GaussianStationary(self.mu_b, self.sigma_b, self.mmin).pdf(m),
            dtype=self.dtype,
        )
        gc = xp.asarray(
            GaussianStationary(self.mu_c, self.sigma_c, self.mmin).pdf(m),
            dtype=self.dtype,
        )
        gd = xp.asarray(
            GaussianStationary(self.mu_d, self.sigma_d, self.mmin).pdf(m),
            dtype=self.dtype,
        )

        return (
            self.mix_alpha * ga
          + self.mix_beta  * gb
          + self.mix_gamma * gc
          + self.mix_delta * gd
        )

    def _compute_log_norm(self):
        """
        Numerically compute log Z on a fixed grid:
            - 1000 points
            - upper bound = 120
        """
        xp = self.xp

        self.m_grid = xp.linspace(
            self.mmin,
            120.0,
            1000,
            dtype=self.dtype,
        )

        dm = self.m_grid[1] - self.m_grid[0]

        energy = self._energy(self.m_grid)

        # log Z = log ∫ exp(E(m)) dm
        self.log_norm = self._logsumexp(energy) + xp.log(dm)

    def _logsumexp(self, x):
        xp = self.xp
        xmax = xp.max(x)
        return xmax + xp.log(xp.sum(xp.exp(x - xmax)))

    def log_pdf(self, m):
        """
        Properly normalized log-PDF.
        """
        return self._energy(m) - self.log_norm

    def pdf(self, m):
        """
        Properly normalized PDF.
        """
        xp = self.xp
        return xp.exp(self.log_pdf(m))
    
    
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

    def __init__(self, n_basis: int = 6, degree: int = 2, spacing: str = "log"):
        # n_basis is the number of interior basis functions supplied by the user;
        # total basis count = interior + 2 (first and last fixed)
        self.n_basis = int(n_basis) + 2
        self.degree = int(degree)
        self.spacing = spacing

        if spacing not in ("uniform", "log"):
            raise ValueError("spacing must be 'uniform' or 'log'")
        
        self.population_parameters = ['mmin', 'mmax'] + [f'c{i}' for i in range(1, self.n_basis-1)]

        # Backend (numpy or cupy) and dtype (float32 on GPU by default)
        self.xp, self.dtype = _set_xp_and_dtype()

    def bspline_basis(self, x, t, k = 3):
        """
        Compute B-spline basis functions using Cox-de Boor recursion.
        """
        xp = self.xp
        x = xp.asarray(x, dtype=self.dtype)
        t = xp.asarray(t, dtype=self.dtype)
        n_basis = len(t) - k - 1
        n_points = len(x)

        # Zeroth-degree basis
        B = xp.zeros((n_points, n_basis), dtype=x.dtype)
        for i in range(n_basis):
            B[:, i] = xp.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
        if x.size and t.size:
            B[x == t[-1], -1] = 1.0

        # Cox-de Boor recursion
        for d in range(1, k + 1):
            t_i = t[:n_basis]
            t_id = t[d:n_basis + d]
            t_ip1 = t[1:n_basis + 1]
            t_ip1d1 = t[d + 1:n_basis + d + 1]

            denom1 = xp.where(t_id - t_i > 0, t_id - t_i, 1.0)
            denom2 = xp.where(t_ip1d1 - t_ip1 > 0, t_ip1d1 - t_ip1, 1.0)

            term1 = ((x[:, None] - t_i[None, :]) / denom1[None, :]) * B
            term1 = xp.where(denom1[None, :] > 0, term1, 0.0)

            term2 = xp.zeros_like(B)
            if n_basis > 1:
                term2[:, :-1] = ((t_ip1d1[None, :-1] - x[:, None]) / denom2[None, :-1]) * B[:, 1:]
                term2 = xp.where(denom2[None, :] > 0, term2, 0.0)

            B = term1 + term2

        return B

    def _setup_grid_and_knots(self):
        """
        Recompute knots and precompute B-spline basis grid.
        Uses self.spacing ("log" or "uniform") to control spacing type.
        """
        xp = self.xp
        spacing = self.spacing
        k = self.degree
        n = self.n_basis

        if spacing == "log":
            self.xmin, self.xmax = xp.log(self.mmin), xp.log(self.mmax)
            from_x = xp.exp
        else:  # uniform
            self.xmin, self.xmax = self.mmin, self.mmax
            from_x = lambda x: x

        # Number of interior knot *locations*
        # This guarantees: len(t) = n + k + 1
        interior = xp.linspace(
            self.xmin,
            self.xmax,
            n - k + 1,
            dtype=self.dtype
        )
        t_start, t_end = xp.repeat(interior[0], k + 1), xp.repeat(interior[-1], k + 1)
        self.t = xp.concatenate([t_start, interior[1:-1], t_end]) # Clamped knot vector

        self._x_grid = xp.linspace(self.xmin, self.xmax, 1000, dtype=self.dtype)
        self._m_grid = from_x(self._x_grid)
        self._B_grid = self.bspline_basis(self._x_grid, self.t, k=k)

    def update(self, **kwargs):
        """
        Update spline parameters and coefficients.
        """
        xp = self.xp
        self.mmin, self.mmax = kwargs['mmin'], kwargs['mmax']
        self._setup_grid_and_knots()

        coeff_keys = [f'c{i}' for i in range(1, self.n_basis - 1)]
        coeffs_list = [0.0] + [kwargs.get(k, 0.0) for k in coeff_keys] + [0.0]
        self.coeffs = xp.asarray(coeffs_list, dtype=self.dtype)

    def eval_spline(self, m):
        """
        Evaluate the spline at mass m.
        """
        xp = self.xp
        m = xp.asarray(m, dtype=self.dtype)
        if self.spacing == "log": x = xp.log(m)
        else:                     x = m
        B = self.bspline_basis(x.ravel(), self.t, k=self.degree)
        coeffs = xp.asarray(self.coeffs, dtype=self.dtype)
        s_flat = B.dot(coeffs)
        return s_flat.reshape(x.shape)

    def logZ(self):
        """
        Compute log-normalization factor.
        """
        xp = self.xp
        coeffs = xp.asarray(self.coeffs, dtype=self.dtype)

        s_grid = self._B_grid.dot(coeffs)
        s_max = xp.max(s_grid)
        
        integrand = xp.exp(s_grid - s_max)
        Z = xp.trapz(integrand, self._m_grid)

        tiny = xp.finfo(self.dtype).tiny
        return xp.log(Z + tiny) + s_max

    def pdf(self, m):
        """
        Evaluate normalized probability density function at m.
        """
        xp = self.xp
        s = self.eval_spline(m)
        lZ = self.logZ()
        return xp.exp(s - lZ)

    def log_pdf(self, m):
        """
        Evaluate log of normalized probability density function at m.
        """
        s = self.eval_spline(m)
        lZ = self.logZ()
        return s - lZ

    # ---------
    # Utilities
    # ---------
    def basis(self, m):
        """
        Return basis evaluated at masses m (shape (len(m), n_basis)).
        Handles log spacing automatically.
        """
        xp = get_module_array(m)
        m = xp.asarray(m, dtype=self.dtype)
        if self.spacing == "log":
            x = xp.log(m)
        else:
            x = m
        # bspline_basis expects x in basis-space and t (knot vector)
        B = self.bspline_basis(x.ravel(), self.t, k=self.degree)
        return B  # shape (len(m), n_basis)
    
    def get_weights(self):
        # coefficients / weights (including boundary zeros)
        if getattr(self, "coeffs", None) is None:
            return None
        try:
            return self.coeffs.get()   # cupy -> host
        except Exception:
            return self.coeffs

    def get_knots(self):
        # knot vector (in basis space: x or log(m) depending on spacing)
        if getattr(self, "t", None) is None:
            return None
        try:
            return self.t.get()
        except Exception:
            return self.t


# PowerLaw models / secondary experiments
class massprior_3PL(pm_prob):
    def __init__(self, flag_powerlaw_smoothing=False):
        self.flag_powerlaw_smoothing = flag_powerlaw_smoothing
        self.population_parameters = [
            'alpha_a', 
            'mmin', 
            'mmax_a', 
            'alpha_b', 
            'mmin_b', 
            'mmax_b', 
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
            mmax_a    = kwargs['mmax_a'], 
            alpha_b   = -kwargs['alpha_b'], 
            mmin_b    = kwargs['mmin_b'], 
            mmax_b    = kwargs['mmax_b'], 
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


class massprior_3PL_global_mmax(pm_prob):
    """
    3 Power-Laws model with a global mmax parameter shared by all PL components.
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


class massprior_3PL_global_mmax_dummy_mmin(pm_prob):
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
##################### SPIN - MASS - REDSHIFT Correlation models ####################

class spinprior_linear_chieff_q(object):
    '''
    Definition
    ----------
    Wrapper for the evolving spin model chi-eff-q inspired from: https://arxiv.org/pdf/2508.18083
    p(chi_eff | q) = Truncated Gaussian on [-1, 1]
    with mean and log-variance linear in q = m2/m1
    
    Parameters
    ----------
    mu_chieff_0, mu_chieff_1: parameters for the linear evolution of the mean of the Gaussian
    ln_sigma_chieff_0,ln_sigma_chieff_1:  parameters for the linear evolution of the ln of the Gaussian width
    x0: pivot value, default is 0.
    
    '''
    def __init__(self):
        self.population_parameters=['mu_chieff_0','mu_chieff_1','ln_sigma_chieff_0','ln_sigma_chieff_1','x0']
        self.event_parameters=['chi_eff','mass_1_source','mass_2_source']

    def update(self,**kwargs):
        self.mu_chieff_0 = kwargs['mu_chieff_0']
        self.mu_chieff_1 = kwargs['mu_chieff_1']
        self.ln_sigma_chieff_0 = kwargs['ln_sigma_chieff_0']
        self.ln_sigma_chieff_1 = kwargs['ln_sigma_chieff_1']
        self.x0 = kwargs['x0']

    def calculate_mu_chieff(self,q):
        return self.mu_chieff_0 + self.mu_chieff_1*(q - self.x0)
        
    def calculate_ln_sigma_chieff(self,q):
        return self.ln_sigma_chieff_0 + self.ln_sigma_chieff_1*(q - self.x0)
    
    def log_pdf(self,chi_eff,mass_1_source,mass_2_source):
        xp = get_module_array(mass_1_source)
        q = mass_2_source / mass_1_source
        
        mu = self.calculate_mu_chieff(q)
        ln_sigma = self.calculate_ln_sigma_chieff(q)
        sigma = xp.exp(ln_sigma)
        # clipping to avoid issue of normalization and dirac like behavior ? sigma = np.exp(np.clip(ln_sigma, -10, 2)) 
        
        dist = TruncatedGaussian(mu,sigma,-1.,1.)
        return dist.log_pdf(chi_eff)
         
    def pdf(self,chi_eff,mass_1_source,mass_2_source):
        xp = get_module_array(mass_1_source)
        return xp.exp(self.log_pdf(chi_eff,mass_1_source,mass_2_source))

class spinprior_linear_chieff_z(object):
    '''
    Definition
    ----------
    Wrapper for the evolving spin model chi-eff-z inspired from: https://arxiv.org/pdf/2508.18083
    p(chi_eff | z) = Truncated Gaussian on [-1, 1]
    with mean and log-variance linear in z
    
    Parameters
    ----------
    mu_chieff_0, mu_chieff_1
        parameters for the linear evolution of the mean of the Gaussian
    ln_sigma_chieff_0,ln_sigma_chieff_1
        parameters for the linear evolution of the ln of the Gaussian width
    x0
        pivot value, default is 0.
    
    '''
    def __init__(self):
        self.population_parameters=['mu_chieff_0','mu_chieff_1','ln_sigma_chieff_0','ln_sigma_chieff_1','x0']
        self.event_parameters=['chi_eff','luminosity_distance']

    def update(self,**kwargs):
        self.mu_chieff_0 = kwargs['mu_chieff_0']
        self.mu_chieff_1 = kwargs['mu_chieff_1']
        self.ln_sigma_chieff_0 = kwargs['ln_sigma_chieff_0']
        self.ln_sigma_chieff_1 = kwargs['ln_sigma_chieff_1']
        self.x0 = kwargs['x0']

    def calculate_mu_chieff(self,z):
        return self.mu_chieff_0 + self.mu_chieff_1*(z - self.x0)
        
    def calculate_ln_sigma_chieff(self,z):
        return self.ln_sigma_chieff_0 + self.ln_sigma_chieff_1*(z - self.x0)

    def log_pdf(self,chi_eff,z):
        xp = get_module_array(chi_eff)
        mu = self.calculate_mu_chieff(z)
        ln_sigma = self.calculate_ln_sigma_chieff(z)
        sigma = xp.exp(ln_sigma)
        
        dist = TruncatedGaussian(mu,sigma,-1.,1.)
        return dist.log_pdf(chi_eff)
         
    def pdf(self,chi_eff,z):
        xp = get_module_array(chi_eff)
        return xp.exp(self.log_pdf(chi_eff,z))



# ------------------------------------ #
#          B-splines models            #
# ------------------------------------ #

class massprior_logBspline(pm_prob):
    def __init__(self, n_basis, degree, spacing, spline_variable):
        self.n_basis = n_basis
        self.degree = degree
        if spacing in {'uniform', 'lin'}: self.spacing = 'lin'
        elif spacing == 'log':            self.spacing = 'log'
        else: raise KeyError("unknown splines spacing option. Choose from uniform, lin, log.")
        if spline_variable in {'uniform', 'lin'}: self.spline_variable = 'lin'
        elif spline_variable == 'log':            self.spline_variable = 'log'
        else: raise KeyError("unknown splines variable option. Choose from uniform, lin, log.")
        self.coeffs_parameters = [f'c{i}' for i in range(1, self.n_basis-1)]
        self.population_parameters = ['mmin', 'mmax'] + self.coeffs_parameters

    def update(self, **kwargs):
        coeffs = {c:kwargs[c] for c in self.coeffs_parameters}
        self.prior = logBspline(
            minval=kwargs['mmin'],
            maxval=kwargs['mmax'],
            n_basis=self.n_basis, 
            degree=self.degree, 
            spacing=self.spacing,
            spline_variable=self.spline_variable,
            **coeffs
        )


class massprior_PowerLawlogBspline(pm_prob):
    def __init__(self, n_basis, degree, spacing, spline_variable):
        self.n_basis = n_basis
        self.degree = degree
        if spacing in {'uniform', 'lin'}: self.spacing = 'lin'
        elif spacing == 'log':            self.spacing = 'log'
        else: raise KeyError("unknown splines spacing option. Choose from uniform, lin, log.")
        if spline_variable in {'uniform', 'lin'}: self.spline_variable = 'lin'
        elif spline_variable == 'log':            self.spline_variable = 'log'
        else: raise KeyError("unknown splines variable option. Choose from uniform, lin, log.")
        self.coeffs_parameters = [f'c{i}' for i in range(1, self.n_basis-1)]
        self.population_parameters = ['mmin', 'mmax', 'alpha'] + self.coeffs_parameters

    def update(self, **kwargs):
        coeffs = {c:kwargs[c] for c in self.coeffs_parameters}
        self.prior = PowerLaw_logBspline(
            minval = kwargs['mmin'],
            maxval = kwargs['mmax'],
            alpha  = - kwargs['alpha'],
            n_basis=self.n_basis, 
            degree=self.degree, 
            spacing=self.spacing,
            spline_variable=self.spline_variable,
            **coeffs
        )







class massratio_EvolvingGaussian():
    '''
    Definition
    ----------
    Conditional mass-ratio distribution
    p(q | m1) = TruncatedGaussian in [0, 1]
    
    Parameters
    ----------
    mu = mu_0 (m1/m0)^a
    sigma = sigma_0 (m1/m0)^b
    TruncatedGaussiaparameters
    '''
    def __init__(self, mw):
        self.population_parameters= ['mu_0', 'm0', 'a','b','sigma_0']
               
    def update(self,**kwargs):
        
        self.mu_0, self.m0 = kwargs['mu_0'], kwargs['m0']
        self.sigma_0, self.a, self.b = kwargs['sigma_0'], kwargs['a'], kwargs['b']
        
    def log_pdf(self, q, mass_1_source):

        mass_2_source = q*mass_1_source
                
        mu = self.mu_0 * ((mass_1_source/self.m0)**self.a)
        sigma = self.sigma_0 * ((mass_1_source/self.m0)**self.b)

        log_out = log_truncnorm_pdf(q, mu, sigma, 0.0, 1.0)
    
        return log_out
        
    def pdf(self,mass_1_source,mass_2_source):
        xp = get_module_array(mass_1_source)
        return xp.exp(self.log_pdf(mass_1_source,mass_2_source))







class spinprior_Gaussian_to_Gaussian_windowGaussian():
    '''
    Wrapper for the evolving spin model chi1-m1
    p(chi_1 | m_1) = w(m_1) * TruncatedGaussian_lowpspinpopulation + (1-w(m_1)) * TruncatedGaussian_highspinpopulation
    with w(m_1) = - sigma_t * sqrt(2pi) * (Guassian window function) + 1
    '''
    def __init__(self, mw):
        self.population_parameters= ['mu_chi_low_1', 'sigma_chi_low_1', 'mu_chi_high_1','sigma_chi_high_1', 'mu_t', 'sigma_t', 'mu_chi_2', 'sigma_chi_2', 'csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2', 'mass_1_source']
               
    def update(self,**kwargs):
        
        self.mu_chi_low_1, self.sigma_chi_low_1 = kwargs['mu_chi_low_1'], kwargs['sigma_chi_low_1']
        self.mu_chi_high_1, self.sigma_chi_high_1 = kwargs['mu_chi_high_1'], kwargs['sigma_chi_high_1']
        self.mu_t, self.sigma_t = kwargs['mu_t'], kwargs['sigma_t']
        self.mu_chi_2, self.sigma_chi_2 = kwargs['mu_chi_2'], kwargs['sigma_chi_2']
        self.csi_spin = kwargs['csi_spin']
        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)
        
    def calculate_gaussian(self, x):
        '''
        only for window function
        '''
        exp = -0.5 * ((x - self.mu_t) / self.sigma_t) ** 2
        return (1 / (self.sigma_t * np.sqrt(2 * np.pi))) * np.exp(exp)

        
    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2, mass_1_source,**kwargs):
        
        xp = get_module_array(chi_1)

        log_angular_part = xp.logaddexp(xp.log1p(-self.csi_spin)+xp.log(0.25),
                                    xp.log(self.csi_spin)+self.aligned_pdf.log_pdf(cos_t_1)+self.aligned_pdf.log_pdf(cos_t_2))
        
        gaussian = self.calculate_gaussian(mass_1_source)
        wz = -(gaussian * (xp.sqrt(2*np.pi) * self.sigma_t)) + 1
        
        pdf_low = xp.exp(log_truncnorm_pdf(chi_1, self.mu_chi_low_1, self.sigma_chi_low_1, 0.0, 1.0))
        pdf_high = xp.exp(log_truncnorm_pdf(chi_1, self.mu_chi_high_1, self.sigma_chi_high_1, 0.0, 1.0))
        log_pdf_1 = xp.log(wz*pdf_low+(1-wz)*pdf_high)

        log_pdf_2 = log_truncnorm_pdf(chi_2, self.mu_chi_2, self.sigma_chi_2, 0.0, 1.0)

        return log_pdf_1 + log_pdf_2 + log_angular_part
        
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source,**kwargs):
        xp = get_module_array(mass_1_source)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2,mass_1_source))





