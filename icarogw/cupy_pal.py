import numpy as np
import scipy as sn

CUPY_LOADED = False

def enable_cupy():
    try:
        import cupy as cp
        import cupyx as _cupyx
        from cupyx.scipy import interpolate
        from cupy import _core

        global cp
        global _cupyx
        global interpolate
        global _core
        global CUPY_LOADED
        
        CUPY_LOADED = True
        print('cupy imported')
    except ImportError:
        print('Error in importing cupy')

def disable_cupy():
    global CUPY_LOADED
    CUPY_LOADED = False
    
def is_there_cupy():
    return CUPY_LOADED

try:
    import config
    print('Config file loaded')
    if config.CUPY:
        enable_cupy()
    else:
        print('Config does not allow CUPY')        
except ImportError:
    print('Config not imported, automatically decides between Numpy and Cupy')
    enable_cupy()


_cupy_functions = {}

def check_bounds_1D(x,minval,maxval):
    if CUPY_LOADED:
        kernel = _cupy_functions.get('check_bounds_1D', None)
        if kernel is None:
            @cp.fuse()
            def check_bounds_sub_1D(x,minval,maxval):
                return (x<minval) | (x>maxval)
            _cupy_functions['check_bounds_1D']=check_bounds_sub_1D
            kernel = check_bounds_sub_1D
        return kernel(x,minval,maxval)
    else:
        return (x<minval) | (x>maxval)

def check_bounds_2D(x1,x2,y):
    if CUPY_LOADED:
        kernel = _cupy_functions.get('check_bounds_2D', None)
        if kernel is None:
            @cp.fuse()
            def check_bounds_sub_2D(x1,x2,y):
                xp = get_module_array(x1)
                return (x1<x2) | xp.isnan(y)
            _cupy_functions['check_bounds_2D']=check_bounds_sub_2D
            kernel = check_bounds_sub_2D
        return kernel(x1,x2,y)
    else:
        return (x1<x2) | np.isnan(y)
    
# LVK Reviewed
def cp2np(array):
    '''Cast any array to numpy'''
    if CUPY_LOADED:
        return cp.asnumpy(array)
    else:
        return array
        
# LVK Reviewed
def np2cp(array):
    '''Cast any array to cupy'''
    if CUPY_LOADED:
        return cp.asarray(array)
    else:
        return array

def get_module_array(array):
    if CUPY_LOADED:
        return cp.get_array_module(array)
    else:
        return np

def get_module_array_scipy(array):
    if CUPY_LOADED:
        return _cupyx.scipy.get_array_module(array)
    else:
        return sn

def iscupy(array):
    if CUPY_LOADED:
        return isinstance(array, (cp.ndarray, _cupyx.scipy.sparse.spmatrix,
                        _core.fusion._FusionVarArray,
                        _core.new_fusion._ArrayProxy))
    else:
        return False

def _detect_xp_and_dtype(array_example):
    """
    Determine xp (numpy or cupy) and a sensible default dtype.

    This uses the existing get_module_array(array) helper to pick the array
    module from a representative array-like `array_example`. If `dtype` is
    provided it is returned unchanged; otherwise the default is:
      - float32 when using CuPy (GPU)
      - float64 when using NumPy (CPU)

    Signature changed to accept a representative array (or value) from which
    get_module_array can infer the correct module.
    """
    xp = get_module_array(array_example)

    # Heuristic: prefer float32 for CuPy, float64 for NumPy.
    # Use getattr to be safe if module doesn't expose float32/float64.
    is_cupy = getattr(xp, "__name__", "") == "cupy" or hasattr(xp, "cuda")
    if is_cupy:
        chosen = getattr(xp, "float32", xp.float32)
    else:
        chosen = getattr(xp, "float64", xp.float64)
    return xp, chosen
