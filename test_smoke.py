from surface_integral import utils
import numpy as np

def param_map(u, v):
    # plane: x=u, y=v, z=0
    return (u, v, np.zeros_like(u))

def f_callable(x, y, z):
    return np.ones_like(x)

res = utils.numeric_surface_integral(f_callable, ((0.0,1.0),(0.0,1.0)), param_map, grid=(50,50))
print('numeric_surface_integral (plane, f=1) =', res)

def P(x,y,z):
    return x
def Q(x,y,z):
    return y
def R(x,y,z):
    return z

flux = utils.numeric_flux_integral(P,Q,R, ((0.0,1.0),(0.0,1.0)), param_map, grid=(50,50))
print('numeric_flux_integral (plane with field [x,y,z]) =', flux)
