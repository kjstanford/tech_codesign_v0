import sympy

def custom_sech(x):
    return 1 / sympy.cosh(x)

def custom_coth(x):
    return 1 / sympy.tanh(x)
