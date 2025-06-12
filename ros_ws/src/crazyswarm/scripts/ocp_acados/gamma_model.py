#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

from acados_template import AcadosModel
from casadi import SX,MX, vertcat, sin, cos

def export_gamma_ode_model(number_of_drones) -> AcadosModel:

    model_name = 'gamma_ode'

    # For 3 drones - [gamma, gamma_dot, gamma_2, gamma_3, l_12, l_13, l_23, a_1, a_2, G_11, G_12, G_13, error]
    gammas =  SX.sym('gammas', number_of_drones + 1) # gamma_{i}, dot_gamma_{i}, gamma_{-i}
    l = SX.sym('l', int(((number_of_drones**2) - number_of_drones)/2)) # l_{ij}
    a = SX.sym('a', number_of_drones - 1) # a_{i}
    g = SX.sym('g', number_of_drones) # G_11
    e = SX.sym('e') # error

    x = vertcat(gammas,l, a, g,e)

    U = SX.sym('U') # gamma_dot_dot - direct control over the acceleration
    u = vertcat(U)

    gammas_dot = SX.sym('gammas_dot', number_of_drones + 1) # gamma_dot
    l_dot = SX.sym('l_dot', int(((number_of_drones**2) - number_of_drones)/2)) # l_dot
    a_dot = SX.sym('a_dot', number_of_drones - 1) # a_dot
    g_dot = SX.sym('g_dot', number_of_drones) # G_dot
    e_dot = SX.sym('e_dot') # error_dot

    xdot = vertcat(gammas_dot, l_dot, a_dot, g_dot, e_dot)
    p = []
    
    # Dynamics
    # Explicit Runge-Kutta 4 integrator (erk) - dot{x} = f(x, u ,p)
    f_expl = vertcat(x[1], U, a, SX.zeros(int(((number_of_drones**2) - number_of_drones)/2) + 2*number_of_drones)) # dx1 = x2, dx2 = U
    f_impl = xdot - f_expl

    # Creat indipendent variables for the model
    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    model.p = p

    return model

