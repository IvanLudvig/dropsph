import os.path
import numpy as np
from cyarray.api import LongArray
from pysph.base.utils import get_particle_array_wcsph as gpa
from pysph.base.kernels import Gaussian, WendlandQuintic, CubicSpline, QuinticSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import WCSPHStep
from pysph.sph.equation import Group
from pysph.sph.wc.transport_velocity import VolumeFromMassDensity, ContinuityEquation, MomentumEquationPressureGradient, MomentumEquationArtificialViscosity
from pysph.sph.wc.basic import TaitEOS
from pysph.sph.basic_equations import XSPHCorrection
import gmsh
import matplotlib.pyplot as plt

# domain and reference values
height = 1.0
gy = -9.8
Vmax = np.sqrt(abs(gy) * height)
c0 = 10 * Vmax
rho0 = 1000.0
gamma = 1.0

# Numerical setup
dx = 1 / 512
hdx = 1.2

# adaptive time steps
h0 = hdx * dx
dt_cfl = 0.25 * h0 / (c0 + Vmax)
dt_force = 0.25 * np.sqrt(h0 / abs(gy))

tf = 5.0
dt = 0.5 * min(dt_cfl, dt_force)
output_at_times = np.arange(0.25, 5.5, 0.25)

wall_thickness = dx


class Drop(Application):
    def create_particles(self):
        gmsh.initialize()
        gmsh.open('wheel.msh')
        nodeTags, nodesCoord, parametricCoord = gmsh.model.mesh.getNodes()

        liquid_x = nodesCoord[0::3]/600
        liquid_y = nodesCoord[2::3]/600
        liquid_y = liquid_y + abs(min(liquid_y)) + height

        liquid_x = liquid_x[0::100]
        liquid_y = liquid_y[0::100]

        min_x = min(liquid_x) - 500*dx
        max_x = max(liquid_x) + 500*dx

        min_y = 0
        max_y = max(liquid_y) + 500*dx

        _x = np.arange(min_x, max_x, dx)
        _y = np.arange(min_y, max_y, dx)
        x, y = np.meshgrid(_x, _y)
        x = x.ravel()
        y = y.ravel()

        walls_x = []
        walls_y = []
        for i in range(x.size):
            if (y[i] < (min_y + wall_thickness)) or (x[i] >= (max_x - wall_thickness)) or (x[i] <= (min_x + wall_thickness)):
                walls_x.append(x[i])
                walls_y.append(y[i])

        plt.scatter(liquid_x, liquid_y)
        plt.scatter(walls_x, walls_y)
        plt.gca().set_aspect('equal')
        plt.show()

        liquid = gpa(name='liquid', x=liquid_x, y=liquid_y)
        walls = gpa(name='walls', x=walls_x, y=walls_y)

        print("Tank :: nliquid = %d, nwalls=%d, dt = %g" % (
            liquid.get_number_of_particles(),
            walls.get_number_of_particles(), dt))

        # particle volume
        liquid.add_property('V')
        walls.add_property('V')

        # kernel sum term for boundary particles
        walls.add_property('wij')

        # advection velocities and accelerations
        for name in ('auhat', 'avhat', 'awhat'):
            liquid.add_property(name)

        liquid.rho[:] = rho0
        walls.rho[:] = rho0

        liquid.rho0[:] = rho0
        walls.rho0[:] = rho0

        # mass is set to get the reference density of rho0
        volume = dx * dx

        # volume is set as dx^2
        liquid.V[:] = 1. / volume
        walls.V[:] = 1. / volume

        liquid.m[:] = volume * rho0
        walls.m[:] = volume * rho0

        # smoothing lengths
        liquid.h[:] = hdx * dx
        walls.h[:] = hdx * dx

        # return the particle list
        return [liquid, walls]

    def create_solver(self):
        kernel = QuinticSpline(dim=2)

        integrator = PECIntegrator(liquid=WCSPHStep())

        solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                        tf=tf, dt=dt, output_at_times=output_at_times)
        return solver

    def create_equations(self):
        # Formulation for REF3
        return [
            # For the multi-phase formulation, we require an estimate of the
            # particle volume. This can be either defined from the particle
            # number density or simply as the ratio of mass to density.
            Group(equations=[
                VolumeFromMassDensity(dest='liquid', sources=None)
            ], ),

            # Equation of state is typically the Tait EOS with a suitable
            # exponent gamma. The solid phase is treated just as a fluid and
            # the pressure and density operations is updated for this as well.
            Group(equations=[
                TaitEOS(
                    dest='liquid',
                    sources=None,
                    rho0=rho0,
                    c0=c0,
                    gamma=gamma),
                TaitEOS(
                    dest='walls',
                    sources=None,
                    rho0=rho0,
                    c0=c0,
                    gamma=gamma),
            ], ),

            # Main acceleration block. The boundary conditions are imposed by
            # peforming the continuity equation and gradient of pressure
            # calculation on the solid phase, taking contributions from the
            # fluid phase
            Group(equations=[

                # Continuity equation
                ContinuityEquation(dest='liquid', sources=['liquid', 'walls']),
                ContinuityEquation(dest='walls', sources=['liquid']),

                # Pressure gradient with acceleration damping.
                MomentumEquationPressureGradient(
                    dest='liquid', sources=['liquid', 'walls'], pb=0.0, gy=gy),

                # artificial viscosity for stability
                MomentumEquationArtificialViscosity(
                    dest='liquid', sources=['liquid', 'walls'], alpha=0.25, c0=c0),

                # Position step with XSPH
                XSPHCorrection(dest='liquid', sources=['liquid'], eps=0.5)
            ]),
        ]


if __name__ == '__main__':
    app = Drop()
    app.run()
