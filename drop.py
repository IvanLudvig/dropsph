import os.path

import numpy as np

# PyZoltan imports
from cyarray.api import LongArray

# PySPH imports
from pysph.base.utils import get_particle_array_wcsph as gpa
from pysph.base.kernels import Gaussian, WendlandQuintic, CubicSpline, QuinticSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import WCSPHStep

# the eqations
from pysph.sph.equation import Group

# Equations for the standard WCSPH formulation and dynamic boundary
# conditions defined in REF3
from pysph.sph.wc.basic import TaitEOS, TaitEOSHGCorrection, MomentumEquation
from pysph.sph.basic_equations import XSPHCorrection, \
    MonaghanArtificialViscosity
# from pysph.tools.gmsh import Gmsh
import gmsh

import matplotlib.pyplot as plt

# domain and reference values
Lx = 2.0
Ly = 1.0
H = 0.9
gy = -1.0
Vmax = np.sqrt(abs(gy) * H)
c0 = 10 * Vmax
rho0 = 1000.0
p0 = c0 * c0 * rho0
gamma = 1.0

# Reynolds number and kinematic viscosity
Re = 100
nu = Vmax * Ly / Re

# Numerical setup
nx = 512
dx = Lx / nx
ghost_extent = 5.5 * dx
hdx = 1.2

# adaptive time steps
h0 = hdx * dx
dt_cfl = 0.25 * h0 / (c0 + Vmax)
dt_viscous = 0.125 * h0**2 / nu
dt_force = 0.25 * np.sqrt(h0 / abs(gy))

tdamp = 1.0
tf = 5.0
dt = 0.75 * min(dt_cfl, dt_viscous, dt_force)
output_at_times = np.arange(0.25, 5.5, 0.25)


def damping_factor(t, tdamp):
    if t < tdamp:
        return 0.5 * (np.sin((-0.5 + t / tdamp) * np.pi) + 1.0)
    else:
        return 1.0



dH = 1.0

class Drop(Application):
    def create_particles(self):
        gmsh.initialize()
        gmsh.open('wheel.msh')
        nodeTags, nodesCoord, parametricCoord = gmsh.model.mesh.getNodes()

        xf = nodesCoord[0::3]/600
        yf = nodesCoord[2::3]/600

        yf = yf + abs(min(yf)) + 0.2

        xf = xf[0::20]
        yf = yf[0::20]

        _x = np.arange(-ghost_extent + min(xf) - dH, max(xf) + ghost_extent + dH, dx)
        _y = np.arange(0, max(yf) + ghost_extent + dH, dx)
        x, y = np.meshgrid(_x, _y)
        x = x.ravel()
        y = y.ravel()

        px = []
        py = []
        for i in range(x.size):
            if (y[i]<(2*dx)) or (x[i]>=(max(xf) + ghost_extent + dH - 2*dx)) or (x[i]<=(-ghost_extent + min(xf) - dH + 2*dx)):
                indices.append(i)
                px.append(x[i])
                py.append(y[i])

        plt.scatter(px, py)
        plt.scatter(xf, yf)
        plt.show()

        solid = gpa(name='solid', x=px, y=py)

        fluid = gpa(name='fluid', x=xf, y=yf)
        fluid.set_name('fluid')

        print("Tank :: nfluid = %d, nsolid=%d, dt = %g" % (
            fluid.get_number_of_particles(),
            solid.get_number_of_particles(), dt))


        # particle volume
        fluid.add_property('V')
        solid.add_property('V')

        # kernel sum term for boundary particles
        solid.add_property('wij')

        # advection velocities and accelerations
        for name in ('auhat', 'avhat', 'awhat'):
            fluid.add_property(name)

        fluid.rho[:] = rho0
        solid.rho[:] = rho0

        fluid.rho0[:] = rho0
        solid.rho0[:] = rho0

        # mass is set to get the reference density of rho0
        volume = dx * dx

        # volume is set as dx^2
        fluid.V[:] = 1. / volume
        solid.V[:] = 1. / volume

        fluid.m[:] = volume * rho0
        solid.m[:] = volume * rho0

        # smoothing lengths
        fluid.h[:] = hdx * dx
        solid.h[:] = hdx * dx # maybe try * 20 ?

        # return the particle list
        return [fluid, solid]

    def create_solver(self):
        kernel = QuinticSpline(dim=2)

        integrator = PECIntegrator(fluid=WCSPHStep())

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
                VolumeFromMassDensity(dest='fluid', sources=None)
            ], ),

            # Equation of state is typically the Tait EOS with a suitable
            # exponent gamma. The solid phase is treated just as a fluid and
            # the pressure and density operations is updated for this as well.
            Group(equations=[
                TaitEOS(
                    dest='fluid',
                    sources=None,
                    rho0=rho0,
                    c0=c0,
                    gamma=gamma),
                TaitEOS(
                    dest='solid',
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
                ContinuityEquation(
                    dest='fluid', sources=[
                        'fluid', 'solid']),
                ContinuityEquation(dest='solid', sources=['fluid']),

                # Pressure gradient with acceleration damping.
                MomentumEquationPressureGradient(
                    dest='fluid', sources=['fluid', 'solid'], pb=0.0, gy=gy,
                    tdamp=tdamp),

                # artificial viscosity for stability
                MomentumEquationArtificialViscosity(
                    dest='fluid', sources=['fluid', 'solid'], alpha=0.25, c0=c0),

                # Position step with XSPH
                XSPHCorrection(dest='fluid', sources=['fluid'], eps=0.5)

            ]),
        ]


    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.tools.interpolator import Interpolator
        from pysph.solver.utils import iter_output
        files = self.output_files
        y = np.linspace(0, 0.9, 20)
        x = np.ones_like(y)
        interp = None
        t, p, p_ex = [], [], []
        for sd, arrays in iter_output(files):
            fluid, solid = arrays['fluid'], arrays['solid']
            if interp is None:
                interp = Interpolator([fluid, solid], x=x, y=y)
            else:
                interp.update_particle_arrays([fluid, solid])
            t.append(sd['t'])
            p.append(interp.interpolate('p'))
            g = 1.0 * damping_factor(t[-1], tdamp)
            p_ex.append(abs(rho0 * H * g))

        t, p, p_ex = list(map(np.asarray, (t, p, p_ex)))
        res = os.path.join(self.output_dir, 'results.npz')
        np.savez(res, t=t, p=p, p_ex=p_ex, y=y)

        import matplotlib
        matplotlib.use('Agg')

        pmax = abs(0.9 * rho0 * gy)

        from matplotlib import pyplot as plt
        plt.plot(t, p[:, 0] / pmax, 'o-')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$p$')
        fig = os.path.join(self.output_dir, 'p_bottom.png')
        plt.savefig(fig, dpi=300)

        plt.clf()
        output_at = np.arange(0.25, 5.5, 0.25)
        count = 0
        for i in range(len(t)):
            if abs(t[i] - output_at[count]) < 1e-8:
                plt.plot(y, p[i] / pmax, 'o', label='t=%.2f' % t[i])
                plt.plot(y, p_ex[i] * (H - y) / (H * pmax), 'k-')
                count += 1
        plt.xlabel('$y$')
        plt.ylabel('$p$')
        plt.legend()
        fig = os.path.join(self.output_dir, 'p_vs_y.png')
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = Drop()
    app.run()
    app.post_process(app.info_filename)
