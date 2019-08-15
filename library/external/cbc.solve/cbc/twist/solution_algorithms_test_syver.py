__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__ = "GNU GPL Version 3 or any later version"

# Modified by Anders Logg, 2010
# Last changed: 2012-05-01

from dolfin import *
from cbc.common import *
from cbc.common.utils import *
from cbc.twist.kinematics import Grad, DeformationGradient
from sys import exit
from numpy import array, loadtxt


def default_parameters():
    "Return default solver parameters."
    p = Parameters("solver_parameters")
    p.add("plot_solution", False)
    p.add("save_solution", True)
    p.add("store_solution_data", False)
    p.add("element_degree", 1)
    return p


def Make_D_function_spaces(space, condition_types):
    D_function_spaces = []
    for condition in condition_types:
        if condition == 'xyz':
            D_function_spaces += [space]
        elif condition == 'x':
            D_function_spaces += [space.sub(0)]
        elif condition == 'y':
            D_function_spaces += [space.sub(1)]
        elif condition == 'z':
            D_function_spaces += [space.sub(2)]
    return D_function_spaces


class StaticMomentumBalanceSolver(CBCSolver):
    "Solves the static balance of linear momentum"

    def __init__(self, problem, parameters):
        """Initialise the static momentum balance solver"""

        # Get problem parameters
        mesh = problem.mesh()

        # Define function spaces
        element_degree = parameters["element_degree"]
        vector = VectorFunctionSpace(mesh, "CG", element_degree)

        # Create boundary conditions
        dirichlet_function_spaces = Make_D_function_spaces(vector, problem.dirichlet_function_spaces())
        bcu = create_dirichlet_conditions(problem.dirichlet_values(),
                                          problem.dirichlet_boundaries(),
                                          dirichlet_function_spaces)

        # Define fields
        # Test and trial functions
        v = TestFunction(vector)
        u = Function(vector)
        du = TrialFunction(vector)

        # Residual stresses
        pre_P = problem.prestress()
        if pre_P == []:
            pre_P = Constant(((0.0,)*vector.mesh().geometry().dim(),)*vector.mesh().geometry().dim())

        # Driving forces
        B = problem.body_force()

        # If no body forces are specified, assume it is 0
        if B == []:
            B = Constant((0,)*vector.mesh().geometry().dim())

        # First Piola-Kirchhoff stress tensor based on the material
        # model
        P = problem.first_pk_stress(u)

        # The variational form corresponding to hyperelasticity
        L = inner(P+pre_P, Grad(v))*dx - inner(B, v)*dx

        # Add contributions to the form from the Neumann boundary
        # conditions

        # Get Neumann boundary conditions on the stress
        neumann_conditions = problem.neumann_conditions()
        # If no Neumann conditions are specified, assume it is 0
        if neumann_conditions == []:
            neumann_conditions = Constant((0,)*vector.mesh().geometry().dim())
        neumann_boundaries = problem.neumann_boundaries()
        boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundary.set_all(len(neumann_boundaries) + 1)
        #dsb = ds[boundary]
        dsb = ds(subdomain_data=boundary)
        for (i, neumann_boundary) in enumerate(neumann_boundaries):
            compiled_boundary = CompiledSubDomain(neumann_boundary)
            compiled_boundary.mark(boundary, i)
            L = L - inner(neumann_conditions[i], v)*dsb(i)

        a = derivative(L, u, du)

        # Setup problem
        problem = NonlinearVariationalProblem(L, u, bcu, a)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-9
        solver.parameters["newton_solver"]["relative_tolerance"] = 1e-9
        solver.parameters["newton_solver"]["maximum_iterations"] = 100

        # Store parameters
        self.parameters = parameters

        # Store variables needed for time-stepping
        # FIXME: Figure out why I am needed
        self.mesh = mesh
        self.equation = solver
        self.u = u

    def solve(self):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Solve problem
        self.equation.solve()

        # Plot solution
        if self.parameters["plot_solution"]:
            plot(self.u, title="Displacement", mode="displacement", rescale=True)
            interactive()

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            displacement_file = File("displacement.pvd")
            displacement_file << self.u

        # Store solution data
        if self.parameters["store_solution_data"]:
            displacement_series = TimeSeries("displacement")
            displacement_series.store(self.u.vector(), 0.0)

        return self.u


class MomentumBalanceSolver(CBCSolver):
    "Solves the quasistatic/dynamic balance of linear momentum"

    def __init__(self, problem, parameters):
        """Initialise the momentum balance solver"""

        # Get problem parameters
        mesh = problem.mesh()
        dt, t_range = timestep_range_cfl(problem, mesh)
        end_time = problem.end_time()

        # Define function spaces
        element_degree = parameters["element_degree"]
        scalar = FunctionSpace(mesh, "CG", element_degree)
        vector = VectorFunctionSpace(mesh, "CG", element_degree)

        # Get initial conditions
        u0, v0 = problem.initial_conditions()

        # If no initial conditions are specified, assume they are 0
        if u0 == []:
            u0 = Constant((0,)*vector.mesh().geometry().dim())
        if v0 == []:
            v0 = Constant((0,)*vector.mesh().geometry().dim())

        # If either are text strings, assume those are file names and
        # load conditions from those files
        if isinstance(u0, str):
            info("Loading initial displacement from file.")
            file_name = u0
            u0 = Function(vector)
            u0.vector()[:] = loadtxt(file_name)[:]
        if isinstance(v0, str):
            info("Loading initial velocity from file.")
            file_name = v0
            v0 = Function(vector)
            v0.vector()[:] = loadtxt(file_name)[:]


        # Define function spaces or subspaces for boundary conditions


        # Create boundary conditions
        dirichlet_values = problem.dirichlet_values()
        dirichlet_function_spaces = Make_D_function_spaces(vector, problem.dirichlet_function_spaces())
        bcu = create_dirichlet_conditions(dirichlet_values,
                                          problem.dirichlet_boundaries(),
                                          dirichlet_function_spaces)

        # Define fields
        # Test and trial functions
        v = TestFunction(vector)
        u1 = Function(vector)
        v1 = Function(vector)
        a1 = Function(vector)
        du = TrialFunction(vector)

        # Initial displacement and velocity
        u0 = interpolate(u0, vector)
        v0 = interpolate(v0, vector)
        u1.assign(u0)
        v1.assign(v0)

        # Residual stresses
        pre_P = problem.prestress()

        if pre_P == []:
            # as_matrix(0.0)
            pre_P = Constant(((0.0,)*vector.mesh().geometry().dim(),)*vector.mesh().geometry().dim())

        # Driving forces
        B = problem.body_force()

        # If no body forces are specified, assume it is 0
        if B == []:
            B = Constant((0,)*vector.mesh().geometry().dim())

        # Parameters pertinent to (HHT) time integration
        # alpha = 1.0
        beta = 0.25
        gamma = 0.5

        """Initial acceleration problem is defined, but never used.
        # Determine initial acceleration
        a0 = TrialFunction(vector)
        P0 = problem.first_pk_stress(u0)
        a_accn = inner(a0, v)*dx
        L_accn = - inner(P0+pre_P, Grad(v))*dx + inner(B, v)*dx

        # Add contributions to the form from the Neumann boundary
        # conditions

        # # Get Neumann boundary conditions on the stress
        neumann_conditions = problem.neumann_conditions()
        # If no Neumann conditions are specified, assume it is 0
        if neumann_conditions == []:
            neumann_conditions = Constant((0,)*vector.mesh().geometry().dim())
        neumann_boundaries = problem.neumann_boundaries()
        boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundary.set_all(len(neumann_boundaries) + 1)
        # dsb = ds[boundary]
        dsb = ds(subdomain_data=boundary)
        for (i, neumann_boundary) in enumerate(neumann_boundaries):
            compiled_boundary = CompiledSubDomain(neumann_boundary)
            compiled_boundary.mark(boundary, i)
            L_accn = L_accn + inner(neumann_conditions[i], v)*dsb(i)

        #problem_accn = LinearVariationalProblem(a_accn, L_accn, a0)
        #solver_accn = LinearVariationalSolver(problem_accn)
        # solver_accn.solve()
        # problem_accn = VariationalProblem(a_accn, L_accn)
        # a0 = problem_accn.solve()

        # MODIF BY ALBAN 10/03/2019, the next 2 lines where missing previously and
        # a0 was still defined as a TrialFunction, which led to arity error when
        # solving the nonlinear problem. If a0 is not computed, it has to be set
        # to a Function since it appear in the variational form L
        """
        a0 = Function(vector)
        #solve(a_accn == L_accn, a0)


        k = Constant(dt)
        a1 = a0*(1.0 - 1.0/(2*beta)) - (u0 - u1 + k*v0)/(beta*k**2)

        # Get reference density
        rho0 = problem.reference_density()

        # If no reference density is specified, assume it is 1.0
        if rho0 == []:
            rho0 = Constant(1.0)

        density_type = str(rho0.__class__)
        if not ("dolfin" in density_type):
            info("Converting given density to a DOLFIN Constant.")
            rho0 = Constant(rho0)

        # Piola-Kirchhoff stress tensor based on the material model
        P = problem.first_pk_stress(u1)

        # FIXME: A general version of the trick below is what should
        # be used instead. The commentend-out lines only work well for
        # quadratically nonlinear models, e.g. St. Venant Kirchhoff.

        # S0 = problem.second_pk_stress(u0)
        # S1 = problem.second_pk_stress(u1)
        # Sm = 0.5*(S0 + S1)
        # Fm = DeformationGradient(0.5*(u0 + u1))
        # P  = Fm*Sm

        # The variational form corresponding to hyperelasticity
        is_dynamic_expression = Constant(int(problem.is_dynamic()))
        L = (is_dynamic_expression*rho0*inner(a1, v)*dx
             + inner(P+pre_P, Grad(v))*dx - inner(B, v)*dx)

        # Add contributions to the form from the Neumann boundary
        # conditions

        # Get Neumann boundary conditions on the stress
        neumann_conditions = problem.neumann_conditions()
        neumann_boundaries = problem.neumann_boundaries()

        boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundary.set_all(len(neumann_boundaries) + 1)

        # dsb = ds[boundary]
        dsb = ds(subdomain_data=boundary)
        for (i, neumann_boundary) in enumerate(neumann_boundaries):
            info("Applying Neumann boundary condition.")
            info(str(neumann_boundary))
            compiled_boundary = CompiledSubDomain(neumann_boundary)
            compiled_boundary.mark(boundary, i)
            L = L - inner(neumann_conditions[i], v)*dsb(i)

        a = derivative(L, u1, du)

        # Store variables needed for time-stepping
        self.dt = dt
        self.k = k
        self.t_range = t_range
        self.end_time = end_time
        self.a = a
        self.L = L
        self.bcu = bcu
        self.u0 = u0
        self.v0 = v0
        self.a0 = a0
        self.u1 = u1
        self.v1 = v1
        self.a1 = a1
        self.k = k
        self.beta = beta
        self.gamma = gamma
        self.vector = vector
        self.B = B
        self.dirichlet_values = dirichlet_values
        self.neumann_conditions = neumann_conditions

        # FIXME: Figure out why I am needed
        self.mesh = mesh
        self.t = 0

        # Empty file handlers / time series
        self.displacement_file = None
        self.velocity_file = None
        self.displacement_series = None
        self.velocity_series = None

        # Store parameters
        self.parameters = parameters

    def solve(self):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Time loop
        for t in self.t_range:
            info("Solving the problem at time t = " + str(self.t))
            self.step(self.dt)
            self.update()

        if self.parameters["plot_solution"]:
            interactive()

        return self.u1

    def step(self, dt):
        """Setup and solve the problem at the current time step"""

        # Update time step
        self.dt = dt
        self.k.assign(dt)

        # FIXME: Setup all stuff in the constructor and call assemble instead of VariationalProblem
        problem = NonlinearVariationalProblem(self.L, self.u1, self.bcu, self.a)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-9
        solver.parameters["newton_solver"]["relative_tolerance"] = 1e-9
        solver.parameters["newton_solver"]["maximum_iterations"] = 100
        solver.solve()

        return self.u1

    def update(self):
        """Update problem at time t"""

        # Compute new accelerations and velocities based on new
        # displacement
        a1 = self.a0*(1.0 - 1.0/(2*self.beta)) \
            - (self.u0 - self.u1 + self.k*self.v0)/(self.beta*self.k**2)
        self.a1 = project(a1, self.vector)
        v1 = self.v0 + self.k*((1 - self.gamma)*self.a1 + self.gamma*self.a0)
        self.v1 = project(v1, self.vector)

        # Propagate the displacements, velocities and accelerations
        self.u0.assign(self.u1)
        self.v0.assign(self.v1)
        self.a0.assign(self.a1)

        # Plot solution
        if self.parameters["plot_solution"]:
            plot(self.u0, title="Displacement", mode="displacement", rescale=True)

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            if self.displacement_file is None:
                self.displacement_file = File("displacement.pvd")
            if self.velocity_file is None:
                self.velocity_file = File("velocity.pvd")
            self.displacement_file << self.u0
            self.velocity_file << self.v0

        # Store solution data
        if self.parameters["store_solution_data"]:
            if self.displacement_series is None:
                self.displacement_series = TimeSeries("displacement")
            if self.velocity_series is None:
                self.velocity_series = TimeSeries("velocity")
            self.displacement_series.store(self.u0.vector(), self.t)
            self.velocity_series.store(self.v0.vector(), self.t)

        # Move to next time step
        self.t = self.t + self.dt

        # Inform time-dependent functions of new time
        for bc in self.dirichlet_values:
            if isinstance(bc, Expression):
                bc.t = self.t
        for bc in self.neumann_conditions:
            bc.t = self.t
        self.B.t = self.t

    def solution(self):
        "Return current solution values"
        return self.u1


class CG1MomentumBalanceSolver(CBCSolver):
    """Solves the dynamic balance of linear momentum using a CG1
    time-stepping scheme"""

    def __init__(self, problem, parameters):
        """Initialise the momentum balance solver"""

        # Get problem parameters
        mesh = problem.mesh()
        dt, t_range = timestep_range_cfl(problem, mesh)
        end_time = problem.end_time()
        info("Using time step dt = %g" % dt)

        # Define function spaces
        element_degree = parameters["element_degree"]
        scalar = FunctionSpace(mesh, "CG", element_degree)
        vector = VectorFunctionSpace(mesh, "CG", element_degree)

        #mixed_element = MixedFunctionSpace([vector, vector])
        mixed_element = FunctionSpace(mesh, MixedElement([vector.ufl_element(), vector.ufl_element()]))
        V = TestFunction(mixed_element)
        dU = TrialFunction(mixed_element)
        U = Function(mixed_element)
        U0 = Function(mixed_element)

        # Get initial conditions
        u0, v0 = problem.initial_conditions()

        # If no initial conditions are specified, assume they are 0
        if u0 == []:
            u0 = Constant((0,)*vector.mesh().geometry().dim())
        if v0 == []:
            v0 = Constant((0,)*vector.mesh().geometry().dim())

        # If either are text strings, assume those are file names and
        # load conditions from those files
        if isinstance(u0, str):
            info("Loading initial displacement from file.")
            file_name = u0
            u0 = Function(vector)
            _u0 = loadtxt(file_name)[:]
            #u0.vector()[0:len(_u0)] = _u0[:]
            u0.vector()[:] = _u0[:]
        if isinstance(v0, str):
            info("Loading initial velocity from file.")
            file_name = v0
            v0 = Function(vector)
            _v0 = loadtxt(file_name)[:]
            #v0.vector()[0:len(_v0)] = _v0[:]
            v0.vector()[:] = _v0[:]

        U.assign(U0)

        # Create boundary conditions
        dirichlet_values = problem.dirichlet_values()
        dirichlet_function_spaces = Make_D_function_spaces(mixed_element.sub(0), problem.dirichlet_function_spaces())
        bcu = create_dirichlet_conditions(dirichlet_values,
                                          problem.dirichlet_boundaries(),
                                          dirichlet_function_spaces)

        # Functions
        xi, eta = split(V)
        u, v = split(U)
        u_plot = Function(vector)

        # Project u0 and v0 into U0
        a_proj = inner(dU, V)*dx
        L_proj = inner(u0, xi)*dx + inner(v0, eta)*dx
        solve(a_proj == L_proj, U0)
        u0, v0 = split(U0)

        # Residual stresses
        pre_P = problem.prestress()
        if pre_P == []:
            pre_P = Constant(((0.0,)*vector.mesh().geometry().dim(),)*vector.mesh().geometry().dim())

        # Driving forces
        B = problem.body_force()
        if B == []:
            B = problem.body_force_u(0.5*(u0 + u))

        # If no body forces are specified, assume it is 0
        if B == []:
            B = Constant((0,)*vector.mesh().geometry().dim())

        # Evaluate displacements and velocities at mid points
        u_mid = 0.5*(u0 + u)
        v_mid = 0.5*(v0 + v)

        # Get reference density
        rho0 = problem.reference_density()

        # If no reference density is specified, assume it is 1.0
        if rho0 == []:
            rho0 = Constant(1.0)

        density_type = str(rho0.__class__)
        if not ("dolfin" in density_type):
            info("Converting given density to a DOLFIN Constant.")
            rho0 = Constant(rho0)

        # Piola-Kirchhoff stress tensor based on the material model
        P = problem.first_pk_stress(u_mid)

        # Convert time step to a DOLFIN constant
        k = Constant(dt)

        # The variational form corresponding to hyperelasticity
        L = rho0*inner(v - v0, xi)*dx + k*inner(P+pre_P, grad(xi))*dx \
            - k*inner(B, xi)*dx + inner(u - u0, eta)*dx \
            - k*inner(v_mid, eta)*dx

        # Add contributions to the form from the Neumann boundary
        # conditions

        # Get Neumann boundary conditions on the stress
        neumann_conditions = problem.neumann_conditions()
        neumann_boundaries = problem.neumann_boundaries()

        boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundary.set_all(len(neumann_boundaries) + 1)

        # dsb = ds[boundary]
        dsb = ds(subdomain_data=boundary)
        for (i, neumann_boundary) in enumerate(neumann_boundaries):
            info("Applying Neumann boundary condition.")
            info(str(neumann_boundary))
            compiled_boundary = CompiledSubDomain(neumann_boundary)
            compiled_boundary.mark(boundary, i)
            L = L - k*inner(neumann_conditions[i], xi)*dsb(i)

        a = derivative(L, U, dU)

        # Store variables needed for time-stepping
        self.dt = dt
        self.k = k
        self.t_range = t_range
        self.end_time = end_time
        self.a = a
        self.L = L
        self.bcu = bcu
        self.U0 = U0
        self.U = U
        self.B = B
        self.dirichlet_values = dirichlet_values
        self.neumann_conditions = neumann_conditions

        # FIXME: Figure out why I am needed
        self.mesh = mesh
        # Kristoffer's fix in order to sync the F and S solvers dt...
        self.t = dt

        # Empty file handlers / time series
        self.displacement_file = None
        self.velocity_file = None
        self.displacement_velocity_series = None
        self.u_plot = u_plot

        # Store parameters
        self.parameters = parameters

    def solve(self):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Time loop
        for t in self.t_range:
            info("Solving the problem at time t = " + str(self.t))
            self.step(self.dt)
            self.update()

        if self.parameters["plot_solution"]:
            interactive()

        return self.U.split(True)[0]

    def step(self, dt):
        """Setup and solve the problem at the current time step"""

        # Update time step
        self.dt = dt
        self.k.assign(dt)

        problem = NonlinearVariationalProblem(self.L, self.U, self.bcu, self.a)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-9
        solver.parameters["newton_solver"]["relative_tolerance"] = 1e-9
        solver.parameters["newton_solver"]["maximum_iterations"] = 100
        solver.solve()
        return self.U.split(True)

    def update(self):
        """Update problem at time t"""

        u, v = self.U.split()

        # Propagate the displacements and velocities
        self.U0.assign(self.U)

        # Plot solution
        if self.parameters["plot_solution"]:
            # Copy to a fixed function to trick Viper into not opening
            # up multiple windows
            assign(self.u_plot, u)   # support fenics 2017
            plot(self.u_plot, title="Displacement", mode="displacement", rescale=True)

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            if self.displacement_file is None:
                self.displacement_file = File("displacement.pvd")
            if self.velocity_file is None:
                self.velocity_file = File("velocity.pvd")
            self.displacement_file << u
            self.velocity_file << v

        # Store solution data
        if self.parameters["store_solution_data"]:
            if self.displacement_velocity_series is None:
                self.displacement_velocity_series = TimeSeries("displacement_velocity")
            self.displacement_velocity_series.store(self.U.vector(), self.t)

        # Move to next time step
        self.t = self.t + self.dt

        # Inform time-dependent functions of new time
        for bc in self.dirichlet_values:
            if isinstance(bc, Expression):
                bc.t = self.t
        for bc in self.neumann_conditions:
            bc.t = self.t
        self.B.t = self.t

    def solution(self):
        "Return current solution values"
        return self.U.split(True)
