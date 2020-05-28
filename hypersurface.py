import numpy as np
import sympy as sp
from manifold import *
from patches import *

# In manifold and type
class Hypersurface(Manifold):

    def __init__(self, coordinates, function, dimensions, n_points):
        super().__init__(dimensions) # Add one more variable for dimension
        self.function = function
        self.coordinates = np.array(coordinates)
        self.conjcoords = sp.conjugate(self.coordinates)
        self.n_points = n_points
        self.__zpairs = self.__generate_random_pair()
        self.points = self.__solve_points()
        self.patches = []
        self.__autopatch()
        self.sections,self.num_sec=self.__sections()
        self.holo_volume_form = self.__get_holvolform() 
    #def HolVolForm(F, Z, j)

=======
        self.coordinates = np.array(coordinates)
        self.conjcoords = sp.conjugate(self.coordinates)
        self.norm_coordinate = norm_coordinate
<<<<<<< Updated upstream
=======
        # The symbolic coordinate is self.coordiante[self.norm_coordiante]
        self.max_grad_coordinate = max_grad_coordinate
        # Range 0 to n-2, this works only on subpatches where max grad is calculated
        # Symbolically self.affin_coordinate[self.max_grad_coordinate]
        if norm_coordinate is not None:
            self.affine_coordinates = np.delete(self.coordinates, norm_coordinate)
        else:
            self.affine_coordinates = self.coordinates
>>>>>>> Stashed changes
        self.patches = []
        if points is None:
            self.points = self.__solve_points(n_pairs)
            self.__autopatch()
        else:
            self.points = points
        self.n_points = len(self.points)
        self.sections,self.num_sec = self.__sections()
        self.initialize_basic_properties()
    
    def initialize_basic_properties(self):
        # This function is necessary because those variables need to be updated on
        # the projective patches after subpatches are created. Then this function will
        # be reinvoked.
        self.grad = self.get_grad()
        self.holo_volume_form = self.get_holvolform()
        #self.transition_function = self.__get_transition_function()
>>>>>>> Stashed changes

    def reset_patchwork(self):
        #self.patches = [None]*n_patches
        self.patches = []

    def set_patch(self, points_on_patch, norm_coordinate):
        #patch.append(point)
        #for points in points_on_patch:
        #    new_patch = Patches(self.coordinates, self.function, self.dimensions, points)
        #    self.patches.append(new_patch)
        new_patch = Patches(self.coordinates, self.function, self.dimensions,
                            points_on_patch, norm_coordinate)
        self.patches.append(new_patch)
        
    def list_patches(self):
        print("Number of Patches:", len(self.patches))
        i = 1
        for patch in self.patches:
            print("Points in patch", i, ":", len(patch.points))
            i = i + 1

    def normalize_point(self, point, coordinate):
        for i in range(len(point)):
            point[i] = sp.simplify(point[i] / point[coordinate])
        return point

    def print_all_points(self):
        print("All points on this hypersurface:")
        print(self.points)

    # def eval_holvolform(self):
    #     holvolform = []
    #     for i in range(len(self.patches)):
    #         holvolform.append(self.patches[i].eval_holvolform())
    #     return holvolform 


    # Private:

    def __generate_random_pair(self):
        z_random_pair = []
        for i in range(self.n_points):
            zv = []
            for j in range(2):
                zv.append([complex(c[0],c[1]) for c in np.random.normal(0.0,1.0,(self.dimensions,2))])
            z_random_pair.append(zv)
        return(z_random_pair)

    def __solve_points(self):
        points = []
<<<<<<< Updated upstream
        for zpair in self.__zpairs:
            a = sp.symbols('a')
            line = [zpair[0][i]+(a*zpair[1][i]) for i in range(self.dimensions)]
            function_eval = self.function.subs([(self.coordinates[i], line[i])
                                                  for i in range(self.dimensions)])
            #print(sp.expand(function_eval))
            #function_lambda = sp.lambdify(a, function_eval, ["scipy", "numpy"])
            #a_solved = fsolve(function_lambda, 1)
            a_solved = sp.polys.polytools.nroots(function_eval)
=======
        zpairs = self.__generate_random_pair(n_pairs)
        f_evaluatable = sp.lamdify(self.coordinates,self.function,"numpy")
        a = sp.symbols('a')
        for zpair in zpairs:
            #a = sp.symbols('a')
            line = [zpair[0][i]+(a*zpair[1][i]) for i in range(self.dimensions)]
            #function_eval = self.function.subs([(self.coordinates[i], line[i])
            #function_eval = self.function.subs([(self.coordinates[i], line[i])
                                                #for i in range(self.dimensions)])
            # This solver uses mpmath package, which should be pretty accurate
            #a_solved = sp.polys.polytools.nroots(function_eval)
            a_solved = sp.polys.polytools.nroots(f_evaluatable(*(line)))
>>>>>>> Stashed changes
            #a_rational = sp.solvers.solve(sp.Eq(sp.nsimplify(function_eval, rational=True)),a)
            # print("Solution for a_lambda:", a_poly)
            # a_solved = sp.solvers.solve(sp.Eq(sp.expand(function_eval)),a)
            for pram_a in a_solved:
                points.append([zpair[0][i]+(pram_a*zpair[1][i])
                               for i in range(self.dimensions)])
        return(points)

    # def __autopatch(self):
    #    self.reset_patchwork()
    #    #self.reset_patchwork(self.dimensions)
    #    #for i in range(self.dimensions):
    #    #     self.patches[i] = []
    #     points_on_patch = [[] for i in range(self.dimensions)]
    #     for point in self.points:
    #         norms = np.absolute(point)
    #         for i in range(self.dimensions):
    #             if norms[i] == max(norms):
    #                 point_normalized = self.normalize_point(point, i)
    #                 points_on_patch[i].append(point_normalized) 
    #     self.set_patch(points_on_patch)
    #                #self.set_patch(point, self.patches[i])
    #                 # remake patch here


    def __autopatch(self):
        self.reset_patchwork()
        for i in range(self.dimensions):
            points_on_patch = []
            for point in self.points:
                norms = np.absolute(point)
                if norms[i] == max(norms):
                    point_normalized = self.normalize_point(point, i)
<<<<<<< Updated upstream
                    points_on_patch.append(point_normalized)
            self.set_patch(points_on_patch, i)

    def __get_holvolform(self):
        holvolform = []
        for i in range(len(self.patches)):
            holvolform.append(self.patches[i].holo_volume_form)
        return holvolform
    #Add class section
    #self. expr = sympy
    #def pt set
    #contains derivatives etc
=======
                    points_on_patch[i].append(point_normalized)
                    continue
        for i in range(self.dimensions):
            self.set_patch(points_on_patch[i], i)
        # Subpatches on each patch
        for patch in self.patches:
            points_on_patch = [[] for i in range(self.dimensions-1)]
            for point in patch.points:
                grad = patch.eval(patch.grad, point)
                grad_norm = np.absolute(grad)
                for i in range(self.dimensions-1):
                    if grad_norm[i] == max(grad_norm):
                        points_on_patch[i].append(point)
                        continue
            for i in range(self.dimensions-1):
                patch.set_patch(points_on_patch[i], patch.norm_coordinate)
            patch.initialize_basic_properties()


     def __sections(self):
        t = sp.symbols('t')
        GenSec = sp.prod(1/(1-(t*zz)) for zz in self.coordinates)
<<<<<<< Updated upstream
        poly = sp.series(GenSec,t,n=self.dimensions+1).coeff(t**(self.dimensions))
        sections = []
        while poly!=0:
            sections.append(sp.LT(poly))
            poly = poly - sp.LT(poly)
        return (np.array(sections),len(sections))
=======
        poly = sp.series(GenSec, t, n=max(self.dimensions+1,k+1)).coeff(t**k)
        while poly!=0:
            sections.append(sp.LT(poly))
            poly = poly - sp.LT(poly)
        n_sections = len(sections)
        sections = np.array(sections)
        return sections, n_sections
    # just one potential
    def kahler_potential(self, h_matrix=None, k=1):
        #need to generalize this for when we start implementing networks
        sections, n_sec = self.get_sections(k)
        if h_matrix is None:
            h_matrix = sp.MatrixSymbol('H', n_sec, n_sec)
        zbar_H_z = np.matmul(sp.conjugate(sections),
                             np.matmul(h_matrix, sections))
        if self.norm_coordinate is not None:
            zbar_H_z = zbar_H_z.subs(self.coordinates[self.norm_coordinate], 1)
        kahler_potential = sp.log(zbar_H_z)
        return kahler_potential

    def kahler_metric(self, h_matrix=None, k=1):
        pot = self.kahler_potential(h_matrix, k)
        metric = []
        #i holomorphc, j anti-hol
        for coord_i in self.affine_coordinates:
            a_holo_der = []
            for coord_j in self.affine_coordinates:
                a_holo_der.append(diff_conjugate(pot, coord_j))
            metric.append([diff(ah, coord_i) for ah in a_holo_der])
        metric = sp.Matrix(metric)
        return metric

    def get_restriction(self, ignored_coord=None):
        if ignored_coord is None:
           ignored_coord = self.max_grad_coordinate
        ignored_coordinate = self.affine_coordinates[ignored_coord]
        local_coordinates = sp.Matrix(self.affine_coordinates).subs(ignored_coordinate,self.function)                                                                   self.function)
        affine_coordinates = sp.Matrix(self.affine_coordinates)
        restriction = local_coordinates.jacobian(affine_coordinates).inv()
        restriction.col_del(ignored_coord)
        return restriction
        # Todo: Add try except in this function 
>>>>>>> Stashed changes

    def KahlerPotential(self):
        ns = self.num_sec
        H = sp.MatrixSymbol('H',ns,ns)
        zbar_H_z = np.matmul(sp.conjugate(self.sections),np.matmul(H,self.sections))
        return sp.log(zbar_H_z)

    def KahlerMetric(self):
        pot = self.KahlerPotential()
        # need to establish diff wrt conjugate
