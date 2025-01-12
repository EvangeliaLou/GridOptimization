import sys
import os
import shutil
import re
import numpy as np
import math
import random
import scipy.optimize as sp_opt
import matplotlib.pyplot as plt
import matplotlib as mpl


class Polygon:
    def __init__(self, x, y):
        '''
        Given that when a polygon is first defined, it is not folded or reversed,
        its surface area must be positively oriented.
        The surface area is calculated by the sum_i of R_i cross R_(i+1). If the initial
        orientation of the nodes is counter-clockwise, the surface area will be positive, according
        the right hand rule. If it is clockwise, the area will be negative.
        '''
        self.n = len(x)
        self.coef = 4*self.n*np.tan(np.pi/self.n)
        self.set(x, y)

    def set(self, x, y):
        self.x, self.y = x, y
        self.r = [vector(x[i], y[i]) for i in range(self.n)]
        self.dr = [self.r[i]-self.r[i-1] for i in range(self.n)]
        self.area = 1/2*sum([self.r[i-1].cross(self.r[i]) for i in range(self.n)])
        self.per = sum([side.norm() for side in self.dr])
        self.R = self.coef*self.area/self.per**2

    def centroid(self):
        '''
        Geometric center:   x = \int x dxdy
                            y = \int y dxdy
        Using Green's Theorem, the above surface integral can be transformed to a line integral.
        '''
        x, y = self.x, self.y
        xc = 1/6*sum([(y[i] - y[i-1])*(x[i]**2 + x[i-1]**2 + x[i]*x[i-1]) for i in range(self.n)])/self.area
        yc = 1/6*sum([(x[i-1] - x[i])*(y[i]**2 + y[i-1]**2 + y[i]*y[i-1]) for i in range(self.n)])/self.area
        return np.array([xc, yc])

    def mean(self):
        '''
        average position = Σ r_i
        '''
        mean = self.r[0]
        for r in self.r[1:]:
            mean += r
        return mean.toarray()/self.n
    
    def grad_A(self, i):
        '''
        The surface derivative with respect to the i-th node
        '''
        return 0.5*(self.r[(i+1)%self.n]-self.r[i-1]).cross('z').toarray()
    
    def grad_P(self, i):
        '''
        The perimeter derivative with respect to the i-th node
        '''
        return (self.dr[i].unit()-self.dr[(i+1)%self.n].unit()).toarray()
    
    def grad_R(self, i):
        '''
        The roundness derivative with respect to the i-th node
        '''
        return self.coef*(self.grad_A(i)/self.per**2 - 2*self.area*self.grad_P(i)/self.per**3)
    
    def is_folded(self):
        if self.area < 0:
            return True
        elif self.n == 4:
            for i in range(self.n):
                A, B = self.r[i], self.r[(i+1)%self.n]
                for j in range(i+2, self.n-(i==0)):
                    C, D = self.r[j%self.n], self.r[(j+1)%self.n]
                    det = (C-B).cross(A-D)
                    if det == 0:
                        return False
                    t1 = (C-B).cross(B-D)/det
                    t2 = (A-D).cross(B-D)/det
                    if 0 < t1 < 1 and 0 < t2 < 1:
                        return True
        return False
    
    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(np.append(self.x, self.x[0]), np.append(self.y, self.y[0]))
        return fig, ax


class vector:
    '''
    Use only for nodes in Polygon class
    '''
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __neg__(self):
        return vector(-self.x, -self.y)

    def __add__(self, other):
        return vector(self.x+other.x, self.y+other.y)

    def __sub__(self, other):
        return vector(self.x-other.x, self.y-other.y)

    def __rmul__(self,other):
        return vector(other*self.x,other*self.y)

    def norm(self):
        return math.sqrt(self.x**2+self.y**2)

    def unit(self):
        norm = self.norm()
        return vector(self.x/norm, self.y/norm)

    def cross(self, other):
        if other == 'z':
            return vector(self.y, -self.x)
        else:
            return self.x*other.y - self.y*other.x

    def toarray(self):
        return np.array([self.x, self.y])

    def __repr__(self):
        r = [self.x, self.y]
        r = [int(i) if int(i)==i else i for i in r]
        return '({0}, {1})'.format(*r)


class grid:
    def __init__(self, file, split=False):
        '''
        FILE MANAGEMENT
        '''
        self.file = file
        self.dir = '{0}/{1}/'.format(sys.path[0], file)
        self.datafile = self.dir+'{0}.txt'.format(file)
        polygon, coords, flag = [], [], []
        with open(self.datafile) as f:
            nd, ns = [int(s) for s in re.findall(r'\b\d+\b', f.readline())]
            for _ in range(ns):
                polygon.append([int(s) for s in re.findall(r'\b\d+\b', f.readline())])
            for _ in range(nd):
                node_data = re.findall(r'-?\d+(?:\.\d+)?', f.readline())
                flag.append(int(node_data[0]))
                coords.append([float(node_data[1]), float(node_data[2])])
        flag = np.array(flag, dtype=int)
        r = np.array(coords)
        out = np.where(flag==0)[0]
        wing = np.where(flag==1)[0]
        ind = np.where(flag==2)[0]
        wall = np.where(np.logical_or(flag==0, flag==1))[0]

        '''
        shapes_with[i] = polygons that contain the i-th node
        '''
        shapes_with = [[] for _ in range(nd)]
        shapes = []
        nodes_of = []
        for i, nu in enumerate(polygon):
            for node in nu:
                shapes_with[node].append(i)
            shapes.append(Polygon(*r[nu].T))
            nodes_of.append(nu)

        '''
        neighbors[i] = all nodes inside the polygons that share the i-th node
        '''
        neighbors = []
        for node in range(nd):
            nbors = []
            for i in shapes_with[node]:
                for node_ in nodes_of[i]:
                    if node_ not in wall and node_ not in nbors:
                        nbors.append(node_)
            neighbors.append(nbors)

        '''
        sides = all sides of all polygons
        Their coordinates will be used for plotting
        '''
        sides, shapes_of_side_dict = [], {}
        for i in range(nd):
            cons = []
            for j in shapes_with[i]:
                shape_nodes = nodes_of[j]
                for node in shape_nodes:
                    if (node > i) and (abs(shape_nodes.index(i)-shape_nodes.index(node)) in (1, len(shape_nodes)-1)):
                        if node not in cons:
                            cons.append(node)
                            shapes_of_side_dict[(i, node)] = [j]
                        else:
                            shapes_of_side_dict[(i, node)].append(j)
            cons = sorted(cons)
            for j in range(len(cons)):
                sides.append((i, cons[j]))
        '''
        shapes_of_side[i] = the shapes that share the i-th side (one or two max)
        '''
        shapes_of_side = []
        for side in sides:
            shapes_of_side.append(shapes_of_side_dict[side])

        '''
        layer[0] = nodes surrounding the wing. layer[1] = nodes surrounding layer[0], and so on...
        '''

        layer = [wing, wing]
        shapes_of_layer = [[]]
        n_tot = 0
        while n_tot < len(ind):
            next_layer, next_shapes = [], []
            for j in layer[-1]:
                for n in neighbors[j]:
                    if n not in layer[-1] and n not in layer[-2]:
                        next_layer.append(n)
                for s in shapes_with[j]:
                    if s not in shapes_of_layer[-1]:
                        next_shapes.append(s)
            next_layer = list(set(next_layer))
            next_shapes = list(set(next_shapes))
            layer.append(np.array(next_layer))
            shapes_of_layer.append(np.array(next_shapes))
            n_tot += len(next_layer)

        '''
        Defining constant attributes of our grid:
        '''
        self.nd = nd
        self.ns = ns
        self.flag = flag
        self.r_initial = r.copy()
        self.wall = wall
        self.wing = wing
        self.out = out
        self.ind = ind
        self.nodes_of = nodes_of
        self.shapes_with = shapes_with
        self.neighbors = neighbors
        self.sides = sides
        self.shapes_of_side = shapes_of_side
        self.layer = layer[2:]
        self.shapes_of_layer = shapes_of_layer[1:]
        self.shapes = shapes #only the coords of each shape's nodes can change
        self.Wing = Polygon(*r[wing].T) #only the coords of the wing's nodes can change
        self.Out = Polygon(*r[out].T)

        '''
        Dynamic variables are defined inside self.reset()
        '''
        self.reset()

    def _set(self, r, nodes=None):
        '''
        set new coordinates for the nodes in our grid. All shapes will be deformed accordingly
        '''
        if nodes is None:
            self.r = r.copy()
        else:
            self.r[nodes] = r
        for i, shape in enumerate(self.shapes):
            shape.set(*self.r[self.nodes_of[i]].T)
        self.Wing.set(*self.r[self.wing].T)
        self.Out.set(*self.r[self.out].T)

    def _set_wing_coords(self, r):
        '''
        set new coordinates for the nodes in our wing only
        '''
        self._set(r, self.wing)
        self.set_bounds()
        if len(self.eff_nodes) < self.nd:
            self.activate_nodes(self.eff_nodes)
        '''
        self.activate_nodes is called only to recalculate self.const_obj_fun,
        although in most cases it will not be affected
        '''

    def activate_nodes(self, nodes):
        '''
        Choose which nodes will be used for minimization
        '''
        self.eff_nodes = nodes.copy()
        self.eff_shapes = self.effective_shapes_of(nodes)
        self.const_obj_fun = 0
        for i, shape in enumerate(self.shapes):
            if i not in self.eff_shapes:
                self.const_obj_fun += self.f(shape.R, self.R_opt[i])

    def update_optimized_state(self):
        self.R_opt = np.array([s.R for s in self.shapes])
        self.r_opt = self.r.copy()
        self.const_obj_fun = 0

    def def_metric_component(self, f, df):
        '''
        The objective function is defined as
        obj_fun = Σ_i f(R_i, R_opt_i)
        Parameters of f and df:
        1) R_i: the roundness of the i-th polygon
        2) R_opt: The roundness of the desired optimized state, stored into the list "self.R_opt"

        f = metric component: By default f(R, R0) = (R-1)^2, because R_max = 1 for all polygons
        df = derivative of f: By default df(R, R0) = 2*(R-1)
        '''
        self.f = f
        self.df = df

    def filter_folded_sides(self):
        '''
        sides in folded shapes are marked with is_bad[i] = True
        '''
        is_bad = []
        for i in range(len(self.sides)):
            is_bad.append(False)
            for j in self.shapes_of_side[i]:
                if self.shapes[j].is_folded():
                    is_bad[-1] = True
                    break
        return is_bad

    def nearest_nodes(self, n, x, y):
        '''
        returns the indices of the first n nearest nodes to the point (x, y)
        '''
        dist = np.array([np.sqrt((r[0]-x)**2 + (r[1]-y)**2) for r in self.r[self.ind]])
        nodes = np.argsort(dist)[:n]
        return self.ind[nodes]

    def folded_nodes(self):
        '''
        returns the indices of all the nodes belonging to folded shapes
        '''
        nodes = []
        for node in self.ind:
            node_shapes = self.shapes_with[node]
            for i in node_shapes:
                if self.shapes[i].is_folded():
                    nodes.append(node)
                    break
        return np.array(nodes)

    def effective_shapes_of(self, nodes):
        shapes = []
        for node in nodes:
            shapes += self.shapes_with[node]
        return list(set(shapes))

    def common_shapes_of(self, i, j):
        '''
        returns the indices of the common shapes of the i-th and j-th node
        '''
        S_i = self.shapes_with[i]
        S_j = self.shapes_with[j]
        return list(set(S_i) & set(S_j))

    def move_wing(self, *delta_r):
        '''
        Move the wing left-right and up-down
        '''
        wing_coords = self.r[self.wing]+vec(*delta_r)
        self.wing_axis += vec(*delta_r)
        self._set_wing_coords(wing_coords)

    def rotate_wing(self, degrees):
        '''
        Rotate the wing by its axis
        '''
        self.wing_rotation += degrees
        new_wing = rotate(degrees, self.wing_axis, self.r[self.wing])
        self.deformation = 'Rotation: {0}'.format(self.wing_rotation)
        self._set_wing_coords(new_wing)

    def deform(self, dev):
        '''
        Move all nodes randomly by +- dev around their current position
        '''
        r = self.r.copy()
        for i in self.ind:
            r[i] += np.array((random.uniform(-dev, dev), random.uniform(-dev, dev)))
        self._set(r[self.ind], self.ind)
        self.deformation = 'Deformation: +-{0}'.format(dev)
        self.set_bounds()

    def obj_fun(self, r_eff = None):
        '''
        Calculate the objective function: Sum of (1-R_i)^2
        '''
        if r_eff is None:
            r_eff = self.r_eff().T.flatten()
        self.set_reff(group(r_eff))
        sum = 0
        for i in self.eff_shapes:
            sum += self.f(self.shapes[i].R, self.R_opt[i])
        return sum + self.const_obj_fun

    def grad(self, r_eff):
        '''
        Calculate the gradient of the objective function
        '''
        self.set_reff(group(r_eff))
        grad = np.zeros(shape=(len(self.eff_nodes), 2))
        for n, i in enumerate(self.eff_nodes):
            for k in self.shapes_with[i]:
                grad[n] += self.df(self.shapes[k].R, self.R_opt[k])*self.shapes[k].grad_R(self.nodes_of[k].index(i))
        return grad.T.flatten()

    def optimizer(self, maxiter, tol = 1e-06):
        r = self.r_eff()
        gradr = self.grad(r)
        f = self.obj_fun(r)
        for _ in range(maxiter):
            if f <= tol:
                break
            else:
                r -= 0.1*gradr.T
                gradr = self.grad(r)
                f = self.obj_fun(r)
        self.r[self.ind] = r

    def activate_nearest_nodes(self, n, center = None):
        '''
        Choose the n closest nodes to the position "center" (e.g. [0, 0]) to
        be used for minimization
        '''
        if center is None:
            center = self.Wing.centroid()
        self.activate_nodes(self.nearest_nodes(n, *center))

    def activate_folded_nodes(self):
        '''
        Choose all nodes inside folded polygons, to be used for minimization
        '''
        self.activate_nodes(self.folded_nodes())

    def set_reff(self, r_eff):
        '''
        change the coordinates of all the nodes that are use in the minimization algorithm
        '''
        self.r[self.eff_nodes] = r_eff
        for i in self.eff_shapes:
            self.shapes[i].set(*self.r[self.nodes_of[i]].T)

    def optimize(self, maxiter=None, iprint=None, bounds=True, gtol=None, ftol=None):
        '''
        Main function that optimizes the grid
        '''
        options = {}
        if maxiter is not None:
            options['maxiter'] = maxiter
        if iprint is not None:
            options['iprint'] = iprint
        if gtol is not None:
            options['gtol'] = gtol
        if ftol is not None:
            options['ftol'] = ftol
        params = dict(fun=self.obj_fun, x0=self.r_eff().T.flatten(), jac=self.grad, method = 'L-BFGS-B')
        if self.bounds is not None and bounds is True:
            params['bounds'] = self.choose_from_bounds(self.eff_nodes)
        self.result = sp_opt.minimize(**params, options=options)

    def optimize_by_steps(self, goal, steps, maxiter=None, iprint=None, bounds = True, view_progress=False, save_progress=False):
        for i in range(1, steps+1):
            if hasattr(goal, '__iter__'):
                self.move_wing(*(goal/steps))
                frac = vector(*(i/steps*goal)), vector(*goal)
            else:
                self.rotate_wing(goal/steps)
                frac = i/steps*goal, goal
            print('Optimizing: {0}% --- {1} / {2}'.format(round(i/steps*100, 2), *frac))
            self.optimize(maxiter=maxiter, iprint=iprint, bounds=bounds)
            if view_progress:
                self.plot()
            if save_progress:
                self.save('{0}_{1}'.format(self.file, round(self.wing_rotation) if self.wing_rotation == int(self.wing_rotation) else self.wing_rotation))
            self.update_optimized_state()

    def rotate(self, deg):
        center = self.Out.centroid()
        new_coords = rotate(deg, center, self.r)
        self._set(new_coords)
        self.r_opt = rotate(deg, center, self.r_opt)
        self.wing_axis = rotate(deg, center, self.wing_axis)
        self.bounds = None
    
    def rotate_eff(self, deg):
        center = self.wing_axis
        new_coords = rotate(deg, center, self.r_eff())
        self.set_reff(new_coords)

    def save(self, file, R = None):
        newdir = '{0}/{1}/'.format(self.dir, file)
        if os.path.exists(newdir):
            shutil.rmtree(newdir)
        os.mkdir(newdir)
        if R is None:
            R = self.r
        with open(newdir+file+'.txt', 'w') as f:
            f.write(grid_text(self.nodes_of, R, self.flag))
        fill, name = [True, False], ['_filled', '']
        for i in range(2):
            fig, _ = self.plot(view=False, fill=fill[i])
            fig.savefig('{0}/{1}{2}.pdf'.format(newdir, file, name[i]))
            plt.close()

    def plot(self, view=True, fill=True, nodes = False, cmap = 'BuPu'):
        '''
        Plot the grid using matplotlib
        '''
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.axis('equal')
        is_bad = self.filter_folded_sides()
        for i, line in enumerate(self.sides):
            if (line[0] in self.wall) and (line[1] in self.wall):
                color, lw, zorder = 'green', 0.8, 3
            elif is_bad[i] is True:
                color, lw, zorder = 'red', 0.3, 2
            else:
                color, lw, zorder = 'black', 0.3, 1
            r1, r2 = self.r[line[0]], self.r[line[1]]
            ax.plot([r1[0], r2[0]], [r1[1], r2[1]], color = color, linewidth = lw, zorder=zorder)
        if nodes is True and len(self.eff_nodes) < len(self.ind):
            ax.scatter(*self.r_eff().T, color='blue', s=30)
        R = [round(s.R, 8) for s in self.shapes]
        if fill is True:
            vmin, vmax = min(R), max(R)
            if vmax-vmin < 0.01:
                vmin, vmax = 0, 1
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cMap = plt.get_cmap(cmap)
            for s in self.shapes:
                ax.fill(s.x, s.y, color=cMap(norm(s.R)), zorder=0)
            fig.colorbar(mappable=plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, ticks=np.linspace(vmin, vmax, num=11))
        ax.set_title('{0} polygons, {1} nodes\n{2}, obj_fun: {3}, folded polygons: {4}\n R_min: {5}, R_max: {6}'.format(
                    self.ns, self.nd, self.deformation, round(self.obj_fun(), 2), len(self.folded_shapes()), min(R), max(R)))
        if view is True:
            plt.show()
        return fig, ax

    def r_eff(self):
        return self.r[self.eff_nodes]

    def folded_shapes(self):
        f = []
        for i in range(len(self.shapes)):
            if self.shapes[i].is_folded() is True:
                f.append(i)
        return f

    def set_bounds(self, bounds=None):
        '''
        The deviation from an optimized state will then be determined based on the movement of the grid,
        starting from that state, to the current state (self.r), so that bounds = (optimized state) +- deviation
        '''
        if bounds is None:
            deviation = max(np.sqrt(((self.r-self.r_opt)**2).sum(axis=1)))*2
            r0 = self.r_opt.T.flatten()
            low, up = r0-deviation, r0+deviation
        else:
            low, up = bounds[0].T.flatten(), bounds[1].T.flatten()
        self.bounds = np.concatenate((np.array([low]), np.array([up])), axis=0).T

    def choose_from_bounds(self, nodes):
        return self.bounds[np.append(nodes, nodes+self.nd)]
    
    def reset(self):
        '''
        Defining dynamic variables of our grid
        '''
        self._set(self.r_initial)
        self.f = lambda R, R_opt: (R-1)**2
        self.df = lambda R, R_opt: 2*(R-1)
        self.R_opt = np.array([s.R for s in self.shapes])
        self.r_opt = self.r_initial.copy()
        self.activate_nodes(self.ind) #defines self.const_obj_fun, self.eff_nodes, self.eff_shapes
        self.wing_rotation = 0
        self.bounds = None
        self.wing_axis = vec(1/4, 0) if self.file in ['n12quads', 'n12hyb'] else self.Wing.centroid()
        self.deformation = 'Deformation: None'

    def activate_layers(self, i):
        self.activate_nodes(np.concatenate(self.layer[:i]))


def group(r):
    return np.reshape(r, (int(len(r)/2), 2), order='F')

def custom_grid(filename, coords, polygons, flags):
    newdir = '{0}/{1}/'.format(sys.path[0], filename)
    if os.path.exists(newdir):
        shutil.rmtree(newdir)
    os.mkdir(newdir)

    with open('{0}/{1}.txt'.format(newdir, filename), 'w') as f:
        f.write(grid_text(polygons, coords, flags))
    return grid(filename)

def Round_grid(filename, r1, r2, n):
    assert type(r1) is int and type(r2) is int
    phi = np.linspace(0, 2*np.pi, n, endpoint=False)
    layers = (r2-r1)+1
    r = np.linspace(r1, r2, layers)
    Nd = n*layers
    coords = []
    for l in range(layers):
        for u in phi:
            coords.append([float(r[l]*np.cos(u)), float(r[l]*np.sin(u))])
    polygons = []
    for l in range(layers-1):
        for i in range(n):
            polygons.append([n*l+i, n*(l+1)+i, n*(l+1)+(i+1)%n, n*l+(1+i)%n])
    flags = np.zeros(Nd, dtype=int)+2
    flags[np.linspace(Nd-n, Nd-1, n, dtype=int)] = 0
    flags[np.linspace(0, n-1, n, dtype=int)] = 1
    return custom_grid(filename, coords, polygons, flags)

def Quad_grid(filename, outer, inner):
    assert outer%2 == inner%2

    d1, d2 = inner, outer
    Nd = (d2+1)**2 - (d1-1)**2
    coords = []
    polygons = []
    for l in range(d1, d2+1, 2):
        for k in range(4):
            for i in range(0, 2*l, 2):
                coord = vec((-1)**(k//2)*l+i*1j**(k+1), (-1)**((k-1)//2)*l+i*1j**k)
                coords.append(tuple(coord))
                a = (l-d1)*(l+d1-2)
                n = a+l*k + int(i/2)
                if l <= d2-1:
                    polygons.append([n, n+4*l+2*k+1, n+4*l+2*k+2, a+(1+l*k+int(i/2))%(4*l)])
            if l <= d2-1:
                polygons.append([a+(k*l+l)%(4*l), n+(4*l+2*k+2), n+(4*l+2*k+3)%(4*(l+2)), n+1+(4*l+2*k+3)%(4*(l+2))])
    flags = np.zeros(Nd, dtype=int)+2
    flags[np.linspace(Nd-4*d2, Nd-1, 4*d2, dtype=int)] = 0
    flags[np.linspace(0, 4*d1-1, 4*d1, dtype=int)] = 1
    return custom_grid(filename, coords, polygons, flags)

def vec(*x):
    return np.array([int(i.real) if i.real==int(i.real) else i.real for i in x])

def rotate(degrees, center, r):
    u = degrees*np.pi/180
    d = r-center
    
    if r.shape == (2,):
        z_cross_d = np.array([-d[1], d[0]])
        return center + np.cos(u)*d + np.sin(u)*z_cross_d
    else:
        z_cross_d = np.array([-d[:,1], d[:,0]]).T
        return np.tile(center, (len(r), 1)) + np.tile(np.cos(u), (2, 1)).T*d + np.tile(np.sin(u), (2, 1)).T*z_cross_d

def grid_text(polygon_nodes, coords, flags):
    nd, ns = len(coords), len(polygon_nodes)
    text = '{0} {1}\n'.format(nd, ns)
    for nodes in polygon_nodes:
        text += ' '.join([str(i) for i in nodes])+'\n'
    for i in range(nd):
        text += ' '.join([str(flags[i])] + ['%.8f'%i for i in [*coords[i]]]) + '\n'
    return text[:-2]
