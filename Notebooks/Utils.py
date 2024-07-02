import numpy as np 
import typing  
import matplotlib.pyplot as plt


class Domain:
    """
    Represents a domain in a multi-dimensional space.

    Args:
        dimension (int): The dimension of the domain.
        level_function (Callable[[np.ndarray], np.ndarray]): A (convex) function that such that the domain is of the form {x:level_function(x) <= level} 
        level (np.ndarray): The level values for the domain 
        bounds (np.ndarray): The bounds of the domain in each dimension e.g., [[a1,b1],[a2,b2],...,[an,bn]].

    Attributes:
        dimension (int): The dimension of the domain.
        level_function (Callable[[np.ndarray], np.ndarray]): A function that maps a point in the domain to a level value.
        level (np.ndarray): The level values for the domain.
        bounds (np.ndarray): The bounds of the domain.

    Methods:
        is_point_inside(point: np.ndarray) -> bool:
            Checks if a given point is inside the domain.
        
        project_point(point: np.ndarray, num_samples: int) -> np.ndarray:
            Projects a point onto the domain, returning the closest point inside the domain.
        
        discretize(num_samples: int) -> np.ndarray:
            Discretizes the domain into a grid of points.
        
        visualize_1dim(num_samples: int = 100, name: str = 'Domain in 1D'):
            Visualizes the domain in 1D.
        
        visualize_2dim(num_samples: int = 10, name: str = 'Domain in 2D'):
            Visualizes the domain in 2D.
    """
    
    def __init__(self, dimension: int, level_function: typing.Callable[[np.ndarray],np.ndarray], level: np.ndarray, bounds: np.ndarray):
        self.dimension = dimension
        self.level_function = level_function
        self.level = level
        self.bounds = bounds 
    
    def is_point_inside(self, point: np.ndarray) -> bool:
        """
        Checks if a given point is inside the domain.

        Args:
            point (np.ndarray): The point to check.

        Returns:
            bool: True if the point is inside the domain, False otherwise.
        """
        if self.dimension == 1:
            return self.level_function(point) <= self.level
        else:
            return (self.level_function(point) <= self.level).all()  
    

    def project_point(self, point: np.ndarray,num_samples: int) -> np.ndarray:
        """
        Projects a point onto the domain, returning the closest point inside the domain.

        Args:
            point (np.ndarray): The point to project.
            num_samples (int): The number of samples to use for discretization.

        Returns:
            np.ndarray: The closest point inside the domain.
        """
        if self.is_point_inside(point):
            return point
        else:
            point_candidates = self.discretize(num_samples) 
        
        distances = np.linalg.norm(point_candidates-point,axis=1)
        closest_point = point_candidates[np.argmin(distances)]
        return closest_point

    def discretize(self, num_samples:int)-> np.ndarray:
        """
        Discretizes the domain into a grid of points.

        Args:
            num_samples (int): The number of samples to use for discretization.

        Returns:
            np.ndarray: The discretized domain as a grid of points.
        """
        if self.dimension == 1:
            X = np.linspace(self.bounds[0],self.bounds[1],num_samples) 
            point_candidates = np.zeros((num_samples,1))
            count = 0
            for i in range(X.shape[0]): 
                if self.is_point_inside(np.array([X[i]])):
                    point_candidates[count] = X[i]
                    count +=1
            discretized_domain = point_candidates[:count] #remove excess 
            return discretized_domain
        else:
            X = np.zeros((self.dimension,num_samples))
            for i in range(self.dimension):
                X[i] = np.linspace(self.bounds[i,0],self.bounds[i,1],num_samples)
            permutation = [i for i in range(1,self.dimension)].append(0)
            mesh = np.transpose(np.meshgrid(*X),axes=permutation)
        
            count = 0
            prod = np.power(num_samples,self.dimension)
            point_candidates = np.empty((prod,self.dimension))
            
            for index in np.ndindex(mesh[:,:,0].shape):
                    point_attempt = np.array(mesh[index])
                    if self.is_point_inside(point_attempt):
                        point_candidates[count] = point_attempt 
                        count += 1
            discretized_domain = point_candidates[:count] #remove excess 

            return discretized_domain
  
    def visualize_1dim(self,num_samples: int=100,name:str='Domain in 1D') :
        """
        Visualizes the domain in 1D.

        Args:
            num_samples (int, optional): The number of samples to use for visualization. Defaults to 100.
            name (str, optional): The name of the plot. Defaults to 'Domain in 1D'.
        """
        assert self.dimension == 1, "Visualization is only supported for 1-dimensional domains" 
        x = np.linspace(self.bounds[0],self.bounds[1],num_samples).squeeze() 
        plt.xlabel('x')
        plt.title(name)
        for i in range(x.shape[0]):
            if self.is_point_inside(np.array([x[i]])):
                plt.scatter(x[i],0,c='blue')
            else:
                plt.scatter(x[i],0,c='red')

        plt.scatter([],[],c='blue',label='Inside Interval')
        plt.scatter([],[],c='red',label='Outside Interval')
        plt.legend()
        plt.show()

    def visualize_2dim(self,num_samples: int=10,name:str='Domain in 2D'):
        """
        Visualizes the domain in 2D.

        Args:
            num_samples (int, optional): The number of samples to use for visualization. Defaults to 10.
            name (str, optional): The name of the plot. Defaults to 'Domain in 2D'.
        """
        assert self.dimension == 2, "Visualization is only supported for 2-dimensional domains" 
        x = np.linspace(self.bounds[0,0],self.bounds[0,1],num_samples)
        y = np.linspace(self.bounds[1,0],self.bounds[1,1],num_samples)
        X,Y = np.meshgrid(x,y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i,j],Y[i,j]])
                if self.is_point_inside(point):
                    plt.scatter(X[i,j],Y[i,j],c='blue')
                else:
                    plt.scatter(X[i,j],Y[i,j],c='red')

        plt.xlabel('x') 
        plt.ylabel('y')
        plt.title(name) 
        plt.scatter([],[],c='blue',label='Inside Domain')
        plt.scatter([],[],c='red',label='Outside Domain')
        plt.legend()
        plt.show()
 

def box_indicator_function(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Computes the indicator function for the box given by bounds input array a, b at point x. 

    Parameters:
    a (np.ndarray): Array of lower bounds.
    b (np.ndarray): Array of upper bounds.
    x (np.ndarray): Input array.

    Returns:
    np.ndarray: Array of box indicator values, where 1 indicates that the corresponding element of x is inside the box defined by a and b, and 100 indicates that it is outside the box. The value 100 is unimportant as anything above 1 suffices for this code, the typical convex indicator is +infty, but I had some issues with np.inf. 
    """

    assert np.shape(a)[0] == np.shape(b)[0] == np.shape(x)[0], "Dimensions of a, b, and x must match"

    if np.shape(a)[0] == 1:
        if a[0] <= x[0] <= b[0]:
            return np.array([1])
        else:
            return 100
    else:
        inside = (x >= a) * (x <= b)
        outside = [100] * ((x < a) + (x > b))
        return inside + outside


    
def create_box(dimension: int, int_bounds: np.ndarray, ext_bounds: np.ndarray)->Domain:
    """
    Create a box object with the given dimension, internal bounds, and external bounds.

    Parameters:
    dimension (int): The dimension of the box.
    int_bounds (np.ndarray): The internal bounds of the box.
    ext_bounds (np.ndarray): The external bounds of the box.

    Returns:
    Box: The created box object.

    Raises:
    AssertionError: If the shape of int_bounds or ext_bounds is not valid.

    """
    if dimension == 1:
        assert np.shape(int_bounds) == (2,) and np.shape(ext_bounds) == (2,), "Bounds must be of shape (2,)"
        a = np.array([int_bounds[0]])
        b = np.array([int_bounds[1]])
    else:
        assert np.shape(int_bounds) == (dimension, 2) and np.shape(ext_bounds) == (dimension, 2), "Bounds must be of shape (dimension,2)"
        a = int_bounds[:, 0]
        b = int_bounds[:, 1]

    box_level_function = lambda x: box_indicator_function(a, b, x)
    Box = Domain(dimension=dimension, level_function=box_level_function, level=1, bounds=ext_bounds)
    return Box


def create_ellipsoid(dimension: int, center: np.ndarray, radii:np.ndarray,bounds: np.ndarray):
    """
    Create an ellipsoid domain object.

    Parameters:
    - dimension (int): The dimension of the ellipsoid.
    - center (np.ndarray): The center of the ellipsoid.
    - radii (np.ndarray): The radii (i.e., y_i =  x_i/r_i parameterizes it to be a circle in y-space) of the ellipsoid. Can be a single value or an array of shape (dimension,).
    - bounds (np.ndarray): The bounds of the ellipsoid.

    Returns:
    - Ellipsoid: The ellipsoid domain object.

    Raises:
    - AssertionError: If the dimensions of center and bounds do not match.
    - AssertionError: If the shape of radii is not a single value or (dimension,).

    """

    assert np.shape(center)[0] == dimension and np.shape(bounds) == (dimension,2), "Dimensions of center, and bounds must match"
    #check if radii is single value
    if np.shape(radii) == ():
        radii = np.array([radii]*dimension)
    else:
        assert np.shape(radii) == (dimension,), "Radii must be a single value or of shape (dimension,)"
    
    ellipsoid_level_function = lambda x: np.sum(((x-center)/radii)**2)
    Ellipsoid = Domain(dimension=dimension,level_function=ellipsoid_level_function,level = 1,bounds = bounds)
    return Ellipsoid

class PSGD:
    """
    Projected Subgradient Gradient Descent (PSGD) class.

    Parameters:
    - C: Domain
        The domain of the optimization problem.
    - f: callable 
        The objective function to be minimized. Should be given f(t,x) where t is the iteration and x is the point. This doesn't have full generality (i.e., function should be specified at each iteration) but it serves for demonstration and was easier. 
    - x0: np.ndarray
        The initial point for optimization.
    - eta: callable
        The learning rate function.
    - projection_fn: callable, optional
        The projection function for constraint handling. Defaults to the projection function of the domain, this is bad in many cases because projection is like O(n^d) in naive case and simple domains (like ball) have easy projection functions. 
    """

    def __init__(self, C: Domain, f: callable, x0: np.ndarray, eta: callable, projection_fn: callable = None): 
        self.f = f
        self.C = C
        if projection_fn is None:           #identity if not specified 
            self.projection = self.C.project_point
        else:
            self.projection = projection_fn

        self.x = np.array([self.projection(x0)])
        self.lr = eta 

    def run(self, t):
        """
        Run the PSGD optimization algorithm for an iteration

        Parameters:
        - t: int
            The current iteration.

        Returns:
        None
        """
        self.x = np.append(self.x,np.array([[0.0, 1.0]]),axis=0) 

        ft = lambda x: self.f(t, x)

        x = self.x[t-2] - self.lr(t) * estimate_gradient(ft, self.x[t-2]) 
        self.x[t-1] = self.projection(x)

def calculate_regret(f: callable, D: Domain, x: np.ndarray, t: int, num_samples: int = 10):
    """
    Calculates the regret of a function f over a given domain D, for a sequence of inputs x.

    Parameters:
    - f (callable): The function to calculate regret for. It should take two arguments: time (int) and input (np.ndarray).
    - D (Domain): The domain over which to calculate regret. It should have a discretize method that returns a list of samples.
    - x (np.ndarray): The sequence of inputs.
    - t (int): The length of the sequence.
    - num_samples (int): The number of samples to use for discretizing the domain. Default is 10.

    Returns:
    - regret (float): The calculated regret.
    - min_x (np.ndarray): The input that minimizes the regret.
    """
    assert np.shape(x)[0] == t 
    assert D is not None    #domain must be specified (bounds needed for discretization)
    raw_regret = 0 
    for i in range(1, t+1):
        raw_regret += f(i, x[i-1]) 
    
    discretized_domain = D.discretize(num_samples)
    min_val = np.inf
    min_x = None 
    for x in discretized_domain:
        val = 0
        for i in range(1, t+1):      #f(time,x) format
            val += f(i, x)
        if val < min_val:
            min_val = val
            min_x = x

    regret = raw_regret - min_val
    return regret, min_x

def estimate_gradient(f: callable, x: np.ndarray, eps=1e-6):
    """
    Estimates the gradient of a function f at a given point x using finite differences.

    Parameters:
        f (callable): The function for which the gradient needs to be estimated.
        x (np.ndarray): The point at which the gradient needs to be estimated.
        eps (float, optional): The step size used for finite differences. Defaults to 1e-6.

    Returns:
        np.ndarray: The estimated gradient of the function at the given point.
    """
    g = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()       #resets each iteration to preserve x 
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        g[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return g

class UniversalAlgorithm: 
    def __init__(self, domain: Domain, T: int, R: float,G: float,x0: np.ndarray,f: callable,list_str_convex: list,list_exp_concave: list,list_convex: list,samples: int):
        """
        Initializes the Universal Algorithm  from Zhang et. al. (2022) Algoritghm 1.

        Args:
        - domain (Domain): The domain on which the algorithm operates.
        - T (int): The number of iterations.
        - R (float): The upper bound on domain radius i.e., l2 norm. 
        - G (float): The upper bound on the gradient norm.
        - x0 (np.ndarray): The initial point.
        - f (callable): The objective function in the form f(t,x).
        - list_str_convex (list): List of strongly convex experts.
        - list_exp_concave (list): List of alpha-exp-concave experts.
        - list_convex (list): List of convex experts.
        - samples (int): The number of samples used for projection.

        Returns:
        - None
        """
        self.x = np.zeros((T, domain.dimension))
        self.domain = domain
        self.T = T 
        self.R = R
        self.G = G
        self.x[0] = x0
        self.f = f 
        self.samples = samples
        k = 1+ int(np.ceil(np.log2(T))) 
        self.str_params = np.array([(2**i)/T for i in range(k)])        #parameters for strongly
        self.exp_params = np.array([(2**i)/T for i in range(k)])        #parameters for alpha-exp-concave 
        all_experts = []
        for expert in list_str_convex:
            for alpha in self.str_params:
                if expert is PSGD:
                    eta_strong = lambda t,alpha=alpha: 1/(alpha*(t-1))
                    projection_func = lambda x: domain.project_point(x,num_samples=samples)
                    all_experts.append(PSGD(C=domain,f=f,eta=eta_strong,x0=x0,projection_fn=projection_func))

        '''
        for expert in list_exp_concave:
            for alpha in self.exp_params:
                #initialize the experts 
        '''

        for expert in list_convex:
            if expert is PSGD: 
                def eta(R,G,time: int)-> float:
                    k = np.floor(np.log2(time)) 
                    return R/(G*np.sqrt(k)) 
                eta_reg = lambda t: eta(R,G,t)
                projection_func = lambda x: domain.project_point(x,num_samples=samples) 
                all_experts.append(PSGD(C=domain,f=f,eta=eta_reg,x0=x0,projection_fn=projection_func))

        self.experts = all_experts


        self.E = len(all_experts)
        self.w = np.ones(self.E)/self.E 
        self.p = self.w
        self.eta =np.ones(self.E) * np.min([.5,np.sqrt(np.log(self.E))]) 
        self.lin_losses = np.zeros((T, self.E))
        self.lin_losses[0] = self.linearized_loss(self.x[0],t=1) 

    def linearized_loss(self, xhat:np.ndarray, t:int):
        """
        Computes the linearized loss at a given point.

        Args:
        - xhat (np.ndarray): The point at which to compute the linearized loss.
        - t (int): The current time step.

        Returns:
        - float: The linearized loss.
        """
        assert xhat.shape[0] == self.domain.dimension 
        ft = lambda x: self.f(t, x)
        grad = estimate_gradient(ft, xhat) 
        return (np.dot(grad, xhat - self.x[t-1])+ self.R * self.G)/(self.R*self.G) 
    

    def run(self, time: int):                       #time starts at 2 since t=1 is initial 
        """
        Runs the algorithm for a given time step.

        Args:
        - time (int): The current time step.

        Returns:
        - None
        """
        p = np.multiply(self.eta,self.w) 
        p = p/np.sum(p) 
        self.p = p
        xi = np.zeros((self.E,self.domain.dimension))
        for i,expert in enumerate(self.experts):
            expert.run(time)
            xi[i] = expert.x[time-1]
            self.x[time-1] += p[i]*xi[i]

        for i in range(self.E):
            self.lin_losses[time-1,i] = self.linearized_loss(xi[i],time)

        new_eta = np.min([.5,np.sqrt(np.log(self.E)/(1+np.sum(self.lin_losses[:time])))])
        w_inside = 1+self.eta*(np.dot(self.lin_losses[time-1,:],p)-self.lin_losses[time-1,:]) 
        self.w = np.power(self.w * w_inside,new_eta/self.eta) 
        self.eta = new_eta

