#%%
#%load_ext autoreload
#%autoreload 2
#import warnings
from utils.essential_imports import *
## Functions to minimize
import utils.funcs_and_grads as objectives
## Baseline algorithms
import utils.true_algos as algos
## Create model
import utils.models as models

######################################################


torch.manual_seed(46845712318)


## Dimension of the problem
#%%
############ Create function to minimize ############
N_samples = 1

from sklearn.datasets import load_svmlight_file
A, b = load_svmlight_file('Data/datasets/mushrooms')
A = torch.tensor(A.todense())
b = torch.tensor(b)
b[b==2] = -1.

M = A.shape[0]
P = A.shape[1]

f = objectives.logistic
xstar = None
lambdamin = 1/M #smallest eigenvalue (comes from the regularization)
lambdamax = 1/2 * torch.sum(A**2, dim=(-1, -2)) + 1/M
func = objectives.function_to_minimize(f=f, A=A, b=b, xstar=xstar, lambdamax=lambdamax)
#%%
########## Prepare initialization #########
xm1 = torch.randn(N_samples, P)
gradxm1 = func.evalgradf(xm1.clone().requires_grad_(True))
gamma_1ststep = 1/(2*func.lambdamax)
if len(gamma_1ststep.shape)>0:
    gamma_1ststep = gamma_1ststep.unsqueeze(-1) #add dim for broadcasting to size P
x0 = xm1 - gamma_1ststep * gradxm1
gradx0 = func.evalgradf(x0.clone().requires_grad_(True))
dm1_0 = x0.clone().detach() - xm1.clone().detach()
DG = gradx0 - gradxm1
BB_stepsize = torch.sum((DG * dm1_0), axis=-1) / torch.sum( (DG * DG), axis=-1)
#Initialize BFGS matrix using the Barzilai-Borwein step-size
first_guess = 0.8 * BB_stepsize[:, None, None] * torch.eye(P)

# %%
########### Initialize the LOA algorithm ###########
gamma = 1.
layers_out_dim_factors = [2]  #only a single linear layer
safe_init = 'BFGS_like'

LAO_BFGS = models.LOA_BFGS_Model(P=P, layers_out_dim_factors=layers_out_dim_factors, gamma=gamma)

LAO_BFGS.load_state_dict(torch.load("Data/pretrained_LOA_BFGS_Model.npy"))



#%%
########## Run the algorithm #########
niter_max = 2000
Wolfe_LS = True

LAO_BFGS.reset_mat(n_traj=N_samples, first_guess=first_guess)
x, gradm1, grad, dm1 = x0.clone(), gradxm1.clone(), gradx0.clone(), dm1_0.clone() 
list_values_LOA = torch.zeros(N_samples, niter_max + 1)

for iteration in range(niter_max):
    current_value = func.evalf(x)  #Compute current value for this func and point
    grad = func.evalgradf(x.clone().detach().squeeze().requires_grad_(True))
    list_values_LOA[:, iteration] = current_value
    previous_stepsize = torch.clone(gamma) if torch.is_tensor(gamma) else gamma
    #set current step-size
    stepsize = gamma #might be overwritten by line-search
    next_step = LAO_BFGS(grad, gradm1, dm1, previous_gamma=previous_stepsize)  #Predict all trajectories at once.
    ## Optionally do line-search
    if Wolfe_LS:
        #Start LS, it should not be part of the graph
        with torch.no_grad():
            c1 = 1e-2  #Params of the linesearch
            step = next_step.clone().detach()
            stepXgrad = torch.sum(step * grad, dim=-1)
            # Do a line search for each element
            n_LS = 0 ; 
            stepsize = torch.ones(N_samples, 1)  #If line-search, start with unit step-size
            Wolfecrit = torch.ones(N_samples, dtype=bool) #mask for successful LS
            temp_values = current_value.clone()
            while n_LS < 5 and torch.sum(Wolfecrit)>0:  #Try 5 times the line-search
                n_LS += 1
                y = x + stepsize * step  #copy x
                temp_values[Wolfecrit] = func.evalf(y)[Wolfecrit] #Not optimized at all!
                Wolfecrit = (temp_values - current_value - c1 * stepsize.squeeze() * stepXgrad) > 0   #this criterion must be negative to exist the LS loop
                stepsize[Wolfecrit] *= 0.1 #Update stepsizes for which it failed
        #Once a new step-size is chosen, it should be part of the graph
        #########################
    #Store step-size for next iter (very important)
    previous_stepsize = stepsize  #store step-size for next iter
    ##update all variables
    dm1 = stepsize * next_step #step between xk and xk+1
    gradm1 = grad.clone().detach() #current grad is stored for next iter
    x = x + dm1 # compute xk+1
list_values_LOA[:, -1] = func.evalf(x) #compute final value
list_values_LOA = list_values_LOA.clone().detach()


#%%
########## Compute baselines ##########
list_values_BFGS = algos.BFGS(x0, xm1, func, gamma_BFGS=gamma, first_guess=None, niter_max=niter_max, BB_init=True, Wolfe_LS=Wolfe_LS)
niter_Newton = 20
list_values_Newton = algos.Newton(x0, func, gamma_Newton=gamma, niter_max=niter_Newton, Wolfe_LS=Wolfe_LS)
list_values_GD = algos.GD(x0, func, gamma=1.99*gamma_1ststep, niter_max=niter_max)
list_values_Nesterov = algos.Nesterov(x0, func, gamma=1.99*gamma_1ststep, alpha=3.01, niter_max=niter_max)
list_values_HB = algos.HeavyBall(x0, func, gamma=1.99*gamma_1ststep, alpha=2*torch.sqrt(torch.tensor(lambdamin)), niter_max=niter_max)



#%%
########### Plot the result ############

def plot_area(xarray, values, ax, label, color=None, zorder=5): 
    below, _ = torch.min(values, dim=0) 
    above, _ = torch.max(values, dim=0)
    ax.plot(xarray, torch.median(values, dim=0)[0], lw=3, color=color, zorder=zorder, label=label)
    ax.fill_between(xarray, below, above, lw=3, color=color, alpha=0.2)
    pass


fig, ax = plt.subplots(figsize=(6,5))
## Newton and QN
fstar = torch.min(list_values_Newton, dim=-1)[0].unsqueeze(-1)
initial_gap = func.evalf(x0).unsqueeze(-1) - fstar

plot_area(np.arange(niter_max+1), (list_values_LOA-fstar)/initial_gap, ax=ax, color='dodgerblue', label='LOA BFGS', zorder=10)
plot_area(np.arange(niter_max+1), (list_values_BFGS-fstar)/initial_gap, ax=ax, color='magenta', label='vanilla BFGS', zorder=4)
plot_area(np.arange(niter_Newton+1), (list_values_Newton-fstar)/initial_gap, ax=ax, color='black', label='Newton', zorder=4)
## First order
plot_area(np.arange(niter_max+1), (list_values_GD-fstar)/initial_gap, ax=ax, color='limegreen', label='Gradient Descent', zorder=4)
plot_area(np.arange(niter_max+1), (list_values_Nesterov-fstar)/initial_gap, ax=ax, color='red', label="Nesterov's Accelerated Gradient", zorder=4)
plot_area(np.arange(niter_max+1), (list_values_HB-fstar)/initial_gap, ax=ax, color='orange', label="Heavy-Ball", zorder=4)


# parameters of the plot
ax.set_xlabel(r'iteration index $k$')
#ax.set_ylabel(r'relative suboptimality $\frac{f(x_k) - f^\star}{f(x_0)-f^\star}$',fontsize=14)
ax.set_title(r'Logistic regression on mushrooms dataset', fontsize=14)
ax.set_xlim(0, niter_max)
ax.set_yscale('log')
#if ax.get_ylim()[0]<1e-12:
#    ax.set_ylim(ymin=1e-12)
#ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig('logistic_mushrooms_comparison.pdf')
fig.show()
# %%

