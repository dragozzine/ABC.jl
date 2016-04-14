# ex1.jl
# by Eric B. Ford
# comments by Darin Ragozzine

# load modules with information on how to run Eric's ABC and statistical distributions
using ABC
using Distributions

# set up 100 data points
num_data_default = 100
# make a function that generates a distribution of num_data_default size 
# from a multivariate normal with widths taken from theta array
# this is the generative ("forward") model for this example
gen_data_normal(theta::Array, n::Integer = num_data_default) = 
	rand(Normal(theta[1],theta[2]),num_data_default)
## note that the "continuation character" in Julia is just to 
## leave the statement unfinished

# set up some simple short quick functions
normalize_theta2_pos(theta::Array) =  theta[2] = abs(theta[2])
is_valid_theta2_pos(theta::Array) =  theta[2]>0.0 ? true : false

# Set the true parameters theta, which are going to be inferred by the code
theta_true = [0.0, 1.0]
# Use as a prior a Gaussian centered on the true value with a width of 1
param_prior = Distributions.MvNormal(theta_true,ones(length(theta_true)))

# set up the ABC run
# gen_data_normal = the function that generates data, e.g., the forward model
#		; in this case, a multivariate normal
# ABC.calc_summary_stats_mean_var = function that is used to calculate the summary
#		statistics; in this case, the summary statistics are the mean and variance
#		of the generated distribution
# ABC.calc_dist_max = the metric function used to calculate the ABC distance
#		; in this case, the metric is the maximum difference between 
#		the two sets of data (or the summary stats?)
# param_prior = the prior for the parameters defined above
# is_valid = the function that will check whether the parameters are 
#            in a valid region of parameter space, in this case, that theta[2] must be >0
# num_max_attempt = maximum number of attempts to generate new parameters that produce
#		epsilon values that are acceptable
abc_plan = abc_pmc_plan_type(gen_data_normal,ABC.calc_summary_stats_mean_var,
ABC.calc_dist_max, param_prior; is_valid=is_valid_theta2_pos,num_max_attempt=10000);
# NOT SET was the number of ABC PMC particles
# NOT SET was the initial epsilon, the epsilon reduction factors, the target epsilon
# NOT SET was the maximum number of population iterations (max_times)


num_param = 2
# generate the true dataset that is going to be fit
data_true = abc_plan.gen_data(theta_true)

# calculate the summary statistics for the true data
# in this case, the mean and variance
ss_true = abc_plan.calc_summary_stats(data_true)
println("theta= ",theta_true," ss= ",ss_true, " d= ", 0.)

# run the ABC algorithm
# use as an input the "plan" as defined above
# use "verbose" to get the output line (1 per generation)
# pop_out is the result and contains the weights
# parameter values, distances, weighted covariance matrix
# (and more?) 
pop_out = run_abc(abc_plan,ss_true;verbose=true);


# make some plots of the output
using PyPlot

# make a histogram of the distribution of weights
PyPlot.hist(pop_out.weights*length(pop_out.weights));
# make a histogram of the distribution of distances (for the final population?) 
PyPlot.hist(pop_out.dist);


# make a 2-d contour plot for the 2 parameters

# setup the 2-d grid array
num_param = 2
num_grid_x = 100
num_grid_y = 100
limit = 1.0
x = linspace(theta_true[1]-limit,theta_true[1]+limit,num_grid_x);
y = linspace(theta_true[2]-limit,theta_true[2]+limit,num_grid_y);
z = zeros(Float64,(num_param,length(x),length(y)))
for i in 1:length(x), j in 1:length(y) 
    z[1,i,j] = x[i]
    z[2,i,j] = y[j]
end
z = reshape(z,(num_param,length(x)*length(y)))

# for each location in the grid, use the ABC parameters, weights, and covariance matrix 
# to determine the probability distribution function for every location in the grid
zz = [ ABC.pdf(ABC.GaussianMixtureModelCommonCovar(pop_out.theta,pop_out.weights,ABC.cov_weighted(pop_out.theta',pop_out.weights)),vec(z[:,i])) for i in 1:size(z,2) ]
zz = reshape(zz ,(length(x),length(y)));

# calculate the 1, 2, 3, 4, 5-sigma equivalent contours based on these probabilities
levels = [exp(-0.5*i^2)/sqrt(2pi^num_param) for i in 5:-1:0];

# make a contour plot of the PDF for the parameters based on the ABC output
PyPlot.contour(x,y,zz',levels);
# add the final draw from the posterior distribution from ABC
plot(pop_out.theta[1,:],pop_out.theta[2,:],".");
# plot the true value of the parameters as an x
plot(theta_true[1],theta_true[2],"x"); # true value

# this is a script, not a function, so no "end" of any kind is needed
