# Load X and y variable
data = readcsv("discrete_hist.csv")
(n,m) = size(data)
@printf("Size: %d x %d\n", n, m)

data = data[shuffle(1:end), :]
train_size = Int(round(n/500))
X = data[1:train_size, 1:end-1]
y = data[1:train_size, m]
Xtest = data[train_size:train_size*10, 1:end-1]
ytest = data[train_size:train_size*10, m]

# Data is sorted, so *randomly* split into train and validation:
n = size(X,1)
perm = randperm(n)
validStart = Int64(Int(round(n/2+1))) # Start of validation indices
validEnd = Int64(n) # End of validation indices
validNdx = perm[validStart:validEnd] # Indices of validation examples
trainNdx = perm[setdiff(1:n,validStart:validEnd)] # Indices of training examples
Xtrain = X[trainNdx,:]
ytrain = y[trainNdx]
Xvalid = X[validNdx,:]
yvalid = y[validNdx]

# Find best value of RBF variance parameter,
#	training on the train set and validating on the test set
include("leastSquares.jl")
include("misc.jl")
minErr = Inf
lambda = 10.0^-12
bestSigma = []

k = 10
sample = n_folds(k, Xtrain, ytrain)
sigma = 2.0.^(-5:0.1:0)
for i in 1:k
	(Xtraining_set, ytraining_set, Xvalidation_set, yvalidation_set) = sample(i) # not random sample, selects a partition deterministically based on i
	# Train on the training set
	model = leastSquaresRBF(Xtraining_set, ytraining_set,sigma[i],lambda)

	# Compute the error on the validation set
	yhat = model.predict(Xvalidation_set)
	validError = sum((yhat - yvalidation_set).^2)/(n/2)
	@printf("With sigma = %.3f, validError = %.2f\n",sigma[i],validError)

	# Keep track of the lowest validation error
	if validError < minErr
		minErr = validError
		bestSigma = sigma[i]
	end

end

# Now fit the model based on the full dataset
model = leastSquaresRBF(X,y,bestSigma,lambda)

# Report the error on the test set
t = size(Xtest,1)
yhat = model.predict(Xtest)
testError = sum((yhat - ytest).^2)/t
@printf("With best sigma of %.3f, testError = %.2f\n",bestSigma,testError)

# Plot model
using PyPlot
figure()

# Fortunately my test data is so dense that it covers most of the domain anyways
title("Crime Heat Map")
xlabel("X")
ylabel("Y")
zlabel("Time")
grid("on")

scatter3D(Xtest[:,2], Xtest[:,3], Xtest[:,1], s=ytest.*1000, c="orange", alpha=0.1)
scatter3D(Xtest[:,2], Xtest[:,3], Xtest[:,1], s=yhat.*1000, c="blue", alpha=0.1)
