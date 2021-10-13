from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import pdb

def estimateQ(model,X):
	"""[summary]

	Args:
		model ([type]): [description]
		X ([type]): [description]

	Returns:
		[type]: [description]
	"""
	Q = 0
	N = np.shape(X)[0]
	M = model.n_components

	postProbs = model.predict_proba(X)
	wLogProbs = model._estimate_weighted_log_prob(X)

	for n in range(N):
		for m in range(M):
			Q = postProbs[n,m]*wLogProbs[n,m]

	return Q

def EMestimation(model,X):
	"""This function implements the EM algorithm to find the weights, means and covariances that fit the data into a GMM with M components

	Args:
		model ([type]): [description]
		X ([type]): [description]

	Returns:
		[type]: [description]
	"""

	M = model.n_components
	N = np.shape(X)[0]
	d = np.shape(X)[1]

	Q = []

	# Obtain the parameters calculated by fitting the model
	weightsA = model.weights_[:]
	meansA = model.means_[:]
	covA = model.covariances_[:]

	# Calculate the first Q
	Q.append(estimateQ(model,X))

	delta = 1e-3 # Minimal distance between Q values to considerate the algorithm has converged
	regConst = 0.1 # Regularization constant

	converged = False
	maxIter = 10
	iter = 0

	print(f"Iter.: {iter}, Q: {Q[-1]}")

	while converged == False and iter < maxIter:
		postProbs = model.predict_proba(X) # Posterior probabilities. Matrix [N x M]

		# Estimate new weights, means and covariance matrices for every component
		weightsB = np.sum(postProbs, axis=0)/N

		meansB = np.zeros(np.shape(meansA))

		for m in range(M):
			den = sum(postProbs[:,m])
			num = 0
			for n in range(N):
				num += X[n]*postProbs[n,m]

		covB = np.zeros(np.shape(covA))

		for m in range(M):
			elem = 0

			for n in range(N):
				arr = X[n]-meansB[m]
				mat = np.zeros((len(arr),len(arr)))
				for i in range(len(arr)):
					mat[i] = arr[i]*arr

				elem += mat*postProbs[n,m]

			covB[m] = (elem*np.eye(d) + regConst*np.eye(d))/(np.sum(postProbs,axis=0)[m] + 1)

		model.weights_ = weightsB[:]
		model.means_ = meansB[:]
		model.covariances_ = covB[:]

		QB = estimateQ(model,X)

		# If the gain of Q has been less than delta or Q has decreased, stop and discard last result
		if (abs(Q[-1]-QB)<= delta) or (QB < Q[-1]):
			converged = True
			model.weights_ = weightsA[:]
			model.means_ = meansA[:]
			model.covariances_ = covA[:]

		# If not, update values and perform another iteration
		else:
			weightsA = weightsB[:]
			meansA = meansB[:]
			covA = covB[:]
			Q.append(QB)
			iter += 1
			print(f"Iter.: {iter} Q: {Q[-1]}")

	print(f"\nN. of iterations: {iter}")
	print(f"Initial Q: {Q[0]}")
	print(f"Best Q: {Q[-1]}")
	print(Q)

	return model

def mergeCriteria(model,X):
	"""This function calculates the merge criteria for every combination of components,
	and returns the results into a symmetric matrix
	It also finds which combination has the maximim vaue.

	Args:
		model ([type]): [description]
		X ([type]): [description]

	Returns:
		[type]: [description]
	"""	""""""
	

	M = model.n_components

	predictProba = model.predict_proba(X)
	mC = np.zeros((M,M))

	for m in range(M):
		for n in range(M):
			prob = predictProba[:,m] @ predictProba[:,n]

			if m == n:
				mC[m,m] = prob

			else:
				mC[m,n] = prob
				mC[n,m] = prob

	result = np.where(mC == np.amax(mC))

	listOfCordinates = list(zip(result[0], result[1]))

	return mC, listOfCordinates[0]

def kl_divergence(p, q):
	"""[summary]

	Args:
		p ([type]): [description]
		q ([type]): [description]

	Returns:
		[type]: [description]
	"""
	return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def splitCriteria(model,X):
	"""[summary]

	Args:
		model ([type]): [description]
		X ([type]): [description]

	Returns:
		[type]: [description]
	"""

	M = model.n_components
	N = np.shape(X)[0]
	dim = np.shape(X)[1]

	sC = np.zeros((M)) # Split criterion array,[M x 1]

	postProbs = model.predict_proba(X) # Posterior probabilities matrix [N x M]
	logProbs = model._estimate_log_prob(X) # Log probabilities matrix [N x M]

	for m in range(M):
		pk = np.zeros(np.shape(X)) # Local data density [N x dim]

		postProbsM = postProbs[:,m] # Posterior probabilities for m component [N x 1]
		gM = np.exp(logProbs[:,m]) # Lineal likehood probabilities for m component [N x 1]

		den = sum(postProbsM) # Denominator of expression for calculating pk

		for n in range(N):
			num = np.zeros((dim))
			for l in range(N):
				num += (X[n] - X[l])*postProbsM[l]
			pk[n] = num/den

		sC[m] = kl_divergence(pk,gM)

	maxComponent = np.argmax(sC)

	return sC[m], maxComponent
