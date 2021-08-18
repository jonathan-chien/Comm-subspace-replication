function loss = RidgeRegressFitAndPredict_jmc...
	(Ytrain, Xtrain, Ytest, Xtest, alpha, varargin)
% 
% loss = RegressFitAndPredict...
% 	(regressFun, Ytrain, Xtrain, Ytest, Xtest, alpha, varargin) fits
% regression model regressFun (with regression parameters alpha) to
% training target and source data (Ytrain and Xtrain) and then predicts
% test target data Ytest using test source data Xtest and the model fit to
% the training data.
% 
%   K:      target data dimensionality
%   p:      source data dimensionality
%   numPar: numbers of regularization parameters to be tested
%   Ntrain: number of training data points
%   Ntest:  number of testing data points
% 
% INPUTS:
% 
% Ytrain     - training target data matrix (Ntrain x K)
% Xtrain     - training source data matrix (Ntrain x p)
% Ytest      - testing target data matrix (Ntest x K)
% Xtest      - testing source data matrix (Ntest x p)
% alpha      - vector containing the regression parameters to be tested
% (1 x numPar)
% varargin   - additional parameters to be passed to either regressFun or
% RegressPredict (Name-Value pairs)
%
% Note (JMC): The original RegressFitAndPredict function's first positional
% arugment was the regression function handle. This is unnecessary here, as
% this function is dedicated to ridge regression. Perhaps if we expand
% someday to other types of parameterized regression (such as LASSO), we
% may have to add this argument back in.
% 
% OUTPUTS:
% 
% loss - Loss incurred when predicting the test target data Ytest using
% the test source data Xtest and the model fit to the training data Ytrain
% and Xtrain, with the regression parameters in alpha. (1 x numPar)
% 
% @ 2018 Joao Semedo -- joao.d.semedo@gmail.com

% Modification to RegressFitAndPredict by JMC: If regressMethod is
% @RidgeRegress for RegressFitAndPredict, we need to select an optimal
% lambda value within the current outer training fold via CV and then test
% the model featuring this optimal value on the held out outer test fold.
% However, the original RegressFitAndPredict function syntax results in all
% lambdas being tested on the held out outer fold (this is likely because
% the cvParameter (which is a vector of lambdas in the ridge regression
% case) is in the case of RRR actually a vector over the range of number of
% predictive dimmensions, and in the case of RRR we indeed want all models
% corresponding to all values of cvParameter to be tested on the held out
% outer test fold so that we can compare performance across the number of
% predictive dimensions; it does not seem that this code pack has been
% written to also handle the ridge regression case properly, and I think it
% is likely that the code used by the actual Semedo et al team is different
% from what was released here), with no way of knowing which was the
% optimal lambda value within each outer training fold, as we simply have
% the performance of each lambda model on the outer test fold (averaged
% across folds). It is unacceptable and extremely biased to simply select
% the lambda-model leading to best performance on the held out test
% fold(s); rather, the best lambda should be selected via inner CV within
% the current outer training fold and then tested on the held out outer
% training fold. (And of course worst of all would be to select the optimal
% lambda via CV over the full dataset in one process, and then test
% performance via CV over the full dataset with the "optimal" lambda in a
% separate process. This is non-nested CV and allows information "leakage"
% that can inflate the performance estimate.) This modification selects an
% optimal lambda-model within the current outer training fold and uses it
% to obtain a mapping matrix, B, which is then passed to the RegressPredict
% function, which calculates the loss. This must be done in a separately
% named function (and not as a direct to modification to
% RegressFitAndPredict) to avoid a series of recursive calls (since the
% RegresModelSelect function also calls RegressFitAndPredict, which would
% call RegressModelSelect, etc.) that escalates through a series of
% increasingly degnerate states leading first to a series of warnings and
% finally an error.

alphaOpt = RegressModelSelect(@RidgeRegress, Ytrain, Xtrain, alpha);

B = RidgeRegress(Ytrain, Xtrain, alphaOpt, varargin{:});

loss = RegressPredict(Ytest, Xtest, B, varargin{:});

end
