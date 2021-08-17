function [optDimFactorRegress,predPerf,predErrors] = TestNumPredDimsFr ...
    (populations,sourceIdx,targetIdx,nvp)
% Takes as input a series of datasets (over various stimulus types), each
% repeated nReps times, and for each repetition of each dataset, calculates
% prediction performance of the source population on the target populations
% over a series of regressors that are factors found using factor analysis.
% (This first finds dimensions of maximal shared variance in the source
% population, with no consideration of a target, and then uses these
% dimenions/factors as regressors). The principal drawback of this method
% (used to make a point in the paper) is that the dimensions that capture
% maximal shared variance amongst a set of predictors may not be the same
% ones that explain the most variance in a response variable. 
%
% PARAMETERS
% ----------
% populations -- nDatasets x 3 cell array, where rows correspond to
%                stimulus types and columns to populations (source V1,
%                targetV1, and V2 for columns 1, 2, and 3 respectively).
%                Note that target V1 and V2 have been
%                mean-matched/subsampled here. Each cell contains an
%                nTimepoints*nStimTrials x nUnits x nRepetitions array of
%                either firing rates or residuals (see the 'residuals'
%                name-value pair under PARAMETERS section of
%                ConstructSourceAndTarget) for the corresponding
%                population.
% sourceIdx   -- Index of source population within "populations" (column
%                index). Should be 1.
% targetIdx   -- Index of target population wihthi "populations" (column
%                index). Set 2 for target V1 and 3 for V2.
% Name-Value Pairs (nvp)
%   'nDimsUsedForPred' -- Vector of number of factors to be tested for
%                         reduced factor regression. Default = 1:10
%   'cvNumFolds'       -- Scalar value that is the number of folds to be
%                         used in cross validation. Default = 10.
%   'useParallel'      -- Logical true or false specifying whether to
%                         enable parallel processing during
%                         cross-validation routine. Default true.
%   'regressMethod'    -- Function handle to be passed to auxiliary
%                         cross-validation function, specifying the type of
%                         regression to be used. Default =
%                         @FactorRegress.
%   'lossMeasure'      -- Loss measure for regression. Default is 'NSE'
%                         (Normalized squared error).
%   'waitbar'          -- Logical true or false specifying whether to enable
%                         waitbar across datasets. Default true.
% 
% RETURNS
% -------
% optDims    -- nDatasets x 1 vector of optimal number of predictive
%               dimensions (averaged across repetitions of each dataset).
%               For each reptitions, the lowest number of predictive
%               dimensions within 1 SEM of peak performance is selected.
% predPerf   -- nDatasets x length(nDimsUsedForPred) matrix of
%               performances of the factor regression modl using the source
%               population activity to predict target population activity
%               (1- cross-validated loss). Each element has been averaged
%               across repetitions.
% predErrors -- nDatasets x length(nDimsUsedForPred) matrix of 1 SEM values
%               of the factor regression model using the source population
%               activity to predict target population activity (1-
%               cross-validated loss). Each element has been averaged
%               across repetitions.
%
% Author: Jonathan Chien 8/13/21. Based on methods from Semedo et al 2019
% "Cortical Areas Interact through a Communication Subspace", Neuron.
% Includes source code from
% https://github.com/joao-semedo/communication-subspace


arguments
    populations
    sourceIdx
    targetIdx
    nvp.nDimsUsedForPrediction = 1:10
    nvp.cvNumFolds = 10
    nvp.useParallel = true
    nvp.regressMethod = @FactorRegress;
    nvp.lossMeasure = 'NSE'
    nvp.waitbar
end

% Initialize default options for cross-validation and enable
% parallelization.
cvOptions = statset('crossval');
if nvp.useParallel, cvOptions.UseParallel = true; end

% Determine number of datasets and number of repetitions of each dataset.
nDatasets = size(populations, 1);
nReps = size(populations{1,1}, 3);

% Preallocate and initialize waitbar.
optDimFactorRegressAll = NaN(nDatasets, nReps);
predPerfAll = NaN(nDatasets, length(nvp.nDimsUsedForPrediction), nReps);
predErrorsAll = NaN(nDatasets, length(nvp.nDimsUsedForPrediction), nReps);
if nvp.waitbar, w = waitbar(0, ''); end

% Define cross-validation auxiliary function. qOptSource is an extra
% argument for FactorRegress. Extra arguments for the regression function
% are passed as name/value pairs after the cross-validation parameter (in
% this case nDimsUsedForPrediction). qOptSource, the optimal factor
% analysis dimensionality for the source activity X, must be provided when
% cross-validating Factor Regression. When absent, Factor Regression will
% automatically determine qOptSource via cross-validation (which will
% generate an error if Factor Regression is itself used within a
% cross-validation procedure). JMC: However, we will set the 'qOpt' nvp to
% be equal to the max of the nDimsUsedForPrediction, ensuring that the same
% number (such as 10) of factors are used as regressors each time;
% otherwise, the CV routine actually ignores the cvParamter
% (nDimsUsedForPredictions) and only tests a number of predictive
% dimensions up to the "optimal" dimensionality. Testing all
% nDimsUsedForPrediction instead makes the data much more uniform and
% easier to deal with; it also more closely matches what was done in the
% paper.
cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
    (nvp.regressMethod, Ytrain, Xtrain, Ytest, Xtest, ...
    nvp.nDimsUsedForPrediction, ...
    'LossMeasure', nvp.lossMeasure, 'qOpt', nvp.nDimsUsedForPrediction(end));

% For each repetition of each dataset, run FA (to assess dimensionality)
% and FR (predictive performance).
for iDataset = 1:nDatasets
    for iRep = 1:nReps
        
        % Update waitbar.
        if nvp.waitbar
            waitbar(iDataset / nDatasets, w,...
                    sprintf('Working on dataset %d of %d...', iDataset, nDatasets));
        end
    
        % Set source and target populations for current dataset and
        % repetition.
        sourcePop = populations{iDataset,sourceIdx}(:,:,iRep);
        targetPop = populations{iDataset,targetIdx}(:,:,iRep);
        
        % Cross-validation routine (loss is in matrix across folds (rows) and
        % predictive dimensions (columns)).
        cvl = crossval(cvFun, targetPop, sourcePop, ...
              'KFold', nvp.cvNumFolds, ...
            'Options', cvOptions);

        % Calculate mean loss and standard error of the mean across folds.
        cvLoss = [ mean(cvl); std(cvl)/sqrt(nvp.cvNumFolds) ];

        % Store Factor Regression cross-validation results for
        % current repetition of current dataset: optimal number of
        % predictive dimensions, mean prediction performance across number
        % of predictive dimensions, and SEM of mean prediction performance
        % across number of predictive dimensions.
        optDimFactorRegressAll(iDataset,iRep) = ModelSelect...
            (cvLoss, nvp.nDimsUsedForPrediction);
        predPerfAll(iDataset,:,iRep) = 1-cvLoss(1,:);
        predErrorsAll(iDataset,:,iRep) = cvLoss(2,:);
    end
end

if nvp.waitbar, close(w); end

% Average over repetitions for each dataset.
optDimFactorRegress = mean(optDimFactorRegressAll, 2);
predPerf = mean(predPerfAll, 3);
predErrors = mean(predErrorsAll, 3);

end