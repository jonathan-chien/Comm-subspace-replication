function [optDimFactorRegress,predPerf,predErrors] = TestNumPredDimsFr ...
    (populations,sourceIdx,targetIdx,nvp)

arguments
    populations
    sourceIdx
    targetIdx
    nvp.nDimsUsedForPrediction = 1:10
    nvp.cvNumFolds = 10
    nvp.regressMethod = @FactorRegress;
    nvp.lossMeasure = 'NSE'
    nvp.useParallel = true
    nvp.waitbar
end

% Initialize default options for cross-validation and enable
% parallelization.
nvp.cvOptions = statset('crossval');
if nvp.useParallel, nvp.cvOptions.UseParallel = true; end

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
            'Options', nvp.cvOptions);

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