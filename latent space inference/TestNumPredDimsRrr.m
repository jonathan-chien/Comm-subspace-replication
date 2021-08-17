function [optDims,predPerf,predErrors] = TestNumPredDimsRrr ...
    (populations,sourceIdx,targetIdx,nvp)
% Takes as input



arguments
    populations
    sourceIdx
    targetIdx
    nvp.nDimsUsedForPrediction = 1:10
    nvp.cvNumFolds = 10
    nvp.useParallel = true
    nvp.regressMethod = @ReducedRankRegress
    nvp.lossMeasure = 'NSE'
    nvp.ridgeInit = true
    nvp.scale = false % advised to specify this in Extract Spikes, if desired.
    nvp.waitbar = true
end

%% Cross-validate Reduced Rank Regression

% Initialize default options for cross-validation and enable parallel
% cross-validation.
cvOptions = statset('crossval');
if nvp.useParallel, cvOptions.UseParallel = true; end

% Regression method to be used.
nvp.regressMethod = @ReducedRankRegress;

% Auxiliary function to be used within the cross-validation routine (type
% 'help crossval' for more information). Briefly, it takes as input the
% the train and test sets, fits the model to the train set and uses it to
% predict the test set, reporting the model's test performance. Here we
% use NSE (Normalized Squared Error) as the performance metric. MSE (Mean
% Squared Error) is also available. JMC: this is the 'fun' arg for MATLAB's
% crossval function.
cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
    (nvp.regressMethod, Ytrain, Xtrain, Ytest, Xtest, ...
    nvp.nDimsUsedForPrediction, 'LossMeasure', nvp.lossMeasure, ...
    'RidgeInit', nvp.ridgeInit, 'Scale', nvp.scale);

% Determine number of datasets and repetitions of each dataset.
nDatasets = size(populations, 1);
nReps = size(populations{1,1}, 3);

% Preallocate for results over all datasets and repetitions, and initialize
% waitbar.
optDimsAll = NaN(nDatasets, nReps);
predPerfAll = NaN(nDatasets, length(nvp.nDimsUsedForPrediction), nReps);
predErrorsAll = NaN(nDatasets, length(nvp.nDimsUsedForPrediction), nReps);
if nvp.waitbar, w = waitbar(0, ''); end

% For each repetition of each dataset, run CV routine with RRR.
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
        
        % Cross-validation routine.
        cvl = crossval(cvFun, targetPop, sourcePop, ...
              'KFold', nvp.cvNumFolds, ...
            'Options', cvOptions);

        % Calculate mean loss and standard error of the mean across folds.
        cvLoss = [ mean(cvl); std(cvl)/sqrt(nvp.cvNumFolds) ];
        
        % Store Reduced Rank Regression cross-validation results for
        % current repetition of current dataset: optimal number of
        % predictive dimensions, mean prediction performance across number
        % of predictive dimensions, and SEM of mean prediction performance
        % across number of predictive dimensions.
        optDimsAll(iDataset,iRep) = ModelSelect(cvLoss, nvp.nDimsUsedForPrediction);
        predPerfAll(iDataset,:,iRep) = 1-cvLoss(1,:);
        predErrorsAll(iDataset,:,iRep) = cvLoss(2,:);
    end
end

if nvp.waitbar, close(w); end

% Average over repetitions for each dataset.
optDims = mean(optDimsAll, 2);
predPerf = squeeze(mean(predPerfAll, 3));
predErrors = squeeze(mean(predErrorsAll, 3));

end