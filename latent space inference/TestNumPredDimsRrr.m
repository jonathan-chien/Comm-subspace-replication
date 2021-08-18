function [optDims,predPerf,predErrors] = TestNumPredDimsRrr ...
    (populations,sourceIdx,targetIdx,nvp)
% Takes as input a series of datasets (over various stimulus types), each
% repeated nReps times, and for each repetition of each dataset, calculates
% prediction performance of the source population on the target populations
% over a series of rank constraints for reduced rank regression.
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
% sourceIdx -- Index of source population within "populations" (column
%              index). Should be 1.
% targetIdx -- Index of target population wihthi "populations" (column
%              index). Set 2 for target V1 and 3 for V2.
% Name-Value Pairs (nvp)
%   'nDimsUsedForPred' -- Vector of rank constraints to be used for reduced
%                         rank regression. Default = 1:10
%   'cvNumFolds'       -- Scalar value that is the number of folds to be
%                         used in cross validation. Default = 10.
%   'useParallel'      -- Logical true or false specifying whether to
%                         enable parallel processing during
%                         cross-validation routine. Default true.
%   'regressMethod'    -- Function handle to be passed to auxiliary
%                         cross-validation function, specifying the type of
%                         regression to be used. Default =
%                         @ReducedRankRegress.
%   'lossMeasure'      -- Loss measure for regression. Default is 'NSE'
%                         (Normalized squared error).
%   'ridgeInit'        -- Logical true or false, specifying whether or not
%                         to use ridge regression as the full-regression
%                         model (if true), for whose reconstruction of the
%                         target population data we seek a low-rank
%                         reconstruction (this reconstruction error is
%                         added to the regression loss function, and the
%                         Eckart-Young theorem guarantees that SVD provides
%                         an analytic and best-possible solution, as
%                         measured by minimizing the Frobenius norm of the
%                         matrix difference between the full rank
%                         regressand reconstruction (projection of response
%                         vectors into the design matrix's column space)
%                         and its low-rank reconstruction (found via
%                         PCA/SVD). If false, an OLS model is fitted, which
%                         obviously is MUCH faster, as there is no
%                         cross-validtion (searching for the correct lambda
%                         in ridge requires a nested CV procedure).
%   'scale'            -- Logical true or false. If true, data is z-scored.
%                         Default false.
%   'waitbar'          -- Logical true or false specifying whether to enable
%                         waitbar across datasets. Default true.
%
% RETURNS
% optDims    -- nDatasets x 1 vector of optimal number of predictive
%               dimensions (averaged across repetitions of each dataset).
%               For each reptitions, the lowest number of predictive
%               dimensions within 1 SEM of peak performance is selected.
% predPerf   -- nDatasets x length(nDimsUsedForPred) matrix of
%               performances of the reduced rank model using the source
%               population activity to predict target population activity
%               (1- cross-validated loss). Each element has been averaged
%               across repetitions.
% predErrors -- nDatasets x length(nDimsUsedForPred) matrix of 1 SEM values
%               of the reduced rank model using the source population
%               activity to predict target population activity (1-
%               cross-validated loss). Each element has been averaged
%               across repetitions.
%
% Author: Jonathan Chien 8/12/21. Based on methods from Semedo et al 2019
% "Cortical Areas Interact through a Communication Subspace", Neuron.
% Includes source code from
% https://github.com/joao-semedo/communication-subspace


arguments
    populations
    sourceIdx
    targetIdx
    nvp.nDimsUsedForPred = 1:10
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

% Auxiliary function to be used within the cross-validation routine (type
% 'help crossval' for more information). Briefly, it takes as input the
% the train and test sets, fits the model to the train set and uses it to
% predict the test set, reporting the model's test performance. Here we
% use NSE (Normalized Squared Error) as the performance metric. MSE (Mean
% Squared Error) is also available. JMC: This is done inside the
% loop because the value of nvp.nDimsUsedForPred will depend on the
% population sizes if nvp.useAllPredDims is true, and the size of
% the current dataset must thus be known.
cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
    (nvp.regressMethod, Ytrain, Xtrain, Ytest, Xtest, ...
    nvp.nDimsUsedForPred, 'LossMeasure', nvp.lossMeasure, ...
    'RidgeInit', nvp.ridgeInit, 'Scale', nvp.scale);

% Determine number of datasets and repetitions of each dataset.
nDatasets = size(populations, 1);
nReps = size(populations{1,1}, 3);

% Preallocate for results over all datasets and repetitions, and initialize
% waitbar.
optDimsAll = NaN(nDatasets, nReps);
predPerfAll = NaN(nDatasets, length(nvp.nDimsUsedForPred), nReps);
predErrorsAll = NaN(nDatasets, length(nvp.nDimsUsedForPred), nReps);
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
        optDimsAll(iDataset,iRep) = ModelSelect(cvLoss, nvp.nDimsUsedForPred);
        predPerfAll(iDataset,:,iRep) = 1-cvLoss(1,:);
        predErrorsAll(iDataset,:,iRep) = cvLoss(2,:);
    end
end

if nvp.waitbar, close(w); end

% Average over repetitions for each dataset.
optDims = mean(optDimsAll, 2);
predPerf = squeeze(mean(predPerfAll, 3));
predErrors = squeeze(mean(predErrorsAll, 3));

if size(predPerfAll, 2) > 1
    a = 1;
end

end