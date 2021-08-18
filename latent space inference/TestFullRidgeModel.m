function [predPerf,predErrors] = TestFullRidgeModel(populations,sourceIdx,targetIdx,nvp)
% Accepts a series of datasets from one session and assesses prediction
% performance of the target population activity based on the source
% poplation activity using cross-validated Ridge Regression.
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
%   'cvNumFolds'    -- Scalar value that is the number of folds to be used
%                      in cross validation. Default = 10.
%   'useParallel'   -- Logical true or false specifying whether to enable
%                      parallel processing during cross-validation routine.
%                      Default true.
%   'calcSource'    -- Logical true or false specifying whether or not to
%                      calculate source dimensionality (this may have been
%                      calculated on a previous run and if we need only to
%                      calculate dimensionality for another target region,
%                      we can set this to false to prevent the source
%                      dimensionality from being recomputed. Default true.
%   'regressMethod' -- Function handle to be passed to auxiliary
%                      cross-validation function, specifying the type of
%                      regression to be used. Default = @RidgeRegress.
%   'scale'         -- Logical true or false. If true, data is z-scored.
%                      Default false.
%   'lossMeasure'   -- Loss measure for regression. Default is 'NSE'
%                      (Normalized squared error).
%   'waitbar'       -- Logical true or false specifying whether to enable
%                      waitbar across datasets. Default true.
% 
% RETURNS
% -------
% predPerf   -- nDatasets x 1 vector of ridge regression performances (1 -
%               cross-validated loss), averaged across reptitions for each
%               dataset.
% predErrors -- nDatasets x 1 vector of 1 SEM of ridge regression
%               performances across folds, averaged across reptitions for
%               each dataset.
%
% Author: Jonathan Chien 8/12/21. Based on methods from Semedo et al 2019
% "Cortical Areas Interact through a Communication Subspace", Neuron.
% Includes source code from
% https://github.com/joao-semedo/communication-subspace

arguments
    populations
    sourceIdx
    targetIdx
    nvp.cvNumFolds = 10
    nvp.useParallel = true
    nvp.regressMethod = @RidgeRegress
    nvp.scale = false % advised to specify this in Extract Spikes, if desired.
    nvp.lossMeasure = 'NSE'
    nvp.waitbar = true
end

% Initialize default options for cross-validation and enable
% parallelization.
cvOptions = statset('crossval');
if nvp.useParallel, cvOptions.UseParallel = true; end

% Determine number of datasets and number of repetitions of each dataset.
nDatasets = size(populations, 1);
nReps = size(populations{1,1}, 3);

% Preallocate over all datasets and repetitions, and initialize waitbar.
predPerfAll = NaN(nDatasets, nReps);
predErrorsAll = NaN(nDatasets, nReps);
if nvp.waitbar, w = waitbar(0, ''); end

% For each repetition of each dataset, run CV routine with RRR.
for iDataset = 1:nDatasets
    for iRep = 1:nReps
        
        % Update waitbar.
        if nvp.waitbar           
            waitbar(iDataset / nDatasets, w,...
                    sprintf('Working on stimulus type %d...', iDataset));
        end
    
        % Set source and target populations.
        sourcePop = populations{iDataset,sourceIdx}(:,:,iRep);
        targetPop = populations{iDataset,targetIdx}(:,:,iRep);
        
        % Calculate range of lambdas to test for current
        % population/repetition.
        lambdas = GetRidgeLambda(.5:.01:1, sourcePop, 'Scale', nvp.scale);
                             
        % Define auxiliary function for cross-validation routine. Must be
        % defined anew for each dataset/repetition because lambda range
        % is determined for each dataset/repetition. Note that we are using
        % RidgeRegressFitAndPredict_jmc rather than RegressFitAndPredict.
        nvp.cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RidgeRegressFitAndPredict_jmc...
            (Ytrain, Xtrain, Ytest, Xtest, ...
             lambdas, 'LossMeasure', nvp.lossMeasure, 'Scale', nvp.scale);
        
        % Cross-validation routine.
        cvl = crossval(nvp.cvFun, targetPop, sourcePop, ...
              'KFold', nvp.cvNumFolds, ...
            'Options', cvOptions);

        % Store cross-validation results: mean loss and standard error of
        % the mean across folds.
        cvLoss = [ mean(cvl); std(cvl)/sqrt(nvp.cvNumFolds) ];
        
        % Store Reduced Rank Regression cross-validation results for
        % current repetition of current dataset. 
        predPerfAll(iDataset,iRep) = 1-cvLoss(1);
        predErrorsAll(iDataset,iRep) = cvLoss(2);
    end
end

if nvp.waitbar, close(w); end


% Average across repetitions for each dataset.
predPerf = mean(predPerfAll, 2);
predErrors = mean(predErrorsAll, 2);

end