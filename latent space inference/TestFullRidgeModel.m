function [predPerf,predErrors] = TestFullRidgeModel(populations,sourceIdx,targetIdx,nvp)
% Accepts a series of datasets from one session and assesses prediction
% performance of the target population activity based on the source
% poplation activity using cross-validated Ridge Regression.
%
% PARAMETERS
% ----------
% populations
% sourceIdx
% targetIdx
% Name-Value Pairs (nvp)
%   'cvNumFolds'

arguments
    populations
    sourceIdx
    targetIdx
    nvp.cvNumFolds = 10
    nvp.useParallel = true
    nvp.regressMethod = @RidgeRegress
    nvp.scale = false % advised to specify this in Extract Spikes, if desired.
    nvp.lossMeasure = 'NSE'
    nvp.waitbar
end

% Initialize default options for cross-validation and enable
% parallelization.
nvp.cvOptions = statset('crossval');
if nvp.useParallel, nvp.cvOptions.UseParallel = true; end

% Regression method to be used.
nvp.regressMethod = @RidgeRegress;

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
            'Options', nvp.cvOptions);

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