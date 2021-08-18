function [predPerf,predErrors] = TestFullRidgeModel2 ...
    (sourcePops,targetPops,nvp)

% Same as TestFullRidgeModel but with modified parameters. Instead of a
% populations cell array containing all three pops (sourceV1, targetV1, and
% V2), this function accepts two 3D arrays, each of which has dimensions
% nObs x nUnits x nReps and which correspond to the nReps reptitions of the
% source and target populations, respectively, for one dataset datasets
% within one session. If called from the dimension_removal_pipeline, the
% datasets additionally correspond to one value of nDimsForPred.

arguments
    sourcePops
    targetPops
    nvp.cvNumFolds = 10
    nvp.useParallel = true
    nvp.regressMethod = @RidgeRegress
    nvp.scale = false % advised to specify this in Extract Spikes, if desired.
    nvp.lossMeasure = 'NSE'
    nvp.waitbar = false
end

% Initialize default options for cross-validation and enable
% parallelization.
nvp.cvOptions = statset('crossval');
if nvp.useParallel, nvp.cvOptions.UseParallel = true; end

% Regression method to be used.
nvp.regressMethod = @RidgeRegress;

% Determine number of repetitions of input dataset.
nReps = size(sourcePops, 3);

% Preallocate over all datasets and repetitions, and initialize waitbar.
predPerfAll = NaN(nReps, 1);
predErrorsAll = NaN(nReps, 1);
if nvp.waitbar, w = waitbar(0, ''); end

% For each repetition of input dataset, run CV routine with RRR.
for iRep = 1:nReps

    % Update waitbar.
    if nvp.waitbar           
        waitbar(iRep / nReps, w,...
                sprintf('Working on reptitions %d of %d...', iRep, nReps));
    end

    % Set source and target populations.
    currSourcePop = squeeze(sourcePops(:,:,iRep));
    currTargetPop = squeeze(targetPops(:,:,iRep));

    % Calculate range of lambdas to test for current
    % population/repetition.
    lambdas = GetRidgeLambda(.5:.01:1, currSourcePop, 'Scale', nvp.scale);

    % Define auxiliary function for cross-validation routine. Must be
    % defined anew for each repetition because lambda range is determined
    % for each repetition. Note that we are using
    % RidgeRegressFitAndPredict_jmc rather than RegressFitAndPredict.
    nvp.cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RidgeRegressFitAndPredict_jmc...
        (Ytrain, Xtrain, Ytest, Xtest, ...
         lambdas, 'LossMeasure', nvp.lossMeasure, 'Scale', nvp.scale);

    % Cross-validation routine.
    cvl = crossval(nvp.cvFun, currTargetPop, currSourcePop, ...
          'KFold', nvp.cvNumFolds, ...
        'Options', nvp.cvOptions);

    % Store cross-validation results: mean loss and standard error of
    % the mean across folds.
    cvLoss = [ mean(cvl); std(cvl)/sqrt(nvp.cvNumFolds) ];

    % Store Reduced Rank Regression cross-validation results for
    % current repetition of current dataset. 
    predPerfAll(iRep) = 1-cvLoss(1);
    predErrorsAll(iRep) = cvLoss(2);
end

if nvp.waitbar, close(w); end

% Average across repetitions for each dataset.
predPerf = mean(predPerfAll);
predErrors = mean(predErrorsAll);

end