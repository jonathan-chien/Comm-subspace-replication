%% Set target region and source and target indices within populations
% Before running, check through entire script and make sure that the paths
% to directories are correct, filenames to be saved are as desired, and
% calculateSource is set as desired (for the Calculate optimal
% dimensionality using Factor Analysis section).

targetArea = 'targetV1';
sourceIdx = 1; % source V1
switch targetArea
    case 'targetV1'
        targetIdx = 2;
    case 'V2'
        targetIdx = 3;
end

CONSTS.nDimsForPred = 1:10;
CONSTS.ridgeInit = false;

% Before running, check through entire script and make sure that the paths
% to directories are correct, filenames to be saved are as desired, and
% calculateSource is set as desired (for the Calculate optimal
% dimensionality using Factor Analysis section).


%% Cross-validated Reduced Rank Regression

% Preallocate across sessions and datasets (repetitions will be averaged
% away within function).
optDimsRrrTargetV1 = NaN(CONSTS.nDatasets, CONSTS.nSessions);
predPerfRrrTargetV1 = NaN(CONSTS.nDatasets, length(CONSTS.nDimsForPred), CONSTS.nSessions);
predErrorsRrrTargetV1 = NaN(CONSTS.nDatasets, length(CONSTS.nDimsForPred), CONSTS.nSessions);
w = waitbar(0, '');

for iSession = 1:CONSTS.nSessions
    
    % Update waitbar.
    waitbar(iSession / CONSTS.nSessions, w, ...
            sprintf('Working on %s...', CONSTS.sessionsToRun{iSession}))
    
    % Test across datasets for current datset.
    [optDimsRrrTargetV1(:,iSession), ...
     predPerfRrrTargetV1(:,:,iSession), ...
     predErrorsRrrTargetV1(:,:,iSession)] ...
        = TestNumPredDimsRrr(populations(:,:,iSession), sourceIdx, targetIdx, ...
                             'nDimsUsedForPred', CONSTS.nDimsForPred, ...
                             'ridgeInit', CONSTS.ridgeInit, 'waitbar', true);
end

close(w)
clear w iSession 

% Change to correct directory and save results.
cd(sprintf(['/home/jonathan/Desktop/Communication subspace/communication-subspace-master/results/'...
            'Latent space inference/RRR/%s'], targetArea))
save('optDimsRrrTargetV1', 'optDimsRrrTargetV1')
save('predPerfRrrTargetV1', 'predPerfRrrTargetV1')
save('predErrorsRrrTargetV1', 'predErrorsRrrTargetV1')

cd ..
cd ..


%% Full ridge regression model performance

% Preallocate across sessions and datasets (repetitions will be averaged
% away within function).
predPerfRidgeTargetV1NormResp = NaN(CONSTS.nDatasets, CONSTS.nSessions);
predErrorsRidgeTargetV1NormResp = NaN(CONSTS.nDatasets, CONSTS.nSessions);
w = waitbar(0, '');

for iSession = 1:CONSTS.nSessions
    
    % Update waitbar.
    waitbar(iSession / CONSTS.nSessions, w, ...
            sprintf('Working on %s...', CONSTS.sessionsToRun{iSession}))
    
    % Test across datasets for current datset.
    [predPerfRidgeTargetV1NormResp(:,iSession), ...
     predErrorsRidgeTargetV1NormResp(:,iSession)] ...
        = TestFullRidgeModel(populations(:,:,iSession), sourceIdx, targetIdx, ...
                             'waitbar', true);

end

close(w)
clear w iSession 

cd(sprintf('Ridge/%s', targetArea))
save('predPerfRidgeTargetV1NormResp', 'predPerfRidgeTargetV1NormResp')
save('predErrorsRidgeTargetV1NormResp', 'predErrorsRidgeTargetV1NormResp')

cd ..
cd ..


%% Calculate optimal dimensionality using Factor Analysis

% Preallocate across sessions and datasets (repetitions will be averaged
% away within function).
qOptSource = NaN(CONSTS.nDatasets, CONSTS.nSessions);
qOptTargetV1NormResp = NaN(CONSTS.nDatasets, CONSTS.nSessions);
w = waitbar(0, '');
CONSTS.calcSourceDim = false;

for iSession = 1:CONSTS.nSessions
    
    % Update waitbar.
    waitbar(iSession / CONSTS.nSessions, w, ...
            sprintf('Working on %s...', CONSTS.sessionsToRun{iSession}))
    
    % Test across datasets for current datset.
    [qOptSource(:,iSession), qOptTargetV1NormResp(:,iSession)] ...
        = CalculateFactorAnDim(populations(:,:,iSession), sourceIdx, targetIdx, ...
                               'calcSource', CONSTS.calcSourceDim, 'waitbar', true);

end

close(w)
clear w iSession 

cd(['/home/jonathan/Desktop/Communication subspace/communication-subspace-master/results/'...
    'Latent space inference/FactorAnDim'])
save('qOptTargetV1NormResp', 'qOptTargetV1NormResp')

cd ..


%% Cross-validate Factor Regression

% Preallocate across sessions and datasets (repetitions will be averaged
% away within function).
optDimsFrTargetV1NormResp = NaN(CONSTS.nDatasets, CONSTS.nSessions);
predPerfFrTargetV1NormResp = NaN(CONSTS.nDatasets, length(CONSTS.nDimsForPred), CONSTS.nSessions);
predErrorsFrTargetV1NormResp = NaN(CONSTS.nDatasets, length(CONSTS.nDimsForPred), CONSTS.nSessions);
w = waitbar(0, '');

for iSession = 1:CONSTS.nSessions
    
    % Update waitbar.
    waitbar(iSession / CONSTS.nSessions, w, ...
            sprintf('Working on %s...', CONSTS.sessionsToRun{iSession}))
    
    % Test across datasets for current datset.
    [optDimsFrTargetV1NormResp(:,iSession), ...
     predPerfFrTargetV1NormResp(:,:,iSession), ...
     predErrorsFrTargetV1NormResp(:,:,iSession)] ...
        = TestNumPredDimsFr(populations(:,:,iSession), sourceIdx, targetIdx, ...
                            'nDimsUsedForPred', CONSTS.nDimsForPred, ...
                            'waitbar', true);
end

close(w)
clear w iSession 

cd(sprintf('FactorRegress/%s', targetArea))
save('optDimsFrTargetV1NormResp', 'optDimsFrTargetV1NormResp')
save('predPerfFrTargetV1NormResp', 'predPerfFrTargetV1NormResp')
save('predErrorsFrTargetV1NormResp', 'predErrorsFrTargetV1NormResp')

