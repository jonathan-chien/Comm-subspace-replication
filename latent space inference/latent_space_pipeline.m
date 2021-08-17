%% Set target region and source and target indices within populations
% Before running, check through entire script and make sure that the paths
% to directories are correct, filenames to be saved are as desired, and
% calculateSource is set as desired (for the Calculate optimal
% dimensionality using Factor Analysis section).

targetArea = 'V2';
sourceIdx = 1; % source V1
switch targetArea
    case 'targetV1'
        targetIdx = 2;
    case 'V2'
        targetIdx = 3;
end

CONSTS.nDimsForPred = 1:10;
CONSTS.ridgeInit = true;

% Before running, check through entire script and make sure that the paths
% to directories are correct, filenames to be saved are as desired, and
% calculateSource is set as desired (for the Calculate optimal
% dimensionality using Factor Analysis section).


% %% Cross-validated Reduced Rank Regression
% 
% % Preallocate across sessions and datasets (repetitions will be averaged
% % away within function).
% optDimsRrr = NaN(CONSTS.nDatasets, CONSTS.nSessions);
% predPerfRrr = NaN(CONSTS.nDatasets, length(CONSTS.nDimsForPred), CONSTS.nSessions);
% predErrorsRrr = NaN(CONSTS.nDatasets, length(CONSTS.nDimsForPred), CONSTS.nSessions);
% w = waitbar(0, '');
% 
% for iSession = 1:CONSTS.nSessions
%     
%     % Update waitbar.
%     waitbar(iSession / CONSTS.nSessions, w, ...
%             sprintf('Working on %s...', CONSTS.sessionsToRun{iSession}))
%     
%     % Test across datasets for current datset.
%     [optDimsRrr(:,iSession), predPerfRrr(:,:,iSession), predErrorsRrr(:,:,iSession)] ...
%         = TestNumPredDimsRrr(populations(:,:,iSession), sourceIdx, targetIdx, ...
%                              'nDimsUsedForPrediction', CONSTS.nDimsForPred, ...
%                              'ridgeInit', CONSTS.ridgeInit, 'waitbar', true);
% end
% 
% close(w)
% clear w iSession 
% 
% % Change to correct directory and save results.
% cd(sprintf(['/home/jonathan/Desktop/Communication subspace/communication-subspace-master/results/'...
%             'Latent space inference/RRR/%s'], targetArea))
% save('optDimsRrrV2', 'optDimsRrr')
% save('predPerfRrrV2', 'predPerfRrr')
% save('predErrorsRrrV2', 'predErrorsRrr')
% 
% cd ..
% cd ..
% 
% 
% %% Full ridge regression model performance
% 
% % Preallocate across sessions and datasets (repetitions will be averaged
% % away within function).
% predPerfRidge = NaN(CONSTS.nDatasets, CONSTS.nSessions);
% predErrorsRidge = NaN(CONSTS.nDatasets, CONSTS.nSessions);
% w = waitbar(0, '');
% 
% for iSession = 1:CONSTS.nSessions
%     
%     % Update waitbar.
%     waitbar(iSession / CONSTS.nSessions, w, ...
%             sprintf('Working on %s...', CONSTS.sessionsToRun{iSession}))
%     
%     % Test across datasets for current datset.
%     [predPerfRidge(:,iSession), predErrorsRidge(:,iSession)] ...
%         = TestFullRidgeModel(populations(:,:,iSession), sourceIdx, targetIdx, ...
%                              'waitbar', true);
% 
% end
% 
% close(w)
% clear w iSession 
% 
% cd(sprintf('Ridge/%s', targetArea))
% save('predPerfRidgeV2', 'predPerfRidge')
% save('predErrorsRidgeV2', 'predErrorsRidge')
% 
% cd ..
% cd ..
% 
% 
% %% Calculate optimal dimensionality using Factor Analysis
% 
% % Preallocate across sessions and datasets (repetitions will be averaged
% % away within function).
% qOptSource = NaN(CONSTS.nDatasets, CONSTS.nSessions);
% qOptTarget = NaN(CONSTS.nDatasets, CONSTS.nSessions);
% w = waitbar(0, '');
% CONSTS.calcSourceDim = false;
% 
% for iSession = 1:CONSTS.nSessions
%     
%     % Update waitbar.
%     waitbar(iSession / CONSTS.nSessions, w, ...
%             sprintf('Working on %s...', CONSTS.sessionsToRun{iSession}))
%     
%     % Test across datasets for current datset.
%     [qOptSource(:,iSession), qOptTarget(:,iSession)] ...
%         = CalculateFactorAnDim(populations(:,:,iSession), sourceIdx, targetIdx, ...
%                                'calcSource', CONSTS.calcSourceDim, 'waitbar', true);
% 
% end
% 
% close(w)
% clear w iSession 
% 
% cd('FactorAnDim')
% save('qOptV2', 'qOptTarget')
% 
% cd ..


%% Cross-validate Factor Regression

% Preallocate across sessions and datasets (repetitions will be averaged
% away within function).
optDimsFr = NaN(CONSTS.nDatasets, CONSTS.nSessions);
predPerfFr = NaN(CONSTS.nDatasets, length(CONSTS.nDimsForPred), CONSTS.nSessions);
predErrorsFr = NaN(CONSTS.nDatasets, length(CONSTS.nDimsForPred), CONSTS.nSessions);
w = waitbar(0, '');

for iSession = 1:CONSTS.nSessions
    
    % Update waitbar.
    waitbar(iSession / CONSTS.nSessions, w, ...
            sprintf('Working on %s...', CONSTS.sessionsToRun{iSession}))
    
    % Test across datasets for current datset.
    [optDimsFr(:,iSession), predPerfFr(:,:,iSession), predErrorsFr(:,:,iSession)] ...
        = TestNumPredDimsFr(populations(:,:,iSession), sourceIdx, targetIdx, ...
                             'nDimsUsedForPrediction', CONSTS.nDimsForPred, ...
                             'waitbar', true);
end

close(w)
clear w iSession 

cd(sprintf('FactorRegress/%s', targetArea))
save('optDimsFrV2', 'optDimsFr')
save('predPerfFrV2', 'predPerfFr')
save('predErrorsFrV2', 'predErrorsFr')

