%% Set target region and source and target indices within populations

sourceIdx = 1; % source V1

removeArea = 'targetV1';
switch removeArea
    case 'targetV1'
        removeIdx = 2;
    case 'V2'
        removeIdx = 3;
end

predictArea = 'V2';
switch predictArea
    case 'targetV1'
        predictIdx = 2;
    case 'V2'
        predictIdx = 3;
end

CONSTS.nDimsForPred = 1:5;
CONSTS.ridgeInit = false;
CONSTS.nReps = 10;


%% Load and pre-process neural data for all sessions

% Preallocate and intialize waitbar.
predPerfDimRem = NaN(length(CONSTS.nDimsForPred), CONSTS.nDatasets, CONSTS.nSessions);
predErrorsDimRem = NaN(length(CONSTS.nDimsForPred), CONSTS.nDatasets, CONSTS.nSessions);
w = waitbar(0, '');
iBar = 0;

for iSession = 1:CONSTS.nSessions
    for iDataset = 1:CONSTS.nDatasets
        
        % Update waitbar.
        iBar = iBar + 1;
        waitbar(iBar / (CONSTS.nSessions*CONSTS.nDatasets), w, ...
                sprintf('Working on dataset %d of %d', ...
                        iBar, CONSTS.nSessions*CONSTS.nDatasets))
        
        % Determine number of units in source population for current
        % dataset.
        nUnitsSource = size(populations{iDataset,1,iSession}, 2);
        
        for iNumDims = CONSTS.nDimsForPred
            
            % Preallocate to hold B_ matrices across all reps for current
            % datset and value of nDimsForPred.
            B_AllReps = NaN(nUnitsSource, CONSTS.nDimsForPred(iNumDims), CONSTS.nReps);
            
            for iRep = 1:CONSTS.nReps
                [~,B_,~] ...
                    = ReducedRankRegress(populations{iDataset,removeIdx,iSession}(:,:,iRep), ...
                                         populations{iDataset,sourceIdx,iSession}(:,:,iRep), ...
                                         CONSTS.nDimsForPred(iNumDims), ...
                                         'RidgeInit', CONSTS.ridgeInit);
                B_AllReps(:,:,iRep) = B_(:,1:CONSTS.nDimsForPred(iNumDims));
            end
            
            % X_hat is nObs x nUnits x nReps for one dataset and value of
            % nDimsForPred, all within current session.
            X_hat = DimensionRemoval(populations{iDataset,sourceIdx,iSession}(:,:,1:CONSTS.nReps), ...
                                     B_AllReps);
                                 
            % Test with ridge right away.
            [predPerfDimRem(iNumDims,iDataset,iSession), ...
             predErrorsDimRem(iNumDims,iDataset,iSession)] ...
                = TestFullRidgeModel2(X_hat, populations{iDataset,predictIdx,iSession}, ...
                                      'cvNumFolds', 10);
        end
    end                   
end

close(w)
clear w iSession iDataset iNumDims iRep iBar B_ B_AllReps

cd('/home/jonathan/Desktop/Communication subspace/communication-subspace-master/results/Dimension removal')
save('predPerfDimRemV1TestOnV2', 'predPerfDimRem')
save('predErrorsDimRemV1TestOnV2', 'predErrorsDimRem')

