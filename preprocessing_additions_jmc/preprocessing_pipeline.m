%% Add data/scripts directories to path 

if strcmp(computer, 'GLNXA64')
    % Set nCores.
    pool = parpool('local', 10);
else
    % Add path to data on local computer.
    addpath('D:\Communication subspace\v1-v2_gratings\v1-v2_gratings\mat_neural_data')
    % Add path to scripts on local computer.
    cd('D:\Communication subspace\communication-subspace-master')
    addpath(genpath(pwd))
    pool = parpool('local', 2);
end

% Terminate existing interactive session.
% delete(gcp('nocreate'))


%% Set constant parameters

clear CONSTS

CONSTS.sessionNames = {'105l001p16.mat', '106r001p26.mat', '106r002p70.mat', ...
                       '107l002p67.mat', '107l003p143.mat'};
CONSTS.sessionsToRun = CONSTS.sessionNames(1:5);
CONSTS.nSessions = length(CONSTS.sessionsToRun);
CONSTS.nDatasets = 8; % number of datasets per session, with each stim type defining a dataset
CONSTS.nRegionsTot = 3;
CONSTS.binWidth = 100; % in ms. Note BLANK_TRIAL_LENGTH = 1500 ms and DRIVEN_TRIAL_LENGTH = 1280 ms
CONSTS.trialPeriod = [1 1000] + 160; 
CONSTS.dropBelow = 0.5; % in Hz
CONSTS.nHistBins = 30;
CONSTS.nReps = 10; % number of repetitions of each dataset
CONSTS.scale = false; % true = z-score firng rates (done before calculation of residuals)


%% Load and pre-process neural data for all sessions

% Preallocate and intialize waitbar.
populations = cell(CONSTS.nDatasets, CONSTS.nRegionsTot, CONSTS.nSessions);
w = waitbar(0, '');

for iSession = 1:CONSTS.nSessions
    
    % Update waitbar.
    waitbar(iSession / CONSTS.nSessions, w, ...
            sprintf('Loading and adding data from %s', CONSTS.sessionsToRun{iSession}))
        
    % Load new session data.
    load(CONSTS.sessionsToRun{iSession})

    % The 'TrialPeriod' option can be set to 'Driven' (trials for which an
    % oriented grating was presented), 'Spontaneous' (trials for which no
    % grating was presented), 'Full' (combine each driven trial with the
    % subsequent spontaneous trial), or an interval in ms (interval must be in
    % the range 1 - 2780 (the first 1280 ms correspond to driven activity, the
    % subsequent 1500s correspond to spontaneous activity). spikes is a 1x2
    % cell array whose first cell contains the nUnits x nTimepoints x nTrials
    % array of firing rates from V1, and whose second cell contains the same as
    % the first cell but for V2.
    [spikes, stim] ...
        = ExtractSpikes_jmc(neuralData, CONSTS.binWidth, ...
                            'TrialPeriod', CONSTS.trialPeriod, ...
                            'Drop', CONSTS.dropBelow, 'Scale', CONSTS.scale);

    clear neuralData
    
    % Creates N_TYPES x 1 cell array, where each cell contains a
    % nTimepoints*nStimTrials x nUnits x nRepetitions matrix of residuals,
    % with nStimTrials = the number of trials featuring one unique stimulus.
    % Note again that spikes{1} is V1 and spikes{2} is V2.
    populations(:,:,iSession) ...
        = ConstructSourceAndTarget(PartitionByType(spikes{1}, stim), ...
                                   PartitionByType(spikes{2}, stim), ...
                                   'nBins', CONSTS.nHistBins, 'nReps', CONSTS.nReps, ...
                                   'residuals', true);
                                   
end

close(w)
clear w spikes stim iSession


