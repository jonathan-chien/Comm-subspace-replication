%% Set constants

clear CONSTS

CONSTS.sessionNames = {'105l001p16.mat', '106r001p26.mat', '106r002p70.mat', ...
                 '107l002p67.mat', '107l003p143.mat'};
CONSTS.sessionsToRun = CONSTS.sessionNames(1:5);
CONSTS.nSessions = length(CONSTS.sessionsToRun);
CONSTS.binWidth = 100;
CONSTS.trialPeriod = [1 1000] + 160;
CONSTS.dropBelow = 0.5; % in Hz


%% Calculate pairwise correlation coefficients across all sessions

% Growing by concatenation is fine because nIterations is tiny.
corrCoefs = cell(1, 2);
w = waitbar(0, '');

for iSession = 1:CONSTS.nSessions
    
    % Update waitbar.
    waitbar(iSession / CONSTS.nSessions, w, ...
            sprintf('Working on data from %s...', CONSTS.sessionsToRun{iSession}))
        
    % Load new session data.
    load(CONSTS.sessionsToRun{iSession})
    
    % Obtain firing rates from each session.
    [spikes, stim] ...
        = ExtractSpikes_jmc(neuralData, CONSTS.binWidth, ...
                            'TrialPeriod', CONSTS.trialPeriod, ...
                            'Drop', CONSTS.dropBelow, 'Scale', false);
    
    % Calculate pairwise correlation coefficients for current session and
    % append to array within cell storing coefficients across all sessions.
    currCorrCoefs = CalculatePairwiseCorr(spikes, stim); 
    corrCoefs{1} = [corrCoefs{1}; currCorrCoefs{1}];
    corrCoefs{2} = [corrCoefs{2}; currCorrCoefs{2}];
end

close(w)
clear w iSession currCorrCoefs

%% Plot histograms

% V1-V1 pairwise correlations.
figure
histogram(corrCoefs{1}, -0.2 : 0.05 : 0.5)
title('V1-V1 pairwise correlations')
ylabel('Pairs')
xlabel('Pairwise correlation coefficient')

% V1-V2 pairwise correlations.
figure
histogram(corrCoefs{2}, -0.2 : 0.05 : 0.5)
title('V1-V2 pairwise correlations')
ylabel('Pairs')
xlabel('Pairwise correlation coefficient')

