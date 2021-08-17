function populations = ConstructSourceAndTarget(V1,V2,nvp)
% Accepts as input V1 and V2, each of which are nUnits x nTimepoints x
% nStimTrials x nStimTypes arrays of firing rates corresponding to one of
% the two neural populations (nStimTrials is the number of trials per
% stimulus type). The two populations are first subsampled via the
% MeanMatch function so that their respective distributions (over neurons)
% of mean firing rates across trials and times (but not across stimulus
% types, i.e. within the same stimulus type) match a greatest common
% distribution. There is an option to repeat the subsampling process to
% generate nRepetitions subsets. There is also an option to subtract the
% PSTH (mean across trials) within stimulus type from each nUnits x
% nTimespoints x nStimTrials x nRepetitions array (for all three
% populations: source V1, V2, and target V1) for each stimulus type.
% Finally, the function reshapes each nUnits x nTimespoints x nStimTrials x
% nRepetitions array (again, for all three populations: source V1, V2, and
% target V1) into an nTimepoints*nStimTrials x nNeurons x nRepetitions
% array, which is assigned into the corresponding cell of populations, a
% nTypes x 1 cell array. It is this cell array that is returned.
%
% PARAMETERS
% ----------
% V1 -- nUnits x nTimepoints x nStimTrials x nStimTypes arrays of firing
%       rates of V1 neurons.
% V2 -- nUnits x nTimepoints x nStimTrials x nStimTypes arrays of firing
%       rates of V2 neurons.
% Name-Value Pairs (nvp)
%   'nBins'     -- Name-value pair for MeanMatch, see that functions'
%                  documentation for more information.
%   'seed'      -- Name-value pair for MeanMatch, see that functions'
%                  documentation for more information.
%   'nReps'     -- Name-value pair for MeanMatch, see that functions'
%                  documentation for more information.
%   'residuals' -- Logical true or false. If true, the PSTHs for each
%                  dataset (each stimulus type defines its own dataset),
%                  derived by averaging across trials (again, trials
%                  sharing the same stimulus type) as a function of time
%                  for each neuron, is subtracted from the single trial
%                  timeseres of each neuron, and the data returned takes
%                  the form of residuals. If false, this behavior is
%                  suppressed, and the returned data are firing rates.
%                  Default true.
%
% RETURNS
% -------
% populations -- nTypes x 2 cell array, where rows correspond to stimulus
%                types and columns to populations (source V1, targetV1, and
%                V2 for columns 1, 2, and 3 respectively). Note that target
%                V1 and V2 have been mean-matched/subsampled here. Each
%                cell contains an nTimepoints*nStimTrials x nUnits x
%                nRepetitions array of either firing rates or residuals
%                (see the 'residuals' name-value pair under PARAMETERS) for
%                the corresponding population.
%
% Author: Jonathan Chien Version 2.0. 8/3/21. Last edit: 8/9/21.
%   Based on Methods from "Cortical Areas Interact through a Communication
%   Subspace", Semedo et al 2019, Neuron.
% Version history: 
%   -- Originally called MeanMatch (found in archive as
%      MeanMatch_arch1.0)

arguments
    V1 
    V2
    nvp.nBins (1,1) {mustBeInteger} = 30 
    nvp.seed = 'shuffle'
    nvp.nReps (1,1) {mustBeInteger} = 25
    nvp.residuals = true
end

% Ensure sizes of population arrays match except for nUnits. If so, derive
% dimension sizes from V1.
sizesV1 = size(V1);
sizesV2 = size(V2);
assert(all(sizesV1(2:end)==sizesV2(2:end)), ...
       'All array dimensions except for nUnits must match between population arrays.')
[~, nTimepts, nStimTrials, nTypes] = size(V1); % here nSimTrials = nTrials within one stim type

% Iterate over each stimulus type, and for each generate nReps mean-matched
% subsets of X and Y based on greatest common distribution.
populations = cell(nTypes, 3);
for iType = 1:nTypes
    
    % Obtain 3D slice corresponding to nUnits x nTimepts x nTrials for
    % current type and mean-match.
    [sourceV1, mmV1, mmV2] = MeanMatch(V1(:,:,:,iType), V2(:,:,:,iType), ...
                                       'nBins', nvp.nBins, 'seed', nvp.seed, ...
                                       'nReps', nvp.nReps);
    
    % Option to convert firing rates into firing rate residuals.
    if nvp.residuals
        
        % Calculate PSTH of mean across trials within current type (with
        % expansion via repmat). 
        psthMmV1 = repmat(mean(mmV1,3), 1, 1, nStimTrials, 1); 
        psthMmV2 = repmat(mean(mmV2,3), 1, 1, nStimTrials, 1);
        psthSourceV1 = repmat(mean(sourceV1,3), 1, 1, nStimTrials);
        
        % Subtract each PSTH from its respective population to obtain
        % residuals.
        mmV1 = mmV1 - psthMmV1;
        mmV2 = mmV2 - psthMmV2;
        sourceV1 = sourceV1 - psthSourceV1;
    end
    
    % Reshape residualsMmV1, residualsMmV2, and residualsSourceV1 from
    % nUnits x nTimepts x nTrials x nReps of residuals into
    % nTimepts*nTrials x nUnits x nReps array.
    nUnitsRetained = size(mmV1, 1);
    mmV1Reshaped = permute(reshape(mmV1, ...
                                  [nUnitsRetained nTimepts*nStimTrials nvp.nReps]), ...
                           [2 1 3]);
    mmV2Reshaped = permute(reshape(mmV2, ...
                                  [nUnitsRetained nTimepts*nStimTrials nvp.nReps]), ...
                           [2 1 3]);
    sourceV1Reshaped = permute(reshape(sourceV1, ...
                                       [sizesV1(1)-nUnitsRetained ...
                                        nTimepts*nStimTrials nvp.nReps]), ...
                               [2 1 3]);
    
    % Assign current type array into cell array (which contains all types).
    populations{iType,1} = sourceV1Reshaped;
    populations{iType,2} = mmV1Reshaped;
    populations{iType,3} = mmV2Reshaped;
end

end