function NxSxTxType = PartitionByType(NxSxT,stim)
% Partitions nUnits x nTimepoints x nTotTrials array into an
% nUnits x nTimepoints x nStimTrials x nTypes array of firing rates, where
% nTotTrials is the total number of trials across all stimulus types, and
% nStimTrials the number of trials featuring each of the unique stimulus
% types (equal to nTotTrials / nTypes).
%
% PARAMETERS
% ----------
% NxSxT -- nUnits x nTimepoints x nTrials array of firing rates. nTrials
%          consists here of all trials across all stimulus types.
% stim  -- nTrials x 1 vector, each of whose elements contains the type
%          label of the stimulus that was shown on that trial.
%         
% RETURNS
% -------
% NxSxTxType -- nUnits x nTimepoints x nStimTrials x nTypes array of firing
%               rates, where nStimTrials is the number of trials featuring
%               each of the unique stimulus types (equal to nTotTrials /
%               nTypes), with each type defined by a unique stimulus
%               condition.
%
% Author: Jonathan Chien 8/3/21.

[nNeurons, nTimepoints, nTotTrials] = size(NxSxT);
nTypes = length(unique(stim));
nStimTrials = nTotTrials / nTypes;

NxSxTxType = NaN(nNeurons, nTimepoints, nStimTrials, nTypes);
for iType = 1:nTypes
    NxSxTxType(:,:,:,iType) = NxSxT(:,:,stim==iType);
end

end