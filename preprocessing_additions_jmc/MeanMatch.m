function [sourcePop1,mmPop1,mmPop2] = MeanMatch(pop1,pop2,nvp)
% Performs mean-matching for a set of two neural populations, pop1 and
% pop2. pop1 is regarded as the source region, and pop2 the target region.
% Generally, nUnits from pop2 << nUnits from pop1. Briefly, a subset of
% neurons from pop1 and pop2 will be subsampled so that their respective
% distributions (over neurons) of mean firing rate across both trials and
% timepoints match a greatest common distribution between pop1 and pop2
% (defined as the minimum bin count between pop1 and pop2 when their
% respective mean firing rates are distributed over the same bins--note
% that here the term "bin" refers to a histogram bin and does NOT refer to
% a time bin). This results in two mean-matched subsets of pop1 and pop2
% (since pop2 is usually much smaller, often all of its units will be
% retained), and any leftover units from pop1 are designated as the source
% population. There is an option to repeat the subsampling process to
% generate nRepetitions subsets (all repetitions operate off of the same
% greatest common distribution, thus it is merely the identity, and not the
% number, of neurons that vary from reptition to reptitions). For each
% repetition, we have a new "dataset," with neurons not subsampled into the
% mean-matched V1 population designated as the V1 source population
% (whereas the mean-matched V1 and V2 popualations are our two respective
% target populations).
%
% PARAMETERS
% ----------
% pop1 -- nUnits x nTimepts x nTrials array of firing rates corresponding
%         population 1. nUnits should be much larger than for pop2.
% pop2 -- nUnits x nTimepts x nTrials array of firing rates corresponding
%         population 2. nUnits should be much smaller than for pop1.
% Name-Value Pairs (nvp)
%   'nBins'     -- Scalar values that is the number of bins when generating
%                  counts for the mean firing rate distributions of neurons
%                  from each of the two populations. These are essentially
%                  the number of bins in the distribution histograms, and
%                  should NOT be confused with time bins. Default = 30.
%   'seed'      -- Value for the seed set/reset before subsampling neurons
%                  from each bin to mean match. Must be an acceptable
%                  argument for MATLAB's rng. Default = 'shuffle'.
%   'nReps'     -- Scalar value specifying number of times to repeat the
%                  subsampling process across bins. Setting 'nReps' = n
%                  generates n datasets which are n subsets of the original
%                  data. Note that each of the subsets is defined from the
%                  same greatest common distribution and therefore has the
%                  same number of neurons; it is merely the identity, and
%                  not the number, of the subsampled neurons that varies
%                  across the nRepetitionsDefault = 25.
% 
% RETURNS
% -------
% sourcePop1 -- nUnits x nTimepoints x nTrials x nReps array of firing
%               rates, with nUnits corresponding to leftover units from
%               pop1 after subsampling to mean-match to greatest common
%               firing distribution between pop1 and pop2.
% mmPop1     -- nUnits x nTimepoints x nTrials x nReps array of firing
%               rates, with nUnits corresponding to units subsampled from
%               pop1 to mean-match to greatest common firing distribution
%               between pop1 and pop2.
% mmPop2     -- nUnits x nTimepoints x nTrials x nReps array of firing
%               rates, with nUnits corresponding to units subsampled from
%               pop2 to mean-match to greatest common firing distribution
%               between pop1 and pop2.
%
% Author: Jonathan Chien Version 1.0. 8/8/21. Last edit: 8/9/21
%   Based on Methods from "Cortical Areas Interact through a Communication
%   Subspace", Semedo et al 2019, Neuron.
% Version history:
%   -- Originally code from ConstructSourceAndTarget function, encapsulated
%      8/8/21.

arguments
    pop1
    pop2
    nvp.nBins (1,1) {mustBeInteger} = 30 
    nvp.seed = 'shuffle'
    nvp.nReps (1,1) {mustBeInteger} = 25
end

% Ensure nTimepoints and nTrials is consistent between populations. If so,
% derive these values from population 1. 
sizes1 = size(pop1);
sizes2 = size(pop2);
assert(all(sizes1(2:end)==sizes2(2:end)), ...
       'All array dimensions except for nUnits must match between population arrays.')
[~, nTimepts, nTrials] = size(pop1);

% Divide firing rate distributions (over neurons) into the same bins.
% histcounts.m unwinds matrix inputs into vectors and is thus the
% bottlneck preventing further vectorization.
pop1bar = mean(pop1, [2 3]);
pop2bar = mean(pop2, [2 3]);
overallMax = max([pop1bar; pop2bar]);
overallMin = min([pop1bar; pop2bar]);
binEdges = linspace(overallMin, overallMax, nvp.nBins+1);
[binCountsPop1,~,binIdcPop1] = histcounts(pop1bar, binEdges);
[binCountsPop2,~,binIdcPop2] = histcounts(pop2bar, binEdges);

% Obtain greatest common distribution of firing rates (binned).
gcdistr = min(binCountsPop1, binCountsPop2);
nUnitsRetained = sum(gcdistr);

% Warn if multiple repetitions requested with same seed.
if nvp.nReps > 1 && ~strcmp(nvp.seed, 'shuffle')
    warning(['Multiple repetitions creating multiple mean-matched ' ...
             'subsets requested, but same seed has been set for  ' ...
             'all repetitions, and all subsets will thus be the  ' ...
             'same. If nReps > 1, set seed to shuffle.'])
end

% Preallocate.
mmPop1 = NaN(nUnitsRetained, nTimepts, nTrials, nvp.nReps);
mmPop2 = NaN(nUnitsRetained, nTimepts, nTrials, nvp.nReps);
sourcePop1 = NaN(sizes1(1)-nUnitsRetained, nTimepts, nTrials, nvp.nReps);

% Randomly drop neurons from appropriate distribution to mean match.
for iRep = 1:nvp.nReps

    % Set/reset seed. Set/reset vars keepIdcV1, keepIdcV2, and dropIdcV1.
    rng(nvp.seed);
    keepIdcPop1 = cell(nvp.nBins, 1);
    keepIdcPop2 = cell(nvp.nBins, 1);
    dropIdcPop1 = cell(nvp.nBins, 1);

    % For each histogram bin, subsample neurons from each population to
    % retain to match greatest common distribution. These are the
    % neurons in target V1 and in V2. Neurons not retained (dropped)
    % are designated as source V1.
    for iBin = 1:nvp.nBins
       keepIdcPop1{iBin} = datasample(find(binIdcPop1==iBin), gcdistr(iBin), ...
                                      'Replace', false);
       keepIdcPop2{iBin} = datasample(find(binIdcPop2==iBin), gcdistr(iBin), ...
                                      'Replace', false);
       dropIdcPop1{iBin} = setdiff(find(binIdcPop1==iBin), keepIdcPop1{iBin});
    end

    % Retain only units selected above. Sorting shouldn't be necessary,
    % but order of neurons will be closer to original (which, again,
    % shouldn't matter).
    keepIdcPop1 = sort(vertcat(keepIdcPop1{:}));  
    keepIdcPop2 = sort(vertcat(keepIdcPop2{:}));  
    dropIdcPop1 = sort(vertcat(dropIdcPop1{:}));
    mmPop1(:,:,:,iRep) = squeeze(pop1(keepIdcPop1,:,:)); 
    mmPop2(:,:,:,iRep) = squeeze(pop2(keepIdcPop2,:,:));
    sourcePop1(:,:,:,iRep) = squeeze(pop1(dropIdcPop1,:,:));
end

end