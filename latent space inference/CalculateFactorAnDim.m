function [qOptSource,qOptTarget] = CalculateFactorAnDim ...
    (populations,sourceIdx,targetIdx,nvp)
% Takes as input a series of datasets (over various stimulus types), each
% repeated nReps times, and for each repetition of each dataset, calculates
% optimal dimensionality of the source and target populations using factor
% analysis (95% variance explained).
%
% PARAMETERS
% ----------
% populations -- nDatasets x 3 cell array, where rows correspond to
%                stimulus types and columns to populations (source V1,
%                targetV1, and V2 for columns 1, 2, and 3 respectively).
%                Note that target V1 and V2 have been
%                mean-matched/subsampled here. Each cell contains an
%                nTimepoints*nStimTrials x nUnits x nRepetitions array of
%                either firing rates or residuals (see the 'residuals'
%                name-value pair under PARAMETERS section of
%                ConstructSourceAndTarget) for the corresponding
%                population.
% sourceIdx   -- Index of source population within "populations" (column
%                index). Should be 1.
% targetIdx   -- Index of target population wihthi "populations" (column
%                index). Set 2 for target V1 and 3 for V2.
% Name-Value Pairs (nvp)
%   'cvNumFolds'   -- Scalar value that is the number of folds to be used
%                     in log-likelihood FA cross validation. Default = 10.
%   'qSourceRange' -- Vector of dimensionalites to test for the source
%                     population. If any of these values exceeds the number
%                     of neurons in the source population, function will
%                     default to testing over 0:nNeuronsPop. Default is
%                     0:30.
%   'qTargetRange' -- Vector of dimensionalites to test for the target
%                     population. If any of these values exceeds the number
%                     of neurons in the target population, function will
%                     default to testing over 0:nNeuronsPop. Default is
%                     0:30.
%   'useParallel'  -- Logical true or false specifying whether to enable
%                     parallel processing during cross-validation routine.
%                     Default true.
%   'waitbar'      -- Logical true or false specifying whether to enable
%                     waitbar across datasets. Default true.
%   'calcSource'   -- Logical true or false specifying whether or not to
%                     calculate source dimensionality (this may have been
%                     calculated on a previous run and if we need only to
%                     calculate dimensionality for another target region,
%                     we can set this to false to prevent the source
%                     dimensionality from being recomputed. Default true.
%
% RETURNS
% -------
% qOptSource -- nDatasets x 1 vector of optimal dimensionalities (averaged
%               across repetitions for each dataset) for the source
%               population.
% qOptTarget -- nDatasets x 1 vector of optimal dimensionalities (averaged
%               across repetitions for each dataset) for the target
%               population.
%
% Author: Jonathan Chien 8/13/21. Based on methods from Semedo et al 2019
% "Cortical Areas Interact through a Communication Subspace", Neuron.
% Includes source code from
% https://github.com/joao-semedo/communication-subspace


arguments
    populations
    sourceIdx
    targetIdx
    nvp.cvNumFolds = 10
    nvp.qSourceRange = 0:30
    nvp.qTargetRange = 0:30
    nvp.useParallel = true
    nvp.waitbar = true
    nvp.calcSource = true
end

% Initialize default options for cross-validation and enable
% parallelization.
nvp.cvOptions = statset('crossval');
if nvp.useParallel, nvp.cvOptions.UseParallel = true; end

% Determine number of datasets and number of repetitions of each dataset.
nDatasets = size(populations, 1);
nReps = size(populations{1,1}, 3);

% Preallocate and initialize waitbar.
qOptSourceAll = NaN(nDatasets, nReps);
qOptTargetAll = NaN(nDatasets, nReps);
if nvp.waitbar, w = waitbar(0, ''); end

% For each repetition of each dataset, run FA (to assess dimensionality)
% and FR (predictive performance).
for iDataset = 1:nDatasets
    for iRep = 1:nReps
        
        % Update waitbar.
        if nvp.waitbar
            waitbar(iDataset / nDatasets, w,...
                    sprintf('Working on dataset %d of %d...', iDataset, nDatasets));
        end
    
        % Set source and target populations for current dataset and
        % repetition.
        sourcePop = populations{iDataset,sourceIdx}(:,:,iRep);
        targetPop = populations{iDataset,targetIdx}(:,:,iRep);
        
        % If e.g. source population dimensionality has already been
        % calculated on a previous run (along with targetV1), and current
        % run is just needed to calculate dimensionality of V2, set
        % 'calcSource' to false to prevent it from being recomputed at
        % significant computational cost.
        if nvp.calcSource
            % Determine optimal dimensionality of source population for the
            % Factor Analysis Model defined as minimum number of latent
            % dimensions needed to explain 95% of variance. % Note that if
            % max(nvp.qSourceRange) > nUnitsSourcePop, indexing error will
            % occur if default nvp.qSourceRange used for source pop CrossValFa,
            % thus we in this case reset qSourceRange, and ensure that its
            % maximum is no more than the neurons in the source population.
            if max(nvp.qSourceRange) > size(sourcePop, 2)
                nvp.qSourceRange = 0:size(sourcePop, 2); 
            end
            nvp.qSourceRange = 0:30;
            qOptSourceAll(iDataset,iRep) = FactorAnalysisModelSelect( ...
                CrossValFa(sourcePop, nvp.qSourceRange, nvp.cvNumFolds, nvp.cvOptions), ...
                nvp.qSourceRange);
        end
        
        % Determine optimal dimensionality of target population for the
        % Factor Analysis Model in same manner as for source population.
        % Note, as with the source population, that if
        % length(nvp.qSourceRange) > nUnitsTargetPop, indexing error will
        % occur if default nvp.qTargetRange used for target pop CrossValFa,
        % and thus we reset qTarget to ensure that its maximum is no more
        % than the neurons in the target population.
        if max(nvp.qTargetRange) > size(targetPop, 2)
            nvp.qTargetRange = 0:size(targetPop, 2); 
        end
        qOptTargetAll(iDataset,iRep) = FactorAnalysisModelSelect( ...
            CrossValFa(targetPop, nvp.qTargetRange, nvp.cvNumFolds, nvp.cvOptions), ...
            nvp.qTargetRange);
    end
end

if nvp.waitbar, close(w); end

% Average over repetitions to get one dimensionality estimate per dataset.
qOptTarget = mean(qOptTargetAll, 2);
qOptSource = mean(qOptSourceAll, 2);

end