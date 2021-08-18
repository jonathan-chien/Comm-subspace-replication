%% Plotting Fig 4A and B

iSession = 5;
iDataset = 2;
PlotPerfVsNumPredDims(predPerfRrrV2(iDataset,:,iSession), ...
                      predErrorsRrrV2(iDataset,:,iSession), ...
                      1:10, ...
                      predPerfRidgeV2(iDataset,iSession), ...
                      'regressMethod', 'Reduced rank', ...
                      'colorIdx', 2)
xticks(1:10)                      




%% Plotting Fig 4C

figure
PlotDimsVsDims(optDimsRrrTargetV1, optDimsRrrV2, [0, 0.4470, 0.7410])
xlabel('Target V1 predictive dimensions')
ylabel('V2 predictive dimensions')
legend('Individual dataset', 'Session mean', 'Location', 'northwest')

% Histogram of delta dimensionality.
figure
histogram(optDimsRrrTargetV1 - optDimsRrrV2, -2:4) % delta nDims
hold on
xlabel('delta dimensionality (target V1 - V2)', 'FontSize', 18)
ylabel('Datasets', 'FontSize', 18)
plot(mean(optDimsRrrTargetV1 - optDimsRrrV2, 'all'), 24, 'v', 'MarkerSize', 18, ...
     'MarkerFaceColor', [0, 0.4470, 0.7410], 'MarkerEdgeColor', [0, 0.4470, 0.7410])


%% Plotting Fig 5a

% Scatter plot
figure
PlotDimsVsDims(qOptTargetV1, qOptV2)
xlabel('Target V1 dimensionality')
ylabel('V2 dimensionality')
legend('Individual dataset', 'Session mean', 'Location', 'southeast')

% Histogram of delta dimensionality.
figure
histogram(qOptTargetV1 - qOptV2, -4:2) % delta nDims
hold on
xlabel('delta dimensionality (target V1 - V2)', 'FontSize', 18)
ylabel('Datasets', 'FontSize', 18)
plot(mean(qOptTargetV1 - qOptV2, 'all'), 15, 'v', 'MarkerSize', 18, ...
     'MarkerFaceColor', [0, 0.4470, 0.7410], 'MarkerEdgeColor', [0, 0.4470, 0.7410])


%% Plotting Fig 5b

figure
PlotDimsVsDims(qOptTargetV1, optDimsRrrTargetV1, [0, 0.4470, 0.7410])
hold on
PlotDimsVsDims(qOptV2, optDimsRrrV2, [0.8500, 0.3250, 0.0980])
xlabel('Target population dimensionality')
ylabel('Number of predictive dimensions')


%% Figure 6A predicting V2
% Paper says that noramalization is by performance of RRR with no source
% activity removed. There is no rank constraint mentioned (and it does not
% seem to make much sense to reduce the rank of Yhat by the same number as
% that by which the rank of Xhat was reduced: removing source activity
% along one predictive dimension may, e.g., reduce the source activity
% rank(X) = 133 to rank(Xhat) = 132, and this does not seem directly
% equivalent in any clear way to reducing the rank of the target population
% reconstruction Yhat from, e.g., 25 to 24. If we interpet the statement
% then as indicating that all columns of V (equal in number to
% nNeuronsTarget because the target regions always had less units) are
% retained, then, VV' = I, and B_rrr = Bols/L2 * I, and the RRR model is
% equivalent to the full ridge model.

% Set indices to control which dataset to plot.
iSession = 4;
iDataset = 1;
plotDims1 = 0:5;
plotDims2 = 0:5;

% Prediction performances against nDimsRemoved, normalized by full ridge
% model. Standard error scales by the same amount.
predPerf1Normed = [predPerfRidgeV2(iDataset,iSession); ...
                   squeeze(predPerfDimRemV1OnV2(plotDims1(2:end),iDataset,iSession))] ...
                  / predPerfRidgeV2(iDataset,iSession);        
predPerf2Normed = [predPerfRidgeV2(iDataset,iSession); ...
                   squeeze(predPerfDimRemV2OnV2(plotDims2(2:end),iDataset,iSession))] ...
                  / predPerfRidgeV2(iDataset,iSession); 
predErrors1Normed = [predErrorsRidgeV2(iDataset,iSession); ...
                     squeeze(predErrorsDimRemV1OnV2(plotDims1(2:end),iDataset,iSession))] ...
                    / predPerfRidgeV2(iDataset,iSession);
predErrors2Normed = [predErrorsRidgeV2(iDataset,iSession); ...
                     squeeze(predErrorsDimRemV2OnV2(plotDims2(2:end),iDataset,iSession))] ...
                    / predPerfRidgeV2(iDataset,iSession);

figure
PlotPredPerfVsPredPerf(predPerf1Normed, ...
                       predErrors1Normed, ...
                       plotDims1, ...
                       predPerf2Normed, ...
                       predErrors2Normed, ...
                       plotDims2, 'legend', {'Remove V1', 'Remove V2'})
title(sprintf('Cross-validated performance predicting V2, session %d dataset %d', ...
              iSession, iDataset))
          
%% Figure 6B predicting V2

predPerf1 = squeeze(predPerfDimRemV2OnV2(2,:,:)) ./ predPerfRidgeV2;
predPerf2 = squeeze(predPerfDimRemV1OnV2(2,:,:)) ./ predPerfRidgeV2;

figure
histogram(predPerf1(:), -0.05 : 0.05 : 0.2)
hold on
histogram(predPerf2(:), 0.1 : 0.05 : 0.45)

xlim([-0.2 0.8])
xticks(-0.2 : 0.2 : 0.8)
yl = ylim;
ylim([yl(1) yl(2)+5])
yl = ylim;
yticks([yl(1) yl(2)])

plot(mean(predPerf1, 'all'), yl(2)-2, 'v', ...
    'MarkerSize', 18, 'MarkerFaceColor', [0, 0.4470, 0.7410])
plot(mean(predPerf2, 'all'), yl(2)-2, 'v', ...
    'MarkerSize', 18, 'MarkerFaceColor', [0.8500, 0.3250, 0.0980])

title('Performance on V2 after predictive dimension removal across datasets')
xlabel('Normalized performance')
ylabel('Datasets')
legend('V2 dims removed', 'V1 dims removed', 'Mean for V2 dims removed', ...
       'Mean for V1 dims removed')


%% Figure 6C predicting target V1

% Set indices to control which dataset to plot.
iSession = 5;
iDataset = 2;
plotDims1 = 0:5;
plotDims2 = 0:2;

% Prediction performances against nDimsRemoved, normalized by full ridge
% model. Standard error scales by the same amount.
predPerf1Normed = [predPerfRidgeTargetV1(iDataset,iSession); ...
                   squeeze(predPerfDimRemV1OnV1(plotDims1(2:end),iDataset,iSession))] ...
                  / predPerfRidgeTargetV1(iDataset,iSession);        
predPerf2Normed = [predPerfRidgeTargetV1(iDataset,iSession); ...
                   squeeze(predPerfDimRemV2OnV1(plotDims2(2:end),iDataset,iSession))] ...
                  / predPerfRidgeTargetV1(iDataset,iSession); 
predErrors1Normed = [predErrorsRidgeV2(iDataset,iSession); ...
                     squeeze(predErrorsDimRemV1OnV1(plotDims1(2:end),iDataset,iSession))] ...
                    / predPerfRidgeV2(iDataset,iSession);
predErrors2Normed = [predErrorsRidgeV2(iDataset,iSession); ...
                     squeeze(predErrorsDimRemV2OnV1(plotDims2(2:end),iDataset,iSession))] ...
                    / predPerfRidgeV2(iDataset,iSession);

figure
PlotPredPerfVsPredPerf(predPerf1Normed, ...
                       predErrors1Normed, ...
                       plotDims1, ...
                       predPerf2Normed, ...
                       predErrors2Normed, ...
                       plotDims2, 'legend', {'Remove V1', 'Remove V2'})
title(sprintf('Cross-validated performance predicting target V1, session %d dataset %d', ...
              iSession, iDataset))
                   
                   
%% Figure 6D predicting target V1

predPerf1 = squeeze(predPerfDimRemV1OnV1(2,:,:)) ./ predPerfRidgeV2;
predPerf2 = squeeze(predPerfDimRemV2OnV1(2,:,:)) ./ predPerfRidgeV2;

figure
histogram(predPerf1(:), -0.05 : 0.05 : 0.2)
hold on
histogram(predPerf2(:), 0.1 : 0.05 : 0.45)

xlim([-0.2 0.8])
xticks(-0.2 : 0.2 : 0.8)
yl = ylim;
ylim([yl(1) yl(2)+5])
yl = ylim;
yticks([yl(1) yl(2)])

plot(mean(predPerf1, 'all'), yl(2)-2, 'v', ...
    'MarkerSize', 18, 'MarkerFaceColor', [0, 0.4470, 0.7410])
plot(mean(predPerf2, 'all'), yl(2)-2, 'v', ...
    'MarkerSize', 18, 'MarkerFaceColor', [0.8500, 0.3250, 0.0980])

title('Performance on target V1 after predictive dimension removal across datasets')
xlabel('Normalized performance')
ylabel('Datasets')
legend('V1 dims removed', 'V2 dims removed', 'Mean for V1 dims removed', ...
       'Mean for V2 dims removed')
   
%% 7A Dominant (factor) vs predictive dimensions for V2

% Set indices to control which dataset to plot.
iSession = 5;
iDataset = 2;
plotDims1 = 1:10;
plotDims2 = 1:3;

figure
PlotPredPerfVsPredPerf(predPerfFrV2(iDataset,plotDims1,iSession), ...
                       predErrorsFrV2(iDataset,plotDims1,iSession), ...
                       plotDims1, ...
                       predPerfRrrV2(iDataset,plotDims2,iSession), ...
                       predErrorsRrrV2(iDataset,plotDims2,iSession), ...
                       plotDims2, ...
                       'legend', {'Dominant dimensions', 'Predictive dimensions'}, ...
                       'ylim', [0.07 0.13])
title(sprintf('Cross-validated performance predicting V2, session %d dataset %d', ...
              iSession, iDataset))
          
%% 7B Dominant (factor) vs predictive dimensions for target V1

% Set indices to control which dataset to plot.
iSession = 5;
iDataset = 2;
plotDims1 = 1:10;
plotDims2 = 1:4;

figure
PlotPredPerfVsPredPerf(predPerfFrV1(iDataset,plotDims1,iSession), ...
                       predErrorsFrV1(iDataset,plotDims1,iSession), ...
                       plotDims1, ...
                       predPerfRrrV1(iDataset,plotDims2,iSession), ...
                       predErrorsRrrV1(iDataset,plotDims2,iSession), ...
                       plotDims2, ...
                       'legend', {'Dominant dimensions', 'Predictive dimensions'}, ...
                       'ylim', [0.05 0.17])
title(sprintf('Cross-validated performance predicting target V1, session %d dataset %d', ...
              iSession, iDataset))
                    
%% 7C

% Set constants.
N_DATASETS = 40;
N_DIMS = 10;

% Set colors.
axesColorOrder = get(0, 'DefaultAxesColorOrder');
COLOR(1,:) = axesColorOrder(1,:);
COLOR(2,:) = axesColorOrder(7,:);
clear axesColorOrder

% Reshape into nDatasets x length(nDimsForPred), where nDimsForPred = 1:4.
predPerfFrV1Reshaped = reshape(permute(predPerfFrV1, [1 3 2]), ...
                               [N_DATASETS N_DIMS]);
predPerfRrrV1Reshaped = reshape(permute(predPerfRrrV1(:,1:4,:), [1 3 2]), ...
                               [N_DATASETS 4]);

predPerfFrV2Reshaped = reshape(permute(predPerfFrV2, [1 3 2]), ...
                               [N_DATASETS N_DIMS]);
predPerfRrrV2Reshaped = reshape(permute(predPerfRrrV2(:,1:4,:), [1 3 2]), ...
                                [N_DATASETS 4]);


minDimsFrV1 = NaN(N_DATASETS, 4);
minDimsFrV2 = NaN(N_DATASETS, 4);
for iNumDims = 1:4
    
% For current number of predictive dimensions, find number of factor
% dimensions yielding better performance and subtract this from 11
% (subtract from 10 and add another one) to find minimum number of factors
% needed to match predictive dim performance, again, for current iNumDims.
minDimsFrV1(:,iNumDims) ...
    = 11 - sum(predPerfFrV1Reshaped > predPerfRrrV1Reshaped(:,iNumDims), 2);
minDimsFrV2(:,iNumDims) ...
    = 11 - sum(predPerfFrV2Reshaped > predPerfRrrV2Reshaped(:,iNumDims), 2);

end



figure
hold on

% Plot individual datasets for V1-V1.
for iNumDims = 1:4
    scatter(randn(N_DATASETS,1)*0.1 + iNumDims, ...
            minDimsFrV1(:,iNumDims), ...
            [], COLOR(1,:), 'LineWidth', 1, 'AlphaData', 0.001*ones(N_DATASETS,1), ...
            'MarkerEdgeAlpha', 'flat');
end

% In the paper, they only plotted nDimForPred = 1:3 for predicting V2, so
% we will drop the last column here.
for iNumDims = 1:3
    scatter(randn(N_DATASETS,1)*0.1 + iNumDims, ...
            minDimsFrV2(:,iNumDims), ...
            [], COLOR(2,:), 'LineWidth', 1, 'AlphaData', 0.001*ones(N_DATASETS,1), ...
            'MarkerEdgeAlpha', 'flat');
end

% Set limits.
xlim([0 6])
ylim([0 10])

% Plot diagonal.
plot([0 6], [0 6], '--', 'Color', 'k')

% Plot mean across datasets with errorbars.
errorbar(1:4, mean(minDimsFrV1), std(minDimsFrV1)/sqrt(N_DATASETS), ...
         'o-', 'Color', COLOR(1,:), ...
         'MarkerFaceColor', COLOR(1,:), ...
         'MarkerSize', 10, ...
         'DisplayName', 'V1-V1')
errorbar(1:3, mean(minDimsFrV2(:,1:3)), std(minDimsFrV2(:,1:3))/sqrt(N_DATASETS),...
         'o-', 'Color', COLOR(2,:), ...
         'MarkerFaceColor', COLOR(2,:), ...
         'MarkerSize', 10, ...
         'DisplayName', 'V1-V2')
     
% Add title and axes labels.
title('Dominant factors vs predictive dimensions')
xlabel('Number of predictive dimensions')
ylabel('Minimum number of dominant dimensions to match')
legend('V1-V1 individual dataset', ...
       '','','','','', ...
       'V1-V2 individual dataset', ...
       '', ...
       'V1-V1 mean + 1 SEM','V2-V2 mean + 1 SEM', ...
       'Location', 'southeast')
   
   
