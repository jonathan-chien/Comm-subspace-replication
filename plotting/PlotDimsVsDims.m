function PlotDimsVsDims(optDimsV1,optDimsV2, color)

arguments
    optDimsV1 % nDatasets x nSessions matrix of prediction performance values
    optDimsV2 % nDatasets x nSessions matrix of prediction performance values
    color
end

% Plot predictive dims for target V1 vs V2.
scatter(optDimsV1(:), optDimsV2(:), 36, color)

% Average across datasets within session.
sessionMeanPredDimsV1 = mean(optDimsV1);
sessionMeanPredDimsV2 = mean(optDimsV2);

% Plot session means.
hold on
scatter(sessionMeanPredDimsV1, sessionMeanPredDimsV2, 24, color, 'filled')

% Plot dotted line bisecting (slope 1).
plot([0 9], [0 9], '--', 'Color', 'k')

xlim([0 9])
ylim([0 9])

end