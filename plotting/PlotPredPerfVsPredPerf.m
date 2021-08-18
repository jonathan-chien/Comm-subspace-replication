function PlotPredPerfVsPredPerf ...
    (predPerf1,predErrors1,dimsForPred1,predPerf2,predErrors2,dimsForPred2,nvp)

arguments
    predPerf1
    predErrors1
    dimsForPred1
    predPerf2
    predErrors2
    dimsForPred2
    nvp.legend = []
    nvp.title = false
    nvp.ylim = []
end

% Set colors.
axesColorOrder = get(0, 'DefaultAxesColorOrder');
COLOR(1,:) = axesColorOrder(1,:);
COLOR(2,:) = axesColorOrder(7,:);
clear axesColorOrder

% Plot with error bars for model series 1.
errorbar(dimsForPred1, predPerf1, predErrors1, ...
         'o--', 'Color', COLOR(1,:), ...
         'MarkerFaceColor', COLOR(1,:), ...
         'MarkerSize', 10, ...
         'DisplayName', nvp.legend{1})
hold on
     
% Plot with error bars for model series 2.
errorbar(dimsForPred2, predPerf2, predErrors2, ...
         'o--', 'Color', COLOR(2,:), ...
         'MarkerFaceColor', COLOR(2,:), ...
         'MarkerSize', 10, ...
         'DisplayName', nvp.legend{2})
     
% Plot horizontal line at 0.
plot(xlim, [0 0], '--', 'Color', 'k', 'HandleVisibility', 'off')
     
% Label axes, set legend, optionally title plot.
xlabel('Number of dimensions')
ylabel('Predictive performance')
ylabel('Predictive performance')
if ~isempty(nvp.ylim)
    ylim(nvp.ylim)
end

if ~isempty(nvp.legend), legend('Location', 'SouthEast'); end
if nvp.title
    title('Cross-validated prediction performance for example dataset')
end

end