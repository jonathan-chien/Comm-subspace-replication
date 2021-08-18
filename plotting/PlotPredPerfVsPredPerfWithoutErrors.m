function PlotPredPerfVsPredPerfWithoutErrors ...
    (predPerf1,dimsForPred1,predPerf2,dimsForPred2,nvp)

arguments
    predPerf1
    dimsForPred1
    predPerf2
    dimsForPred2
    nvp.legend = []
    nvp.title = false
end


% Set colors.
axesColorOrder = get(0, 'DefaultAxesColorOrder');
COLOR(1,:) = axesColorOrder(1,:);
COLOR(2,:) = axesColorOrder(7,:);
clear axesColorOrder


% Plot with error bars for model series 1.
plot(dimsForPred1, predPerf1, ...
     'o--', 'Color', COLOR(1,:), ...
     'MarkerFaceColor', COLOR(1,:), ...
     'MarkerSize', 10, ...
     'DisplayName', nvp.legend{1})         
hold on
     
% Plot with error bars for model series 2.
plot(dimsForPred2, predPerf2, ...
     'o--', 'Color', COLOR(2,:), ...
     'MarkerFaceColor', COLOR(2,:), ...
     'MarkerSize', 10, ...
     'DisplayName', nvp.legend{2})
 
% Plot horizontal line at 0.
plot(xlim, [0 0], '--', 'Color', 'k')
     
% Label axes, set legend, optionally title plot.
xlabel('Number of predictive dimensions')
ylabel('Predictive performance')
yticks([0 0.5 1])
if ~isempty(nvp.legend), legend; end
if nvp.title
    title('Cross-validated prediction performance for example dataset')
end

end