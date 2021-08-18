function PlotPerfVsNumPredDims(predPerf,predErrors,dimsUsedForPred,fullPredPerf,nvp)
% Replicates figure 4 A and B from Communication subspace, Semedo et al
% 2019. Plots predictive performance against number of predictive
% dimensions.
%
% PARAMETERS
% ----------
% predPerf        -- n-vector whose elements are the predictive performance
%                    of the regression model over n different numbers of
%                    predictive dimensions used.
% predErrors      -- n-vector whose elements are the predictive performance
%                    errors of the regression model over n different
%                    numbers of predictive dimensions used.
% dimsUsedForPred -- n-vector whose elements are the numbers of predictive
%                    dimensions used for the n models.
% fullPredPerf    -- Scalar value that is the predictive performance of the
%                    full-dimensionality model. 
% Name-Value Pairs (nvp)
%   'regressMethod' -- String value, either 'Reduced rank' or 'Factor'.
%                      Aids in generation of legend and is otherwise
%                      unused. There is no default, and an error will be
%                      thrown if a correct value is not supplied.
%   'subplot'       -- Logical true or false (default). Set true to
%                      suppress generation of a new figure, allowing this
%                      function to be used with existing subplot axes.
%   'title'         -- Logical true (default) or false. If true, will
%                      generate a pre-set title for the plot.
%   'colorIdx'      -- Scalar value, either 1 or 2, specifing whether to
%                      plot in blue (1) or red (2).
%
% RETURNS
% -------
% 2D plot -- Predictive performance of cross-validated regression model
%            plotted against number of dimensions used for prediction (as
%            regressors). Full model (with all dimensions used) displayed
%            as well.
%
% Author: Jonathan Chien. Version 1.0. 8/7/21. Last edit:.

arguments
    predPerf 
    predErrors
    dimsUsedForPred
    fullPredPerf
    nvp.regressMethod = []
    nvp.subplot = false
    nvp.title = true
    nvp.colorIdx = 1
end

% For creation of legend.
if ~strcmp(nvp.regressMethod, 'Reduced rank') ...
        && ~strcmp(nvp.regressMethod, 'Factor')
    error("Must specify 'regressMethod' as either 'Reduced rank' or 'Factor'.")
end

% Set colors.
axesColorOrder = get(0, 'DefaultAxesColorOrder');
COLOR(1,:) = axesColorOrder(1,:);
COLOR(2,:) = axesColorOrder(7,:);
clear axesColorOrder


if ~nvp.subplot, figure; end

% Plot with error bars.
errorbar(dimsUsedForPred, predPerf, predErrors, ...
         'o--', 'Color', COLOR(nvp.colorIdx,:), ...
         'MarkerFaceColor', COLOR(nvp.colorIdx,:), ...
         'MarkerSize', 10, ...
         'DisplayName', 'Reduced rank regression model')

% Plot performance of full ridge regression model.
hold on
plot(0, fullPredPerf, 'v', ...
     'MarkerFaceColor', COLOR(nvp.colorIdx,:), ...
     'MarkerSize', 10, ...
     'DisplayName', 'Ridge regression (full) model')

% Label axes, set legend, optionally title plot.
xlabel('Number of predictive dimensions')
ylabel('Predictive performance')
legend('Location', 'southeast')
if nvp.title
    title('Cross-validated prediction performance for example dataset')
end

end
