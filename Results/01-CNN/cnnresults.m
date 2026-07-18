%% CNN Parametric Analysis
clear; close all; clc;

T = readtable("CNNFullSweep.xlsx", "VariableNamingRule", "preserve");

outDir = "CNN_parametric_outputs";
if ~exist(outDir, "dir"); mkdir(outDir); end

predictors = {'N','filter_size','num_filters','lr'};
metrics = {'Accuracy %','Inference 10k s'};

summaryMetrics = {'Accuracy %','Precision %','Recall %','F1 %','Inference 10k s'};

%% Summary table by sample number
summaryByN = groupsummary(T, "N", {'mean','std','min','max'}, summaryMetrics);
writetable(summaryByN, fullfile(outDir, "CNN_summary_by_N.xlsx"));

%% Summary table for every parameter
allSummaries = table();

for p = 1:numel(predictors)
    pred = predictors{p};

    G = groupsummary(T, pred, {'mean','std','min','max'}, summaryMetrics);

    G.Parameter = repmat(string(pred), height(G), 1);
    G.Level = string(G.(pred));
    G.(pred) = [];

    G = movevars(G, {'Parameter','Level'}, 'Before', 1);

    allSummaries = [allSummaries; G];
end

writetable(allSummaries, fullfile(outDir, "CNN_summary_by_parameter.xlsx"));

%% Overall ANOVA, Tukey-Kramer, and Spearman correlation
statsOverall = table();
tukeyAll = table();

for p = 1:numel(predictors)
    pred = predictors{p};

    for m = 1:numel(metrics)
        metric = metrics{m};

        x = T.(pred);
        y = T.(metric);

        group = categorical(string(x));
        validGroups = categories(group);

        [pAnova, ~, stats] = anova1(y, group, "off");

        if numel(validGroups) >= 2
            C = multcompare(stats, "CType", "tukey-kramer", "Display", "off");

            tukeyTbl = array2table(C, ...
                "VariableNames", {'Group1','Group2','LowerCI','Difference','UpperCI','pValue'});

            tukeyTbl.Parameter = repmat(string(pred), height(tukeyTbl), 1);
            tukeyTbl.Metric = repmat(string(metric), height(tukeyTbl), 1);
            tukeyTbl.Group1Label = string(stats.gnames(tukeyTbl.Group1));
            tukeyTbl.Group2Label = string(stats.gnames(tukeyTbl.Group2));

            tukeyAll = [tukeyAll; tukeyTbl];
        end

        rho = NaN;
        pSpearman = NaN;

        if isnumeric(x) || islogical(x)
            [rho, pSpearman] = corr(double(x), y, ...
                "Type", "Spearman", "Rows", "complete");
        end

        statsOverall = [statsOverall; table( ...
            string(pred), string(metric), pAnova, rho, pSpearman, ...
            'VariableNames', {'Parameter','Metric','ANOVA_p','Spearman_rho','Spearman_p'})];
    end
end

writetable(statsOverall, fullfile(outDir, "CNN_overall_stats.xlsx"));
writetable(tukeyAll, fullfile(outDir, "CNN_tukey_posthoc.xlsx"));

%% Same analysis within each sample number, N
statsByN = table();
tukeyByN = table();

Ns = unique(T.N);

for i = 1:numel(Ns)
    Ti = T(T.N == Ns(i), :);

    for p = 1:numel(predictors)
        pred = predictors{p};

        if strcmp(pred, "N")
            continue;
        end

        for m = 1:numel(metrics)
            metric = metrics{m};

            x = Ti.(pred);
            y = Ti.(metric);

            group = categorical(string(x));
            validGroups = categories(group);

            [pAnova, ~, stats] = anova1(y, group, "off");

            if numel(validGroups) >= 2
                C = multcompare(stats, "CType", "tukey-kramer", "Display", "off");

                tukeyTbl = array2table(C, ...
                    "VariableNames", {'Group1','Group2','LowerCI','Difference','UpperCI','pValue'});

                tukeyTbl.N = repmat(Ns(i), height(tukeyTbl), 1);
                tukeyTbl.Parameter = repmat(string(pred), height(tukeyTbl), 1);
                tukeyTbl.Metric = repmat(string(metric), height(tukeyTbl), 1);
                tukeyTbl.Group1Label = string(stats.gnames(tukeyTbl.Group1));
                tukeyTbl.Group2Label = string(stats.gnames(tukeyTbl.Group2));

                tukeyByN = [tukeyByN; tukeyTbl];
            end

            rho = NaN;
            pSpearman = NaN;

            if isnumeric(x) || islogical(x)
                [rho, pSpearman] = corr(double(x), y, ...
                    "Type", "Spearman", "Rows", "complete");
            end

            statsByN = [statsByN; table( ...
                Ns(i), string(pred), string(metric), pAnova, rho, pSpearman, ...
                'VariableNames', {'N','Parameter','Metric','ANOVA_p','Spearman_rho','Spearman_p'})];
        end
    end
end

writetable(statsByN, fullfile(outDir, "CNN_stats_by_N.xlsx"));
writetable(tukeyByN, fullfile(outDir, "CNN_tukey_by_N.xlsx"));

%% Paired comparisons for binary parameters
binaryPredictors = {'num_filters','lr'};
pairedResults = table();

for p = 1:numel(binaryPredictors)
    pred = binaryPredictors{p};

    for m = 1:numel(metrics)
        metric = metrics{m};
        pairedResults = [pairedResults; pairedBinaryTest(T, pred, metric)];
    end
end

writetable(pairedResults, fullfile(outDir, "CNN_paired_binary_tests.xlsx"));

%% Top model tables by N
topAcc = table();
topTime = table();

for i = 1:numel(Ns)
    Ti = T(T.N == Ns(i), :);

    [~, ia] = max(Ti.("Accuracy %"));
    [~, it] = min(Ti.("Inference 10k s"));

    topAcc = [topAcc; Ti(ia,:)];
    topTime = [topTime; Ti(it,:)];
end

writetable(topAcc, fullfile(outDir, "CNN_top_models_by_accuracy.xlsx"));
writetable(topTime, fullfile(outDir, "CNN_top_models_by_time.xlsx"));

%% Plots: sample number
plotBox(T, "N", "Accuracy %", "Accuracy (%)", ...
    fullfile(outDir, "cnn_acc_v_N.png"));

plotBox(T, "N", "Inference 10k s", "10k Inference Time (s)", ...
    fullfile(outDir, "cnn_time_v_N.png"));

%% Plots: each CNN parameter split by N
paramsNoN = {'filter_size','num_filters','lr'};

for p = 1:numel(paramsNoN)
    pred = paramsNoN{p};

    plotMeanSDByN(T, pred, "Accuracy %", "Accuracy (%)", ...
        fullfile(outDir, "cnn_acc_v_" + pred + ".png"));

    plotMeanSDByN(T, pred, "Inference 10k s", "10k Inference Time (s)", ...
        fullfile(outDir, "cnn_time_v_" + pred + ".png"));
end

disp("Done. Results saved in: " + outDir);

%% ---------- Local functions ----------

function plotBox(T, pred, metric, ylab, saveName)
    figure('Color','w','Units','inches','Position',[1 1 5 4]);
    boxplot(T.(metric), categorical(T.(pred)));
    xlabel(pred, 'Interpreter','none');
    ylabel(ylab);
    grid on;
    set(gca, 'FontSize', 12);
    exportgraphics(gcf, saveName, 'Resolution', 300);
end

function plotMeanSDByN(T, pred, metric, ylab, saveName)
    Ns = unique(T.N);

    figure('Color','w','Units','inches','Position',[1 1 6 4]);
    hold on;

    for i = 1:numel(Ns)
        Ti = T(T.N == Ns(i), :);
        G = groupsummary(Ti, pred, {'mean','std'}, metric);

        xRaw = G.(pred);
        y = G.("mean_" + metric);
        e = G.("std_" + metric);

        if isnumeric(xRaw) || islogical(xRaw)
            x = double(xRaw);
            [x, idx] = sort(x);
            y = y(idx);
            e = e(idx);

            errorbar(x, y, e, '-o', ...
                'LineWidth', 1.5, ...
                'MarkerSize', 7);
        else
            x = 1:height(G);
            errorbar(x, y, e, '-o', ...
                'LineWidth', 1.5, ...
                'MarkerSize', 7);
            xticks(x);
            xticklabels(string(G.(pred)));
        end
    end

    xlabel(pred, 'Interpreter','none');
    ylabel(ylab);
    legend("N = " + string(Ns), 'Location','best');
    grid on;
    set(gca, 'FontSize', 12);
    exportgraphics(gcf, saveName, 'Resolution', 300);
end

function R = pairedBinaryTest(T, pred, metric)
    vals = unique(string(T.(pred)));
    vals = sort(vals);

    if numel(vals) ~= 2
        R = table(string(pred), string(metric), NaN, NaN, NaN, NaN, "", "", ...
            'VariableNames', {'Parameter','Metric','MeanDifference','SDDifference', ...
            'MedianDifference','pValue','Level_A','Level_B'});
        return;
    end

    otherVars = {'N','filter_size','num_filters','lr'};
    otherVars(strcmp(otherVars, pred)) = [];

    A = T(string(T.(pred)) == vals(1), [otherVars, {metric}]);
    B = T(string(T.(pred)) == vals(2), [otherVars, {metric}]);

    A.Properties.VariableNames(end) = {'Metric_A'};
    B.Properties.VariableNames(end) = {'Metric_B'};

    M = innerjoin(A, B, 'Keys', otherVars);

    d = M.Metric_B - M.Metric_A;

    if isempty(d)
        pVal = NaN;
    else
        pVal = signrank(d);
    end

    R = table(string(pred), string(metric), mean(d,'omitnan'), std(d,'omitnan'), ...
        median(d,'omitnan'), pVal, string(vals(1)), string(vals(2)), ...
        'VariableNames', {'Parameter','Metric','MeanDifference','SDDifference', ...
        'MedianDifference','pValue','Level_A','Level_B'});
end