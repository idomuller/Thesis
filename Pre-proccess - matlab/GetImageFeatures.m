function [intens, entropy, kurtos, skewne, histogram] = GetImageFeatures(data)

bins=100:10:250;

% Histogram
histogram = histcounts(data, bins);

% Intense
intens = mean(data);

% Entropy
entropy = CalcEntropy(data);

% Kurtosis
kurtos = kurtosis(data);

% Skewness
skewne = skewness(data);

