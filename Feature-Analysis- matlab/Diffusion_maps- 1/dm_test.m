load  wine_data.csv;

% This dataset holds information about 3 types of wines. 
% the label 1/2/3 is in the first col. 
% Each row (cols 2-14) holds chemical test results taken from the wine.

% we reduce the dimension of the data from 13 to 2 and color it by the
% label.

lbl = wine_data(:,1);
data = wine_data(:,2:end);

% diffusion maps
%
% 1. build the pairwise Euclidean distances between the wines (I normalized with zscore)

dist = squareform(pdist(zscore(wine_data)));

% 2. Apply diffusion maps to reduce the dimension from 13 to 2

[ U, d_A, v_A, sigma ] = AnalyzeGraphAlpha(  dist,1, 'max', 10, 10);

% 3. plot - color by the label, in Diffusion maps the first coordinate is
% left out

figure; scatter(v_A(:,2), v_A(:,3), 30, lbl, 'filled')
