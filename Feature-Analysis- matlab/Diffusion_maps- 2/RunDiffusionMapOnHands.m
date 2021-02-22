%% Run mainfold based on the Thesis: Manifold Learning with Application to Deep Brain Stimulation (DBS)
clear all
% create the matrix again if not exist
if ~exist('data','var') || isempty(data)
    folder = 'C:\Projects\Thesis\Feature-Analysis- matlab\Resize Feature\';
    allFiles = struct2cell(dir(folder))';
    allFiles(cellfun(@(x) isempty(x), ...
        cellfun(@(x) strfind(x, '.xlsx'),allFiles(:,1),'UniformOutput',false)),:)=[];
    allFilesPath = cellfun(@(x) [folder, x], allFiles(:,1), 'UniformOutput', false);
    
    featureSelected = {'Intence'};
    
    inv_c = nan(39, 39, size(allFilesPath,1));
    data = [];
    indc=0;
    for currentPath = allFilesPath'
        indc=indc+1
        %% Open file and make sure that is sorted
        currentAllFeatures = [];
        filePath = currentPath{:};
        FeatureTable = openSubjectFile(filePath);
        
        
        %% Take relevant features - UPDATE ACCORDING TO THE DESIRED FEATURES
        relevantInds = find(all([contains(FeatureTable.Properties.VariableNames, featureSelected);...
            contains(FeatureTable.Properties.VariableNames, 'proxy');...
            contains(FeatureTable.Properties.VariableNames, featureSelected{:})],1));
        
        handFeatures = table2array(FeatureTable(:,relevantInds));
        handFeatures = handFeatures'; % Transpose to be time x ROIs
        
        %% convert to Ratios- COMMENT OUT IF NEEDED
        handFeatures = (handFeatures - handFeatures(:,1))./handFeatures;
        
        %% Calculate metric across all ROIs at every time
        dataFeatures = mean(handFeatures,1);
        data = [data; dataFeatures];
        
        %% Find hand Cov and inverse Cov matrix
        c = cov(handFeatures);
        inv_c(:,:,indc) = pinv(c);
        
    end
end
%% compute the pairwise distances based on the inverse covariance matrix of each point

Dis = zeros(size(data,1), size(data,1));
d_size = size(data,1);
for i = 1 : d_size
    for j = 1 : d_size
        
         Dis(i,j) = [data(i,:) - data(j,:)] * [inv_c(:,:,j)+inv_c(:,:,i)]*0.5 * [data(i,:) - data(j,:)]';

    end
end

ep_factor = 5;
ep = 1000;
alpha = 1;

[U, d_A, v_A] = GetDiffuisionMap (Dis, ep_factor, ep, alpha);
figure
scatter(d_A(2,2)*v_A(:,2), d_A(4,4)*v_A(:,4), 30,[1:d_size],'filled');

