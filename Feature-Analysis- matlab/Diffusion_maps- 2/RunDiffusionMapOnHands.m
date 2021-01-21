%% Run mainfold based on the Thesis

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
    featureTable = readtable(filePath);
    timeColIdx = find(strcmp(featureTable.Properties.VariableNames, 'Time'), 1);
    
    sInd = strfind(filePath,'s_');
    dashInd = strfind(filePath,'_');
    id = filePath(sInd:sInd+6);%dashInd(find(dashInd>sInd,1,'first')));
    tempTimeChar = featureTable.Time;
    
    Time = tempTimeChar;
    dt = [];
    baseRowIdx = find(featureTable.phase==0);
    baseRows = sortrows( featureTable(baseRowIdx, :),timeColIdx);
    dt = [dt; baseRows.dT];
    
    gRoewsIdx = find(featureTable.phase==1);
    gravitationRows = sortrows( featureTable(gRoewsIdx, :),timeColIdx);
    dt = [dt; gravitationRows.dT];
    RecoveryInd = size(dt,1);
    
    rRoewsIdx = find(featureTable.phase==2);
    recoveryRows = sortrows( featureTable(rRoewsIdx, :),timeColIdx);
    dt = [dt; recoveryRows.dT];
    dt = round(dt);
    
    sortedFeatureTable = [baseRows; gravitationRows; recoveryRows];

    %% Take relevant features - UPDATE ACCORDING TO THE DESIRED FEATURES
    relevantInds = find(all([contains(sortedFeatureTable.Properties.VariableNames, featureSelected);...
        contains(sortedFeatureTable.Properties.VariableNames, 'proxy');...
        contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
    
    handFeatures = table2array(sortedFeatureTable(:,relevantInds));
    handFeatures = handFeatures'; % Transpose to be time x ROIs
    
    dataFeatures = mean(handFeatures,1);
    data = [data; dataFeatures];
    
    %% Find hand Cov and inverse Cov matrix 
    c = cov(handFeatures);
    inv_c(:,:,indc) = pinv(c);    
    
end

%% compute the pairwise distances based on the inverse covariance matrix of each point

Dis = zeros(size(data,1), size(data,1));
for i = 1:size(data,1)
    for j = 1:size(data,1)
        
         Dis(i,j) = [data(i,:) - data(j,:)] * [inv_c(:,:,j)+inv_c(:,:,i)]*0.5 * [data(i,:) - data(j,:)]';

    end
end

%% now the gaussian kernel
ker = exp(-Dis/(ep_factor*ep));

% normalization
sum_alpha = sum( ker,2).^alpha;
symmetric_sum = sum_alpha*sum_alpha';
ker = ker./symmetric_sum;
    
   
% second normalization to make it row - stochatic         
sum_c = (sum( ker,2));        
for i=1:size(ker,1)
    ker(i,:) = ker(i,:) / sum_c(i);
end;

A = ker ;
 disp('doing an eigen value decomposition\n\n')

[ U, d_A, v_A ] = svd( double( A)); %Yuri


