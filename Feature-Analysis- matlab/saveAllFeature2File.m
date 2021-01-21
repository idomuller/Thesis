allFilesPath = 'C:\Users\ido\Google Drive\Thesis\Data\Processed Data\JPG Feature tables\';

allFilesT = struct2cell(dir(allFilesPath))';
isXLS = cellfun(@(x) contains(x, '.xlsx'), allFilesT(:,1));
allFiles = allFilesT(isXLS,1);

variableNames = {};
correspondedId = {};
correspondedHandSide = {};
allFeatureMat = nan(150, 231, size(allFiles,1));

for fileInd =1:size(allFiles,1)
    filePath = [allFilesPath allFiles{fileInd}];
    featureTable = readtable(filePath);
    
    if isempty(variableNames)
        variableNames = featureTable.Properties.VariableNames;
        variableNames{2}='phase';
        variableNames = [variableNames(2:3), 'dT', variableNames(4:end)];
    end
    
    timeColIdx = find(strcmp(variableNames, 'Time'), 1);
    
    sInd = strfind(filePath,'s_');
    dashInd = strfind(filePath,'_');
    try
        id = str2double(filePath(sInd+2:sInd+3));
    catch
        error('Error in line 26: patient Id not as expected');
    end
    
    if contains(filePath, 'right')
        correspondedHandSide = [correspondedHandSide; 'right'];
    else
        correspondedHandSide = [correspondedHandSide; 'left'];
    end
        
    
    tempTimeChar = featureTable.Time;
    if any(contains(tempTimeChar,':'))
        [sortedFeatureTable, idx] = sortrows(featureTable, timeColIdx);
        timeChar = sortedFeatureTable.Time;
        times = cellfun(@(x) datenum(x,'HH:MM:SS'),timeChar);
        
        dt = [0; times(2:end,1)-times(1,1)];
        
        critdT = dt(2);
        while critdT> 2/1440
            sortedFeatureTable(1,:)=[];
            times(1) = [];
            dt = [0; times(2:end,1)-times(1,1)];
            critdT = dt(2);           
            
        end
            
        %Find trial phase
        reverseDt = [times(1:end-1,1) - (times(end,1)-10/1440);1];
        midInd = find(sign(reverseDt(1:end-1,1).*reverseDt(2:end,1))<0);
        try
        dtDiff = [zeros(midInd-3,1); diff(times(midInd-3:midInd+3,1)); zeros(size(times,1)-(midInd+3),1)];
        catch
            x=1;
        end
        [~, mInd] = max(dtDiff);
        RecoveryInd = mInd-1;

    else
        Time = cellfun(@(x) str2double(x), tempTimeChar);
        times = [];
        baseRowIdx = find(cellfun(@(x) contains(lower(x),'_b_'), featureTable.Date_Time));
        baseRows = sortrows( featureTable(baseRowIdx, :),timeColIdx);
        times = [times; cellfun(@(x) str2num(x), baseRows.Time)];
        
        gRoewsIdx = find(cellfun(@(x) contains(lower(x),'_g_'), featureTable.Date_Time));
        gravitationRows = sortrows( featureTable(gRoewsIdx, :),timeColIdx);
        times = [times; cellfun(@(x) str2num(x), gravitationRows.Time)];
        RecoveryInd = size(times,1);
        
        rRoewsIdx = find(cellfun(@(x) contains(lower(x),'_re_'), featureTable.Date_Time));
        recoveryRows = sortrows( featureTable(rRoewsIdx, :),timeColIdx);
        times = [times; cellfun(@(x) str2num(x), recoveryRows.Time)+times(end)+10];
        
        tempFeatureTable = [baseRows; gravitationRows; recoveryRows];
        [times, sortedTimesIDs] = sortrows(times);
        sortedFeatureTable = tempFeatureTable(sortedTimesIDs,:);
        timeChar = num2str(times);
        dt =times;
        
    end
    phase = zeros(size(sortedFeatureTable,1),1);
    phase(2:RecoveryInd,1) = 1;
    phase(RecoveryInd+1:end,1)=2;
    
    correspondedId = [correspondedId; filePath(sInd:sInd+6)];
    
    allFeatureMat(1:size(sortedFeatureTable,1),1,fileInd) = phase;
    allFeatureMat(1:size(sortedFeatureTable,1),2,fileInd) = times;
    allFeatureMat(1:size(sortedFeatureTable,1),3,fileInd) = dt*24*60*60;
    allFeatureMat(1:size(sortedFeatureTable,1),4:end,fileInd) = table2array(sortedFeatureTable(:,4:end));
end

save('C:\Projects\Thesis\Feature-Analysis- matlab\featureDataFile.mat', 'variableNames', 'allFeatureMat', 'correspondedId', 'correspondedHandSide');
