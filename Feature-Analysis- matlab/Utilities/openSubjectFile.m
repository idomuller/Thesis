function sortedFeatureTable = openSubjectFile(filePath)

% Get table from file and make sure that the table is sorted

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