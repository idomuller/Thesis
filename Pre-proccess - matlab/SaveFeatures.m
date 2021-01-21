function featureTable = SaveFeatures(image, imageData, featureTable)


if isempty(featureTable)
    try
        featureTable=readtimetable('C:\Users\ido\Google Drive\Thesis\Project\Pre-Process\Hands_extraction_algorithm\Workspace\all features.xlsx');
    catch
        varNames = {'subject_ID', 'Date_Time','Time'};
        allROINames = {'Thumbs_dist_', 'Thumbs_proxy_',...
            'Index_dist_', 'Index_proxy_',....
            'Middle_dist_', 'Middle_proxy_',...
            'Ring_dist_', 'Ring_proxy_',...
            'Pinky_dist_', 'Pinky_proxy_',...
            'Palm_arch_', 'Palm_Center_'};
        featuresNames = {'Intence', 'Entropy', 'kurtos', 'Skeweness'};
        bins=110:10:250;% DONT FORGET TO MATCH BINS INSIDE 'GetImageFeatures' FUNCTION
        %and to make sure the number of bins here will be length(bins)-1 !!
        
        for num = bins
            featuresNames= [featuresNames, ['Hist_' num2str(num)]];
        end
        
        for ROIName = allROINames
            for featName = featuresNames
                varNames = [varNames, [ROIName{:} featName{:}]];
            end
        end
        
        featureTable=cell2table(cell(1,length(varNames)),'VariableNames', varNames);
    end
end

firstPalmInd = find(strcmp(featureTable.Properties.VariableNames, 'Palm_arch_Intence'), 1);
featureTable(end+1,'subject_ID')= {imageData.subjectID} ;
% featureTable(end,'Trial_Phase')= {imageData.trialPhase} ;
featureTable(end,'Date_Time') = {imageData.imageTime};
imageTime = imageData.imageTime(strfind(imageData.imageTime, 'T')+1:end);
if length(imageTime)==6
    featureTable(end,'Time')= {[imageTime(1:2), ':',imageTime(3:4), ':',imageTime(5:6)]} ;
else
    featureTable(end,'Time')= {imageTime} ;
end

fingersFeatures = GetFingersFeatures(image, imageData.fingers.fingersTips,...
    imageData.fingers.fingersBase, imageData.fingers.props.fingerNames);
featureTable{end,4:firstPalmInd-1} = (num2cell(fingersFeatures));

palmFeatures = GetPalmFeatures(image, imageData.palm.elipse,...
    imageData.palm.mainBox);
featureTable{end,firstPalmInd:end} = (num2cell(palmFeatures));