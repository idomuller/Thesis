function features = GetPalmFeatures(image, elipse,...
     palmBox)

imageSize = size(image);

%% Palm elipse/arch features
% Get all elipse region pixels
allElipseIdx = FillBoundary(elipse, imageSize);

relevantElipsePixels = image(allElipseIdx);

[elipseIntens, elipseEntropy, elipseKurtos, elipseKkewne, elipseHistogram] = GetImageFeatures(relevantElipsePixels);

%% Palm box/center features
palmBoxLines = palmBox';
minX = min(palmBoxLines(:,1));
maxX = max(palmBoxLines(:,1));
minY = min(palmBoxLines(:,2));
maxY = max(palmBoxLines(:,2));
fillLine = [minY:maxY]';
palmBoxLines = [palmBoxLines; [ones(size(fillLine,1),1).*minX, fillLine]];
palmBoxLines = [palmBoxLines; [ones(size(fillLine,1),1).*maxX, fillLine]];

allBoxIdx = FillBoundary(palmBoxLines, imageSize);

relevantBOXPixels = image(allBoxIdx);

[boxIntens, boxEntropy, boxKurtos, boxSkewne, boxHistogram] = GetImageFeatures(relevantBOXPixels);

features = [elipseIntens, elipseEntropy, elipseKurtos, elipseKkewne, elipseHistogram,...
    boxIntens, boxEntropy, boxKurtos, boxSkewne, boxHistogram];


function allIdx = FillBoundary(boundaryInds, imageSize)

allIdx=[];
while ~isempty(boundaryInds)
    lineInd=[];
    currentYind = find(boundaryInds(:,2)==max(boundaryInds(:,2))); 
    
    [~, xMinIndT]= min(boundaryInds(currentYind,1));
    xMinInd = boundaryInds(currentYind(xMinIndT),1); 
    [~, xMaxIndT]= max(boundaryInds(currentYind,1));
    xMaxInd = boundaryInds(currentYind(xMaxIndT),1);
    
    if xMinInd < xMaxInd
        lineInd = [xMinInd:xMaxInd]';
        lineInd = [lineInd, ones(size(lineInd,1),1).*boundaryInds(currentYind(1),2)];
        idx=sub2ind(imageSize,lineInd(:,2), lineInd(:,1));
        allIdx = [allIdx; idx];
    else
        x=1;
    end
    boundaryInds(currentYind,:) = [];

end


