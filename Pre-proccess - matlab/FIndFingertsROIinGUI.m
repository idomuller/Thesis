function FIndFingertsROIinGUI (fingerName)

figure (20);
global imageProccess
fingersBW = imageProccess.fingers.BW;
handSide = imageProccess.handSide;
imshow(imageProccess.fingers.BW);hold on
titleTxt = sprintf('Choose ROI in %s', fingerName);
title(titleTxt);

if ~isfield(imageProccess,'fingers') || ~isfield(imageProccess.fingers,'props')
    %% Extract Centroid
    % New section that need to be completed- find fingers ROIS
    
    tempFingersProps = regionprops(fingersBW,'centroid','PixelList','Area',...
        'Circularity', 'SubarrayIdx', 'Orientation', 'MinorAxisLength', 'MajorAxisLength');
    
    
    %% Find fingers ROI by the 5 highest areas in fingers image
    fingersPropsTable = struct2table(tempFingersProps);
    fingersArea = fingersPropsTable.Area;
    [~, highestInds] = sort(fingersArea, 'descend');
    fingersProps = fingersPropsTable(highestInds(1:5),:);
    
    fingersCentroid = fingersProps.Centroid;
    fingersCentroid = round(fingersCentroid);
    colCenter = fingersCentroid(:,1);
    rowCenter = fingersCentroid(:,2);
    imageSize = size(fingersBW);
    
    fingerNames = {'Thumbs', 'Index', 'Middle', 'Ring', 'Pinky'};
    if strcmp(handSide, 'left')
        [~, fingerPositionInds] = sort(colCenter, 'descend');
        fingersProps.fingerNames(fingerPositionInds)= fingerNames';
    else
        [~, fingerPositionInds] = sort(colCenter, 'ascend');
        fingersProps.fingerNames(fingerPositionInds)= fingerNames';
    end
    imageProccess.fingers.props = fingersProps;
end

if ~isfield(imageProccess.fingers,'fingersBase')
    imageProccess.fingers.fingersBase = zeros(5,3);
end
   
if ~isfield(imageProccess.fingers,'fingersTips') 
    imageProccess.fingers.fingersTips = zeros(5,3);
end

if ~isfield(imageProccess,'fingers') || ~isfield(imageProccess.fingers,'Boundary')||...
        isempty(imageProccess.fingers.Boundary.x)
    fingerBoundary = bwboundaries(imageProccess.fingers.BW);
    imageProccess.fingers.Boundary.x=[];
    imageProccess.fingers.Boundary.y=[];
    for ind = 1:size(fingerBoundary,1)
        b = fingerBoundary{ind}; % Extract from cell.
        imageProccess.fingers.Boundary.x = [imageProccess.fingers.Boundary.x; b(:, 2)];
        imageProccess.fingers.Boundary.y = [imageProccess.fingers.Boundary.y; b(:, 1)];
    end
end

[Userx,Usery] = ginput(1);
% Get distances from center to each of the edge pixels.
distances = sqrt((imageProccess.fingers.Boundary.x - Userx).^2 +...
    (imageProccess.fingers.Boundary.y - Usery).^2);
% Find the min distance.
[minDistance, indexOfMin] = min(distances);

fingersROI=[Userx, Usery, floor(minDistance)-3];
ang=0:0.01:2*pi;
xp=fingersROI(3)*cos(ang);
yp=fingersROI(3)*sin(ang);
plot(fingersROI(1)+xp,fingersROI(2)+yp);%,'Parent',handles.FingersAxes);

currentFingerInd = cellfun(@(x) ~isempty(strfind(x, fingerName(1:1:4))), imageProccess.fingers.props.fingerNames);
switch fingerName
    case{'ThumbsProx'}
        imageProccess.fingers.fingersBase(currentFingerInd,:) = fingersROI;
    case{'ThumbsDist'}
        imageProccess.fingers.fingersTips(currentFingerInd,:) = fingersROI;
    case{'IndexProx'}
        imageProccess.fingers.fingersBase(currentFingerInd,:) = fingersROI;
    case{'IndexDist'}
        imageProccess.fingers.fingersTips(currentFingerInd,:) = fingersROI;
    case{'MiddleProx'}
        imageProccess.fingers.fingersBase(currentFingerInd,:) = fingersROI;
    case{'MiddleDist'}
        imageProccess.fingers.fingersTips(currentFingerInd,:) = fingersROI;
    case{'RingProx'}
        imageProccess.fingers.fingersBase(currentFingerInd,:) = fingersROI;
    case{'RingDist'}
        imageProccess.fingers.fingersTips(currentFingerInd,:) = fingersROI;
    case{'PinkyProx'}
        imageProccess.fingers.fingersBase(currentFingerInd,:) = fingersROI;
    case{'PinkyDist'}
        imageProccess.fingers.fingersTips(currentFingerInd,:) = fingersROI;
end

close(20)


   
%     %find closest center
%     distFromCenter = sqrt((colCenter-Userx(fingerInd)).^2+(rowCenter-Usery(fingerInd)).^2);
%     [~,minId] = min(distFromCenter);
%     
%     
%     %%
%     % find rectangle arround the center of palm- bu minimal distance from center
%     % Take the largest blob only.
%     edtImage = bwdist(~palmBW);
%     
%     % Find the max of the EDT:
%     maxDistance = max(edtImage(:));
%     [rowCenter, colCenter] = find(edtImage == maxDistance);
%     
%     % Get the boundary of the blob.
%     boundaries = bwboundaries(palmBW);
%     b = boundaries{1}; % Extract from cell.
%     x = b(:, 2);
%     y = b(:, 1);
%     % Get distances from center to each of the edge pixels.
%     distances = sqrt((x - colCenter).^2 + (y - rowCenter).^2);
%     % Find the min distance.
%     [~, indexOfMin] = min(distances);
%     % Find x and y of the min
%     xMin = x(indexOfMin);
%     yMin = y(indexOfMin);
%     
%     % Get the delta x and delta y from center to corner
%     dx = abs(colCenter - xMin);
%     dy = abs(rowCenter - yMin);
%     % Get edges of rectangle by adding and subtracting deltas from center.
%     row1 = rowCenter - dy;
%     row2 = rowCenter + dy;
%     col1 = colCenter - dx;
%     col2 = colCenter + dx;
%     % Make a box so we can plot it.
%     xBox = [col1, col2, col2, col1, col1];
%     yBox = [row1, row1, row2, row2, row1];
%     
%     plamBox = [x
%     %%
    
    