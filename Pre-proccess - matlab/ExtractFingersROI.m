function [fingersTipROI, fingersBaseROI, fingersProps] = ExtractFingersROI (fingersBW, handSide, palmCenter, plotFlag)
wInd=0;
w=waitbar(wInd, 'Please wait, starting fingers segmentation...');
pause(0.01);

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
imageSize = size(fingersBW);

fingerNames = {'Thumbs', 'Index', 'Middle', 'Ring', 'Pinky'};
if strcmp(handSide, 'left')
    [~, fingerPositionInds] = sort(colCenter, 'descend');
    fingersProps.fingerNames(fingerPositionInds)= fingerNames';
else
    [~, fingerPositionInds] = sort(colCenter, 'ascend');
    fingersProps.fingerNames(fingerPositionInds)= fingerNames';
end
% % There are ROIS that need to be remove- so try to identify which one is
% % fingerr
% yDist = palmCenter(2)-rowCenter;
% distFromCenter = sqrt((palmCenter(2)-colCenter).^2+(palmCenter(1)-rowCenter).^2);
%
% relevantInds = yDist>0; % only centers above the palm center- if minus its not relevant to finger
% if sum(relevantInds) < 5
%     % Need to add the thumd
%
% elseif sum(relevantInds) > 5
%     % There are more regions above the palm center that needs to be removed
%
% end


%% Detect fingertips
fingersTipROI = zeros(5,3);
fingersBaseROI = zeros(5,3);

for fInd = fingerPositionInds'
         

    waitbar(wInd/length(fingerPositionInds),w, 'Find fingers ROIs');
    pause(0.01);

    %Find finger region
    fingerRegion = zeros(imageSize);
    pixelList = cell2mat(fingersProps(fInd,:).PixelList);
    idx=sub2ind(imageSize,pixelList(:,2), pixelList(:,1));
    fingerRegion (idx)=1;
    fingerOrientation = deg2rad(fingersProps(fInd,:).Orientation);
    fingerCenter = fingersProps(fInd,:).Centroid;fingerWide = fingersProps(fInd,:).MinorAxisLength;
    fingerLength = fingersProps(fInd,:).MajorAxisLength;
    
    % Find the line between the finger center and top of the finger
    diagxDist = (20:10:fingerLength/2);
    dotOutIn = 0;
    dotOutDist = 0;
    dotInIn = 0;
    dotInDist = 0;
    diagIn=[];
    diagOut=[];
    
%     f=figure(fInd+3);
%     imshow(fingerRegion); hold on
    allOrientations = fingerOrientation-pi/2:0.2182:fingerOrientation+pi/2;% 15deg = 0.2618
    for tOrentation = allOrientations
        
        diagT=[];
        
        diagT(:,1) = fingerCenter(1)+sin(tOrentation).*diagxDist;
        diagT(:,2) = fingerCenter(2)- cos(tOrentation).*diagxDist;
        
        mask = zeros(imageSize);
        idx=sub2ind(imageSize,round(diagT(:,2)),round(diagT(:,1)));
        mask(idx) = 1;
        dotInT = sum(sum(fingerRegion.*mask));
        dotDistT = pdist([palmCenter(2), diagT(end,2); palmCenter(1), diagT(end,1)], 'euclidean');
%         plot(diagT(:,1), diagT(:,2),'-b','LineWidth',1);
    
        if all(diagT(:,2) < fingerCenter(2))
            
            if dotInT > dotOutIn %&&... % all point inside
                    %dotDistT > dotOutDist   % and far from the palm center
                diagOut = diagT;
                dotOutDist = dotDistT;
                dotOutIn = dotInT;
                orientationOut = tOrentation;
            end
        else
%             if dotInT > dotInIn %&&... % all point inside
% %                     -dotDistT < dotInDist   % and far from the palm center
%                 diagIn= diagT;
%                 dotInDist = dotDistT;
%                 dotInIn = dotInT;
%             end
        end
    end
    
    diagIn = [[fingerCenter(1)+sin(orientationOut+pi).*diagxDist]',...
        [fingerCenter(2)- cos(orientationOut+pi).*diagxDist]'];
%     diagIn = [diagIn];
   
%     plot(diagOut(:,1), diagOut(:,2),'-y','LineWidth',2);
%     plot(diagIn(:,1), diagIn(:,2),'-g','LineWidth',2); 
    
    %% Start to find the possibles circles
    radi = 20:2:fingerWide*0.5;
    tempCircles = zeros(0,4);
    for diaInd =size(diagOut,1):-1:1
        cx = diagOut(diaInd,1);
        cy = diagOut(diaInd,2);
        
        for radiInd = length(radi):-1:1
            
            mask = createCirclesMask(imageSize, [cx, cy], radi(radiInd));
            
            if ~any(any(mask.*~fingerRegion))
                tempCircles = [tempCircles; cx, cy, radi(radiInd), diagxDist(diaInd)];
                break
            end
        end
        
    end
    normalizedRadi = (tempCircles(:,3)-min(tempCircles(:,3)))./(max(tempCircles(:,3))-min(tempCircles(:,3)));
    normalizedDist = (tempCircles(:,4)-min(tempCircles(:,4)))./(max(tempCircles(:,4))-min(tempCircles(:,4)));
    
    circleScore = normalizedRadi .* normalizedDist ;
    [~, maxScore] = max( circleScore);
    try
        fingersTipROI (fInd,:)= tempCircles(maxScore,1:3);
    catch
        x=1;
    end
    
    clear normalizedRadi normalizedDist
    
    
    %% Find the second circle from the center
    tempCircles=[];
    radi = fingerWide*0.3:5:fingerWide*0.5;
    for diaInd =size(diagIn,1):-1:1
        cx = diagIn(diaInd,1);
        cy = diagIn(diaInd,2);
        for radiInd = length(radi):-1:1
            
            mask = createCirclesMask(imageSize, [cx, cy], radi(radiInd));
            
            if ~any(any(mask.*~fingerRegion))
                tempCircles = [tempCircles; cx, cy, radi(radiInd), diagxDist(diaInd)];
                break
            end
        end
%         if ~isempty(tempCircles)
%             secondRegionInds = find(mask);
%             if ~ismember(secondRegionInds, circleInds)
%                 fingersBaseROI(fInd,:) = tempCircles;
%             end
%         end
    end
    if ~isempty(tempCircles)
        normalizedRadi = (tempCircles(:,3)-min(tempCircles(:,3)))./(max(tempCircles(:,3))-min(tempCircles(:,3)));
        normalizedDist = (tempCircles(:,4)-min(tempCircles(:,4)))./(max(tempCircles(:,4))-min(tempCircles(:,4)));
        
        circleScore = normalizedRadi .* normalizedDist ;
        [~, maxScore] = max( circleScore);
        try
            fingersBaseROI (fInd,:)= tempCircles(maxScore,1:3);
        catch
            x=1;
        end
    end
%     PlotCircle(fingersTipROI(fInd,1),fingersTipROI(fInd,2),fingersTipROI(fInd,3), f);
%     PlotCircle(fingersBaseROI(fInd,1),fingersBaseROI(fInd,2),fingersBaseROI(fInd,3), f);
    clear diagxDist tempCircles
    wInd = wInd+1;
end
% Reduce radius by 1 - just to make sure all inside
fingersTipROI (:,3)= round(fingersTipROI (:,3))-5;
fingersBaseROI (:,3)= round(fingersBaseROI(:,3))-5;
close(w);
pause(0.01);
% c=zeros(0,4);
% itter = 1;
% thresh = 0.35;
% while size(c,1)~=5 && itter<10
%     c = houghcircles(fingersBW,10,70,thresh ,50);%houghcircles(fingersBW,8,65,0.45,25); %imfindcircles(fingersBW, [30,70],'sensitivity', 0.95);%[h, margin] = circle_hough(fingersBW, [30, 70]);
%     if size(c,1)<5
%         break
%     end
%     thresh = thresh+0.01;
%     itter=itter+1;
% end

%% Find finger edges coordinate
% boundaries = bwboundaries(fingerRegion);
% x=[];
% y=[];
% for ind = 1:size(boundaries,1)
%     b = boundaries{ind}; % Extract from cell.
%     x = [x; b(:, 2)];
%     y = [y; b(:, 1)];
% end

%% Find the minimum circle arround the center of the finger tip
% fingersROI = zeros(5,3);
% for ind = 1: size(c,1)
%     colCenter = c(ind,1);
%     rowCenter = c(ind,2);
%     % Get distances from center to each of the edge pixels.
%     distances = sqrt((x - colCenter).^2 + (y - rowCenter).^2);
%     % Find the min distance.
%     [minDistance, indexOfMin] = min(distances);
%     % Find x and y of the min
%     xMin = x(indexOfMin);
%     yMin = y(indexOfMin);
%
%     fingersROI(ind,:)=[colCenter,rowCenter, floor(minDistance)];
% end
% fingersTipROI = c;
% fingersTipROI(all(fingersTipROI==0,1),:)=[];
% 
% if exist('plotFlag','var') && plotFlag
%     f=figure;
%     imshow(fingersBW);
%     title('Fingers Circles ROI');
%     hold on;
%     plot(c(:,1), c(:,2), '*r')
%     for ind =1 :size(fingersTipROI,1)
%         PlotCircle(fingersTipROI(ind,1),fingersTipROI(ind,2),fingersTipROI(ind,3), f);
%         
%     end
% end
%% Find the minimum rectangle arround the center of the finger tip
% for ind =1 : size(c,1)
%     colCenter = c(ind,1);
%     rowCenter = c(ind,2);
%     % Get distances from center to each of the edge pixels.
%     distances = sqrt((x - colCenter).^2 + (y - rowCenter).^2);
%     % Find the min distance.
%     [~, indexOfMin] = min(distances);
%     % Find x and y of the min
%     xMin = x(indexOfMin);
%     yMin = y(indexOfMin);
%     plot(xMin, yMin, 'co', 'MarkerSize', 10, 'LineWidth', 2);
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
%     plot(xBox, yBox, 'r-', 'LineWidth', 2);% Now, for fun, plot it over the binary image.
% end