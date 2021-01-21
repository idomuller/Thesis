function [palmBox, palmElipse] = ExtractPalmROI(palmBW, palmProps, plotFlag)

%% find rectangle arround the center of palm- by enlarging the rectangle
% Take the largest blob only.
imageMask = zeros(size(palmBW));

%Extract Centroid

palmCentroid = cat(1, palmProps.Centroid);
palmCentroid = round(palmCentroid);
colCenter = palmCentroid(1);
rowCenter = palmCentroid(2);
wide = 2;

flag = true;%sumOfMask == sum(MaskOnBW(:));
while flag
    
    imageMask(palmCentroid(2)-wide:palmCentroid(2)+wide,palmCentroid(1)-wide:palmCentroid(1)+wide)=1;
    sumOfMask = sum(imageMask(:));
    MaskOnBW = imageMask.*palmBW;
    flag = sumOfMask == sum(MaskOnBW(:));
    wide = wide+5;
    
end
wide = wide-25;
imageMask = zeros(size(palmBW));
imageMask(palmCentroid(2)-wide:palmCentroid(2)+wide,palmCentroid(1)-wide:palmCentroid(1)+wide)=1;

% edtImage = bwdist(~imageMask);
% boundaries = bwboundaries(edtImage);
% b= boundaries{1};
% x = b(:, 2);
% y = b(:, 1);  

[I,J]=find(imageMask>max(imageMask(:))/2);
IJ=[I,J];
[~,idx]=min(IJ*[1 1; -1 -1; 1 -1; -1 1].');
corners=IJ(idx,:);
corners=sortrows(corners,1);
corners(end+1,:) = corners(1, :);
palmBox = [corners(:,2)';corners(:,1)'];

xBox = palmBox(1,:);
yBox = palmBox(2,:);

%% Create an ellipse with specified
% semi-major and semi-minor axes, center, and image size.
% From the FAQ: https://matlab.wikia.com/wiki/FAQ#How_do_I_create_an_ellipse.3F
xRadius = wide+10;% palmProps.MinorAxisLength / 2;
yRadius = palmProps.MajorAxisLength / 2-30;
% Make an angle array of about the same number of angles as there are pixels in the perimeter.
theta = linspace(0, 2*pi, ceil(palmProps.Perimeter));
x = xRadius * cos(theta);
y = yRadius * sin(theta);
% Now we might need to rotate the coordinates slightly.  Make a rotation matrix
% Reference: https://en.wikipedia.org/wiki/Rotation_matrix
angleInDegrees = palmProps.Orientation - 90;
rotationMatrix = [cosd(angleInDegrees), -sind(angleInDegrees); sind(angleInDegrees), cosd(angleInDegrees)];
% Now do the rotation
xy = [x', y'];
xyRotated = xy * rotationMatrix;
x = round(xyRotated(:, 1)+ colCenter);
y = round(xyRotated(:, 2)+ rowCenter);
x(y>(min(yBox)-10))=[];
x(end+1) = x(1);
y(y>(min(yBox)-10))=[];
y(end+1) = y(1);
palmElipse = [x, y];
% closeElipseLine = [x(1)+1:x(end)-1]';
% closeElipseLine = [closeElipseLine, ones(size(closeElipseLine,1),1).*y(1)];
% palmElipse = [palmElipse; closeElipseLine];
%% 
% %% find rectangle arround the center of palm- bu minimal distance from center
% % Take the largest blob only.
% edtImage = bwdist(~palmBW);
% 
% % Find the max of the EDT:
% maxDistance = max(edtImage(:));
% [rowCenter, colCenter] = find(edtImage == maxDistance);
% 
% % Get the boundary of the blob.
% boundaries = bwboundaries(palmBW);
% b = boundaries{1}; % Extract from cell.
% x = b(:, 2);
% y = b(:, 1);
% % Get distances from center to each of the edge pixels.
% distances = sqrt((x - colCenter).^2 + (y - rowCenter).^2);
% % Find the min distance.
% [~, indexOfMin] = min(distances);
% % Find x and y of the min
% xMin = x(indexOfMin);
% yMin = y(indexOfMin);
% 
% % Get the delta x and delta y from center to corner
% dx = abs(colCenter - xMin);
% dy = abs(rowCenter - yMin);
% % Get edges of rectangle by adding and subtracting deltas from center.
% row1 = rowCenter - dy;
% row2 = rowCenter + dy;
% col1 = colCenter - dx;
% col2 = colCenter + dx;
% % Make a box so we can plot it.
% xBox = [col1, col2, col2, col1, col1];
% yBox = [row1, row1, row2, row2, row1];
% 
% plamBox = [xBox; yBox];

if plotFlag
    % Display the image.
    figure;
    subplot(1, 2, 1);
    imshow(edtImage, []);
    axis on;
    caption = sprintf('Distance Transform Image');
    title(caption, 'FontSize', 13, 'Interpreter', 'None');
    drawnow;
    hold on;
    plot(colCenter, rowCenter, 'r+', 'MarkerSize', 30, 'LineWidth', 2);
%     plot(xMin, yMin, 'co', 'MarkerSize', 10, 'LineWidth', 2);
    plot(xBox, yBox, 'r-', 'LineWidth', 2);% Now, for fun, plot it over the binary image.
    % Display the image.
    subplot(1, 2, 2);
    imshow(palmBW, []);
    axis on;
    caption = sprintf('Binary Image with largest rectangle');
    title(caption, 'FontSize', 12, 'Interpreter', 'None');
    hold on;
    plot(xBox, yBox, 'r-', 'LineWidth', 2);
    hold off
end