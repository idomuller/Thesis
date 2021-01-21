function [imageOut, theta] = fixImageOrientation(imageIn)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Make sure the image orientation is as wanted: from buton to top, when the
% fingers pointing up
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% imageIn(all(imageIn==0,2),:)=[];
% imageIn(:, all(imageIn==0,1))=[];

%Get image edges
edges = edge(imageIn);

% Take background out
numAdgesCols = sum(edges,1);
numAdgesRows = sum(edges,2);

% Find center of mass/ palm
centers = regionprops(imageIn, 'centroid');
centroids = cat(1,centers.Centroid);
mainCenter = round(centroids(1,:));

halfUp = edges(1:mainCenter(2), :);
halfDown = edges(mainCenter(2)+1:end, :);

numAdgesHalfUp = sum(sum(halfUp, 2));
numAdgesHalfDown = sum(sum(halfDown, 2));

halfRight =edges(:, mainCenter(1)+1:end);
halfLeft = edges(:, 1:mainCenter(1));

numAdgesHalfRight = sum(sum(halfRight, 1));
numAdgesHalfLeft = sum(sum(halfLeft, 1));

[~, maxEdgesID] = max([numAdgesHalfUp,...
    numAdgesHalfDown,...
    numAdgesHalfRight,...
    numAdgesHalfLeft]);

% Check if up/down or left/right
switch maxEdgesID
    % If more adges on rows than edges on colum than it's up/down
    case 1
        % More edges on top- do nothing
        
        theta = 0;
        imageOut = imageIn;
        
    case 2
        % More edges on left- rotate 180degree right
        
        theta = 180;
        imageOut = imrotate(imageIn,theta);
    case 3
        % More edges on right- rotate 90degree left
        theta = 90;
        imageOut = imrotate(imageIn,theta);
    case 4
        % More edges on left- rotate 90degree right
        theta = -90;
        imageOut = imrotate(imageIn,theta);
end


%% Old way 
% if sum(numAdgesRows==2)>sum(numAdgesCols==2)
%     % If more adges on rows than edges on colum than it's up/down
%     halfUp = edges(1:mainCenter(2), :);
%     halfDown = edges(mainCenter(2)+1:end, :);
%     
%     numAdgesHalfUp = sum(halfUp, 2);
%     numAdgesHalfDown = sum(halfDown, 2);
%     
%     if sum(numAdgesHalfUp) > sum(numAdgesHalfDown)
%         % More edges on top- do nothing
%         theta = 0;
%         imageOut = imageIn;
%         
%     elseif sum(numAdgesHalfUp) < sum(numAdgesHalfUp)
%         % More edges on left- rotate 180degree right
%         theta = 180;
%         imageOut = imrotate(imageIn,theta);
%     else
%         
%     end
%     
% elseif sum(numAdgesRows==2)<sum(numAdgesCols==2)
%     % If more adges on rows than edges on colum than it's left/right
%     halfRight =edges(:, mainCenter(1)+1:end);
%     halfLeft = edges(:, 1:mainCenter(1));
%     
%     numAdgesHalfRight = sum(halfRight, 1);
%     numAdgesHalfLeft = sum(halfLeft, 1);
%     
%     if sum(numAdgesHalfRight) > sum(numAdgesHalfLeft)
%         % More edges on right- rotate 90degree left
%         theta = 90;
%         imageOut = imrotate(imageIn,theta);
%     elseif sum(numAdgesHalfRight) < sum(numAdgesHalfLeft)
%         % More edges on left- rotate 90degree right
%         theta = -90;
%         imageOut = imrotate(imageIn,theta);
%     else
%         
%     end
%     
% else
%     error('Couldn''d find orientation: same adges size')
%     
% end