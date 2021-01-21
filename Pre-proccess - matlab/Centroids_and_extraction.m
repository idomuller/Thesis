%---------------------Extract centroids of palm blob----------------------%

palmCC = bwconncomp(palmBW);
palmProps = regionprops(palmCC,'Area');
palmAreas = cat(1,palmProps.Area);

%keep only largest blob if more then 1 have been retained 
if size(palmAreas,1) > 1
    [removeThresh,index] = max(palmAreas);
    removeThresh = removeThresh - 1;
    palmBW = bwareaopen(palmBW,removeThresh);
end

%Extract Centroid
palmProps = regionprops(palmBW,'centroid');
palmCentroid = cat(1, palmProps.Centroid);
palmCentroid = round(palmCentroid);

% ExtractPalmROI(palmBW);

%-------------------Extract centroids of fingers blob---------------------%

ExtractFingersROI (fingersBW)

fingersCC = bwconncomp(fingersBW);
fingersProps = regionprops(fingersCC,'Area');
fingersAreas = cat(1,fingersProps.Area);

%keep only 5 largest blobs if more then 5 have been retained
if size(fingersAreas,1) > 5
    for i = 1:5
        [removeThresh,index] = max(fingersAreas);
        fingersAreas(index,:) = [];
    end
    removeThresh = removeThresh - 1;
    fingersBW = bwareaopen(fingersBW,removeThresh);
    fingersCC = bwconncomp(fingersBW);
end

%Extract centroids 
fingersProps = regionprops(fingersCC,'centroid');
fingersCentroids = cat(1, fingersProps.Centroid);
fingersCentroids = round(fingersCentroids);

%make background pixels equal to 0, fingers pixels equal to 256 and palm
%pixels = 150
origBW = BW;
BW = uint8(BW);
BW(BW == 1) = 256;
BW(palmBW == 1) = 150;

%Compute gradients between palm centroid and all fingers centroids
gradientsT = ((fingersCentroids(:,2)-palmCentroid(1,2))./(fingersCentroids(:,1)-palmCentroid(1,1)));

%join all points and gradients in one matrix
points_matrix = [fingersCentroids, palmCentroid(1).*ones(size(fingersCentroids,1),1),...
    palmCentroid(2).*ones(size(fingersCentroids,1),1), gradientsT];

%identify thumb
Identify_thumb

%compute angles between two lines between thumb and all remaining lines
theta = zeros(size(points_matrix,1),1);
DirVector1=[thumb(1,1),thumb(1,2)]-[thumb(1,3),thumb(1,4)];
for i = 1:size(points_matrix,1)
    DirVector2=[points_matrix(i,1),points_matrix(i,2)]-[points_matrix(i,3),points_matrix(i,4)];
    theta(i,1)=acos( dot(DirVector1,DirVector2)/norm(DirVector1)/norm(DirVector2) );
end
points_matrix = [points_matrix,abs(theta)];


%identify little finger
points_matrix = sortrows(points_matrix,6);
little = points_matrix(end,:);
points_matrix(end,:) = [];

%identify index finger
index = points_matrix(1,:);
points_matrix(1,:) = [];

%compute intersection point for three lines (little, thumb and index) using y= mx+c
c = (thumb(1,2))-(thumb(1,5)*thumb(1,1));
thumb = [thumb,c];

c = (little(1,2))-(little(1,5)*little(1,1));
little = [little,c];

c = (index(1,2))-(index(1,5)*index(1,1));
index= [index,c];

%Extract all pixels which fall on the three lines
extract_points_lines

%move through the points on the line and find point which lies on the edge
%between the palm and the fingers
%thumb
i = 1;
while BW(round(thumb_points(i,2)),round(thumb_points(i,1))) ~= 150
i = i+1;
end
thumb_point = thumb_points(i,:);
thumb = [thumb,thumb_point];

%index
i = 1;
while BW(round(index_points(i,2)),round(index_points(i,1))) ~= 150
    i = i+1;
end
index_point = index_points(i,:);
index = [index,index_point];

%little
i = 1;
while BW(round(little_points(i,2)),round(little_points(i,1))) == BW(round(little_points(1,2)),round(little_points(1,1)))
    i = i+1;
end
little_point = little_points(i,:);
little = [little,little_point];

%Load file with template image and moving points
if contains(filename,'_L_')%strcmpi(filename, 'left hand.ptw')
    load('\\tsclient\c\Projects\Thesis_local\Hands_extraction_algorithm\Left_Hand_Template\Left_hand_template.mat');
elseif contains(filename,'_R_')%strcmpi(filename, 'right Hand.ptw')
    load('\\tsclient\c\Projects\Thesis_local\\Hands_extraction_algorithm\Right_Hand_Template\Right_hand_template.mat');
end
%set fixed points to identified points
%Moving points are set as thumb,index,little. set fixed points in a similar manner
fixed = round([thumb(1,7),thumb(1,8);index(1,8),index(1,9);little(1,8),little(1,9);thumb(1,3),thumb(1,4)]);

% template_fig = imrotate(template_fig,0);
% templateCC = regionprops(template_fig,'Area','centroid');
% templateAreas = cat(1,templateCC.Area);
% [~,areaSizeSortedInds]=sort(templateAreas);
% templateCentroid = cat(1,templateCC.Centroid);
% moving = templateCentroid(areaSizeSortedInds(1:4),:);

tform = fitgeotrans(moving,fixed,'affine');
template_tformed = imwarp(template_fig,tform);

%identify 3 control points on transformed image
templateCC = bwconncomp(template_tformed);
templateProps = regionprops(templateCC,'Area','centroid');
templateAreas = cat(1,templateProps.Area);
templateCentroid = cat(1,templateProps.Centroid);

% moving_tformed = [0,0];
% for i=1:size(templateAreas,1)
%     if templateAreas(i,1) <= 20
%         moving_tformed = [moving_tformed;templateCentroid(i,:)];
%     end
% end
% moving_tformed = round(moving_tformed(2:end,:));

[~, areaSizeInd]= sort(templateAreas);
holeSize = ceil(templateAreas(areaSizeInd(4)) + 1);
moving_tformed = round(templateCentroid(areaSizeInd(1:4),:));


%identify offset from transformed image to original image
fixed = sortrows(fixed,2);
moving_tformed = sortrows(moving_tformed,2);

Xoffset = fixed(1,1) - moving_tformed(1,1);
Yoffset = fixed(1,2) - moving_tformed(1,2);
offset = [Xoffset,Yoffset];


%apply offset to transformed template
if Xoffset > 0
    Xappend = zeros(size(template_tformed,1),Xoffset);
    template_tformed = [Xappend,template_tformed];
else
    template_tformed = template_tformed(:,(abs(Xoffset+1)):end);
end

if Yoffset > 0
    Yappend = zeros(Yoffset,size(template_tformed,2));
    template_tformed = [Yappend;template_tformed];
else
    template_tformed = template_tformed((abs(Yoffset)+1):end,:);
end

%append template with zeros at end to make size equal to original image
%Rows
if size(template_tformed,1) < size(BW,1)
    rows = zeros(size(BW,1)-size(template_tformed,1),size(template_tformed,2));
    template_tformed = [template_tformed;rows];
else
    template_tformed = template_tformed(1:(size(BW,1)),:);
end

%Columns
if size(template_tformed,2) < size(BW,2)
    cols = zeros(size(template_tformed,1),size(BW,2)-size(template_tformed,2));
    template_tformed = [template_tformed,cols];
else
    template_tformed = template_tformed(:,1:(size(BW,2)));
end

%remove control points from template and fill holes
template_tformed = bwareaopen(template_tformed,holeSize);% 20);
template_tformed = imfill(template_tformed,'holes');

%mark regions of template on control image
template_tformed2 = bwmorph(template_tformed,'remove');
grey(template_tformed2 == 1) = 256;

%Detect fingertips
c=zeros(0,4);
itter = 1;
thresh = 0.35;
while size(c,1)~=5 && itter<10
    c = houghcircles(fingersBW,10,70,thresh ,50);%houghcircles(fingersBW,8,65,0.45,25); %imfindcircles(fingersBW, [30,70],'sensitivity', 0.95);%[h, margin] = circle_hough(fingersBW, [30, 70]);
    thresh = thresh+0.01;
    itter=itter+1;
end
figure;imshow(fingersBW);hold on; plot(c(:,1), c(:,2),'or');
fingersBW2 = bwmorph(fingersBW,'remove');

% houghcircles(fingersBW2,8,65,0.55,25);
% houghcircles(fingersBW,8,65,0.45,25);

[rows cols] = meshgrid(1:320,1:256);
C = zeros(256,320);
for i = 1:size(c,1)
    C = C | sqrt((rows-round(c(i,1))).^2+(cols-round(c(i,2))).^2)<=(round(c(i,3))-2);
end
C2 = bwmorph(C,'remove');
grey(C2 == 1) = 256;

%----------------------Figures------------------------%
figure;imagesc(grey)
colormap jet
axis off
axis image

hold on

plot(palmCentroid(:,1),palmCentroid(:,2), 'g*');
plot(thumb(:,1),thumb(:,2), 'b*');
plot(little(:,1),little(:,2), 'b*');
plot(index(:,1),index(:,2), 'b*');
plot(thumb(:,7),thumb(:,8), 'r*');
plot(little(:,8),little(:,9), 'r*');
plot(index(:,8),index(:,9), 'r*');

line([thumb(1,3),thumb(1,1)],[thumb(1,4),thumb(1,2)]);
line([little(1,3),little(1,1)],[little(1,4),little(1,2)]);
line([index(1,3),index(1,1)],[index(1,4),index(1,2)]);

hold off
%----------------------------------------------------%

