numLines = size(points_matrix,1);
theta_mat = zeros(numLines-1,1);

%compute from all lines to all lines
for i = 1:numLines
    testLine = points_matrix(1,:);
    points_matrix(1,:) = [];
    theta = zeros(size(points_matrix,1),1);
    DirVector1=[testLine(1,1),testLine(1,2)]-[testLine(1,3),testLine(1,4)];
    for j = 1:size(points_matrix,1)
        DirVector2=[points_matrix(j,1),points_matrix(j,2)]-[points_matrix(j,3),points_matrix(j,4)];
        theta(j,1)=acos( dot(DirVector1,DirVector2)/norm(DirVector1)/norm(DirVector2) );
    end
    points_matrix = [points_matrix;testLine];
    theta_mat = [theta_mat,theta];
end

%remove extra column
theta_mat = theta_mat(:,2:end);
%find largest angle
max_all_cols = max(theta_mat);
max_angle = max(max_all_cols);

%find two lines between which largest angle falls
for i = 1:size(theta_mat,1)
    for j = 1:size(theta_mat,2)
        if theta_mat(i,j) == max_angle
            point = [i,j];
        end
    end
end

point1 = point(1,2);
point2 = mod(point(1,1)+point(1,2),numLines);

if point2 == 0
    point2 = numLines;
end

%compute angle between the two identified lines
angle = atand((points_matrix(point1,5)-points_matrix(point2,5))/(1-points_matrix(point1,5)*points_matrix(point2,5)));

%identify thumb by identifying if angle between two lines is clockwise (-ve) or
%anticlockwise (+ve)
if contains(filename, '_L_')%strcmpi(filename, 'left hand.ptw')
    if angle > 0
        thumb = points_matrix(point2,:);
        points_matrix(point2,:) = [];
    else
        thumb = points_matrix(point1,:);
        points_matrix(point1,:) = [];
    end
elseif contains(filename, '_R_')%strcmpi(filename, 'right Hand.ptw')
    if angle < 0
        thumb = points_matrix(point2,:);
        points_matrix(point2,:) = [];
    else
        thumb = points_matrix(point1,:);
        points_matrix(point1,:) = [];
    end
end