%Participants to exclude
%if index_file ~= 151|| index_file ~= 61

% filename = files(index_file).name;

% imageTemp = frames(:,:,index_file);
imageTemp  = xlsread('\\tsclient\c\Projects\Thesis_local\Data\No11\Left\No11_L_G_T120.csv');
%normalize temperature data to visualize greyscale image
grey = imageTemp;
minV = min(imageTemp(:));
tempRange = range(imageTemp(:));

for i = 1:size(grey,1);
    for j = 1:size(grey,2);
        grey(i,j) = round(((imageTemp(i,j)-minV)/(tempRange))*255);
    end
end
grey = uint8(grey);

%Create Segmentation mask using mean method
BW = imageTemp;

threshold = (mean(imageTemp(:)));
BW(BW<threshold)=0;
BW(BW>=threshold)=1;
BW = bwareaopen(BW,9000);   %remove small spurious regions
BW = imfill(BW,'holes');   %Fill holes inside main connected component

figure,imshow(BW)

grey2 = grey;
grey2(BW == 0) = 0;
figure,imagesc(grey2);


%compute width of hand to limit iterations
bbox = regionprops(BW,'BoundingBox');
maxRad = max(bbox.BoundingBox(3),bbox.BoundingBox(4));

areaVec = [0,0];
areaTot = 0;

%Figure
% figure('Name',['File ', filename],'NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
% subplot(2,3,1),imshow(grey),title('Original Thermal Image');
% subplot(2,3,2),title('Change in area (blue) and first derivative (red)');
% subplot(2,3,3),title('Derivative of area and selected points');
% subplot(2,3,4),imshow(BW),title('Segmented image with morphological opening');
% subplot(2,3,5),imshow(BW),title('Segmented fingers');
% subplot(2,3,6),imshow(BW),title('Segmented palm');

%iterate until radius equal maximum limit
for i = 1:1:maxRad
    SE = strel('disk',i);  %create disk shaped structuring element radius equal to i
    BW2 = imopen(BW,SE);   %morphological opening using disk shaped structuring element
    area = regionprops(BW2,'Area');  %extract area properties
    
    %if foreground area has been completely removed break iterations
    if size(area,1) == 0
        break
    else
        %compute total number of pixels in current binary image
        for j = 1:size(area,1)
            areaTot = areaTot + getfield(area,{j},'Area');
        end
        areaVec = [areaVec;areaTot,i];  %add to vector containing areas through iterations including radius
        areaTot = 0; %reset area variable
    end
%     figure,imshow(BW2);
% pause;
end
areaVec = areaVec(2:end,:);   %remove extra zero at the begining

%figure
% subplot(2,3,4),imshow(BW),title('Segmented image with morphological opening');

%differentiate number of pixels at each iteration with radius of SE used
areas = areaVec(:,1);
radii = areaVec(:,2);
dy = abs(diff(areas)./diff(radii));

%find the peaks in the first derivative of the area
[peaks,locations] = findpeaks(dy,radii(2:end,1));
pks = [peaks,locations];
nonPks = pks;


%extract peaks which fall above 30% of range of peaks
pk = peaks(:,1) > 0.3*((min(peaks(:,1)))+range(peaks(:,1)));
pks(~pk,:) = [];
nonPks(pk,:) = [];
%compute mean of remaining peaks
lvl = round(mean(nonPks(:,1)));

%compute spread of peaks (X-axis range)
spread = range(pks(:,2));

%remove peaks which are close to each other to end up with only 2 peaks
if size(pks,1) == 3
    for i = 1:2
        if abs(pks(i,2)-pks(i+1,2)) < 0.5*spread
            pks(i+1,:) = [];
            break;
        end
    end
elseif size(pks,1) == 4
    for i = 1:2
        if abs(pks(i,2)-pks(i+1,2)) < 0.5*spread
            pks(i+1,:) = [];
        end
    end
    %in some cases last peak is not detected, in this case add manually
elseif size(pks,1) == 1
    pks = [pks;dy(end),radii(end)];
end

%starting from the two peaks, identify 2 points at each end of each
%peak which first fall under lvl
k = size(pks,1);
for i = 1:k
    j=0;
    while dy(pks(i,2)-1-j) > lvl
        j = j+1;
    end
    pks = [pks;dy(pks(i,2)-1-j),pks(i,2)-j];
end

for i = 1:k
    j=0;
    while dy(pks(i,2)-1+j) > lvl
        j = j+1;
        if pks(i,2)-1+j > size(dy,1)
            j = 0;
            break
        end
    end
    pks = [pks;dy(pks(i,2)-1+j),pks(i,2)+j];
end

figure
% subplot(2,3,2);
[hAx,hline1,hline2]  = plotyy(radii(2:end,1),dy,radii(2:end,1),areas(2:end,1));
set(hline1,'LineWidth',2);
set(hline2,'LineWidth',2);
set(hAx,'LineWidth',2);
set(hAx,'FontSize',24)
ylabel(hAx(1),'dA/dr','FontSize',24) % left y-axis
ylabel(hAx(2),'Area in number of pixels','FontSize',24) % right y-axis
xlabel('radius of structuring element','FontSize',24)
% title('Change in area (blue) and first derivative (red)');

hold on
% figure
% subplot(2,3,3);
% plot(radii(2:end,1),dy);
% hold on
plot(pks(:,2),pks(:,1),'*r')
% title('Derivative of area and selected points');

%sort peaks and extracted points
pks = sortrows(pks,2);

%segment fingers from first peaks
SE = strel('disk',pks(1,2));
BW3 = imopen(BW,SE);
SE = strel('disk',pks(3,2));
BW4 = imopen(BW,SE);
fingersBW = BW3 - BW4;
% subplot(2,3,5),imshow(fingersBW)%,title('Segmented fingers');
% axis off
%segment palm from second peaks
SE = strel('disk',pks(5,2));
palmBW = imopen(BW,SE);
% subplot(2,3,6),imshow(palmBW)%,title('Segmented palm');
% axis off

% %save figure
% indexString = num2str(index_file+238);
% name = strcat(indexString, '.jpg');
% saveas(gcf,fullfile('C:\Users\Jean\Google Drive\Masters\Code\Single RUN\Hands_extraction_algorithm\Workspace\Palm_finger_identification\Phase 3', name));
% close
