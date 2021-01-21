function PalmAndFingerIntification(filePath)
filePath = '\\tsclient\c\Projects\Thesis_local\Data\No11\Left\No11_L_G_T50.csv';%files(index_file).name;
[~, filename, ~]= fileparts(filePath);

% imageTemp = frames(:,:,index_file);
imageTemp  = xlsread(filePath);%'\\tsclient\c\Projects\Thesis_local\Data\No11\Left\No11_L_G_T120.csv');
% Remove rows and cols with zeros val - due to scv defects
imageTemp(all(imageTemp==0,2),:)=[];
imageTemp(all(imageTemp==0,1),:)=[];

%normalize temperature data to visualize greyscale image
grey = imageTemp;
minV = min(imageTemp(:));
tempRange = range(imageTemp(:));

gerey = round(255*(grey-minV)/tempRange);

%Create Segmentation mask using mean method
BW1=imageTemp;
threshold = mean(imageTemp(:));
BW1(BW1<threshold)=0;
BW1(imageTemp>=threshold)=1;
BW1 = bwareaopen(BW1,9000);   %remove small spurious regions
BW1 = imfill(BW1,'holes');   %Fill holes inside main connected component

grey2 = grey;
grey2(BW1 == 0) = 0;

%Second thresholding- IDO
imageTempSegmented=imageTemp;
imageTempSegmented(BW1==0)=nan;
thresold2=nanmean(imageTempSegmented(:))-nanstd(imageTempSegmented(:));
BW = imageTemp;
BW(BW1==0 | imageTemp<thresold2)=false;
BW(BW>=thresold2)=1;
BW = bwareafilt(logical(BW),9000);   %remove small spurious regions
BW = imfill(BW,'holes');   %Fill holes inside main connected component

BW = fixImageOrientation(BW);

grey3 = grey;
grey3(BW == 0) = 0;

%compute width of hand to limit iterations
bbox = regionprops(BW,'BoundingBox');
% maxRad = max(bbox.BoundingBox(3),bbox.BoundingBox(4));
maxRad = max(struct2array(bbox));

areaVec=nan(maxRad+1,2);
areaVec(1,:) = [0,0];
areaTot = 0;

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
        areaVec(i+1,:) = [areaTot,i];  %add to vector containing areas through iterations including radius
        areaTot = 0; %reset area variable
    end
%     figure(6),imshow(BW2);
% pause;
end
areaVec = areaVec(2:end,:);   %remove extra zero at the begining
areaVec(isnan(areaVec(:,1)),:)=[];%remove extra variables at the end (if nan remain)


%differentiate number of pixels at each iteration with radius of SE used
areas = areaVec(:,1);
radii = areaVec(:,2);
dy = abs(diff(areas)./diff(radii));

%% IDO
localSTD = stdfilt(dy);
[peaks,locations] = findpeaks(localSTD,radii(2:end,1),'MinPeakHeight',mean(localSTD)+std(localSTD)/2);

% %find the peaks in the first derivative of the area
% [peaks,locations] = findpeaks(dy,radii(2:end,1));
pks = [peaks,locations];
nonPks = pks;


% %extract peaks which fall above 30% of range of peaks
% pk = peaks(:,1) > 0.3*((min(peaks(:,1)))+range(peaks(:,1)));
% pks(~pk,:) = [];
% nonPks(pk,:) = [];
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
n_pks=[];
for i = 1:k
    j=0;
    while dy(pks(i,2)-1-j) > lvl
        j = j+1;
    end
    n_pks = [n_pks;dy(pks(i,2)-1-j),pks(i,2)-j];
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
    n_pks = [n_pks;dy(pks(i,2)-1+j),pks(i,2)+j];
end
n_pks=sortrows(n_pks,2);
bigDiffInd = find(diff(n_pks(:,2))>20, 1, 'first');
firstPick = [n_pks(1,2), n_pks(bigDiffInd,2), n_pks(bigDiffInd+1,2)] ;
%sort peaks and extracted points
% pks = sortrows(pks,2);

%segment fingers from first peaks
SE = strel('disk',firstPick(1)-1);
BW3 = imopen(BW,SE);
SE = strel('disk',firstPick(2));
BW4 = imopen(BW,SE);
fingersBW = BW3 - BW4;
fingersBW = bwareaopen(fingersBW,50);

%segment palm from second peaks
SE = strel('disk',firstPick(3));
palmBW = imopen(BW,SE);


%% Figure
figure('Name',['File ', filename],'NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
subplot(2,3,1),imagesc(imageTemp),title('Original Thermal Image');%imshow(grey)

subplot(2,3,2);
title('Change in area (blue) and first derivative (red)');
[hAx,hline1,hline2]  = plotyy(radii(2:end,1),dy,radii(2:end,1),areas(2:end,1));
set(hline1,'LineWidth',1);
set(hline2,'LineWidth',1);
set(hAx,'LineWidth',1);
set(hAx,'FontSize',8)
ylabel(hAx(1),'dA/dr','FontSize',8) % left y-axis
ylabel(hAx(2),'Area in number of pixels','FontSize',8) % right y-axis
xlabel('radius of structuring element','FontSize',8)
title('Change in area (blue) and first derivative (red)');

subplot(2,3,3);
title('Derivative of area and selected points');
plot(radii(2:end,1),dy);
hold on
plot(pks(:,2),pks(:,1),'*r')
title('Derivative of area and selected points');


subplot(2,3,4);
imshow(BW);
title('Segmented image with morphological opening');

subplot(2,3,5);
imshow(fingersBW);
title('Segmented fingers');
axis off

subplot(2,3,6),imshow(BW),title('Segmented palm');
subplot(2,3,6),imshow(palmBW)%,title('Segmented palm');
axis off

% %save figure
% indexString = num2str(index_file+238);
% name = strcat(indexString, '.jpg');
% saveas(gcf,fullfile('C:\Users\Jean\Google Drive\Masters\Code\Single RUN\Hands_extraction_algorithm\Workspace\Palm_finger_identification\Phase 3', name));
% close
