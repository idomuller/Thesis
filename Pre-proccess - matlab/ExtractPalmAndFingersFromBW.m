function [fingersBW, palmBW, changeInAreaUtils] = ExtractPalmAndFingersFromBW(BW)

%compute width of hand to limit iterations
bbox = regionprops(BW,'BoundingBox');
% maxRad = max(bbox.BoundingBox(3),bbox.BoundingBox(4));
maxRad = max(struct2array(bbox));

areaVec=nan(round(maxRad)+1,2);
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
for i = 2:k
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
rad = 20;
firstPick=[];
while length(firstPick)<2 && rad>1
    bigDiffInd = find(diff(n_pks(:,2))>rad, 1, 'first');
    firstPick = [n_pks(1,2), n_pks(bigDiffInd,2), n_pks(bigDiffInd+1,2)] ;
    rad = rad-1;
end
%sort peaks and extracted points
% pks = sortrows(pks,2);

fingersBW = SegmentFingersFromPicks(BW, firstPick(1:2));

palmBW = SegmentPalmFromPick(BW, firstPick(3));

changeInAreaUtils.radii = radii;
changeInAreaUtils.dy = dy;
changeInAreaUtils.areas = areas;
changeInAreaUtils.pks = firstPick;