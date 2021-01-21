function [BW, grey, theta] = ImageSegmentation(image, theta)

% Remove rows and cols with zeros val - due to scv defects
imageCompy = image;
imageCompy(all(imageCompy==0,2),:)=[];
imageCompy(:, all(imageCompy==0,1))=[];

%Create Segmentation mask using mean method
BW1=image;
prc = prctile(imageCompy(:),[10,90]);
threshold = mean(imageCompy(imageCompy(:)>=prc(1)&imageCompy(:)<=prc(2)));
BW1(BW1<threshold)=0;
BW1(image>=threshold)=1;
BW1 = bwareafilt(logical(BW1),1); %BW1 = bwareaopen(BW1,9000);   %remove small spurious regions
BW1 = imfill(BW1,'holes');   %Fill holes inside main connected component

% %Second thresholding- IDO
% imageSegmented=imageCompy;
% imageSegmented(imageCompy<threshold)=nan;
% thresold2=nanmean(imageSegmented(:))-nanstd(imageSegmented(:));
% BW = image;
% BW(BW1==0 | image<thresold2)=false;
% BW(BW>=thresold2)=1;
% BW = bwareafilt(logical(BW),1); %BW = bwareafilt(logical(BW),9000);   %remove small spurious regions
% BW = imfill(BW,'holes');   %Fill holes inside main connected component
BW=BW1;
grey = image;
minV = min(image(:));
tempRange = range(image(:));

grey = round(255*(grey-minV)/tempRange);

if ~exist('theta','var')
    [BW, theta] = fixImageOrientation(BW);
end
grey = imrotate(grey,theta);


