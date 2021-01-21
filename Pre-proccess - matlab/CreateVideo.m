function outputfile = CreateVideo (folder, fileType)

if ~exist('fileType','var')
    fileType = '.jpg';
end

allFiles = dir(folder);
allFiles = allFiles(cellfun(@(x) ~x, {allFiles.isdir})); 
relevantFiles = struct2cell(allFiles(cellfun(@(x) contains(x, fileType), {allFiles.name})))'; 
SortedFiles = sort(relevantFiles(:,1));

%% Get file parts
if contains(lower(folder),'left')
    
    handSideIn = strfind(lower(folder), 'left');
    handSideIn = [handSideIn-1, handSideIn+4];
elseif contains(lower(folder),'right')
    
    handSideIn = strfind(lower(folder), 'right');
    handSideIn = [handSideIn-1, handSideIn+5];
    
end

dashInds = strfind(folder,'\');
subjectFirstInd = find(dashInds<handSideIn(1),1,'last');
subjectID = folder(dashInds(subjectFirstInd)+1:handSideIn(1)-1);

%% read all images and capture their sizes
for ind=1:size(SortedFiles,1) %where N is the number of images
    currentImage = [folder '\' SortedFiles{ind}];
    allImages{ind} = imread(currentImage); %read the next image
    allRows(ind,1) = size(allImages{ind},1);
    allCols(ind,1) = size(allImages{ind},2);
end

%% Complete images to match the bigger image size
maxRows = max(allRows);
completeRowsInd = find(allRows<maxRows);
for ind = completeRowsInd'
    
    numRowsToadd = maxRows- allRows(ind);
    addedrows = zeros(numRowsToadd,size(allImages{ind},2));
    a = allImages{ind}(:,:,1);
    a=[a; addedrows];
    b=allImages{ind}(:,:,2);
    b=[b; addedrows];
    c=allImages{ind}(:,:,3);
    c=[c; addedrows];
    
    allImages{ind}=[];
    allImages{ind}(:,:,1)=a;
    allImages{ind}(:,:,2)=b;
    allImages{ind}(:,:,3)=c;
    allImages{ind} = uint8(allImages{ind});
    clear a b c
end

maxCols = max(allCols);
completeColsInd = find(allCols<maxCols);
for ind = completeColsInd'
    
    numColsToadd = maxCols- allCols(ind);
    addedrows = zeros(size(allImages{ind},1), numColsToadd);
    a = allImages{ind}(:,:,1);
    a=[a, addedrows];
    b=allImages{ind}(:,:,2);
    b=[b, addedrows];
    c=allImages{ind}(:,:,3);
    c=[c, addedrows];
    
    allImages{ind}=[];
    allImages{ind}(:,:,1)=a;
    allImages{ind}(:,:,2)=b;
    allImages{ind}(:,:,3)=c;
    allImages{ind} = uint8(allImages{ind});
    clear a b c
end

outputfile = [folder '\' subjectID '_framevideo.avi'];
video = VideoWriter(outputfile); %create the video object
video.FrameRate = 3;
open(video); %open the file for writing

for ind=1:size(SortedFiles,1)
    fullName = SortedFiles{ind};
    imageTime = fullName(strfind(fullName, 'T')+1:end-4);
    if length(imageTime)==6
        imageTimeTxt = [imageTime(1:2), ':',imageTime(3:4), ':',imageTime(5:6)] ;
    else
        imageTimeTxt = [imageTime] ;
    end
    
    
    image = allImages{ind};
    image_with_txt = insertText(image,[size(image,2)-250,size(image,1)-250] ,imageTimeTxt,'FontSize',36,'TextColor', 'white', 'BoxColor','black');
    writeVideo(video,image_with_txt); %write the image to file
end
close(video); %close the file