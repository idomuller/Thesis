clc
close all
%% Deal with baseline image first

% Best example
% folderPath = '\\tsclient\c\Projects\Thesis_local\Data\No11\Left\';
% baseLineImage = 'No11_L_G_T10.csv';

% Fine example
% folderPath = '\\tsclient\c\Projects\Thesis_local\Data\No11\Right\';
% baseLineImage = 'No11_R_B_T0.csv';


% Not good example
folderPath = 'C:\Users\ido\Google Drive\Thesis\Data\No14\Left\';
baseLineImage = 'No14_L_G_T10.csv';

% Bad- with GUI
% folderPath = 'C:\Users\ido\Google Drive\Thesis\Data\No11\Left\';
% baseLineImage = 'No11_L_B_T0.csv';

fixedImage  = xlsread([folderPath baseLineImage]);
plotFlag = true;

[fingersBW, palmBW, BW, fixedGrey, ~, theta] = PalmAndFingerIdentification(fixedImage, plotFlag);

[palmCentroid, plamBox]= ExtractPalmFeatures (palmBW, plotFlag);

fingersROI = ExtractFingersROI (fingersBW, plotFlag);

% allImagesPath = '\\tsclient\c\Projects\Thesis_local\Data\No14\Right\';
allImages = struct2cell(dir(folderPath))';

for index_file =6: size(allImages,1)
    
    if ~allImages{index_file, 5}
        if ~strcmp(allImages{index_file, 1}, baseLineImage)
            currentfile = [folderPath allImages{index_file, 1}];
            
            movingImageOrig  = xlsread(currentfile);
            [~, movingGrey, theta] = ImageSegmentation(movingImageOrig);
            movingImageOrig = imrotate(movingImageOrig, theta);
            
            %             % Show base image and new image alignment
            %             moving_c = imhistmatch(movingGrey,fixedGrey);
            %             [D,movingReg] = imregdemons(movingGrey,fixedGrey,...
            %                 'AccumulatedFieldSmoothing',1.3);
            
            
            % Show base image and new image alignment
            moving_c = imhistmatch(uint8(movingGrey),uint8(fixedGrey));
            [D,movingReg] = imregdemons(moving_c,uint8(fixedGrey),[500 400 200],...
                'AccumulatedFieldSmoothing',1.3);
            
            figure('NumberTitle', 'off', 'Name', 'Non- rigid registration');
            subplot(2,2,1); imagesc(uint8(fixedGrey)); title('Base Image');
            subplot(2,2,2); imagesc(uint8(movingGrey)); title('Moving Image');
            subplot(2,2,3); imshowpair(uint8(fixedGrey), uint8(movingGrey)); title('Pair images');
            subplot(2,2,4); imagesc(movingReg); title('Registered Image');            
            
            movingRegistered = IntensityRegistration (uint8(movingGrey), uint8(fixedGrey), 'affine',...
                false);
            figure('NumberTitle', 'off', 'Name', 'Rigid registration');
            subplot(2,2,1); imagesc(uint8(fixedGrey)); title('Base Image');
            subplot(2,2,2); imagesc(uint8(movingGrey)); title('Moving Image');
            subplot(2,2,3); imshowpair(uint8(fixedGrey), uint8(movingGrey)); title('Pair images');
            subplot(2,2,4); imagesc(movingRegistered); title('Registered Image');            
            
%             [movingBW] = ImageSegmentation(movingRegistered);
%             figure;imshow(movingBW);hold on
%             
%             plot(plamBox(1,:), plamBox(2,:), 'r-', 'LineWidth', 2);% Now, for fun, plot it over the binary image.
%             for ind =1 :size(fingersROI,1)
%                 PlotCircle(fingersROI(ind,1),fingersROI(ind,2),fingersROI(ind,3));
%                 
%             end
            
            
            %Feature extraction
             pointsFixed = detectSURFFeatures(fixedGrey);
             pointsMoving = detectSURFFeatures(movingGrey);
%             [fFixed, vptsFixed] = extractFeatures(fixedGrey, pointsFixed);
%             [fMoving, vptsMoving] = extractFeatures(movingGrey, pointsMoving);
%             indexPairs = matchFeatures(fFixed, fMoving) ;
%             
        end
    end
end



% - use this to find the center of the hand : regionprops(imageIn)

