function [fingersBW, palmBW, BW, grey, changeInAreaUtils, theta] = PalmAndFingerIdentification(image, plotFlag)

[BW, grey, theta] = ImageSegmentation(image);

[fingersBW, palmBW, changeInAreaUtils] = ExtractPalmAndFingersFromBW(BW);

%% Figure
if exist('plotFlag', 'var') && plotFlag

    figure('Name','Baseline Image','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
    subplot(2,3,1),imagesc(image),title('Original Thermal Image');%imshow(grey)
    
    subplot(2,3,2);
    title('Change in area (blue) and first derivative (red)');
    yyaxis left
    plot(changeInAreaUtils.radii(2:end,1),changeInAreaUtils.dy,'LineWidth',1);hold on
    plot(changeInAreaUtils.radii(changeInAreaUtils.pks,1),...
        changeInAreaUtils.dy(changeInAreaUtils.pks,1),'*r')
    ylabel('dA/dr','FontSize',8)
    xlabel('radius of structuring element','FontSize',8);
    hold off
    
    yyaxis right
    plot(changeInAreaUtils.radii(2:end,1),changeInAreaUtils.areas(2:end,1),...
        'LineWidth',1);
    ylabel('Area in number of pixels','FontSize',8) % right y-axis
    title('Change in area (blue) and first derivative (red)');

    subplot(2,3,3);
    title('Derivative of area and selected points');
    plot(changeInAreaUtils.radii(2:end,1),changeInAreaUtils.dy);
    hold on
    plot(changeInAreaUtils.radii(changeInAreaUtils.pks,1),...
        changeInAreaUtils.dy(changeInAreaUtils.pks,1),'*r');
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
end

