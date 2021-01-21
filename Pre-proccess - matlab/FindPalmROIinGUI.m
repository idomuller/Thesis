function FindPalmROIinGUI()

figure (2);
global imageProccess
palmBox = []; 
palmElipse = [];

if ~isfield(imageProccess,'palm') && ~isfield(imageProccess.palm,'props')
    %Extract Centroid
    palmCC = bwconncomp(imageProccess.palm.BW);
    palmProps = regionprops(palmCC,'Area');
    palmAreas = cat(1,palmProps.Area);
    
    %keep only largest blob if more then 1 have been retained
    if size(palmAreas,1) > 1
        [removeThresh] = max(palmAreas);
        removeThresh = removeThresh - 1;
        imageProccess.palm.BW = bwareaopen(imageProccess.palm.BW,removeThresh);
    end
    
    %Extract props
    palmProps = regionprops(imageProccess.palm.BW,'centroid', 'MinorAxisLength',...
        'MajorAxisLength', 'Perimeter', 'Orientation');
    imageProccess.palm.props = palmProps;
end

imshow(imageProccess.palm.BW);
hold on
[Userx,Usery] = ginput(1);
imageProccess.palm.props.Centroid = [Userx, Usery];
[palmBox, palmElipse] = ExtractPalmROI(imageProccess.palm.BW,...
    imageProccess.palm.props, false);

plot(palmBox(1,:), palmBox(2,:), 'r-', 'LineWidth', 2);
plot(palmElipse(:,1), palmElipse(:,2), 'r-', 'LineWidth', 2);
hold off
close(2)

imageProccess.palm.elipse = palmElipse;
imageProccess.palm.mainBox = palmBox;
