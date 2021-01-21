function fetures = GetFingersFeatures(image, fingersTipsROI, fingerBasesROI, fingerNams)

fetures = [];

fingerNames = {'Thumbs', 'Index', 'Middle', 'Ring', 'Pinky'};

imageSize = size(image);

figure(23);
imagesc(image); hold on
for currnetFingure = fingerNames
    
    currnetInd = strcmp(fingerNams, currnetFingure);
    
    %% Finger tip/dist features
    tipMask = createCirclesMask(imageSize, [fingersTipsROI(currnetInd,1),...
        fingersTipsROI(currnetInd,2)], fingersTipsROI(currnetInd,3));
    
    tipsROIInd = find(tipMask);
    [row,col] = ind2sub(size(tipMask),tipsROIInd);
    plot(col,row, 'k*');
    relevantTipsPixels = image(tipsROIInd);
    
    [tipIntens, tipEntropy, tipKurtos, tipKkewne, tipHistogram] = GetImageFeatures(relevantTipsPixels);
    
    %% Finger base/proxy features    
    baseMask = createCirclesMask(imageSize, [fingerBasesROI(currnetInd,1),...
        fingerBasesROI(currnetInd,2)], fingerBasesROI(currnetInd,3));
    
    baseROIInd = find(baseMask);
    [row,col] = ind2sub(size(baseMask),baseROIInd);
    plot(col,row, 'k*');
    relevantBasePixels = image(baseROIInd);
    
    [baseIntens, baseEntropy, baseKurtos, baseSkewne, baseHistogram] = GetImageFeatures(relevantBasePixels);
    
    fetures = [fetures, tipIntens, tipEntropy, tipKurtos, tipKkewne, tipHistogram,...
        baseIntens, baseEntropy, baseKurtos, baseSkewne, baseHistogram];
    
end
hold off

% close (figure(23));


