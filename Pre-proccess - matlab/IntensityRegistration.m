function movingRegistered = IntensityRegistration (moving, fixed, transformType,...
    plotingFlag)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform intensity registration between fixed and moving images.
% This is done using imregister function.
%

%% Check Inoputs
if ~exist('transformType','var')
    transformType = 'affine';
    
else
    if strcmp(transformType, 'affine') &&...
            strcmp(transformType, 'translation')&&...
            strcmp(transformType, 'rigid')&&...
            strcmp(transformType, 'similarity')
       error('Error in movingRegistered: unknow transformType. use ''affine'', ''translation'', ''rigid'' or ''similarity'' only');
    end
end

if ~exist('plotingFlag', 'var')
    plotingFlag = false;
end

%% Perform the registration
[optimizer, metric] = imregconfig('multimodal');
% tform = imregtform(moving,fixed,transformType,optimizer,metric);

% movingRegistered = imwarp(moving,tform,'OutputView',imref2d(size(fixed)));

[movingRegistered , R_reg]= imregister(moving, fixed, 'rigid', optimizer, metric);
% B = imwarp(moving,tform);

%% Plot result
if plotingFlag
    
    figure;
    subplot(1,2,1);
    imshowpair(fixed, moving,'Scaling','joint');
    
    subplot(1,2,2);
    imshowpair(fixed, movingRegistered,'Scaling','joint')
    
end
