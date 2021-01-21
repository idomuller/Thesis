function registeredImage = RegisterImage(fixImage, movingImage, registrationType)

wb = waitbar(0, 'Registration in progress, please wait ...','WindowStyle', 'modal');
wbch = allchild(wb);
jp = wbch(1).JavaPeer;
jp.setIndeterminate(1);
    
if ~exist('registrationType', 'var') || registrationType
    
    registeredImage = IntensityRegistration (uint8(movingImage), uint8(fixImage), 'affine',...
                false);
            
else
    
    moving_c = imhistmatch(uint8(movingImage),uint8(fixImage));
    [D,registeredImage] = imregdemons(moving_c,uint8(fixedGrey),[500 400 200],...
        'AccumulatedFieldSmoothing',1.3);
            
end

close(wb);
