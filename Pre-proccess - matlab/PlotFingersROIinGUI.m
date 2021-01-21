function PlotFingersROIinGUI (figHandle, BWImage, fingersTips, fingersBase)%handles)

% global imageProccess

% fingersTips = imageProccess.fingers.fingersTips;
% fingersBase = imageProccess.fingers.fingersBase;

axes(figHandle);
cla reset
imagesc(BWImage, 'Parent', figHandle);%imshow(BWImage, 'Parent', figHandle);
hold(figHandle, 'on');
ang=0:0.01:2*pi;
for ind =1 :size(fingersTips,1)
xp=fingersTips(ind,3)*cos(ang);
yp=fingersTips(ind,3)*sin(ang);
plot(fingersTips(ind,1)+xp,fingersTips(ind,2)+yp,'Parent',figHandle,'Color', 'r');  
end

for ind =1 :size(fingersBase,1)
xp=fingersBase(ind,3)*cos(ang);
yp=fingersBase(ind,3)*sin(ang);
plot(fingersBase(ind,1)+xp,fingersBase(ind,2)+yp,'Parent',figHandle, 'Color', 'r');  
end

hold(figHandle, 'off');