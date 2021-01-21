function PlotAreaDiffVsRadius (changeInAreaUtils, handles)

axes(handles.ShowGradientGraph); % Make averSpec the current axes.
cla(handles.ShowGradientGraph,'reset')

maxY = max(changeInAreaUtils.dy(:,1));
minMax = [0, maxY+100];
yyaxis(handles.ShowGradientGraph, 'left')
plot(changeInAreaUtils.radii(2:end,1),changeInAreaUtils.dy,'LineWidth',1,...
    'Parent', handles.ShowGradientGraph);hold on
for ind=1:length(changeInAreaUtils.pks)
    xDot = changeInAreaUtils.radii(changeInAreaUtils.pks(ind),1);
    plot([xDot, xDot], minMax,'--k','Parent', handles.ShowGradientGraph)
end
% plot(changeInAreaUtils.radii(changeInAreaUtils.pks,1),...
%     changeInAreaUtils.dy(changeInAreaUtils.pks,1),'*r')
ylabel('dA/dr','FontSize',8)
xlabel('radius of structuring element','FontSize',8);
hold off

yyaxis right
plot(changeInAreaUtils.radii(2:end,1),changeInAreaUtils.areas(2:end,1),...
    'LineWidth',1);
ylabel('Area in number of pixels','FontSize',8) % right y-axis
title('Change in area (blue) and first derivative (red)');