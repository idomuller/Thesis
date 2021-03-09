function varargout = plotFeatures(varargin)
% PLOTFEATURES MATLAB code for plotFeatures.fig
%      PLOTFEATURES, by itself, creates a new PLOTFEATURES or raises the existing
%      singleton*.
%
%      H = PLOTFEATURES returns the handle to a new PLOTFEATURES or the handle to
%      the existing singleton*.
%
%      PLOTFEATURES('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PLOTFEATURES.M with the given input arguments.
%
%      PLOTFEATURES('Property','Value',...) creates a new PLOTFEATURES or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before plotFeatures_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to plotFeatures_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help plotFeatures

% Last Modified by GUIDE v2.5 07-Nov-2020 22:20:42

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @plotFeatures_OpeningFcn, ...
                   'gui_OutputFcn',  @plotFeatures_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before plotFeatures is made visible.
function plotFeatures_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to plotFeatures (see VARARGIN)

% Choose default command line output for plotFeatures
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes plotFeatures wait for user response (see UIRESUME)
% uiwait(handles.figure1);
image = imread('originalImage.JPG');
image=imresize(image,[125, 125]);
imshow(imread('handImage.JPG'), 'Parent', handles.handImage);

set(handles.thumbDistCheckBox, 'value',true);
set(handles.thumbProxyCheckBox, 'value',true);
set(handles.indexDistCheckBox, 'value',true);
set(handles.indexProxyCheckBox, 'value',true);
set(handles.middleDistCheckBox, 'value',true);
set(handles.middleProxyCheckBox, 'value',true);
set(handles.ringDistCheckBox, 'value',true);
set(handles.ringProxyCheckBox, 'value',true);
set(handles.pinkyDistCheckBox, 'value',true);
set(handles.pinkyProxyCheckBox, 'value',true);
set(handles.palmCenterCheckBox, 'value',true);
set(handles.palmArchCheckBox, 'value',true);
set(handles.allDist, 'value',true);
set(handles.allProxy, 'value',true);

defaultPath='C:\Users\ido\Google Drive\Thesis\Data\Processed Data\JPG Feature tables\';
if isfolder(defaultPath)
    set(handles.pathBrowser, 'String',defaultPath);
else
    set(handles.pathBrowser, 'String','Add path');
end

% --- Outputs from this function are returned to the command line.
function varargout = plotFeatures_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


function pathBrowser_Callback(hObject, eventdata, handles)
% hObject    handle to pathBrowser (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of pathBrowser as text
%        str2double(get(hObject,'String')) returns contents of pathBrowser as a double


% --- Executes during object creation, after setting all properties.
function pathBrowser_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pathBrowser (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in browseBtn.
function browseBtn_Callback(hObject, eventdata, handles)
% hObject    handle to browseBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
defaultPath='C:\Users\ido\Google Drive\Thesis\Data\Processed Data\';

% if isfolder(defaultPath)
%     [file,path] = uigetfile([defaultPath '*.xlsx'],'Select an image file');
% else
%     [file,path] = uigetfile('*.xlsx','Select an image file');
% end
% set( handles.pathBrowser, 'String', [path file]);

allFilesT = struct2cell(dir(get(handles.pathBrowser, 'String')))';
isXLS = cellfun(@(x) contains(x, '.xlsx'), allFilesT(:,1));
allFiles = allFilesT(isXLS,1);
set(handles.fileList, 'string',allFiles );

% --- Executes during object creation, after setting all properties.
function handImage_CreateFcn(hObject, eventdata, handles)
% hObject    handle to handImage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate handImage


% --- Executes on button press in thumbProxyCheckBox.
function thumbProxyCheckBox_Callback(hObject, eventdata, handles)
% hObject    handle to thumbProxyCheckBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of thumbProxyCheckBox
plotHelper(handles)


% --- Executes on button press in indexProxyCheckBox.
function indexProxyCheckBox_Callback(hObject, eventdata, handles)
% hObject    handle to indexProxyCheckBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of indexProxyCheckBox
plotHelper(handles)


% --- Executes on button press in middleProxyCheckBox.
function middleProxyCheckBox_Callback(hObject, eventdata, handles)
% hObject    handle to middleProxyCheckBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of middleProxyCheckBox
plotHelper(handles)


% --- Executes on button press in ringProxyCheckBox.
function ringProxyCheckBox_Callback(hObject, eventdata, handles)
% hObject    handle to ringProxyCheckBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ringProxyCheckBox
plotHelper(handles)


% --- Executes on button press in pinkyProxyCheckBox.
function pinkyProxyCheckBox_Callback(hObject, eventdata, handles)
% hObject    handle to pinkyProxyCheckBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of pinkyProxyCheckBox
plotHelper(handles)


% --- Executes on button press in pinkyDistCheckBox.
function pinkyDistCheckBox_Callback(hObject, eventdata, handles)
% hObject    handle to pinkyDistCheckBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of pinkyDistCheckBox
plotHelper(handles)


% --- Executes on button press in ringDistCheckBox.
function ringDistCheckBox_Callback(hObject, eventdata, handles)
% hObject    handle to ringDistCheckBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ringDistCheckBox
plotHelper(handles)


% --- Executes on button press in middleDistCheckBox.
function middleDistCheckBox_Callback(hObject, eventdata, handles)
% hObject    handle to middleDistCheckBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of middleDistCheckBox
plotHelper(handles)


% --- Executes on button press in indexDistCheckBox.
function indexDistCheckBox_Callback(hObject, eventdata, handles)
% hObject    handle to indexDistCheckBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of indexDistCheckBox
plotHelper(handles)


% --- Executes on button press in thumbDistCheckBox.
function thumbDistCheckBox_Callback(hObject, eventdata, handles)
% hObject    handle to thumbDistCheckBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of thumbDistCheckBox
plotHelper(handles)


% --- Executes on button press in palmCenterCheckBox.
function palmCenterCheckBox_Callback(hObject, eventdata, handles)
% hObject    handle to palmCenterCheckBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of palmCenterCheckBox
plotHelper(handles)


% --- Executes on button press in palmArchCheckBox.
function palmArchCheckBox_Callback(hObject, eventdata, handles)
% hObject    handle to palmArchCheckBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of palmArchCheckBox
plotHelper(handles)


% --- Executes on button press in plotBtn.
function plotBtn_Callback(hObject, eventdata, handles)
% hObject    handle to plotBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global featureToPlot
featureToPlot=struct;
allFilse = get(handles.fileList, 'String');
filePath = [get( handles.pathBrowser, 'String'),'\', allFilse{get(handles.fileList, 'value')}];
featureTable = readtable(filePath);
timeColIdx = find(strcmp(featureTable.Properties.VariableNames, 'Time'), 1);

sInd = strfind(filePath,'s_');
dashInd = strfind(filePath,'_');
featureToPlot.id = filePath(sInd:sInd+6);%dashInd(find(dashInd>sInd,1,'first')));
tempTimeChar = featureTable.Time;
if any(contains(tempTimeChar,':'))
    [featureToPlot.sortedFeatureTable, idx] = sortrows(featureTable, timeColIdx);
    timeChar = featureToPlot.sortedFeatureTable.Time;
    times = cellfun(@(x) datenum(x,'HH:MM:SS'),timeChar);

    dt = [0; times(2:end,1)-times(1,1)];
    
    %Find trial phase
    reverseDt = [times(1:end-1,1) - (times(end,1)-10/1440);1];
    midInd = find(sign(reverseDt(1:end-1,1).*reverseDt(2:end,1))<0);
    
    dtDiff = [zeros(midInd-3,1); diff(times(midInd-3:midInd+3,1)); zeros(size(times,1)-(midInd+3),1)];
    [~, mInd] = max(dtDiff);
    featureToPlot.RecoveryInd = mInd-1;
else
    featureTable.Time = cellfun(@(x) str2num(x), tempTimeChar);
    times = [];
    baseRowIdx = find(cellfun(@(x) contains(lower(x),'_b_'), featureTable.Date_Time));
    baseRows = sortrows( featureTable(baseRowIdx, :),timeColIdx);
    times = [times; baseRows.Time];
    
    gRoewsIdx = find(cellfun(@(x) contains(lower(x),'_g_'), featureTable.Date_Time));
    gravitationRows = sortrows( featureTable(gRoewsIdx, :),timeColIdx);
    times = [times; gravitationRows.Time+times(end)+10];
    featureToPlot.RecoveryInd = size(times,1);
    
    rRoewsIdx = find(cellfun(@(x) contains(lower(x),'_re_'), featureTable.Date_Time));
    recoveryRows = sortrows( featureTable(rRoewsIdx, :),timeColIdx);
    times = [times; recoveryRows.Time+times(end)+10];

    featureToPlot.sortedFeatureTable = [baseRows; gravitationRows; recoveryRows];
    timeChar = num2str(times);
    dt =times;
    
end


featureToPlot.featuresToChoose = {'Intence', 'Entropy', 'kurtos', 'Skeweness'};
featureToPlot.dt=dt;
featureToPlot.timeChar=timeChar;

featureToPlot.yLim = [0, 250;...
                      0, 10;...
                       -5, 5;...
                       -5, 5];
if ~any(featureTable.Index_dist_Intence>50)
    featureToPlot.yLim(1,:) = [0, 40];
end

plotHelper(handles);

% h1  =plot(dt, cell2mat(sortedFeatureTable.


% --- Executes on selection change in selectFeaturePopUp.
function selectFeaturePopUp_Callback(hObject, eventdata, handles)
% hObject    handle to selectFeaturePopUp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns selectFeaturePopUp contents as cell array
%        contents{get(hObject,'Value')} returns selected item from selectFeaturePopUp
plotHelper(handles)


% --- Executes during object creation, after setting all properties.
function selectFeaturePopUp_CreateFcn(hObject, eventdata, handles)
% hObject    handle to selectFeaturePopUp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

 
function plotHelper(handles)

global featureToPlot
sortedFeatureTable = featureToPlot.sortedFeatureTable;
featuresToChoose = featureToPlot.featuresToChoose;
dt = featureToPlot.dt;
colors = [0 0.4470 0.7410;...
    0.8500 0.3250 0.0980;...
    0.9290 0.6940 0.1250;...
    0.4940 0.1840 0.5560;...
    0.4660 0.6740 0.1880;...
    0.3010 0.7450 0.9330;...
    0.6350 0.0780 0.1840];
tixc = 1:3:length(dt);

featureID = get(handles.selectFeaturePopUp, 'value');
featureSelected = featuresToChoose(featureID);
movingAveWinTgl = get(handles.movingAveWinTgl,'value');
legednTxt = {};

%% Secondary Plot
axes(handles.secondaryPlot);
cla(handles.secondaryPlot, 'reset');
% hold on
axes(handles.mainPlot);
cla(handles.mainPlot, 'reset');

%Thumb
if get(handles.thumbProxyCheckBox,'value')
    Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Thumbs');...
        contains(sortedFeatureTable.Properties.VariableNames, 'proxy');...
        contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
    legednTxt=[legednTxt; 'Thumb_proxy'];

    data = sortedFeatureTable{:,Ind};
    plotLine(handles, dt, data, movingAveWinTgl, colors(1,:), '-.');
    
end

if get(handles.thumbDistCheckBox,'value')
    Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Thumbs');...
        contains(sortedFeatureTable.Properties.VariableNames, 'dist');...
        contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
    data = sortedFeatureTable{:,Ind};
    plotLine(handles, dt, data, movingAveWinTgl, colors(1,:), '--');
    legednTxt=[legednTxt; 'Thumb_dist'];
end

%Index
if get(handles.indexProxyCheckBox,'value')
    Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Index');...
        contains(sortedFeatureTable.Properties.VariableNames, 'proxy');...
        contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
    data = sortedFeatureTable{:,Ind};
    plotLine(handles, dt, data, movingAveWinTgl, colors(2,:), '-.');
    legednTxt=[legednTxt; 'Index_proxy'];
end
   
if get(handles.indexDistCheckBox,'value')
    Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Index');...
        contains(sortedFeatureTable.Properties.VariableNames, 'dist');...
        contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
    data = sortedFeatureTable{:,Ind};
    plotLine(handles, dt, data, movingAveWinTgl, colors(2,:), '--');
    legednTxt=[legednTxt; 'Index_dist'];
end

%Middle
if get(handles.middleProxyCheckBox,'value')
    Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Middle');...
        contains(sortedFeatureTable.Properties.VariableNames, 'proxy');...
        contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
    data = sortedFeatureTable{:,Ind};
    plotLine(handles, dt, data, movingAveWinTgl, colors(3,:), '-.');
    legednTxt=[legednTxt; 'Middle_proxy'];
end

if get(handles.middleDistCheckBox,'value')
    Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Middle');...
        contains(sortedFeatureTable.Properties.VariableNames, 'dist');...
        contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
    data = sortedFeatureTable{:,Ind};
    plotLine(handles, dt, data, movingAveWinTgl, colors(3,:), '--');
    legednTxt=[legednTxt; 'Middle_dist'];
end

% Ring
if get(handles.ringProxyCheckBox,'value')
    Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Ring');...
        contains(sortedFeatureTable.Properties.VariableNames, 'proxy');...
        contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
    data = sortedFeatureTable{:,Ind};
    plotLine(handles, dt, data, movingAveWinTgl, colors(4,:), '-.');
    legednTxt=[legednTxt; 'Ring_proxy'];
end

if get(handles.ringDistCheckBox,'value')
    Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Ring');...
        contains(sortedFeatureTable.Properties.VariableNames, 'dist');...
        contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
    data = sortedFeatureTable{:,Ind};
    plotLine(handles, dt, data, movingAveWinTgl, colors(4,:), '--');
    legednTxt=[legednTxt; 'Ring_dist'];
end

% Pinky
if get(handles.pinkyProxyCheckBox,'value')
    Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Pinky');...
        contains(sortedFeatureTable.Properties.VariableNames, 'proxy');...
        contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
    data = sortedFeatureTable{:,Ind};
    plotLine(handles, dt, data, movingAveWinTgl, colors(5,:), '-.');
    legednTxt=[legednTxt; 'Pinky_proxy'];
end

if get(handles.pinkyDistCheckBox,'value')
    Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Pinky');...
        contains(sortedFeatureTable.Properties.VariableNames, 'dist');...
        contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
    data = sortedFeatureTable{:,Ind};
    plotLine(handles, dt, data, movingAveWinTgl, colors(5,:), '--');
    legednTxt=[legednTxt; 'Pinky_dist'];
end

% Palm center
if get(handles.palmCenterCheckBox,'value')
    Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Palm_Center');...
        contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
    data = sortedFeatureTable{:,Ind};
    plotLine(handles, dt, data, movingAveWinTgl, colors(6,:), '--');
    legednTxt=[legednTxt; 'Palm_center'];
end

% Palm arch
if get(handles.palmArchCheckBox,'value')
    Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Palm_arch');...
        contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
    data = sortedFeatureTable{:,Ind};
    plotLine(handles, dt, data, movingAveWinTgl, colors(7,:), '--');
    legednTxt=[legednTxt; 'Palm_arch'];
end
axes(handles.secondaryPlot);
hold on
xticks(dt(tixc));
xticklabels(featureToPlot.timeChar(tixc));
xtickangle(45);

if get(handles.yLim2FigTgl, 'Value')
    t_min=[];
    t_max=[];
    for ind=1:size(handles.secondaryPlot.Children,1)
        t_min = min([t_min, handles.secondaryPlot.Children(ind).YData]);
        t_max = max([t_max, handles.secondaryPlot.Children(ind).YData]);
    end
    uYLim = [t_min*1.1, t_max*1.1];
else
    uYLim = [-3, 3];
    
end
ylim(uYLim);
plot([dt(featureToPlot.RecoveryInd,1), dt(featureToPlot.RecoveryInd,1)],...
    uYLim, 'Color', [0.48, 0.48, 0.48]);
legend(legednTxt, 'Interpreter', 'none','location', 'eastoutside');
legend('hide');

hold off

axes(handles.mainPlot);
hold on
xticks([]);%dt(tixc));
xticklabels(featureToPlot.timeChar(tixc));
xtickangle(45);
if get(handles.yLim2FigTgl, 'value')
    t_min=[];
    t_max=[];
    for ind=1:size(handles.mainPlot.Children,1)
        t_min = min([t_min, handles.mainPlot.Children(ind).YData]);
        t_max = max([t_max, handles.mainPlot.Children(ind).YData]);
    end
    uYLim = [t_min*0.9, t_max*1.1];
else
    uYLim = featureToPlot.yLim(featureID,:);
end
ylim(uYLim);

plot([dt(featureToPlot.RecoveryInd,1), dt(featureToPlot.RecoveryInd,1)],...
    uYLim, 'Color', [0.48, 0.48, 0.48]);
legend(legednTxt, 'Interpreter', 'none','location','eastoutside');
hold off

% %% Main Plot
% axes(handles.mainPlot);
% cla(handles.mainPlot);
% hold on
% 
% %Thumb
% if get(handles.thumbProxyCheckBox,'value')
%     Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Thumbs');...
%         contains(sortedFeatureTable.Properties.VariableNames, 'proxy');...
%         contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
%     legednTxt=[legednTxt; 'Thumb_proxy'];
%     plot(dt, sortedFeatureTable{:,Ind},'-.','Color', colors(1,:));
% end
% 
% if get(handles.thumbDistCheckBox,'value')
%     Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Thumbs');...
%         contains(sortedFeatureTable.Properties.VariableNames, 'dist');...
%         contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
%     plot(dt, sortedFeatureTable{:,Ind}, '--','Color', colors(1,:));
%     legednTxt=[legednTxt; 'Thumb_dist'];
% end
% 
% %Index
% if get(handles.indexProxyCheckBox,'value')
%     Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Index');...
%         contains(sortedFeatureTable.Properties.VariableNames, 'proxy');...
%         contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
%     plot(dt, sortedFeatureTable{:,Ind},'-.','Color', colors(2,:));
%     legednTxt=[legednTxt; 'Index_proxy'];
% end
%    
% if get(handles.indexDistCheckBox,'value')
%     Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Index');...
%         contains(sortedFeatureTable.Properties.VariableNames, 'dist');...
%         contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
%     plot(dt, sortedFeatureTable{:,Ind},'--','Color', colors(2,:));
%     legednTxt=[legednTxt; 'Index_dist'];
% end
% 
% %Middle
% if get(handles.middleProxyCheckBox,'value')
%     Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Middle');...
%         contains(sortedFeatureTable.Properties.VariableNames, 'proxy');...
%         contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
%     plot(dt, sortedFeatureTable{:,Ind},'-.','Color', colors(3,:));
%     legednTxt=[legednTxt; 'Middle_proxy'];
% end
% 
% if get(handles.middleDistCheckBox,'value')
%     Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Middle');...
%         contains(sortedFeatureTable.Properties.VariableNames, 'dist');...
%         contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
%     plot(dt, sortedFeatureTable{:,Ind},'--','Color', colors(3,:));
%     legednTxt=[legednTxt; 'Middle_dist'];
% end
% 
% % Ring
% if get(handles.ringProxyCheckBox,'value')
%     Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Ring');...
%         contains(sortedFeatureTable.Properties.VariableNames, 'proxy');...
%         contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
%     plot(dt, sortedFeatureTable{:,Ind},'-.','Color', colors(4,:));
%     legednTxt=[legednTxt; 'Ring_proxy'];
% end
% 
% if get(handles.ringDistCheckBox,'value')
%     Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Ring');...
%         contains(sortedFeatureTable.Properties.VariableNames, 'dist');...
%         contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
%     plot(dt, sortedFeatureTable{:,Ind},'--','Color', colors(4,:));
%     legednTxt=[legednTxt; 'Ring_dist'];
% end
% 
% % Pinky
% if get(handles.pinkyProxyCheckBox,'value')
%     Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Pinky');...
%         contains(sortedFeatureTable.Properties.VariableNames, 'proxy');...
%         contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
%     plot(dt, sortedFeatureTable{:,Ind},'-.','Color', colors(5,:));
%     legednTxt=[legednTxt; 'Pinky_proxy'];
% end
% 
% if get(handles.pinkyDistCheckBox,'value')
%     Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Pinky');...
%         contains(sortedFeatureTable.Properties.VariableNames, 'dist');...
%         contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
%     plot(dt, sortedFeatureTable{:,Ind},'--','Color', colors(5,:));
%     legednTxt=[legednTxt; 'Pinky_dist'];
% end
% 
% % Palm center
% if get(handles.palmCenterCheckBox,'value')
%     Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Palm_Center');...
%         contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
%     plot(dt, sortedFeatureTable{:,Ind},'--','Color', colors(6,:));
%     legednTxt=[legednTxt; 'Palm_center'];
% end
% 
% % Palm arch
% if get(handles.palmArchCheckBox,'value')
%     Ind = find(all([contains(sortedFeatureTable.Properties.VariableNames, 'Palm_arch');...
%         contains(sortedFeatureTable.Properties.VariableNames, featureSelected{:})],1));
%     plot(dt, sortedFeatureTable{:,Ind},'--','Color', colors(7,:));
%     legednTxt=[legednTxt; 'Palm_arch'];
% end
% xticks(dt(tixc));
% xticklabels(featureToPlot.timeChar(tixc));
% xtickangle(45);
% 
% ylim(featureToPlot.yLim(featureID,:));
% plot([dt(featureToPlot.RecoveryInd,1), dt(featureToPlot.RecoveryInd,1)],...
%     featureToPlot.yLim(featureID,:), 'Color', [0.48, 0.48, 0.48]);
% legend(legednTxt);
% hold off

function plotLine(handles, dt, data, movingAveWinTgl, colors, LineStyle)
if movingAveWinTgl
    data = movmean(data,3);
end

axes(handles.secondaryPlot); hold on
plot(dt, data./data(1)-1,LineStyle,'Color', colors(1,:));
hold off

axes(handles.mainPlot); hold on
plot(dt, data,LineStyle,'Color', colors(1,:));
hold off
    
% --- Executes on button press in refreshPlot.
function refreshPlot_Callback(hObject, eventdata, handles)
% hObject    handle to refreshPlot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
plotHelper(handles)


% --- Executes on button press in saveFigBtn.
function saveFigBtn_Callback(hObject, eventdata, handles)
% hObject    handle to saveFigBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global featureToPlot
% save main plot
filePath = get( handles.pathBrowser, 'String');
dashId = strfind(filePath, '\');
savePath = [filePath(1:dashId(end)) featureToPlot.id '.fig'];

Fig1 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8],'Visible','off');
newAxes = copyobj(handles.mainPlot,Fig1); % Copy the appropriate axes
set(newAxes,'Position',get(groot,'DefaultAxesPosition')); % The original position is copied too, so adjust it.
set(Fig1,'CreateFcn','set(gcbf,''Visible'',''on'')'); % Make it visible upon loading
savefig(Fig1,savePath);
delete(Fig1);

% Save secondary plot
savePath = [filePath(1:dashId(end)) featureToPlot.id '_Ratios.fig'];

Fig2 = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8],'Visible','off');
newAxes = copyobj(handles.secondaryPlot,Fig2); % Copy the appropriate axes
set(newAxes,'Position',get(groot,'DefaultAxesPosition')); % The original position is copied too, so adjust it.
set(Fig2,'CreateFcn','set(gcbf,''Visible'',''on'')'); % Make it visible upon loading
savefig(Fig2,savePath);
delete(Fig2);

% --- Executes on button press in allDist.
function allDist_Callback(hObject, eventdata, handles)
% hObject    handle to allDist (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of allDist
if get(handles.allDist,'value')
    
    set(handles.thumbDistCheckBox, 'value',true);
    set(handles.indexDistCheckBox, 'value',true);
    set(handles.middleDistCheckBox, 'value',true);
    set(handles.ringDistCheckBox, 'value',true);
    set(handles.pinkyDistCheckBox, 'value',true);
    
else
    set(handles.thumbDistCheckBox, 'value',false);
    set(handles.indexDistCheckBox, 'value',false);
    set(handles.middleDistCheckBox, 'value',false);
    set(handles.ringDistCheckBox, 'value',false);
    set(handles.pinkyDistCheckBox, 'value',false);
end
plotHelper(handles)

% --- Executes on button press in allProxy.
function allProxy_Callback(hObject, eventdata, handles)
% hObject    handle to allProxy (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of allProxy
if get(handles.allProxy,'value')
    
    set(handles.thumbProxyCheckBox, 'value',true);
    set(handles.indexProxyCheckBox, 'value',true);
    set(handles.middleProxyCheckBox, 'value',true);
    set(handles.ringProxyCheckBox, 'value',true);
    set(handles.pinkyProxyCheckBox, 'value',true);
    
else
    set(handles.thumbProxyCheckBox, 'value',false);
    set(handles.indexProxyCheckBox, 'value',false);
    set(handles.middleProxyCheckBox, 'value',false);
    set(handles.ringProxyCheckBox, 'value',false);
    set(handles.pinkyProxyCheckBox, 'value',false);
end
plotHelper(handles)


% --- Executes on button press in movingAveWinTgl.
function movingAveWinTgl_Callback(hObject, eventdata, handles)
% hObject    handle to movingAveWinTgl (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of movingAveWinTgl


% --- Executes on button press in yLim2FigTgl.
function yLim2FigTgl_Callback(hObject, eventdata, handles)
% hObject    handle to yLim2FigTgl (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of yLim2FigTgl


% --- Executes on selection change in fileList.
function fileList_Callback(hObject, eventdata, handles)
% hObject    handle to fileList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns fileList contents as cell array
%        contents{get(hObject,'Value')} returns selected item from fileList


% --- Executes during object creation, after setting all properties.
function fileList_CreateFcn(hObject, eventdata, handles)
% hObject    handle to fileList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
