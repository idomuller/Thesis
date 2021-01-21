function varargout = ManualROIsExtractionGUI(varargin)
% MANUALROISEXTRACTIONGUI MATLAB code for ManualROIsExtractionGUI.fig
%      MANUALROISEXTRACTIONGUI, by itself, creates a new MANUALROISEXTRACTIONGUI or raises the existing
%      singleton*.
%
%      H = MANUALROISEXTRACTIONGUI returns the handle to a new MANUALROISEXTRACTIONGUI or the handle to
%      the existing singleton*.
%
%      MANUALROISEXTRACTIONGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MANUALROISEXTRACTIONGUI.M with the given input arguments.
%
%      MANUALROISEXTRACTIONGUI('Property','Value',...) creates a new MANUALROISEXTRACTIONGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ManualROIsExtractionGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ManualROIsExtractionGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ManualROIsExtractionGUI

% Last Modified by GUIDE v2.5 30-Sep-2020 23:37:33

% Begin initialization code - DO NOT EDIT
global imageStructOut

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ManualROIsExtractionGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @ManualROIsExtractionGUI_OutputFcn, ...
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


% --- Executes just before ManualROIsExtractionGUI is made visible.
function ManualROIsExtractionGUI_OpeningFcn(hObject, eventdata, handles, imageStructIn, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ManualROIsExtractionGUI (see VARARGIN)

% Choose default command line output for ManualROIsExtractionGUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes ManualROIsExtractionGUI wait for user response (see UIRESUME)
set(handles.radius, 'string','12');
global imageStructOut

imageStructOut = imageStructIn; 
srtToShow = [imageStructIn.subjectID, ' -> ' imageStructIn.imageTime];
set(handles.imagePath,'string',srtToShow);

if ~isfield(imageStructOut,'fingers') ||...
        ~isfield( imageStructOut.fingers,'props')
    imageStructOut.fingers.props.fingerNames = {'Thumbs', 'Index', 'Middle', 'Ring', 'Pinky'};
else
end

if ~isfield(imageStructOut.fingers,'fingersBase')
    imageStructOut.fingers.fingersBase = zeros(5,3);
end
   
if ~isfield(imageStructOut.fingers,'fingersTips') 
    imageStructOut.fingers.fingersTips = zeros(5,3);
end

if ~isfield(imageStructOut,'palm')
    imageStructOut.palm.elipse = zeros(1,2);
end

if ~isfield(imageStructOut.palm,'mainBox')
    imageStructOut.palm.mainBox = zeros(2,1);
end

PlotAllROIs(handles);

imshow(imread('handImage.JPG'), 'Parent', handles.handROImage);
uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = ManualROIsExtractionGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
global imageStructOut

varargout{1} = imageStructOut;%handles.output;


% --- Executes on button press in ThumbsDistBtn.
function ThumbsDistBtn_Callback(hObject, eventdata, handles)
% hObject    handle to ThumbsDistBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% if handles
global imageStructOut

axes(handles.plotImage);hold on;
[Userx,Usery] = ginput(1);
radi = str2double(get(handles.radius,'string'));

fingerROI=[round(Userx), round(Usery), floor(radi)];

currentFingerInd = cellfun(@(x) contains(x, 'Thumb'), imageStructOut.fingers.props.fingerNames);
imageStructOut.fingers.fingersTips(currentFingerInd,:) = fingerROI;
PlotAllROIs(handles)

% --- Executes on button press in IndexDistBtn.
function IndexDistBtn_Callback(hObject, ~, handles)
% hObject    handle to IndexDistBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global imageStructOut

axes(handles.plotImage);hold on;
[Userx,Usery] = ginput(1);
radi = str2double(get(handles.radius,'string'));

fingerROI=[round(Userx), round(Usery), floor(radi)];

currentFingerInd = cellfun(@(x) contains(x, 'Index'), imageStructOut.fingers.props.fingerNames);
imageStructOut.fingers.fingersTips(currentFingerInd,:)= fingerROI;
PlotAllROIs(handles) 


% --- Executes on button press in MiddleDistBtn.
function MiddleDistBtn_Callback(hObject, eventdata, handles)
% hObject    handle to MiddleDistBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global imageStructOut

axes(handles.plotImage);hold on;
[Userx,Usery] = ginput(1);
radi = str2double(get(handles.radius,'string'));

fingerROI=[round(Userx), round(Usery), floor(radi)];

currentFingerInd = cellfun(@(x) contains(x, 'Middle'), imageStructOut.fingers.props.fingerNames);
imageStructOut.fingers.fingersTips(currentFingerInd,:)= fingerROI;
PlotAllROIs(handles) 


% --- Executes on button press in RingDistBtn.
function RingDistBtn_Callback(hObject, eventdata, handles)
% hObject    handle to RingDistBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageStructOut

axes(handles.plotImage);hold on;
[Userx,Usery] = ginput(1);
radi = str2double(get(handles.radius,'string'));

fingerROI=[round(Userx), round(Usery), floor(radi)];

currentFingerInd = cellfun(@(x) contains(x, 'Ring'), imageStructOut.fingers.props.fingerNames);
imageStructOut.fingers.fingersTips(currentFingerInd,:)= fingerROI;
PlotAllROIs(handles) 


% --- Executes on button press in PinkyDistBtn.
function PinkyDistBtn_Callback(hObject, eventdata, handles)
% hObject    handle to PinkyDistBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageStructOut

axes(handles.plotImage);hold on;
[Userx,Usery] = ginput(1);
radi = str2double(get(handles.radius,'string'));

fingerROI=[round(Userx), round(Usery), floor(radi)];

currentFingerInd = cellfun(@(x) contains(x, 'Pinky'), imageStructOut.fingers.props.fingerNames);
imageStructOut.fingers.fingersTips(currentFingerInd,:)= fingerROI;
PlotAllROIs(handles) 


% --- Executes on button press in ThumbsProxBtn.
function ThumbsProxBtn_Callback(hObject, eventdata, handles)
% hObject    handle to ThumbsProxBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageStructOut

axes(handles.plotImage);hold on;
[Userx,Usery] = ginput(1);
radi = str2double(get(handles.radius,'string'));

fingerROI=[round(Userx), round(Usery), floor(radi)];

currentFingerInd = cellfun(@(x) contains(x, 'Thumbs'), imageStructOut.fingers.props.fingerNames);
imageStructOut.fingers.fingersBase(currentFingerInd,:)= fingerROI;
PlotAllROIs(handles) 

% --- Executes on button press in IndexProxBtn.
function IndexProxBtn_Callback(hObject, eventdata, handles)
% hObject    handle to IndexProxBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageStructOut

axes(handles.plotImage);hold on;
[Userx,Usery] = ginput(1);
radi = str2double(get(handles.radius,'string'));

fingerROI=[round(Userx), round(Usery), floor(radi)];

currentFingerInd = cellfun(@(x) contains(x, 'Index'), imageStructOut.fingers.props.fingerNames);
imageStructOut.fingers.fingersBase(currentFingerInd,:)= fingerROI;
PlotAllROIs(handles) 


% --- Executes on button press in MiddleProxBtn.
function MiddleProxBtn_Callback(hObject, eventdata, handles)
% hObject    handle to MiddleProxBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageStructOut

axes(handles.plotImage);hold on;
[Userx,Usery] = ginput(1);
radi = str2double(get(handles.radius,'string'));

fingerROI=[round(Userx), round(Usery), floor(radi)];

currentFingerInd = cellfun(@(x) contains(x, 'Middle'), imageStructOut.fingers.props.fingerNames);
imageStructOut.fingers.fingersBase(currentFingerInd,:)= fingerROI;
PlotAllROIs(handles) 

% --- Executes on button press in RingProxBtn.
function RingProxBtn_Callback(hObject, eventdata, handles)
% hObject    handle to RingProxBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageStructOut

axes(handles.plotImage);hold on;
[Userx,Usery] = ginput(1);
radi = str2double(get(handles.radius,'string'));

fingerROI=[round(Userx), round(Usery), floor(radi)];

currentFingerInd = cellfun(@(x) contains(x, 'Ring'), imageStructOut.fingers.props.fingerNames);
imageStructOut.fingers.fingersBase(currentFingerInd,:)= fingerROI;
PlotAllROIs(handles)


% --- Executes on button press in PinkyProxBtn.
function PinkyProxBtn_Callback(hObject, eventdata, handles)
% hObject    handle to PinkyProxBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageStructOut

axes(handles.plotImage);hold on;
[Userx,Usery] = ginput(1);
radi = str2double(get(handles.radius,'string'));

fingerROI=[round(Userx), round(Usery), floor(radi)];

currentFingerInd = cellfun(@(x) contains(x, 'Pinky'), imageStructOut.fingers.props.fingerNames);
imageStructOut.fingers.fingersBase(currentFingerInd,:)= fingerROI;
PlotAllROIs(handles)

% --- Executes on button press in palm1Btn.
function palm1Btn_Callback(hObject, eventdata, handles)
% hObject    handle to palm1Btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageStructOut
axes(handles.plotImage);hold on;
[Userx,Usery] = ginput(1);

if ~isfield(imageStructOut.palm,'props')
    imageStructOut.palm.props.Centroid(1) = min(imageStructOut.palm.mainBox(1,:))+...
        round((max(imageStructOut.palm.mainBox(1,:))-min(imageStructOut.palm.mainBox(1,:)))/2);
    imageStructOut.palm.props.Centroid(2) = min(imageStructOut.palm.mainBox(2,:))+...
        round((max(imageStructOut.palm.mainBox(2,:))-min(imageStructOut.palm.mainBox(2,:)))/2);

end
xDist = round(Userx - imageStructOut.palm.props.Centroid(1));
yDist = round(Usery - imageStructOut.palm.props.Centroid(2));

imageStructOut.palm.props.Centroid = round([Userx, Usery]);

imageStructOut.palm.mainBox(1,:) = imageStructOut.palm.mainBox(1,:) + xDist;
imageStructOut.palm.mainBox(2,:) = imageStructOut.palm.mainBox(2,:) + yDist;
imageStructOut.palm.elipse(:,1) = imageStructOut.palm.elipse(:,1) + xDist;
imageStructOut.palm.elipse(:,2) = imageStructOut.palm.elipse(:,2) + yDist;

PlotAllROIs(handles)

% --- Executes on button press in SaveAndClose.
function SaveAndClose_Callback(hObject, eventdata, handles)
% hObject    handle to SaveAndClose (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageStructOut

varargout{1} = imageStructOut;
% The figure can be deleted now
delete(handles.figure1);


% --- Executes during object creation, after setting all properties.
function imagePath_CreateFcn(hObject, eventdata, handles)
% hObject    handle to imagePath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called



function radius_Callback(hObject, eventdata, handles)
% hObject    handle to radius (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of radius as text
%        str2double(get(hObject,'String')) returns contents of radius as a double


% --- Executes during object creation, after setting all properties.
function radius_CreateFcn(hObject, eventdata, handles)
% hObject    handle to radius (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function PlotAllROIs(handles)

global imageStructOut
axes(handles.plotImage);
if isfield(imageStructOut, 'registeredImage')
    imagesc(imageStructOut.registeredImage, 'Parent', handles.plotImage);    
else
    imagesc(imageStructOut.fixedImage, 'Parent', handles.plotImage);
end
hold on
for currentFingerInd =1:5
    PlotROIOnFingers(imageStructOut.fingers.fingersBase(currentFingerInd,:));
    
    PlotROIOnFingers(imageStructOut.fingers.fingersTips(currentFingerInd,:));
end

plot(imageStructOut.palm.mainBox(1,:), imageStructOut.palm.mainBox(2,:), 'r-', 'LineWidth', 1);
plot(imageStructOut.palm.elipse(:,1), imageStructOut.palm.elipse(:,2), 'r-', 'LineWidth', 1);

hold off

function PlotROIOnFingers(fingerROI)

ang=0:0.01:2*pi;
xp=fingerROI(3)*cos(ang);
yp=fingerROI(3)*sin(ang);
plot(fingerROI(1)+xp,fingerROI(2)+yp,'r');


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
if isequal(get(hObject, 'waitstatus'), 'waiting')
    % The GUI is still in UIWAIT, us UIRESUME
    uiresume(hObject);
else
    % The GUI is no longer waiting, just close it
    delete(hObject);
end
