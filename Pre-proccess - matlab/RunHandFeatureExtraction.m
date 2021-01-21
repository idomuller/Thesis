function varargout = RunHandFeatureExtraction_first(varargin)
% RunHandFeatureExtraction_first MATLAB code for RunHandFeatureExtraction_first.fig
%      RunHandFeatureExtraction_first, by itself, creates a new RunHandFeatureExtraction_first or raises the existing
%      singleton*.
%
%      H = RunHandFeatureExtraction_first returns the handle to a new RunHandFeatureExtraction_first or the handle to
%      the existing singleton*.
%
%      RunHandFeatureExtraction_first('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RunHandFeatureExtraction_first.M with the given input arguments.
%
%      RunHandFeatureExtraction_first('Property','Value',...) creates a new RunHandFeatureExtraction_first or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before RunHandFeatureExtraction_first_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to RunHandFeatureExtraction_first_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help RunHandFeatureExtraction_first

% Last Modified by GUIDE v2.5 12-May-2020 21:58:00

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @RunHandFeatureExtraction_first_OpeningFcn, ...
                   'gui_OutputFcn',  @RunHandFeatureExtraction_first_OutputFcn, ...
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


% --- Executes just before RunHandFeatureExtraction_first is made visible.
function RunHandFeatureExtraction_first_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to RunHandFeatureExtraction_first (see VARARGIN)

% Choose default command line output for RunHandFeatureExtraction_first
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes RunHandFeatureExtraction_first wait for user response (see UIRESUME)
% % uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = RunHandFeatureExtraction_first_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Get default command line output from handles structure
varargout{1} = handles.output;
upperFigure = findobj(0, 'tag', 'upperFigure');
set(upperFigure, 'Visible','off');
set(handles.upperFigureTitle,'Visible', 'off');

CurrentRight = findobj(0, 'tag', 'CurrentRight');
set(CurrentRight,'Visible', 'off');
set(handles.CurrnetRightTitle,'Visible', 'off');

CurrentMid = findobj(0, 'tag', 'CurrentMid');
set(CurrentMid,'Visible', 'off');
set(handles.CurrentMidTitle,'Visible', 'off');

CurrentLeft = findobj(0, 'tag', 'CurrentLeft');
set(CurrentLeft,'Visible', 'off');
set(handles.CurrentLeftTitle,'Visible', 'off');

ButtomLeftFigure = findobj(0, 'tag', 'ButtomLeftFigure');
set(ButtomLeftFigure, 'Visible', 'off');

set(handles.pastMid,'Visible', 'off');
set(handles.pastRight,'Visible', 'off');

set(handles.autoDoneTXT,'Visible','off');
set(handles.DoneRefineTXT,'Visible','off');

set(handles.UpperFingersFigure, 'Visible', 'off');
set(handles.UpperPalmFigure, 'Visible', 'off');



function baseLinePath_Callback(hObject, eventdata, handles)
% hObject    handle to baseLinePath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of baseLinePath as text
%        str2double(get(hObject,'String')) returns contents of baseLinePath as a double


% --- Executes during object creation, after setting all properties.
function baseLinePath_CreateFcn(hObject, eventdata, handles)
% hObject    handle to baseLinePath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function upperFigure_CreateFcn(hObject, eventdata, handles)
% hObject    handle to baseLinePath (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.


% --- Executes on button press in baseLineBrowserBtn.
function baseLineBrowserBtn_Callback(hObject, eventdata, handles)
% hObject    handle to baseLineBrowserBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[file,path] = uigetfile;
set( handles.baseLinePath, 'String', [path file]);


% --- Executes on button press in runAutoSegmentation.
function runAutoSegmentation_Callback(hObject, eventdata, handles)
% hObject    handle to runAutoSegmentation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess
set(handles.autoDoneTXT,'Visible','off');

set(handles.UpperFingersFigure, 'Visible', 'off');
set(handles.UpperPalmFigure, 'Visible', 'off');

imageProccess.fixedImage  = xlsread(handles.baseLinePath.String);
[imageProccess.fingersBW, imageProccess.palmBW, imageProccess.BW, imageProccess.fixedGrey,...
    imageProccess.changeInAreaUtils] = PalmAndFingerIdentification...
    (imageProccess.fixedImage);

set(handles.firstLinePickSTR,'String',num2str(imageProccess.changeInAreaUtils.pks(1)));
set(handles.secondLineSTR,'String',num2str(imageProccess.changeInAreaUtils.pks(2)));
set(handles.thirdLineSTR,'String',num2str(imageProccess.changeInAreaUtils.pks(3)));


set(handles.ButtomLeftFigure,'Visible', 'on');
imagesc(imageProccess.fixedImage, 'Parent', handles.ButtomLeftFigure);
% Plot changes Vs radius
PlotAreaDiffVsRadius (imageProccess.changeInAreaUtils, handles);

set(handles.CurrentLeftTitle,'Visible', 'on','String', 'Full segmented hand');
set(handles.CurrentLeft,'Visible', 'on');
imshow(imageProccess.BW, 'Parent', handles.CurrentLeft);

set(handles.CurrentMidTitle,'Visible', 'on','String', 'Fingures');
set(handles.CurrentMid,'Visible', 'on');
imshow(imageProccess.fingersBW, 'Parent', handles.CurrentMid);

set(handles.CurrnetRightTitle,'Visible', 'on','String', 'Palm');
set(handles.CurrentRight,'Visible', 'on');
imshow(imageProccess.palmBW, 'Parent', handles.CurrentRight);

set(handles.autoDoneTXT,'Visible','on');


% --- Executes during object creation, after setting all properties.
function upperFigureTitle_CreateFcn(hObject, eventdata, handles)
% hObject    handle to upperFigureTitle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function CurrentLeftTitle_CreateFcn(hObject, eventdata, handles)
% hObject    handle to CurrentLeftTitle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function CurrentMidTitle_CreateFcn(hObject, eventdata, handles)
% hObject    handle to CurrentMidTitle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function CurrnetRightTitle_CreateFcn(hObject, eventdata, handles)
% hObject    handle to CurrnetRightTitle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function CurrentLeft_CreateFcn(hObject, eventdata, handles)
% hObject    handle to CurrentLeft (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate CurrentLeft


% --- Executes during object creation, after setting all properties.
function CurrentMid_CreateFcn(hObject, eventdata, handles)
% hObject    handle to CurrentMid (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate CurrentMid


% --- Executes during object creation, after setting all properties.
function CurrentRight_CreateFcn(hObject, eventdata, handles)
% hObject    handle to CurrentRight (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate CurrentRight


% --- Executes during object creation, after setting all properties.
function ButtomLeftFigure_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ButtomLeftFigure (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate ButtomLeftFigure


% --- Executes during object creation, after setting all properties.
function firstLinePickSTR_CreateFcn(hObject, eventdata, handles)
% hObject    handle to firstLinePickSTR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function secondLineSTR_CreateFcn(hObject, eventdata, handles)
% hObject    handle to secondLineSTR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function thirdLineSTR_CreateFcn(hObject, eventdata, handles)
% hObject    handle to thirdLineSTR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in thirdLineDecrease.
function thirdLineDecrease_Callback(hObject, eventdata, handles)
% hObject    handle to thirdLineDecrease (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
setValue = str2double(get(handles.thirdLineSTR, 'String')) -1;
set(handles.thirdLineSTR, 'String', num2str(setValue));

% --- Executes on button press in thirdLineIncrease.
function thirdLineIncrease_Callback(hObject, eventdata, handles)
% hObject    handle to thirdLineIncrease (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
setValue = str2double(get(handles.thirdLineSTR, 'String')) +1;
set(handles.thirdLineSTR, 'String', num2str(setValue));


% --- Executes on button press in firstLineDecrease.
function firstLineDecrease_Callback(hObject, eventdata, handles)
% hObject    handle to firstLineDecrease (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
setValue = str2double(get(handles.firstLinePickSTR, 'String')) -1;
set(handles.firstLinePickSTR, 'String', num2str(setValue));


% --- Executes on button press in FirstLineIncrease.
function FirstLineIncrease_Callback(hObject, eventdata, handles)
% hObject    handle to FirstLineIncrease (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
setValue = str2double(get(handles.firstLinePickSTR, 'String')) +1;
set(handles.firstLinePickSTR, 'String', num2str(setValue));


% --- Executes on button press in secondLineIncrease.
function secondLineIncrease_Callback(hObject, eventdata, handles)
% hObject    handle to secondLineIncrease (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
setValue = str2double(get(handles.secondLineSTR, 'String')) +1;
set(handles.secondLineSTR, 'String', num2str(setValue));


% --- Executes on button press in secondLineDecrease.
function secondLineDecrease_Callback(hObject, eventdata, handles)
% hObject    handle to secondLineDecrease (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
setValue = str2double(get(handles.secondLineSTR, 'String')) -1;
set(handles.secondLineSTR, 'String', num2str(setValue));

% --- Executes on button press in refineSementation.
function refineSementation_Callback(hObject, eventdata, handles)
% hObject    handle to refineSementation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.DoneRefineTXT,'Visible','off');
global imageProccess
if ~isfield(imageProccess.changeInAreaUtils,'originalPicks')
    imageProccess.changeInAreaUtils.originalPicks = imageProccess.changeInAreaUtils.pks;
    set(handles.pastMid,'Visible', 'on');
    imshow(imageProccess.fingersBW , 'Parent', handles.pastMid);
    set(handles.pastRight,'Visible', 'on');
    imshow(imageProccess.palmBW, 'Parent', handles.pastRight);
end

imageProccess.changeInAreaUtils.pks(1) = str2double(get(handles.firstLinePickSTR,'String'));
imageProccess.changeInAreaUtils.pks(2) = str2double(get(handles.secondLineSTR,'String'));
imageProccess.changeInAreaUtils.pks(3) = str2double(get(handles.thirdLineSTR,'String'));

imageProccess.palmBW = SegmentPalmFromPick(imageProccess.BW, imageProccess.changeInAreaUtils.pks(3));
imageProccess.fingersBW = SegmentFingersFromPicks(imageProccess.BW, imageProccess.changeInAreaUtils.pks(1:2));

imshow(imageProccess.fingersBW, 'Parent', handles.CurrentMid);

imshow(imageProccess.palmBW, 'Parent', handles.CurrentRight);

set(handles.DoneRefineTXT,'Visible','on');

% --- Executes during object creation, after setting all properties.
function pastMid_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pastMid (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate pastMid


% --- Executes during object creation, after setting all properties.
function pastRight_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pastRight (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate pastRight


% --- Executes during object creation, after setting all properties.
function autoDoneTXT_CreateFcn(hObject, eventdata, handles)
% hObject    handle to autoDoneTXT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function DoneRefineTXT_CreateFcn(hObject, eventdata, handles)
% hObject    handle to DoneRefineTXT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in ResetLinesButton.
function ResetLinesButton_Callback(hObject, eventdata, handles)
% hObject    handle to ResetLinesButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global imageProccess
if isfield(imageProccess.changeInAreaUtils,'originalPicks')
    
    set(handles.firstLinePickSTR,'String',num2str(imageProccess.changeInAreaUtils.originalPicks(1)));
    imageProccess.changeInAreaUtils.pks(1) = imageProccess.changeInAreaUtils.originalPicks(1);

    set(handles.secondLineSTR,'String',num2str(imageProccess.changeInAreaUtils.originalPicks(2)));
    imageProccess.changeInAreaUtils.pks(2) = imageProccess.changeInAreaUtils.originalPicks(2);
    
    set(handles.thirdLineSTR,'String',num2str(imageProccess.changeInAreaUtils.originalPicks(3)));
    imageProccess.changeInAreaUtils.pks(3) = imageProccess.changeInAreaUtils.originalPicks(3);

end


% --- Executes on button press in FindFingersROIButton.
function FindFingersROIButton_Callback(hObject, eventdata, handles)
% hObject    handle to FindFingersROIButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% 
global imageProccess

fingersROI = ExtractFingersROI (imageProccess.fingersBW, true);

set(handles.upperFigure, 'Visible','off');
% set(handles.upperFigureTitle,'Visible', 'off');
% 
% set(handles.UpperFingersFigure, 'Visible', 'on');

% --- Executes on button press in FindPalmROIButton.
function FindPalmROIButton_Callback(hObject, eventdata, handles)
% hObject    handle to FindPalmROIButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global imageProccess

[palmCentroid, plamBox]= ExtractPalmFeatures (imageProccess.palmBW, true);
% set(handles.upperFigure, 'Visible','off');
% set(handles.UpperPalmFigure, 'Visible', 'on');

% --- Executes during object creation, after setting all properties.
function UpperFingersFigure_CreateFcn(hObject, eventdata, handles)
% hObject    handle to UpperFingersFigure (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate UpperFingersFigure
