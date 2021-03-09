function varargout = RunExtractHandFeatures(varargin)
% RUNEXTRACTHANDFEATURES MATLAB code for RunExtractHandFeatures.fig
%      RUNEXTRACTHANDFEATURES, by itself, creates a new RUNEXTRACTHANDFEATURES or raises the existing
%      singleton*.
%
%      H = RUNEXTRACTHANDFEATURES returns the handle to a new RUNEXTRACTHANDFEATURES or the handle to
%      the existing singleton*.
%
%      RUNEXTRACTHANDFEATURES('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RUNEXTRACTHANDFEATURES.M with the given input arguments.
%
%      RUNEXTRACTHANDFEATURES('Property','Value',...) creates a new RUNEXTRACTHANDFEATURES or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before RunExtractHandFeatures_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to RunExtractHandFeatures_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help RunExtractHandFeatures

% Last Modified by GUIDE v2.5 01-Mar-2021 12:56:33

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @RunExtractHandFeatures_OpeningFcn, ...
                   'gui_OutputFcn',  @RunExtractHandFeatures_OutputFcn, ...
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


% --- Executes just before RunExtractHandFeatures is made visible.
function RunExtractHandFeatures_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to RunExtractHandFeatures (see VARARGIN)

% Choose default command line output for RunExtractHandFeatures
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes RunExtractHandFeatures wait for user response (see UIRESUME)
% uiwait(handles.figure1);
clearBtn_Callback(hObject, eventdata, handles)

%%Set arrawos images on btns
ararawImage = imread('leftBtn.JPG');
ararawImage=imresize(ararawImage,[35, 35]);
set(handles.blackLeft,'CData', ararawImage);
ararawImage=imrotate(ararawImage,90);
set(handles.blackDown,'CData', ararawImage);
ararawImage=imrotate(ararawImage,90);
set(handles.blackRight,'CData', ararawImage);
ararawImage=imrotate(ararawImage,90);
set(handles.blackUp,'CData', ararawImage);

% --- Outputs from this function are returned to the command line.
function varargout = RunExtractHandFeatures_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



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


% --- Executes on button press in baseLineBrowserBtn.
function baseLineBrowserBtn_Callback(hObject, eventdata, handles)
% hObject    handle to baseLineBrowserBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global imageProccess

[file,path] = uigetfile('C:\Users\ido.DM\Google Drive\Thesis\Data\JPG-Recover\*.jpg','Select an image file');%'C:\Users\ido\Google Drive\Thesis\Data\CSV-Recover\*.csv','Select an image file');
set( handles.baseLinePath, 'String', [path file]);
try
    if contains(file, '.csv') || contains(file, '.xlsx')
        imageProccess.originalImage  = xlsread(handles.baseLinePath.String);
        im(:,:,1) = uint8(255.*(imageProccess.originalImage-min(imageProccess.originalImage(:)))./...
            (max(imageProccess.originalImage(:))-min(imageProccess.originalImage(:))));
        im(:,:,2)=im(:,:,1);
        im(:,:,3)=im(:,:,1);
        set(handles.clearBtn,'CData', ...
            imresize(im, [125, 125]));
        
    elseif contains(file, '.jpg')
        imageProccess.originalImage = imread(handles.baseLinePath.String);
        set(handles.clearBtn,'CData', imresize(imageProccess.originalImage, [125, 125]))
        
    end
    
catch
    error('problem with browsing for file. baseLineBrowserBtn_Callback: Lines 109- 119')
    
end


% --- Executes on button press in LoadImageBtn.
function LoadImageBtn_Callback(hObject, eventdata, handles)
% hObject    handle to LoadImageBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess
imageProccess = struct;
clear global featureTable
clearBtn_Callback(hObject, eventdata, handles)

%% Get original image
if ~isempty(handles.baseLinePath.String) %&&...
    %         (~isfield(imageProccess, 'originalImage') || isempty(imageProccess.originalImage))
    file=handles.baseLinePath.String;
    imageProccess.filePath = file;
    if contains(file, '.csv') || contains(file, '.xlsx')
        imageProccess.originalImage  = xlsread(file);
        imageProccess.fixedImage= imageProccess.originalImage;
        image(:,:,1) = (imageProccess.originalImage-min(imageProccess.originalImage(:)))./...
            (max(imageProccess.originalImage(:))-min(imageProccess.originalImage(:)));
        image(:,:,2) = image(:,:,1);
        image(:,:,3) = image(:,:,1);
        image = uint8(image);
    elseif contains(file, '.jpg')
        imageProccess.originalImage = imread(file);
        imageProccess.fixedImage = rgb2gray(imageProccess.originalImage);
        image = imageProccess.originalImage;
    end
    reImage=imresize(image,[125, 125]);
    set(handles.clearBtn,'CData', reImage);
else
    baseLineBrowserBtn_Callback
end

%% Get hand side
if contains(lower(get(handles.baseLinePath,'String')),'left')
    
    imageProccess.handSide = 'left';
    handSideIn = strfind(lower(file), 'left');
    handSideIn = [handSideIn-1, handSideIn+4];
elseif contains(lower(get(handles.baseLinePath,'String')),'right')
    
    imageProccess.handSide = 'right';
    handSideIn = strfind(lower(file), 'right');
    handSideIn = [handSideIn-1, handSideIn+5];
else
    
end

%% Get file parts
dashInds = strfind(file,'\');
subjectFirstInd = find(dashInds<handSideIn(1),1,'last');
imageProccess.subjectID = file(dashInds(subjectFirstInd)+1:handSideIn(1)-1);
imageProccess.trialPhase = file(handSideIn(2)+1:dashInds(end)-1);
dotInd = strfind(file,'.');
imageProccess.imageTime = file(dashInds(end)+6:dotInd(end)-1);
imageProccess.originalFixGrey = imageProccess.fixedImage;
imageProccess.blackDown =0; 
imageProccess.blackUp =0;
imageProccess.blackLeft=0;
imageProccess.blackRight=0;
imageProccess.rotate = 0;

set(handles.BWTitle,'Visible', 'on','String', 'Full segmented hand');
set(handles.BWAxes,'Visible', 'on');
imagesc(imageProccess.fixedImage, 'Parent', handles.BWAxes);

% --- Executes on button press in runAutoSegmentation.
function runAutoSegmentation_Callback(hObject, eventdata, handles)
% hObject    handle to runAutoSegmentation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess

wb = waitbar(0, 'Segmentation in progress, please wait ...','WindowStyle', 'modal');
wbch = allchild(wb);
jp = wbch(1).JavaPeer;
jp.setIndeterminate(1);

set(handles.autoDoneTXT,'Visible','on','String','Processing...',...
    'ForegroundColor', [0.1, 0.15, 0.88]);

%% Segmentation- palm and fingers identifier
[imageProccess.fingers.BW, imageProccess.palm.BW, imageProccess.BW, imageProccess.fixedGrey,...
    imageProccess.changeInAreaUtils,theta] = PalmAndFingerIdentification...
    (double(imageProccess.fixedImage));

imageProccess.fixedImage = imrotate(imageProccess.fixedImage,theta);

imageProccess.changeInAreaUtils.original = imageProccess.changeInAreaUtils;

%     % Plot changes Vs radius
%     PlotAreaDiffVsRadius (imageProccess.changeInAreaUtils, handles);
%     if get(handles.ShowGradientGraphBtn,'Value')
%         set(handles.ShowGradientGraph,'Visible','on');
%     else
%         set(handles.ShowGradientGraph,'Visible','off');
%     end
%     
set(handles.firstLinePickSTR,'String',num2str(imageProccess.changeInAreaUtils.pks(1)));
set(handles.secondLinePickSTR,'String',num2str(imageProccess.changeInAreaUtils.pks(2)));
set(handles.thirdLinePickSTR,'String',num2str(imageProccess.changeInAreaUtils.pks(3)));

set(handles.BWTitle,'Visible', 'on','String', 'Full segmented hand');
set(handles.BWAxes,'Visible', 'on');
imshow(imageProccess.BW, 'Parent', handles.BWAxes);

set(handles.FingersTitle,'Visible', 'on','String', 'Fingures/ palm seperation');
set(handles.FingersAxes,'Visible', 'on');
image2Show = zeros([size(imageProccess.fingers.BW),3]);
image2Show (:,:,3) = imageProccess.fingers.BW.*255;
% imshow(image2Show, 'Parent', handles.FingersAxes);

% set(handles.PalmTitle,'Visible', 'on','String', 'Palm');
% set(handles.PalmAxes,'Visible', 'on');
% imshow(imageProccess.palm.BW, 'Parent', handles.PalmAxes);
image2Show (:,:,1) = imageProccess.palm.BW.*255;
imageProccess.fingersPalmsSeperated = image2Show;
imshow(image2Show, 'Parent', handles.FingersAxes);

set(handles.FirstLineSlider,'min',min(imageProccess.changeInAreaUtils.radii),...
    'max',max(imageProccess.changeInAreaUtils.radii), 'Value',imageProccess.changeInAreaUtils.pks(1),...
    'SliderStep',[1/(max(imageProccess.changeInAreaUtils.radii)-1), 0.1]);

set(handles.SecondLineSlider,'min',min(imageProccess.changeInAreaUtils.radii),...
    'max',max(imageProccess.changeInAreaUtils.radii), 'Value',imageProccess.changeInAreaUtils.pks(2),...
    'SliderStep',[1/(max(imageProccess.changeInAreaUtils.radii)-1) , 0.1 ]);

set(handles.ThirdLineSlider,'min',min(imageProccess.changeInAreaUtils.radii),...
    'max',max(imageProccess.changeInAreaUtils.radii), 'Value',imageProccess.changeInAreaUtils.pks(3),...
    'SliderStep',[1/(max(imageProccess.changeInAreaUtils.radii)-1) , 0.1 ]);

set(handles.autoDoneTXT,'Visible','on','String','Done',...
    'ForegroundColor', [0.467, 0.675, 0.188]);

close(wb);

% --- Executes on button press in ShowGradientGraphBtn.
% function ShowGradientGraphBtn_Callback(hObject, eventdata, handles)
% % hObject    handle to ShowGradientGraphBtn (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% 
% global imageProccess
% 
% if get(handles.ShowGradientGraphBtn,'Value')
%     try
%         % Plot changes Vs radius
%         axes(handles.ShowGradientGraph);
%         PlotAreaDiffVsRadius (imageProccess.changeInAreaUtils, handles);
%         set(handles.ShowGradientGraph,'Visible','On');
%     catch
%         
%     end
% else
%     axes(handles.ShowGradientGraph);
%     cla reset
%     set(handles.ShowGradientGraph,'Visible','Off');
%     
% end


% Hint: get(hObject,'Value') returns toggle state of ShowGradientGraphBtn


% --- Executes during object creation, after setting all properties.
function ShowGradientGraph_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ShowGradientGraph (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate ShowGradientGraph


% --- Executes during object creation, after setting all properties.
function FingersAxes_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FingersAxes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate FingersAxes


% --- Executes during object creation, after setting all properties.
function PalmAxes_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PalmAxes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate PalmAxes


% --- Executes during object creation, after setting all properties.
function BWAxes_CreateFcn(hObject, eventdata, handles)
% hObject    handle to BWAxes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate BWAxes


% --- Executes during object creation, after setting all properties.
function BWTitle_CreateFcn(hObject, eventdata, handles)
% hObject    handle to BWTitle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function FingersTitle_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FingersTitle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function PalmTitle_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PalmTitle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function firstLinePickSTR_CreateFcn(hObject, eventdata, handles)
% hObject    handle to firstLinePickSTR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in rotateLeftBtn.
function rotateLeftBtn_Callback(hObject, eventdata, handles)
% hObject    handle to rotateLeftBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
theta = 90;
global imageProccess
imageProccess.rotate = imageProccess.rotate+theta;
imageProccess.fixedImage = imrotate(imageProccess.fixedImage,theta);

if isfield(imageProccess,'BW')
    imageProccess.BW = imrotate(imageProccess.BW,theta);
    
    set(handles.BWAxes,'Visible', 'on');
    imshow(imageProccess.BW, 'Parent', handles.BWAxes);

else
    set(handles.BWAxes,'Visible', 'on');
    imshow(imageProccess.fixedImage, 'Parent', handles.BWAxes);

end

if isfield(imageProccess,'BW')
    imageProccess.fingers.BW = imrotate(imageProccess.fingers.BW,theta);
end

if isfield(imageProccess,'fixedGrey')
    imageProccess.fixedGrey = imrotate(imageProccess.fixedGrey,theta);
end
if isfield(imageProccess,'fingersPalmsSeperated')
    imageProccess.fingersPalmsSeperated = imrotate(imageProccess.fingersPalmsSeperated,theta);
    set(handles.FingersAxes,'Visible', 'on');
    imshow(imageProccess.fingersPalmsSeperated, 'Parent', handles.FingersAxes);
end

% if isfield(imageProccess,'palm')
%     imageProccess.palm.BW = imrotate(imageProccess.palm.BW,theta);
%     set(handles.PalmAxes,'Visible', 'on');
%     imshow(imageProccess.palm.BW, 'Parent', handles.PalmAxes);
% 
% end


% --- Executes on button press in rotateRightBtn.
function rotateRightBtn_Callback(hObject, eventdata, handles)
% hObject    handle to rotateRightBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
theta = -90;
global imageProccess
imageProccess.rotate = imageProccess.rotate+theta;

imageProccess.fixedImage = imrotate(imageProccess.fixedImage,theta);

if isfield(imageProccess,'BW')
    imageProccess.BW = imrotate(imageProccess.BW,theta);
    
    set(handles.BWAxes,'Visible', 'on');
    imshow(imageProccess.BW, 'Parent', handles.BWAxes);

else
    set(handles.BWAxes,'Visible', 'on');
    imshow(imageProccess.fixedImage, 'Parent', handles.BWAxes);

end

if isfield(imageProccess,'BW')
    imageProccess.fingers.BW = imrotate(imageProccess.fingers.BW,theta);
end

if isfield(imageProccess,'fixedGrey')
    imageProccess.fixedGrey = imrotate(imageProccess.fixedGrey,theta);
end
if isfield(imageProccess,'fingersPalmsSeperated')
    imageProccess.fingersPalmsSeperated = imrotate(imageProccess.fingersPalmsSeperated,theta);
    set(handles.FingersAxes,'Visible', 'on');
    imshow(imageProccess.fingersPalmsSeperated, 'Parent', handles.FingersAxes);
end
   

% --- Executes during object creation, after setting all properties.
function secondLinePickSTR_CreateFcn(hObject, eventdata, handles)
% hObject    handle to secondLinePickSTR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function thirdLinePickSTR_CreateFcn(hObject, eventdata, handles)
% hObject    handle to thirdLinePickSTR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in ResetSegmentationBtn.
function ResetSegmentationBtn_Callback(hObject, eventdata, handles)
% hObject    handle to ResetSegmentationBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global imageProccess
imageProccess.blackRight=0;
imageProccess.blackLeft=0;
imageProccess.blackDown=0;
imageProccess.blackUp=0;

if isfield(imageProccess,'changeInAreaUtils')
       
    if isfield(imageProccess.changeInAreaUtils,'original')
        
        set(handles.firstLinePickSTR,'String',num2str(imageProccess.changeInAreaUtils.original.pks(1)));
        set(handles.FirstLineSlider, 'Value', imageProccess.changeInAreaUtils.original.pks(1));
        imageProccess.changeInAreaUtils.pks(1) = imageProccess.changeInAreaUtils.original.pks(1);
        
        set(handles.secondLinePickSTR,'String',num2str(imageProccess.changeInAreaUtils.original.pks(2)));
        set(handles.SecondLineSlider, 'Value', imageProccess.changeInAreaUtils.original.pks(2));
        imageProccess.changeInAreaUtils.pks(2) = imageProccess.changeInAreaUtils.original.pks(2);
        
        set(handles.thirdLinePickSTR,'String',num2str(imageProccess.changeInAreaUtils.original.pks(3)));
        set(handles.ThirdLineSlider, 'Value', imageProccess.changeInAreaUtils.original.pks(3));
        imageProccess.changeInAreaUtils.pks(3) = imageProccess.changeInAreaUtils.original.pks(3);
    end
    
    imageProccess.palm.BW = SegmentPalmFromPick(imageProccess.BW, imageProccess.changeInAreaUtils.pks(3));
    
    axes(handles.PalmAxes);
    imshow(imageProccess.palm.BW, 'Parent', handles.PalmAxes);
    
    imageProccess.fingers.BW = SegmentFingersFromPicks(imageProccess.BW, imageProccess.changeInAreaUtils.pks(1:2));
    
    image2Show = zeros([size(imageProccess.fingers.BW),3]);
    image2Show (:,:,3) = imageProccess.fingers.BW.*255;
    image2Show (:,:,1) = imageProccess.palm.BW.*255;
    imageProccess.fingersPalmsSeperated = image2Show;

    axes(handles.FingersAxes);
    imshow(image2Show, 'Parent', handles.FingersAxes);

else
    if size(imageProccess.originalImage,3)==1
        imageProccess.fixedImage= imageProccess.originalImage;
    else
        imageProccess.fixedImage= rgb2gray(imageProccess.originalImage);
    end
    set(handles.BWAxes,'Visible', 'on');
    imshow(imageProccess.fixedImage, 'Parent', handles.BWAxes);
    
end


% --- Executes during object creation, after setting all properties.
function handROImage_CreateFcn(hObject, eventdata, handles)
% hObject    handle to handROImage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate handROImage


% --- Executes on button press in ThumbsDistBtn.
function ThumbsDistBtn_Callback(hObject, eventdata, handles)
% hObject    handle to ThumbsDistBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% if handles
global imageProccess
FIndFingertsROIinGUI('ThumbsDist');
PlotFingersROIinGUI (handles.PalmAxes, imageProccess.fixedGrey,  imageProccess.fingers.fingersTips, imageProccess.fingers.fingersBase);

% --- Executes on button press in IndexDistBtn.
function IndexDistBtn_Callback(hObject, eventdata, handles)
% hObject    handle to IndexDistBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess
FIndFingertsROIinGUI('IndexDist');
PlotFingersROIinGUI (handles.PalmAxes, imageProccess.fixedGrey,  imageProccess.fingers.fingersTips, imageProccess.fingers.fingersBase);

% --- Executes on button press in MiddleDistBtn.
function MiddleDistBtn_Callback(hObject, eventdata, handles)
% hObject    handle to MiddleDistBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess
FIndFingertsROIinGUI('MiddleDist');
PlotFingersROIinGUI (handles.PalmAxes, imageProccess.fixedGrey,  imageProccess.fingers.fingersTips, imageProccess.fingers.fingersBase);

% --- Executes on button press in RingDistBtn.
function RingDistBtn_Callback(hObject, eventdata, handles)
% hObject    handle to RingDistBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess
FIndFingertsROIinGUI('RingDist');
PlotFingersROIinGUI (handles.PalmAxes, imageProccess.fixedGrey,  imageProccess.fingers.fingersTips, imageProccess.fingers.fingersBase);

% --- Executes on button press in PinkyDistBtn.
function PinkyDistBtn_Callback(hObject, eventdata, handles)
% hObject    handle to PinkyDistBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess
FIndFingertsROIinGUI('PinkyDist');
PlotFingersROIinGUI (handles.PalmAxes, imageProccess.fixedGrey,  imageProccess.fingers.fingersTips, imageProccess.fingers.fingersBase);

% --- Executes on button press in ThumbsProxBtn.
function ThumbsProxBtn_Callback(hObject, eventdata, handles)
% hObject    handle to ThumbsProxBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess
FIndFingertsROIinGUI('ThumbsProx');
PlotFingersROIinGUI (handles.PalmAxes, imageProccess.fixedGrey,  imageProccess.fingers.fingersTips, imageProccess.fingers.fingersBase);

% --- Executes on button press in IndexProxBtn.
function IndexProxBtn_Callback(hObject, eventdata, handles)
% hObject    handle to IndexProxBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess
FIndFingertsROIinGUI('IndexProx');
PlotFingersROIinGUI (handles.PalmAxes, imageProccess.fixedGrey,  imageProccess.fingers.fingersTips, imageProccess.fingers.fingersBase);


% --- Executes on button press in MiddleProxBtn.
function MiddleProxBtn_Callback(hObject, eventdata, handles)
% hObject    handle to MiddleProxBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess
FIndFingertsROIinGUI('MiddleProx');
PlotFingersROIinGUI (handles.PalmAxes, imageProccess.fixedGrey,  imageProccess.fingers.fingersTips, imageProccess.fingers.fingersBase);

% --- Executes on button press in RingProxBtn.
function RingProxBtn_Callback(hObject, eventdata, handles)
% hObject    handle to RingProxBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess
FIndFingertsROIinGUI('RingProx');
PlotFingersROIinGUI (handles.PalmAxes, imageProccess.fixedGrey,  imageProccess.fingers.fingersTips, imageProccess.fingers.fingersBase);

% --- Executes on button press in PinkyProxBtn.
function PinkyProxBtn_Callback(hObject, eventdata, handles)
% hObject    handle to PinkyProxBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess
FIndFingertsROIinGUI('PinkyProx');
PlotFingersROIinGUI (handles.PalmAxes, imageProccess.fixedGrey,  imageProccess.fingers.fingersTips, imageProccess.fingers.fingersBase);

% --- Executes on button press in palm1Btn.
function palm1Btn_Callback(hObject, eventdata, handles)
% hObject    handle to palm1Btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
FindPalmROIinGUI;
global imageProccess

axes(handles.PalmAxes);
% hold on
% imshow(imageProccess.palm.BW, 'Parent', handles.PalmAxes);
hold on
plot(imageProccess.palm.mainBox(1,:), imageProccess.palm.ROI.mainBox(2,:), 'r-', 'LineWidth', 2);
plot(imageProccess.palm.elipse(:,1), imageProccess.palm.ROI.elipse(:,2), 'r-', 'LineWidth', 2);
hold off

% --- Executes on button press in palm2Btn.
function palm2Btn_Callback(hObject, eventdata, handles)
% hObject    handle to palm2Btn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in FindFingersROISBtn.
function FindFingersROISBtn_Callback(hObject, eventdata, handles)
% hObject    handle to FindFingersROISBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess


% --- Executes on button press in FindPalmRoiBtn.
function FindPalmRoiBtn_Callback(hObject, eventdata, handles)
% hObject    handle to FindPalmRoiBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess

[palmProps, palmElipse, palmBox]= ExtractPalmFeatures (imageProccess.palm.BW, false);
imageProccess.palm.props = palmProps;
imageProccess.palm.elipse = palmElipse;
imageProccess.palm.mainBox = palmBox;

[imageProccess.fingers.fingersTips,...
    imageProccess.fingers.fingersBase,...
    imageProccess.fingers.props] = ExtractFingersROI (imageProccess.fingers.BW,...
    imageProccess.handSide, imageProccess.palm.props.Centroid);

PlotFingersROIinGUI (handles.PalmAxes, imageProccess.fixedGrey,  imageProccess.fingers.fingersTips, imageProccess.fingers.fingersBase);
% axes(handles.PalmAxes); % Make averSpec the current axes.
% imshow(imageProccess.palm.BW, 'Parent', handles.PalmAxes);
hold(handles.PalmAxes, 'on');
set(gca,'xtick',[]);
set(gca,'visible','off');
plot(palmBox(1,:), palmBox(2,:), 'r-', 'LineWidth', 1, 'Parent', handles.PalmAxes);
plot(palmElipse(:,1), palmElipse(:,2), 'r-', 'LineWidth', 1);
hold(handles.PalmAxes, 'off');


% --- Executes on button press in clearBtn.
function clearBtn_Callback(hObject, eventdata, handles)
% hObject    handle to clearBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.autoDoneTXT,'Visible','off');

global imageProccess
imageProccess=struct;
axes(handles.PalmAxes); cla reset
set(handles.PalmAxes , 'Visible', 'Off');

axes(handles.FingersAxes); cla reset
set(handles.FingersAxes , 'Visible', 'Off');

axes(handles.BWAxes); cla reset
set(handles.BWAxes , 'Visible', 'Off');

axes(handles.ShowGradientGraph); cla reset
set(handles.ShowGradientGraph,'Visible','Off');

set(handles.autoDoneTXT,'Visible','Off');
% axes(handles.originalImage); % Make averSpec the current axes.
% cla reset;
% set(handles.originalImage,'Visible', 'Off');
image = imread('originalImage.JPG');
image=imresize(image,[125, 125]);
set(handles.clearBtn,'CData', image);
imshow(imread('handImage.JPG'), 'Parent', handles.handROImage);


% --- Executes during object creation, after setting all properties.
function originalImage_CreateFcn(hObject, eventdata, handles)
% hObject    handle to originalImage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate originalImage


% --- Executes on slider movement.
function FirstLineSlider_Callback(hObject, eventdata, handles)
% hObject    handle to FirstLineSlider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
global imageProccess
if ~isfield(imageProccess.changeInAreaUtils,'original')
    return
end

setValue = get(handles.FirstLineSlider, 'Value');
set(handles.firstLinePickSTR, 'String', num2str(setValue));

imageProccess.changeInAreaUtils.pks(1) = round(str2double(get(handles.firstLinePickSTR,'String')));

imageProccess.fingers.BW = SegmentFingersFromPicks(imageProccess.BW, imageProccess.changeInAreaUtils.pks(1:2));

imageProccess.fingersPalmsSeperated(:,:,3) = imageProccess.fingers.BW.*255;
imshow(imageProccess.fingersPalmsSeperated, 'Parent', handles.FingersAxes);
% 
% axes(handles.FingersAxes);
% imshow(imageProccess.fingers.BW, 'Parent', handles.FingersAxes);




% --- Executes during object creation, after setting all properties.
function FirstLineSlider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FirstLineSlider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



% --- Executes on slider movement.
function SecondLineSlider_Callback(hObject, eventdata, handles)
% hObject    handle to SecondLineSlider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
global imageProccess
if ~isfield(imageProccess.changeInAreaUtils,'original')
    return
end

setValue = get(handles.SecondLineSlider, 'Value');
set(handles.secondLinePickSTR, 'String', num2str(setValue));

imageProccess.changeInAreaUtils.pks(2) = round(str2double(get(handles.secondLinePickSTR,'String')));

imageProccess.fingers.BW = SegmentFingersFromPicks(imageProccess.BW, imageProccess.changeInAreaUtils.pks(1:2));

axes(handles.FingersAxes);
imageProccess.fingersPalmsSeperated(:,:,3) = imageProccess.fingers.BW.*255;
imshow(imageProccess.fingersPalmsSeperated, 'Parent', handles.FingersAxes);


% --- Executes during object creation, after setting all properties.
function SecondLineSlider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SecondLineSlider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function ThirdLineSlider_Callback(hObject, eventdata, handles)
% hObject    handle to ThirdLineSlider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
global imageProccess
if ~isfield(imageProccess.changeInAreaUtils,'original')
    return
end

setValue = get(handles.ThirdLineSlider, 'Value');
set(handles.thirdLinePickSTR, 'String', num2str(setValue));

imageProccess.changeInAreaUtils.pks(3) = round(str2double(get(handles.thirdLinePickSTR,'String')));

imageProccess.palm.BW = SegmentPalmFromPick(imageProccess.BW, imageProccess.changeInAreaUtils.pks(3));

axes(handles.FingersAxes);
imageProccess.fingersPalmsSeperated(:,:,1) = imageProccess.palm.BW.*255;
imshow(imageProccess.fingersPalmsSeperated, 'Parent', handles.FingersAxes);

% imshow(imageProccess.palm.BW, 'Parent', handles.PalmAxes);

% --- Executes during object creation, after setting all properties.
function ThirdLineSlider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ThirdLineSlider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in SaveFeatures.
function SaveFeatures_Callback(hObject, eventdata, handles)
% hObject    handle to SaveFeatures (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global imageProccess
global featureTable
featureTable = SaveFeatures(imageProccess.fixedImage, imageProccess, featureTable);


% --- Executes on button press in blackLeft.
function blackLeft_Callback(hObject, eventdata, handles)
% hObject    handle to blackLeft (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess

imageProccess.blackLeft = imageProccess.blackLeft+10;
imageProccess.fixedImage(:,1:imageProccess.blackLeft,:)=0;

set(handles.BWAxes,'Visible', 'on');
imagesc(imageProccess.fixedImage, 'Parent', handles.BWAxes);

% --- Executes on button press in blackDown.
function blackDown_Callback(hObject, eventdata, handles)
% hObject    handle to blackDown (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess

imageProccess.blackDown = imageProccess.blackDown+10;
imageProccess.fixedImage(end-imageProccess.blackDown:end,:,:)=0;

set(handles.BWAxes,'Visible', 'on');
imagesc(imageProccess.fixedImage, 'Parent', handles.BWAxes);



% --- Executes on button press in blackUp.
function blackUp_Callback(hObject, eventdata, handles)
% hObject    handle to blackUp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess

imageProccess.blackUp = imageProccess.blackUp+10;
imageProccess.fixedImage(1:imageProccess.blackUp,:,:)=0;

set(handles.BWAxes,'Visible', 'on');
imagesc(imageProccess.fixedImage, 'Parent', handles.BWAxes);

% --- Executes on button press in blackRight.
function blackRight_Callback(hObject, eventdata, handles)
% hObject    handle to blackRight (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess

imageProccess.blackRight = imageProccess.blackRight+10;
imageProccess.fixedImage(:,end-imageProccess.blackRight:end,:)=0;

set(handles.BWAxes,'Visible', 'on');
imagesc(imageProccess.fixedImage, 'Parent', handles.BWAxes);


% --- Executes on button press in runRegistrationBtn.
function runRegistrationBtn_Callback(hObject, eventdata, handles)
% hObject    handle to runRegistrationBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
RunRegistration


% --- Executes on button press in manualROIsPickBtn.
function manualROIsPickBtn_Callback(hObject, eventdata, handles)
% hObject    handle to manualROIsPickBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess
imageProccess = ManualROIsExtractionGUI(imageProccess);

PlotFingersROIinGUI (handles.PalmAxes, imageProccess.fixedGrey,  imageProccess.fingers.fingersTips, imageProccess.fingers.fingersBase);

hold(handles.PalmAxes, 'on');
set(gca,'xtick',[]);
set(gca,'visible','off');
plot(imageProccess.palm.mainBox(1,:), imageProccess.palm.mainBox(2,:), 'r-', 'LineWidth', 1, 'Parent', handles.PalmAxes);
plot(imageProccess.palm.elipse(:,1), imageProccess.palm.elipse(:,2), 'r-', 'LineWidth', 1);
hold(handles.PalmAxes, 'off');
