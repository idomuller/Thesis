function varargout = RunRegistration(varargin)
% RUNREGISTRATION MATLAB code for RunRegistration.fig
%      RUNREGISTRATION, by itself, creates a new RUNREGISTRATION or raises the existing
%      singleton*.
%
%      H = RUNREGISTRATION returns the handle to a new RUNREGISTRATION or the handle to
%      the existing singleton*.
%
%      RUNREGISTRATION('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RUNREGISTRATION.M with the given input arguments.
%
%      RUNREGISTRATION('Property','Value',...) creates a new RUNREGISTRATION or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before RunRegistration_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to RunRegistration_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help RunRegistration

% Last Modified by GUIDE v2.5 06-Oct-2020 09:08:45

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @RunRegistration_OpeningFcn, ...
                   'gui_OutputFcn',  @RunRegistration_OutputFcn, ...
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


% --- Executes just before RunRegistration is made visible.
function RunRegistration_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to RunRegistration (see VARARGIN)

% Choose default command line output for RunRegistration
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes RunRegistration wait for user response (see UIRESUME)
% uiwait(handles.figure1);
%%Set arrawos images on btns
ararawImage = imread('leftBtn.JPG');
ararawImage=imresize(ararawImage,[35, 35]);
set(handles.moveLeftBtn,'CData', ararawImage);
ararawImage=imrotate(ararawImage,90);
set(handles.moveDownBtn,'CData', ararawImage);
ararawImage=imrotate(ararawImage,90);
set(handles.moveRightBtn,'CData', ararawImage);
ararawImage=imrotate(ararawImage,90);
set(handles.moveUpBtn,'CData', ararawImage);

global imageProccess
global movingImage
movingImage = struct;
movingImage.rotate =0;

%% Load next image
filePath = imageProccess.filePath;
if contains(lower(filePath),'.csv')
    fileType = '.csv';
    dashInds = strfind(lower(filePath),'\');
    path = filePath(1:dashInds(end));
elseif contains(lower(filePath),'.jpg') 
    fileType = '.jpg';
    dashInds = strfind(lower(filePath),'\');
    path = filePath(1:dashInds(end));
end
imageProccess.path = path;
set(handles.workingFolderSTR,'String',path);

allFiles = dir(path);
allFiles = allFiles(cellfun(@(x) ~x, {allFiles.isdir})); 
%Remove the base image from list off all images
allFiles = allFiles(cellfun(@(x) ~contains(x, imageProccess.imageTime), {allFiles.name})); 
relevantFiles = struct2cell(allFiles(cellfun(@(x) contains(x, fileType), {allFiles.name})))'; 
movingImage.relevantFiles = sort(relevantFiles(:,1));

movingImage.imageInd = 1;

printNumIndSTR(handles, movingImage.relevantFiles, movingImage.imageInd);

currnetImagePath = [path, cell2mat(movingImage.relevantFiles(movingImage.imageInd,:))];
if contains(fileType, '.csv') 
    movingImage.originalImage  = xlsread(currnetImagePath);
    movingImage.fixedImage= movingImage.originalImage;
elseif contains(fileType, '.jpg')
    movingImage.originalImage = imread(currnetImagePath);
    movingImage.fixedImage = double(rgb2gray(movingImage.originalImage));
end

%% Get file parts
if contains(lower(currnetImagePath),'left')
    
    handSideIn = strfind(lower(currnetImagePath), 'left');
    handSideIn = [handSideIn-1, handSideIn+4];
elseif contains(lower(currnetImagePath),'right')
    
    handSideIn = strfind(lower(currnetImagePath), 'right');
    handSideIn = [handSideIn-1, handSideIn+5];
else
    
end

subjectFirstInd = find(dashInds<handSideIn(1),1,'last');
movingImage.subjectID = currnetImagePath(dashInds(subjectFirstInd)+1:handSideIn(1)-1);
if ~strcmp(imageProccess.subjectID, movingImage.subjectID)
    error ('mismatch in subject ID');
end
movingImage.trialPhase = currnetImagePath(handSideIn(2)+1:dashInds(end)-1);
dotInd = strfind(currnetImagePath,'.');
movingImage.imageTime = currnetImagePath(dashInds(end)+6:dotInd(end)-1);

movingImage.fileType=fileType;
movingImage.size = size(movingImage.fixedImage);

movingImage.palm.mainBox = imageProccess.palm.mainBox; 
movingImage.palm.elipse = imageProccess.palm.elipse;
movingImage.fingers.fingersBase = imageProccess.fingers.fingersBase;
movingImage.fingers.fingersTips = imageProccess.fingers.fingersTips;
movingImage.fingers.props.fingerNames = imageProccess.fingers.props.fingerNames;
movingImage.palm.props.Centroid(1) = min(movingImage.palm.mainBox(1,:))+...
    round((max(movingImage.palm.mainBox(1,:))-min(movingImage.palm.mainBox(1,:)))/2);
movingImage.palm.props.Centroid(2) = min(movingImage.palm.mainBox(2,:))+...
    round((max(movingImage.palm.mainBox(2,:))-min(movingImage.palm.mainBox(2,:)))/2);

movingImage.AllROIS = {'DateAndTime','mainBox','elipse','fingerBase','fingerTips','fingerNames'};
movingImage.AllROIS = [movingImage.AllROIS; {imageProccess.imageTime}, {imageProccess.palm.mainBox},...
    {imageProccess.palm.elipse}, {imageProccess.fingers.fingersBase}, {imageProccess.fingers.fingersTips},...
    {imageProccess.fingers.props.fingerNames}];

% Moving image segmentation
[movingImage.BW, movingImage.fixedGrey] = ImageSegmentation(movingImage.fixedImage, imageProccess.rotate);

% fixGreyOrig = imrotate(imageProccess.originalFixGrey, imageProccess.rotate);
movingImage.registeredImage = movingImage.fixedImage;
plotRegistration (handles, imageProccess.fixedImage, movingImage.fixedImage, movingImage.registeredImage,...
    movingImage.palm.mainBox, movingImage.palm.elipse, movingImage.fingers.fingersBase, movingImage.fingers.fingersTips);

cla(handles.aligmentFig)
set(handles.alignmentDiffSTR,'Visible','off');

axes(handles.baseImageFig);
PlotFingersROIinGUI (handles.baseImageFig, imageProccess.fixedImage, imageProccess.fingers.fingersTips, imageProccess.fingers.fingersBase);
hold(handles.baseImageFig, 'on');
set(gca,'xtick',[]);
set(gca,'visible','off');
plot(imageProccess.palm.mainBox(1,:), imageProccess.palm.mainBox(2,:), 'r-', 'LineWidth', 1, 'Parent', handles.baseImageFig);
plot(imageProccess.palm.elipse(:,1), imageProccess.palm.elipse(:,2), 'r-', 'LineWidth', 1);
hold(handles.baseImageFig, 'off');


% --- Outputs from this function are returned to the command line.
function varargout = RunRegistration_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in moveLeftBtn.
function moveLeftBtn_Callback(hObject, eventdata, handles)
% hObject    handle to moveLeftBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global movingImage
global imageProccess

padS = 3;

movingImage.palm.mainBox(1,:) = movingImage.palm.mainBox(1,:)-padS; 
movingImage.palm.elipse(:,1) = movingImage.palm.elipse(:,1)-padS;
movingImage.fingers.fingersBase(:,1) = movingImage.fingers.fingersBase(:,1)-padS;
movingImage.fingers.fingersTips(:,1) = movingImage.fingers.fingersTips(:,1)-padS;
movingImage.palm.props.Centroid(1) = movingImage.palm.props.Centroid(1)-padS;

% padImag = movingImage.fixedGrey(:,padS:end);
% padImag=[padImag, zeros(size(movingImage.fixedGrey,1), padS-1)];
% 
% movingImage.fixedGrey = padImag;
% 
% axes(handles.movingImageFig);
% imagesc(movingImage.fixedGrey, 'Parent', handles.movingImageFig);
%    
% padMovingImage = movingImage.registeredImage(:,padS:end);
% padMovingImage = [padMovingImage, zeros(size(movingImage.registeredImage,1), padS-1)];
% movingImage.registeredImage = padMovingImage;

plotRegistration (handles, [], movingImage.fixedImage, movingImage.registeredImage,...
    movingImage.palm.mainBox, movingImage.palm.elipse, movingImage.fingers.fingersBase, movingImage.fingers.fingersTips);


% --- Executes on button press in moveDownBtn.
function moveDownBtn_Callback(hObject, eventdata, handles)
% hObject    handle to moveDownBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global movingImage
global imageProccess

padS = 3;

movingImage.palm.mainBox(2,:) = movingImage.palm.mainBox(2,:)+padS; 
movingImage.palm.elipse(:,2) = movingImage.palm.elipse(:,2)+padS;
movingImage.fingers.fingersBase(:,2) = movingImage.fingers.fingersBase(:,2)+padS;
movingImage.fingers.fingersTips(:,2) = movingImage.fingers.fingersTips(:,2)+padS;
movingImage.palm.props.Centroid(2) = movingImage.palm.props.Centroid(2)+padS;

% padS = 3;
% padImag = movingImage.fixedGrey(1:end-padS,:);
% padImag=[zeros(padS-1, size(movingImage.fixedGrey,2)); padImag];
% 
% movingImage.fixedGrey = padImag;
% 
% axes(handles.movingImageFig);
% imagesc(movingImage.fixedGrey, 'Parent', handles.movingImageFig);
% 
% padMovingImage = movingImage.registeredImage(1:end-padS,:);
% padMovingImage = [zeros(padS-1, size(movingImage.registeredImage,2)); padMovingImage];
% movingImage.registeredImage = padMovingImage;

plotRegistration (handles, [], movingImage.fixedImage, movingImage.registeredImage,...
    movingImage.palm.mainBox, movingImage.palm.elipse, movingImage.fingers.fingersBase, movingImage.fingers.fingersTips);


% --- Executes on button press in moveUpBtn.
function moveUpBtn_Callback(hObject, eventdata, handles)
% hObject    handle to moveUpBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global movingImage
global imageProccess

padS = 3;

movingImage.palm.mainBox(2,:) = movingImage.palm.mainBox(2,:)-padS; 
movingImage.palm.elipse(:,2) = movingImage.palm.elipse(:,2)-padS;
movingImage.fingers.fingersBase(:,2) = movingImage.fingers.fingersBase(:,2)-padS;
movingImage.fingers.fingersTips(:,2) = movingImage.fingers.fingersTips(:,2)-padS;
movingImage.palm.props.Centroid(2) = movingImage.palm.props.Centroid(2)-padS;
% 
% padS = 3;
% padImag = movingImage.fixedGrey(padS:end,:);
% padImag=[padImag; zeros(padS-1, size(movingImage.fixedGrey,2))];
% 
% movingImage.fixedGrey = padImag;
% 
% axes(handles.movingImageFig);
% imagesc(movingImage.fixedGrey, 'Parent', handles.movingImageFig);
% 
% padMovingImage = movingImage.registeredImage(padS:end,:);
% padMovingImage = [padMovingImage; zeros(padS-1, size(movingImage.registeredImage,2))];
% movingImage.registeredImage = padMovingImage;
% 
plotRegistration (handles, [], movingImage.fixedImage, movingImage.registeredImage,...
    movingImage.palm.mainBox, movingImage.palm.elipse, movingImage.fingers.fingersBase, movingImage.fingers.fingersTips);


% --- Executes on button press in moveRightBtn.
function moveRightBtn_Callback(hObject, eventdata, handles)
% hObject    handle to moveRightBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global movingImage
global imageProccess

padS = 3;

movingImage.palm.mainBox(1,:) = movingImage.palm.mainBox(1,:)+padS; 
movingImage.palm.elipse(:,1) = movingImage.palm.elipse(:,1)+padS;
movingImage.fingers.fingersBase(:,1) = movingImage.fingers.fingersBase(:,1)+padS;
movingImage.fingers.fingersTips(:,1) = movingImage.fingers.fingersTips(:,1)+padS;
movingImage.palm.props.Centroid(1) = movingImage.palm.props.Centroid(1)+padS;
% 
% padS = 3;
% padImag = movingImage.fixedGrey(:,1:end-padS);
% padImag=[zeros(size(movingImage.fixedGrey,1), padS-1), padImag];
% 
% movingImage.fixedGrey = padImag;
% 
% axes(handles.movingImageFig);
% imagesc(movingImage.fixedGrey, 'Parent', handles.movingImageFig);
% 
% padMovingImage = movingImage.registeredImage(:,1:end-padS);
% padMovingImage = [zeros(size(movingImage.registeredImage,1), padS-1), padMovingImage];
% movingImage.registeredImage = padMovingImage;

plotRegistration (handles, [], movingImage.fixedImage, movingImage.registeredImage,...
    movingImage.palm.mainBox, movingImage.palm.elipse, movingImage.fingers.fingersBase, movingImage.fingers.fingersTips);


% --- Executes on button press in rotateLeftBtn.
function rotateLeftBtn_Callback(hObject, eventdata, handles)
% hObject    handle to rotateLeftBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
theta = 90;
global movingImage

movingImage.BW = imrotate(movingImage.BW,theta);
movingImage.fixedGrey = imrotate(movingImage.fixedGrey, theta);
movingImage.fixedImage = imrotate(movingImage.fixedImage, theta);
movingImage.rotate = movingImage.rotate + theta;
movingImage.registeredImage = imrotate(movingImage.registeredImage, theta);

plotRegistration (handles, [], movingImage.fixedImage, movingImage.registeredImage,...
    movingImage.palm.mainBox, movingImage.palm.elipse, movingImage.fingers.fingersBase, movingImage.fingers.fingersTips);


% --- Executes on button press in rotateRightBtn.
function rotateRightBtn_Callback(hObject, eventdata, handles)
% hObject    handle to rotateRightBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
theta = -90;
global movingImage

movingImage.BW = imrotate(movingImage.BW,theta);
movingImage.fixedGrey = imrotate(movingImage.fixedGrey, theta);
movingImage.fixedImage = imrotate(movingImage.fixedImage, theta);
movingImage.rotate = movingImage.rotate + theta;
movingImage.registeredImage = imrotate(movingImage.registeredImage, theta);

plotRegistration (handles, [], movingImage.fixedImage, movingImage.registeredImage,...
    movingImage.palm.mainBox, movingImage.palm.elipse, movingImage.fingers.fingersBase, movingImage.fingers.fingersTips);


function printNumIndSTR(handles, relevantFiles, imageInd)

imagesProgressSTR = sprintf('image %d / %d',imageInd, size(relevantFiles,1));
set(handles.imagesProgress,'string', imagesProgressSTR);
set(handles.imagesProgress,'string', imagesProgressSTR);
set(handles.movingImagenameSTR, 'string', relevantFiles(imageInd,:));


% --- Executes on button press in ExtractFeaturesBtn.
function ExtractFeaturesBtn_Callback(hObject, eventdata, handles)
% hObject    handle to ExtractFeaturesBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global movingImage
global featureTable

featureTable = SaveFeatures(movingImage.registeredImage, movingImage, featureTable);

movingImage.AllROIS = [movingImage.AllROIS; {movingImage.imageTime}, {movingImage.palm.mainBox},...
    {movingImage.palm.elipse}, {movingImage.fingers.fingersBase}, {movingImage.fingers.fingersTips},...
    {movingImage.fingers.props.fingerNames}];
nextImageBtn_Callback(hObject, eventdata, handles);

% --- Executes on button press in nextImageBtn.
function nextImageBtn_Callback(hObject, eventdata, handles)
% hObject    handle to nextImageBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global movingImage
global imageProccess

movingImage.imageInd=movingImage.imageInd+1;

if size(movingImage.relevantFiles,1)>=movingImage.imageInd
    printNumIndSTR(handles, movingImage.relevantFiles, movingImage.imageInd);
else 
    set(handles.movingImagenameSTR, 'string', 'Done');
    return
end

filePath = imageProccess.filePath;
if contains(lower(filePath),'.csv')
    fileType = '.csv';
    dashInds = strfind(lower(filePath),'\');
    path = filePath(1:dashInds(end));
elseif contains(lower(filePath),'.jpg')
    fileType = '.jpg';
    dashInds = strfind(lower(filePath),'\');
    path = filePath(1:dashInds(end));
end
set(handles.workingFolderSTR,'String',path);

currnetImagePath = [path cell2mat(movingImage.relevantFiles(movingImage.imageInd,:))];
if contains(movingImage.fileType, '.csv') 
    movingImage.originalImage  = xlsread(currnetImagePath);
    movingImage.fixedImage= movingImage.originalImage;
elseif contains(movingImage.fileType, '.jpg')
    movingImage.originalImage = imread(currnetImagePath);
    movingImage.fixedImage = double(rgb2gray(movingImage.originalImage));
end

%% Get file parts
if contains(lower(currnetImagePath),'left')
    
    handSideIn = strfind(lower(currnetImagePath), 'left');
    handSideIn = [handSideIn-1, handSideIn+4];
elseif contains(lower(currnetImagePath),'right')
    
    handSideIn = strfind(lower(currnetImagePath), 'right');
    handSideIn = [handSideIn-1, handSideIn+5];
else
    
end

subjectFirstInd = find(dashInds<handSideIn(1),1,'last');
movingImage.subjectID = currnetImagePath(dashInds(subjectFirstInd)+1:handSideIn(1)-1);
if ~strcmp(imageProccess.subjectID, movingImage.subjectID)
    error ('mismatch in subject ID');
end
movingImage.trialPhase = currnetImagePath(handSideIn(2)+1:dashInds(end)-1);
dotInd = strfind(currnetImagePath,'.');
movingImage.imageTime = currnetImagePath(dashInds(end)+6:dotInd(end)-1);

movingImage.fileType=fileType;
movingImage.size = size(movingImage.fixedImage);

% Moving image segmentation
[movingImage.BW, movingImage.fixedGrey] = ImageSegmentation(movingImage.fixedImage, movingImage.rotate);

movingImage.fixedImage = imrotate(movingImage.fixedImage, movingImage.rotate);
movingImage.registeredImage = movingImage.fixedImage;
plotRegistration (handles, [], movingImage.fixedImage, movingImage.registeredImage,...
    movingImage.palm.mainBox, movingImage.palm.elipse, movingImage.fingers.fingersBase, movingImage.fingers.fingersTips);

cla(handles.aligmentFig)
set(handles.alignmentDiffSTR,'Visible','off');

axes(handles.baseImageFig);
PlotFingersROIinGUI (handles.baseImageFig, imageProccess.fixedImage, imageProccess.fingers.fingersTips, imageProccess.fingers.fingersBase);
hold(handles.baseImageFig, 'on');
set(gca,'xtick',[]);
set(gca,'visible','off');
plot(imageProccess.palm.mainBox(1,:), imageProccess.palm.mainBox(2,:), 'r-', 'LineWidth', 1, 'Parent', handles.baseImageFig);
plot(imageProccess.palm.elipse(:,1), imageProccess.palm.elipse(:,2), 'r-', 'LineWidth', 1);
hold(handles.baseImageFig, 'off');


function plotRegistration(handles, fixedGrey, movingGrey, registeredImage,...
    mainBoxIndices, elipseIndices, fingersBaseROI, fingersTipsROI)

set(handles.aligmentFig, 'visible','on');
set(handles.movingImageROIs,'visible','on');
set(handles.alignmentDiffSTR,'visible','on');
set(handles.movingImageROISTR,'visible','on');

% Original/base image- in grey scale 
if ~isempty(fixedGrey)
    axes(handles.baseImageFig);
    imagesc(fixedGrey, 'Parent', handles.baseImageFig);
    set(handles.baseImageFig, 'visible', 'off');
end

if ~isempty(movingGrey)
    % Moving image - in grey scale
    axes(handles.movingImageFig);
    imagesc(movingGrey, 'Parent', handles.movingImageFig);
    set(handles.movingImageFig, 'visible', 'off');
end

if ~isempty(fixedGrey)
% Show aligment
axes(handles.aligmentFig);
imshowpair(uint8(fixedGrey), uint8(registeredImage), 'method', 'falsecolor', 'Parent', handles.aligmentFig);
end

if ~isempty(registeredImage)
    % Show resitered image with ROI on it
%     maskedImage = movingBW.*registeredImage;%fixedBW.*
%     maskedImage(maskedImage>0)=1;
%     maskedImage = movingGrey;
    PlotFingersROIinGUI (handles.movingImageROIs, registeredImage, fingersTipsROI, fingersBaseROI)
    
%     PlotFingersROIinGUI (handles.movingImageROIs, maskedImage, fingersTipsROI, fingersBaseROI)
end
axes(handles.movingImageROIs);
hold on
plot(mainBoxIndices(1,:), mainBoxIndices(2,:), 'r-', 'LineWidth', 1);% Now, for fun, plot it over the binary image.
plot(elipseIndices(:,1), elipseIndices(:,2), 'r-', 'LineWidth', 1);


% --- Executes on button press in runSegmentationBtn.
function runSegmentationBtn_Callback(hObject, eventdata, handles)
% hObject    handle to runSegmentationBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global imageProccess
global movingImage

cla(handles.aligmentFig)
cla(handles.movingImageROIs)
set(handles.alignmentDiffSTR,'Visible','off');
set(handles.movingImageROISTR,'Visible','off');
pause(0.1);
fixGreyOrig = imrotate(imageProccess.originalFixGrey, imageProccess.rotate);

movingImage.registeredImage = double(RegisterImage(fixGreyOrig, movingImage.fixedImage));

plotRegistration (handles, [], movingImage.fixedImage, movingImage.registeredImage,...
    movingImage.palm.mainBox, movingImage.palm.elipse, movingImage.fingers.fingersBase, movingImage.fingers.fingersTips);

movingImage.fingers.fingersTips = imageProccess.fingers.fingersTips;
movingImage.fingers.fingersBase = imageProccess.fingers.fingersBase;
movingImage.fingers.props.fingerNames = imageProccess.fingers.props.fingerNames;
movingImage.palm.elipse = imageProccess.palm.elipse;
movingImage.palm.mainBox = imageProccess.palm.mainBox;

% --- Executes on button press in resetImageBtn.
function resetImageBtn_Callback(hObject, eventdata, handles)
% hObject    handle to resetImageBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global movingImage
if contains(movingImage.fileType, '.csv') 
    movingImage.fixedGrey= movingImage.originalImage;
    movingImage.fixedImage= movingImage.originalImage;

elseif contains(movingImage.fileType, '.jpg')
    movingImage.fixedGrey = double(rgb2gray(movingImage.originalImage));
    movingImage.fixedImage = double(rgb2gray(movingImage.originalImage));
end
axes(handles.movingImageFig);
imagesc(movingImage.fixedImage, 'Parent', handles.movingImageFig);
set(handles.movingImageFig, 'visible', 'off');

cla(handles.aligmentFig)
cla(handles.movingImageROIs)
set(handles.alignmentDiffSTR,'Visible','off');
set(handles.movingImageROISTR,'Visible','off');


% --- Executes on button press in SaveFeaturesToFileBtn.
function SaveFeaturesToFileBtn_Callback(hObject, eventdata, handles)
% hObject    handle to SaveFeaturesToFileBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global featureTable 
global imageProccess 
global movingImage
featureTable(1,:)=[];

if contains(movingImage.fileType, '.csv')
    defaultPath = 'C:\Users\ido\Google Drive\Thesis\Data\Processed Data\CSV Feature tables\';
else
    defaultPath = 'C:\Users\ido\Google Drive\Thesis\Data\Processed Data\JPG Feature tables\';
end
if exist(defaultPath, 'dir')
    saveTablePath = [defaultPath imageProccess.subjectID '_' imageProccess.handSide '_Feature_Table.xlsx'];
else 
    saveTablePath = [imageProccess.path imageProccess.subjectID '_' imageProccess.handSide '_Feature_Table.xlsx'];
end

writetable(featureTable, saveTablePath, 'Sheet','all');
save([imageProccess.path 'data.mat'], 'featureTable', 'imageProccess', 'movingImage');

% --- Executes on button press in ApproveNextBtn.
function ApproveNextBtn_Callback(hObject, eventdata, handles)
% hObject    handle to ApproveNextBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

ExtractFeaturesBtn_Callback;
nextImageBtn_Callback(hObject, eventdata, handles);
runSegmentationBtn_Callback(hObject, eventdata, handles);


% --- Executes on button press in manualROIsPickBtn.
function manualROIsPickBtn_Callback(hObject, eventdata, handles)
% hObject    handle to manualROIsPickBtn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global movingImage
movingImage = ManualROIsExtractionGUI(movingImage);

PlotFingersROIinGUI (handles.movingImageROIs, movingImage.fixedImage, movingImage.fingers.fingersTips, movingImage.fingers.fingersBase)

hold(handles.movingImageROIs, 'on');
set(gca,'xtick',[]);
set(gca,'visible','off');
plot(movingImage.palm.mainBox(1,:), movingImage.palm.mainBox(2,:), 'r-', 'LineWidth', 1, 'Parent', handles.movingImageROIs);
plot(movingImage.palm.elipse(:,1), movingImage.palm.elipse(:,2), 'r-', 'LineWidth', 1);
hold(handles.movingImageROIs, 'off');
