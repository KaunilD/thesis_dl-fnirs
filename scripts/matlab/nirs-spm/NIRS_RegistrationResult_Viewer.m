function varargout = NIRS_RegistrationResult_Viewer(varargin)
% NIRS_REGISTRATIONRESULT_VIEWER M-file for NIRS_RegistrationResult_Viewer.fig
%      NIRS_REGISTRATIONRESULT_VIEWER, by itself, creates a new NIRS_REGISTRATIONRESULT_VIEWER or raises the existing
%      singleton*.
%
%      H = NIRS_REGISTRATIONRESULT_VIEWER returns the handle to a new NIRS_REGISTRATIONRESULT_VIEWER or the handle to
%      the existing singleton*.
%
%      NIRS_REGISTRATIONRESULT_VIEWER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in NIRS_REGISTRATIONRESULT_VIEWER.M with the given input arguments.
%
%      NIRS_REGISTRATIONRESULT_VIEWER('Property','Value',...) creates a new NIRS_REGISTRATIONRESULT_VIEWER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before NIRS_RegistrationResult_Viewer_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to NIRS_RegistrationResult_Viewer_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help NIRS_RegistrationResult_Viewer

% Last Modified by GUIDE v2.5 03-Sep-2010 21:07:52

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @NIRS_RegistrationResult_Viewer_OpeningFcn, ...
                   'gui_OutputFcn',  @NIRS_RegistrationResult_Viewer_OutputFcn, ...
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


% --- Executes just before NIRS_RegistrationResult_Viewer is made visible.
function NIRS_RegistrationResult_Viewer_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;
rendered_MNI = varargin{1};
Nch = size(rendered_MNI{1}.rchn,1);

load Split
kk = 2; % dorsal view
brain = rendered_MNI{kk}.ren;
brain = brain.* 0.5;
sbar = linspace(0, 1, 128);
sbrain = ((-sbar(1) + sbar(64))/(0.5)).* brain + sbar(1);
sbrain(1,1) = 1;
axes(handles.axes_brain);

rchn = rendered_MNI{kk}.rchn;
cchn = rendered_MNI{kk}.cchn;
for jj = 1:Nch
    if rchn(jj) ~= -1 && cchn(jj) ~= -1 %% updated 2009-02-25
        if rchn(jj) < 6 || cchn(jj) < 6
            sbrain(rchn(jj), cchn(jj)) = 0.9; % 0.67
        else
            sbrain(rchn(jj)-5:rchn(jj)+5, cchn(jj)-5:cchn(jj)+5) = 0.9;
        end
    end
end
imagesc(sbrain);
colormap(split);
axis image;
axis off;

for jj = 1:Nch
    if rchn(jj) ~= -1 && cchn(jj) ~= -1 %% updated 2009-02-25
        text(cchn(jj)-5, rchn(jj), num2str(jj), 'color', 'r');
    end
end

set(handles.slider_brainview, 'enable', 'on', 'sliderstep', [1/5, 1/5], 'max', 6, 'min', 1, 'value', 3);
handles.rendered_MNI = rendered_MNI;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes NIRS_RegistrationResult_Viewer wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = NIRS_RegistrationResult_Viewer_OutputFcn(hObject, eventdata, handles) 
% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on slider movement.
function slider_brainview_Callback(hObject, eventdata, handles)
kk = round(get(handles.slider_brainview, 'value'));
set(handles.slider_brainview, 'value', kk);

rendered_MNI = handles.rendered_MNI;
Nch = size(rendered_MNI{1}.rchn,1);

load Split
brain = rendered_MNI{kk}.ren;
brain = brain.* 0.5;
sbar = linspace(0, 1, 128);
sbrain = ((-sbar(1) + sbar(64))/(0.5)).* brain + sbar(1);
sbrain(1,1) = 1;
axes(handles.axes_brain);
rchn = rendered_MNI{kk}.rchn;
cchn = rendered_MNI{kk}.cchn;
for jj = 1:Nch
    if rchn(jj) ~= -1 && cchn(jj) ~= -1 %% updated 2009-02-25
        if rchn(jj) < 6 || cchn(jj) < 6
            sbrain(rchn(jj), cchn(jj)) = 0.9; % 0.67
        else
            sbrain(rchn(jj)-5:rchn(jj)+5, cchn(jj)-5:cchn(jj)+5) = 0.9;
        end
    end
end
imagesc(sbrain);
colormap(split);
axis image;
axis off;

for jj = 1:Nch
    if rchn(jj) ~= -1 && cchn(jj) ~= -1 %% updated 2009-02-25
        text(cchn(jj)-5, rchn(jj), num2str(jj), 'color', 'r');
    end
end

% --- Executes during object creation, after setting all properties.
function slider_brainview_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider_brainview (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
