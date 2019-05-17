function varargout = NIRS_Rendered_MNI_Viewer(varargin)

% Last Modified by GUIDE v2.5 03-Dec-2008 13:10:11

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @NIRS_Rendered_MNI_Viewer_OpeningFcn, ...
    'gui_OutputFcn',  @NIRS_Rendered_MNI_Viewer_OutputFcn, ...
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


% --- Executes just before NIRS_Rendered_MNI_Viewer is made visible.
function NIRS_Rendered_MNI_Viewer_OpeningFcn(hObject, eventdata, handles, varargin)

% Choose default command line output for NIRS_Rendered_MNI_Viewer
handles.output = hObject;

rendered_MNI = varargin{1};
Nch = size(rendered_MNI{1}.rchn,1);

load Split
for kk = 1:6
    brain = rendered_MNI{kk}.ren;
    brain = brain.* 0.5;
    sbar = linspace(0, 1, 128);
    sbrain = ((-sbar(1) + sbar(64))/(0.5)).* brain + sbar(1);
    sbrain(1,1) = 1;
    
    switch kk
        case 1
            axes(handles.axes_ventral_view);
        case 2
            axes(handles.axes_dorsal_view);
        case 3
            axes(handles.axes_lateral_view_right);
        case 4
            axes(handles.axes_lateral_view_left);
        case 5
            axes(handles.axes_frontal_view);
        case 6
            axes(handles.axes_occipital_view);
    end
    
    imagesc(sbrain);
    for jj = 1:Nch
        rchn = rendered_MNI{kk}.rchn(jj);
        cchn = rendered_MNI{kk}.cchn(jj);        
        if rchn ~= -1 && cchn ~= -1 %% updated 2009-02-25
            if rchn < 6 || cchn < 6
                sbrain(rchn, cchn) = 0.67;
            else                
                sbrain(rchn-5:rchn+5, cchn-5:cchn+5) = 0.67;
                text(cchn, rchn, num2str(jj), 'color', 'r');           
            end
        end
    end
    imagesc(sbrain);
    colormap(split);
    axis image
    axis off
end
% Update handles structure
guidata(hObject, handles);


% --- Outputs from this function are returned to the command line.
function varargout = NIRS_Rendered_MNI_Viewer_OutputFcn(hObject, eventdata, handles)

% Get default command line output from handles structure
varargout{1} = handles.output;
