function varargout = NIRS_SPM(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @NIRS_SPM_OpeningFcn, ...
                   'gui_OutputFcn',  @NIRS_SPM_OutputFcn, ...
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

% --- Executes just before NIRS_SPM is made visible.
function NIRS_SPM_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;

guidata(hObject, handles);

clc
disp('NIRS-SPM: Statistical Parametric Mapping for Near Infrared Spectroscopy');
disp('      ');
disp('NIRS-SPM present working directory:');
current_dir = cd;
disp(['        ' current_dir]);

% --- Outputs from this function are returned to the command line.
function varargout = NIRS_SPM_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

% --- Executes on button press in push_quit.
function push_quit_Callback(hObject, eventdata, handles)
close(gcf);

% --- Executes on button press in push_resultsfMRI.
function push_resultsfMRI_Callback(hObject, eventdata, handles)
global h_fmriviewer;
global handles_fmriviewer;
[SPM, xSPM] = spm_getSPM_modified;
fmri_result = spm_render_modified(struct('XYZ',xSPM.XYZ,'t',xSPM.Z,'mat',xSPM.M,'dim',xSPM.DIM));
fmri_result{1}.thresDesc = xSPM.thresDesc;
fmri_result{1}.swd = xSPM.swd;
fmri_result{1}.title = xSPM.title;

global fmri_result_path
fmri_result_path = fmri_result{1}.swd;

for kk = 1:length(fmri_result)
    temp = fmri_result{kk}.img;
    temp2 = fmri_result{kk}.data;
    fmri_result{kk}.img = temp(size(fmri_result{kk}.img,1):-1:1,:);
    fmri_result{kk}.data = temp2(size(fmri_result{kk}.data,1):-1:1,:);
end


cur_dir = cd;
cd(fmri_result{1}.swd);
save('fmri_result','fmri_result');
cd(cur_dir);

[h_fmriviewer handles_fmriviewer] = fMRI_Results_Viewer;


% --- Executes on button press in push_resultsNIRS.
function push_resultsNIRS_Callback(hObject, eventdata, handles)
NIRS_Results_Viewer

% --- Executes on button press in push_spec1st.
function push_spec1st_Callback(hObject, eventdata, handles)
NIRS_Specification;

% --- Executes on button press in push_est.
function push_est_Callback(hObject, eventdata, handles)
NIRS_Estimation;


% --- Executes on button press in push_chloc.
function push_chloc_Callback(hObject, eventdata, handles)
h = get(handles.checkbox_standalone,'value');
h2 = get(handles.checkbox_MRI, 'value');
if h == 1 & h2 == 0
    global handles_standalone;
    [temp1 temp2] = NIRS_Registration_Standalone;
    handles_standalone{1} = temp1;
    handles_standalone{2} = temp2;
elseif h == 0 & h2 == 1
    global handles_registration_NIRS_MRI;
    [temp1 temp2] = NIRS_Import_Indicator_Loc;
    handles_registration_NIRS_MRI{1} = temp1;
    handles_registration_NIRS_MRI{2} = temp2;   
end


% --- Executes on button press in push_convert.
function push_convert_Callback(hObject, eventdata, handles)
NIRS_Data_Conversion;


% --- Executes on button press in push_NIRS_timeseries.
function push_NIRS_timeseries_Callback(hObject, eventdata, handles)
NIRS_TimeSeries_Viewer


% --- Executes on button press in push_inquiry.
function push_inquiry_Callback(hObject, eventdata, handles)
helpdlg('To read other NIRS data formats from other venders, please send a data set and file format to shtak@kaist.ac.kr. We will update NIRS-SPM packages to include the data formats.');


% --- Executes on button press in checkbox_MRI.
function checkbox_MRI_Callback(hObject, eventdata, handles)
h = get(handles.checkbox_MRI, 'value');
if h == 0 
    set(handles.checkbox_standalone, 'value', 1);
elseif h == 1
    set(handles.checkbox_standalone, 'value', 0);
end

% --- Executes on button press in checkbox_standalone.
function checkbox_standalone_Callback(hObject, eventdata, handles)
h = get(handles.checkbox_standalone, 'value');
if h == 0
    set(handles.checkbox_MRI, 'value', 1);
elseif h == 1
    set(handles.checkbox_MRI, 'value', 0);
end


% --- Executes on button press in push_CMRO2_Est.
function push_CMRO2_Est_Callback(hObject, eventdata, handles)
CMRO2_Est
