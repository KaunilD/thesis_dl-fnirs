function varargout = NIRS_Specification(varargin)

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @NIRS_Specification_OpeningFcn, ...
    'gui_OutputFcn',  @NIRS_Specification_OutputFcn, ...
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


% --- Executes just before NIRS_Specification is made visible.
function NIRS_Specification_OpeningFcn(hObject, eventdata, handles, varargin)
% Choose default command line output for NIRS_Specification
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = NIRS_Specification_OutputFcn(hObject, eventdata, handles)


varargout{1} = handles.output;



function edit_nirsfname_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function edit_nirsfname_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in push_nirsfile.
function push_nirsfile_Callback(hObject, eventdata, handles)

[filen, pathn] = uigetfile('*.mat','Select  file');
if filen(1) == 0 | pathn(1) == 0
    return
else
    path_file_n = [pathn filen];
    set(handles.edit_nirsfname','string',path_file_n);
end


% --- Executes on button press in push_spec.
function push_spec_Callback(hObject, eventdata, handles)
nirs_fname = get(handles.edit_nirsfname,'string');
spm_dir = get(handles.edit_spmdir','string');
h1 = get(handles.checkbox_HbO, 'value');
h2 = get(handles.checkbox_HbR, 'value');
h3 = get(handles.checkbox_HbT, 'value');

if h1 == 0 & h2 == 0 & h3 == 0
    errordlg('Please select the hemoglobin type for specifying the design matrix','Selection Error');
    return
end

if h1 == 1 % HbO    
    SPM_nirs = spm_nirs_design_final(nirs_fname, 1, 'GLM_specification');
    SPM_nirs.nirs.Hb = 'HbO';
    SPM_nirs.nirs.level = 'individual';
    save([spm_dir filesep 'SPM_indiv_HbO.mat'], 'SPM_nirs');
elseif h2 == 1 % HbR
    SPM_nirs = spm_nirs_design_final(nirs_fname, 2, 'GLM_specification');
    SPM_nirs.nirs.Hb = 'HbR';
    SPM_nirs.nirs.level = 'individual';
    save([spm_dir filesep 'SPM_indiv_HbR.mat'], 'SPM_nirs');
elseif h3 == 1 % HbT
    SPM_nirs = spm_nirs_design_final(nirs_fname, 1, 'GLM_specification');
    SPM_nirs.nirs.Hb = 'HbT';
    SPM_nirs.nirs.level = 'individual';
    save([spm_dir filesep 'SPM_indiv_HbT.mat'], 'SPM_nirs');
end




% --- Executes on button press in push_spmdir.
function push_spmdir_Callback(hObject, eventdata, handles)

cur_dir = cd;
pathn = uigetdir(cur_dir);
if pathn == 0
    return
else 
    set(handles.edit_spmdir,'string',pathn);
end


function edit_spmdir_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_spmdir_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in checkbox_HbO.
function checkbox_HbO_Callback(hObject, eventdata, handles)
h = get(handles.checkbox_HbO, 'value');

if h == 1
    set(handles.checkbox_HbR, 'value', 0);
    set(handles.checkbox_HbT, 'value', 0);
end

    
% --- Executes on button press in checkbox_HbR.
function checkbox_HbR_Callback(hObject, eventdata, handles)
h = get(handles.checkbox_HbR, 'value');
if h == 1
    set(handles.checkbox_HbO, 'value', 0);
    set(handles.checkbox_HbT, 'value', 0);
end

% --- Executes on button press in checkbox_HbT.
function checkbox_HbT_Callback(hObject, eventdata, handles)
h = get(handles.checkbox_HbT, 'value');
if h == 1
set(handles.checkbox_HbO, 'value', 0);
set(handles.checkbox_HbR, 'value', 0);
end
