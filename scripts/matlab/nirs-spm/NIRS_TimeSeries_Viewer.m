function varargout = NIRS_TimeSeries_Viewer(varargin)

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @NIRS_TimeSeries_Viewer_OpeningFcn, ...
    'gui_OutputFcn',  @NIRS_TimeSeries_Viewer_OutputFcn, ...
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


function NIRS_TimeSeries_Viewer_OpeningFcn(hObject, eventdata, handles, varargin)

% Choose default command line output for NIRS_TimeSeries_Viewer
handles.output = hObject;

set(handles.edit_raw_ch, 'enable', 'inactive', 'string', '');
set(handles.edit_process_ch, 'enable', 'inactive', 'string', '');
set(handles.slider_raw_ch, 'enable', 'inactive');
set(handles.slider_process_ch, 'enable','inactive');
set(handles.push_ROI_analysis,'enable','off');
set(handles.checkbox_ROI_HbO,'enable','off', 'value', 1);
set(handles.checkbox_ROI_HbR,'enable','off', 'value', 1);
set(handles.checkbox_ROI_HbT,'enable','off', 'value', 1);
set(handles.checkbox_model, 'enable', 'off');
set(handles.checkbox_model, 'value', 0);
set(handles.checkbox_HbO,'enable', 'off','value', 1);
set(handles.checkbox_HbR,'enable', 'off','value', 1);
set(handles.checkbox_HbT,'enable', 'off','value', 1);
set(handles.checkbox_BOLD_raw, 'enable', 'off', 'value', 0);
set(handles.checkbox_BOLD_filt, 'enable', 'off', 'value', 0);
set(handles.checkbox_process_HbO, 'enable', 'off','value',0)
set(handles.checkbox_process_HbR, 'enable', 'off','value',0);
set(handles.checkbox_process_HbT, 'enable', 'off','value',0);
set(handles.push_model_response, 'enable', 'off');
set(handles.push_filtering, 'enable','off');
set(handles.popup_num_model, 'enable', 'off');
set(handles.push_save, 'enable', 'off');
set(handles.checkbox_norm_raw, 'enable', 'off', 'value', 0);
set(handles.checkbox_ROI_BOLD, 'enable', 'off', 'value', 0);
set(handles.push_filter_BOLD, 'enable', 'off');
set(handles.checkbox_axislimit_raw, 'enable', 'off', 'value', 0);
set(handles.text25, 'enable', 'off');
set(handles.checkbox_axislimit_filt, 'enable', 'off', 'value', 0);

str_list{1,1} = 'The summary of results';
str_list{1,1} = 'Name of files';
set(handles.listbox_fname, 'value', 1, 'string', str_list);
str_list{1,1} = 'NIRS processing info.';
set(handles.listbox_NIRS_info, 'value', 1, 'string', str_list);
str_list{1,1} = 'fMRI processing info.';
set(handles.listbox_fMRI_info, 'value', 1, 'string', str_list);

cla(handles.axes_nirs_timeseries, 'reset');
cla(handles.axes_result, 'reset');


% for raw data
try
    handles = rmfield(handles, 'nirs_data');
    handles = rmfield(handles, 'total_Hb');
end

try
    handles = rmfield(handles, 'fMRI_data');
end

% for filtered data
try
    handles = rmfield(handles, 'fnirs_data');
    handles = rmfield(handles, 'ftotal_Hb');
end

% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = NIRS_TimeSeries_Viewer_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;


% --- Executes on button press in checkbox_HbO.
function checkbox_HbO_Callback(hObject, eventdata, handles)

if isfield(handles, 'fMRI_data') == 0 && get(handles.checkbox_HbO, 'value') == 0 && get(handles.checkbox_HbR, 'value') == 0 && get(handles.checkbox_HbT, 'value') == 0
    cla(handles.axes_nirs_timeseries, 'reset');
    helpdlg('Please select the hemoglobin type to be displayed.');
    return;
elseif isfield(handles, 'fMRI_data') == 1 && get(handles.checkbox_HbO, 'value') == 0 && get(handles.checkbox_HbR, 'value') == 0 && get(handles.checkbox_HbT, 'value') == 0 && get(handles.checkbox_BOLD_raw,'value') ==0
    cla(handles.axes_nirs_timeseries, 'reset');
    helpdlg('Please select the hemoglobin type or BOLD to be displayed.');
    return;
end


% load the nirs_data
if get(handles.checkbox_HbO, 'value') == 1 || get(handles.checkbox_HbR, 'value') == 1 || get(handles.checkbox_HbT, 'value') == 1
    try % exist?
        nirs_data = handles.nirs_data;
        total_Hb = handles.total_Hb;
    catch % not exist?
        set(handles.checkbox_HbO, 'enable', 'off', 'value', 0);
        set(handles.checkbox_HbR, 'enable', 'off', 'value', 0);
        set(handles.checkbox_HbT, 'enable', 'off', 'value', 0);
        errordlg('NIRS data does not exist. Please load the NIRS data.');
        return;
    end
end

% load the fMRI data
if get(handles.checkbox_BOLD_raw, 'value') == 1
    try % exist?
        fMRI_data = handles.fMRI_data;
    catch
        set(handles.checkbox_BOLD_raw, 'enable', 'off', 'value', 0);
        errordlg('fMRI_data does not exist. Please load the fMRI data.');
        return;
    end
end

% obtain the number of channels from NIRS or BOLD data format
try
    nch = size(nirs_data.oxyData,2);
catch
    nch = size(fMRI_data.bold, 2);
end


h = get(handles.slider_raw_ch, 'value');

time = handles.time;
set(handles.edit_raw_ch, 'string', [num2str(h) ' / ' num2str(nch)]);

axes(handles.axes_nirs_timeseries);
hold off

if get(handles.checkbox_axislimit_raw, 'value') == 1
    limit_xaxis = handles.limit_xaxis;
    limit_yaxis = handles.limit_yaxis;

    axes(handles.axes_nirs_timeseries);
    default_axis = axis;
    if isempty(limit_xaxis) == 0
        default_axis(1:2) = limit_xaxis(1:2);
    end
    if isempty(limit_yaxis) == 0
        default_axis(3:4) = limit_yaxis(1:2);
    end
end

% plot the normalized signals ([0 1])
if get(handles.checkbox_norm_raw, 'value') == 1
    if get(handles.checkbox_HbO, 'value') == 1
        plot(time, nirs_data.oxyData(:,h)./max(nirs_data.oxyData(:,h)),'r');
        hold on
    end
    if get(handles.checkbox_HbR, 'value') == 1
        plot(time, nirs_data.dxyData(:,h)./max(nirs_data.dxyData(:,h)), 'b');
        hold on
    end
    if get(handles.checkbox_HbT, 'value') == 1
        plot(time, total_Hb(:,h)./max(total_Hb(:,h)), 'color', [0 127/255 0]);
        hold on
    end
    if get(handles.checkbox_BOLD_raw, 'value') == 1
        plot(handles.time_b, fMRI_data.bold(:,h)./max(fMRI_data.bold(:,h)), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
elseif get(handles.checkbox_norm_raw, 'value') == 0
    if get(handles.checkbox_HbO, 'value') == 1
        plot(time, nirs_data.oxyData(:,h), 'r');
        hold on
    end
    if get(handles.checkbox_HbR, 'value') == 1
        plot(time, nirs_data.dxyData(:,h), 'b');
        hold on
    end
    if get(handles.checkbox_HbT, 'value') == 1
        plot(time, total_Hb(:,h), 'color', [0 127/255 0]);
        hold on
    end
    if get(handles.checkbox_BOLD_raw, 'value') == 1
        plot(handles.time_b, fMRI_data.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
end
xlabel('Time (s)');
set(handles.axes_nirs_timeseries, 'FontSize', 9);

if get(handles.checkbox_axislimit_raw, 'value') == 0
    axes(handles.axes_nirs_timeseries);
    default_axis = axis;
    if ismember('time', who) == 1
        default_axis(1:2) = [min(time) max(time)];
    end
end
axis(default_axis);

if get(handles.checkbox_model, 'value') == 1
    try
        num_model = get(handles.popup_num_model, 'value');
        scale_model = linspace(default_axis(3), default_axis(4), 6);
        min_model = min(handles.SPM.xX.X(:,num_model));
        max_model = max(handles.SPM.xX.X(:,num_model));
        model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
        hold on
        plot(time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
    catch
        set(handles.checkbox_model, 'value', 0);
        helpdlg('The predicted model response does not exist.');
    end
end

guidata(hObject, handles);


% --- Executes on button press in checkbox_HbR.
function checkbox_HbR_Callback(hObject, eventdata, handles)
if isfield(handles, 'fMRI_data') == 0 && get(handles.checkbox_HbO, 'value') == 0 && get(handles.checkbox_HbR, 'value') == 0 && get(handles.checkbox_HbT, 'value') == 0
    cla(handles.axes_nirs_timeseries, 'reset');
    helpdlg('Please select the hemoglobin type to be displayed.');
    return;
elseif isfield(handles, 'fMRI_data') == 1 && get(handles.checkbox_HbO, 'value') == 0 && get(handles.checkbox_HbR, 'value') == 0 && get(handles.checkbox_HbT, 'value') == 0 && get(handles.checkbox_BOLD_raw,'value') ==0
    cla(handles.axes_nirs_timeseries, 'reset');
    helpdlg('Please select the hemoglobin type or BOLD to be displayed.');
    return;
end


% load the nirs_data
if get(handles.checkbox_HbO, 'value') == 1 || get(handles.checkbox_HbR, 'value') == 1 || get(handles.checkbox_HbT, 'value') == 1
    try % exist?
        nirs_data = handles.nirs_data;
        total_Hb = handles.total_Hb;
    catch % not exist?
        set(handles.checkbox_HbO, 'enable', 'off', 'value', 0);
        set(handles.checkbox_HbR, 'enable', 'off', 'value', 0);
        set(handles.checkbox_HbT, 'enable', 'off', 'value', 0);
        errordlg('NIRS data does not exist. Please load the NIRS data.');
        return;
    end
end

% load the fMRI data
if get(handles.checkbox_BOLD_raw, 'value') == 1
    try % exist?
        fMRI_data = handles.fMRI_data;
    catch
        set(handles.checkbox_BOLD_raw, 'enable', 'off', 'value', 0);
        errordlg('fMRI_data does not exist. Please load the fMRI data.');
        return;
    end
end

% obtain the number of channels from NIRS or BOLD data format
try
    nch = size(nirs_data.oxyData,2);
catch
    nch = size(fMRI_data.bold, 2);
end


h = get(handles.slider_raw_ch, 'value');

time = handles.time;

set(handles.edit_raw_ch, 'string', [num2str(h) ' / ' num2str(nch)]);

axes(handles.axes_nirs_timeseries);
hold off

if get(handles.checkbox_axislimit_raw, 'value') == 1
    limit_xaxis = handles.limit_xaxis;
    limit_yaxis = handles.limit_yaxis;

    axes(handles.axes_nirs_timeseries);
    default_axis = axis;
    if isempty(limit_xaxis) == 0
        default_axis(1:2) = limit_xaxis(1:2);
    end
    if isempty(limit_yaxis) == 0
        default_axis(3:4) = limit_yaxis(1:2);
    end
end

% plot the normalized signals ([0 1])
if get(handles.checkbox_norm_raw, 'value') == 1
    if get(handles.checkbox_HbO, 'value') == 1
        plot(time, nirs_data.oxyData(:,h)./max(nirs_data.oxyData(:,h)),'r');
        hold on
    end
    if get(handles.checkbox_HbR, 'value') == 1
        plot(time, nirs_data.dxyData(:,h)./max(nirs_data.dxyData(:,h)), 'b');
        hold on
    end
    if get(handles.checkbox_HbT, 'value') == 1
        plot(time, total_Hb(:,h)./max(total_Hb(:,h)), 'color', [0 127/255 0]);
        hold on
    end
    if get(handles.checkbox_BOLD_raw, 'value') == 1
        plot(handles.time_b, fMRI_data.bold(:,h)./max(fMRI_data.bold(:,h)), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
elseif get(handles.checkbox_norm_raw, 'value') == 0
    if get(handles.checkbox_HbO, 'value') == 1
        plot(time, nirs_data.oxyData(:,h), 'r');
        hold on
    end
    if get(handles.checkbox_HbR, 'value') == 1
        plot(time, nirs_data.dxyData(:,h), 'b');
        hold on
    end
    if get(handles.checkbox_HbT, 'value') == 1
        plot(time, total_Hb(:,h), 'color', [0 127/255 0]);
        hold on
    end
    if get(handles.checkbox_BOLD_raw, 'value') == 1
        plot(handles.time_b, fMRI_data.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
end
xlabel('Time (s)');
set(handles.axes_nirs_timeseries, 'FontSize', 9);

if get(handles.checkbox_axislimit_raw, 'value') == 0
    axes(handles.axes_nirs_timeseries);
    default_axis = axis;
    if ismember('time', who) == 1
        default_axis(1:2) = [min(time) max(time)];
    end
end
axis(default_axis);

if get(handles.checkbox_model, 'value') == 1
    try
        num_model = get(handles.popup_num_model, 'value');
        scale_model = linspace(default_axis(3), default_axis(4), 6);
        min_model = min(handles.SPM.xX.X(:,num_model));
        max_model = max(handles.SPM.xX.X(:,num_model));
        model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
        hold on
        plot(time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
    catch
        set(handles.checkbox_model, 'value', 0);
        helpdlg('The predicted model response does not exist.');
    end
end

guidata(hObject, handles);

% --- Executes on button press in checkbox_HbT.
function checkbox_HbT_Callback(hObject, eventdata, handles)
if isfield(handles, 'fMRI_data') == 0 && get(handles.checkbox_HbO, 'value') == 0 && get(handles.checkbox_HbR, 'value') == 0 && get(handles.checkbox_HbT, 'value') == 0
    cla(handles.axes_nirs_timeseries, 'reset');
    helpdlg('Please select the hemoglobin type to be displayed.');
    return;
elseif isfield(handles, 'fMRI_data') == 1 && get(handles.checkbox_HbO, 'value') == 0 && get(handles.checkbox_HbR, 'value') == 0 && get(handles.checkbox_HbT, 'value') == 0 && get(handles.checkbox_BOLD_raw,'value') ==0
    cla(handles.axes_nirs_timeseries, 'reset');
    helpdlg('Please select the hemoglobin type or BOLD to be displayed.');
    return;
end


% load the nirs_data
if get(handles.checkbox_HbO, 'value') == 1 || get(handles.checkbox_HbR, 'value') == 1 || get(handles.checkbox_HbT, 'value') == 1
    try % exist?
        nirs_data = handles.nirs_data;
        total_Hb = handles.total_Hb;
    catch % not exist?
        set(handles.checkbox_HbO, 'enable', 'off', 'value', 0);
        set(handles.checkbox_HbR, 'enable', 'off', 'value', 0);
        set(handles.checkbox_HbT, 'enable', 'off', 'value', 0);
        errordlg('NIRS data does not exist. Please load the NIRS data.');
        return;
    end
end

% load the fMRI data
if get(handles.checkbox_BOLD_raw, 'value') == 1
    try % exist?
        fMRI_data = handles.fMRI_data;
    catch
        set(handles.checkbox_BOLD_raw, 'enable', 'off', 'value', 0);
        errordlg('fMRI_data does not exist. Please load the fMRI data.');
        return;
    end
end

% obtain the number of channels from NIRS or BOLD data format
try
    nch = size(nirs_data.oxyData,2);
catch
    nch = size(fMRI_data.bold, 2);
end


h = get(handles.slider_raw_ch, 'value');

time = handles.time;
set(handles.edit_raw_ch, 'string', [num2str(h) ' / ' num2str(nch)]);

axes(handles.axes_nirs_timeseries);
hold off

if get(handles.checkbox_axislimit_raw, 'value') == 1
    limit_xaxis = handles.limit_xaxis;
    limit_yaxis = handles.limit_yaxis;

    axes(handles.axes_nirs_timeseries);
    default_axis = axis;
    if isempty(limit_xaxis) == 0
        default_axis(1:2) = limit_xaxis(1:2);
    end
    if isempty(limit_yaxis) == 0
        default_axis(3:4) = limit_yaxis(1:2);
    end
end

% plot the normalized signals ([0 1])
if get(handles.checkbox_norm_raw, 'value') == 1
    if get(handles.checkbox_HbO, 'value') == 1
        plot(time, nirs_data.oxyData(:,h)./max(nirs_data.oxyData(:,h)),'r');
        hold on
    end
    if get(handles.checkbox_HbR, 'value') == 1
        plot(time, nirs_data.dxyData(:,h)./max(nirs_data.dxyData(:,h)), 'b');
        hold on
    end
    if get(handles.checkbox_HbT, 'value') == 1
        plot(time, total_Hb(:,h)./max(total_Hb(:,h)), 'color', [0 127/255 0]);
        hold on
    end
    if get(handles.checkbox_BOLD_raw, 'value') == 1
        plot(handles.time_b, fMRI_data.bold(:,h)./max(fMRI_data.bold(:,h)), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
elseif get(handles.checkbox_norm_raw, 'value') == 0
    if get(handles.checkbox_HbO, 'value') == 1
        plot(time, nirs_data.oxyData(:,h), 'r');
        hold on
    end
    if get(handles.checkbox_HbR, 'value') == 1
        plot(time, nirs_data.dxyData(:,h), 'b');
        hold on
    end
    if get(handles.checkbox_HbT, 'value') == 1
        plot(time, total_Hb(:,h), 'color', [0 127/255 0]);
        hold on
    end
    if get(handles.checkbox_BOLD_raw, 'value') == 1
        plot(handles.time_b, fMRI_data.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
end
xlabel('Time (s)');
set(handles.axes_nirs_timeseries, 'FontSize', 9);

if get(handles.checkbox_axislimit_raw, 'value') == 0
    axes(handles.axes_nirs_timeseries);
    default_axis = axis;
    if ismember('time', who) == 1
        default_axis(1:2) = [min(time) max(time)];
    end
end
axis(default_axis);

if get(handles.checkbox_model, 'value') == 1
    try
        num_model = get(handles.popup_num_model, 'value');
        scale_model = linspace(default_axis(3), default_axis(4), 6);
        min_model = min(handles.SPM.xX.X(:,num_model));
        max_model = max(handles.SPM.xX.X(:,num_model));
        model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
        hold on
        plot(time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
    catch
        set(handles.checkbox_model, 'value', 0);
        helpdlg('The predicted model response does not exist.');
    end
end

guidata(hObject, handles);

function edit_raw_ch_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_raw_ch_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in push_nirsfile.
function push_nirsfile_Callback(hObject, eventdata, handles)
str = 'Loading?';
str_load = {'NIRS', 'NIRS&fMRI'};
str_load = spm_input(str, 1, 'b', str_load);

str_list = {};

%==
%  setup for NIRS data
spm_input('Setup for NIRS data', '+2', 'd');
[path_file_n, sts] = spm_select(1, 'mat', 'Select NIRS data');
if sts == 0
    return;
end
[tmp, disp_fname] = fileparts(path_file_n);
spm_input(['File name: ' disp_fname], '+1', 'd');
% for listbox
str_list = [str_list; {['File name (NIRS): ' disp_fname]}];

handles.fname_nirs = path_file_n;

% setup for fMRI data
if strcmp(str_load, 'NIRS&fMRI') == 1
    spm_input('Setup for fMRI data', '+2', 'd');
    [path_file_b, sts] = spm_select_modified(Inf, 'image', 'Select fMRI data (.img, .nii, .mat)');
    if sts == 0
        return;
    end
    RT = spm_input('Interscan interval [secs]', '+1', 'r', ' ', 1);
    [tmp, disp_fname, ext] = fileparts(path_file_b(1,:));
    if strcmpi(ext(1, 1:4), '.mat') == 1
        spm_input(['File name: ' disp_fname], '+1', 'd');
        str_list = [str_list; {['File name (fMRI): ' disp_fname]}];
    else
        nscan_b = size(path_file_b, 1);
        spm_input(['File info.: ' num2str(nscan_b) ' scans, ' ext(1,1:4) ' format'], '+1','d');
        str_list = [str_list; {['File name (fMRI): ' disp_fname(1,:)]}];
        str_list = [str_list; {['File info.: ' num2str(nscan_b) ' scans, ' ext(1,1:4) ' format']}];
    end
    spm_input('NIRS-fMRI alignment', '+2', 'd');
    [path_file_ch, sts] = spm_select(1, 'mat', 'Select a file for NIRS channel locations');
    if sts ==  0
        return;
    end
    [tmp, disp_fname] = fileparts(path_file_ch);
    spm_input(['File name: ' disp_fname], '+1', 'd');
    str_list = [str_list; {['File name (Ch.): ' disp_fname]}];
end
set(handles.listbox_fname, 'string', str_list, 'value', 1);

%==
% check if filtering has been applied.
% for NIRS
load(path_file_n);
nscan_n = size(nirs_data.oxyData,1);

str_process = '';
if isfield(nirs_data, 'cL') == 1 || isfield(nirs_data, 'cH') == 1
    str_process = [str_process 'filtering'];
end
if isfield(nirs_data, 'baseline') == 1
    str_process = [str_process '  baseline-correction'];
end
if isempty(str_process) == 0
    helpdlg(['NIRS: Note that temporal processing (' str_process ') has been already applied to the loaded data.']);
end

% read the fMRI time-series
if strcmp(str_load, 'NIRS&fMRI') == 1
    load(path_file_ch);
    nch = size(preproc_info.ch_MNI_mm, 2);
    if strcmpi(ext(1, 1:4), '.mat') == 1 % from .mat format
        load(path_file_b(1:end-2));
        if ismember('fMRI_data', who) == 0
            errordlg('The variable, fMRI_data, which saves fMRI time-series does not exist. Please confirm it.');
            return;
        elseif ismember('fMRI_data', who) == 1
            if isfield(fMRI_data, 'cL') == 1 || isfield(fMRI_data, 'cH') == 1 || isfield(fMRI_data, 'baseline') == 1
                errordlg('Please load the raw data (fMRI) without filtering.');
                return;
            end
        end
    else % from .img and .nii format
        fMRI_data.bold = zeros(nscan_b, nch);
        try
            ch_vx = preproc_info.ch_vx;
        catch
            ch_vx = inv(preproc_info.wT1_info.mat) * preproc_info.ch_MNI_mm;
        end
        h_wait = waitbar(0, 'Reading fMRI time-series...');
        for kk = 1:nscan_b
            D = spm_vol(path_file_b(kk,:));
            fMRI_data.bold(kk,:) = spm_get_data(D, ch_vx);
            waitbar(kk/nscan_b);
        end
        close(h_wait);
    end
    fMRI_data.fname_bold = path_file_b; % fMRI file name
    fMRI_data.fname_ch = path_file_ch; % channel file name
    fMRI_data.RT = RT; % interscan interval
end

% ==
time = linspace(0, size(nirs_data.oxyData,1)./nirs_data.fs, size(nirs_data.oxyData,1));
ch = 1;
try
    total_Hb = nirs_data.tHbData;
catch
    total_Hb = nirs_data.oxyData + nirs_data.dxyData;
end

try
    handles = rmfield(handles, 'nirs_data');
    handles = rmfield(handles, 'total_Hb');
end
try
    handles = rmfield(handles, 'fMRI_data');
end

try
    handles = rmfield(handles, 'fnirs_data');
    handles = rmfield(handles, 'ftotal_Hb');
end

try
    handles = rmfield(handles, 'rfMRI_data');
end

try
    handles = rmfield(handles, 'SPM');
end

try
    handles = rmfield(handles, 'index_start');
    handles = rmfield(handles, 'index_end');
end

try
    handles = rmfield(handles, 'limit_xaxis');
    handles = rmfield(handles, 'limit_yaxis');
end

try
    handles = rmfield(handles, 'limit_xaxis_filt');
    handles = rmfield(handles, 'limit_yaxis_filt');
end

% plot the time series of BOLD
axes(handles.axes_nirs_timeseries);
hold off;

if strcmp(str_load, 'NIRS&fMRI') == 1 && ismember('fMRI_data', who) == 1
    % if both NIRS and fMRI exist, default of display mode is normalization
    time_b = linspace(0, size(fMRI_data.bold, 1).*fMRI_data.RT, size(fMRI_data.bold,1));
    plot(time_b, fMRI_data.bold(:,ch)./max(fMRI_data.bold(:,ch)), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
    hold on
    plot(time, nirs_data.oxyData(:,ch)./max(nirs_data.oxyData(:,ch)), 'r');
    plot(time, nirs_data.dxyData(:,ch)./max(nirs_data.dxyData(:,ch)), 'b');
    plot(time, total_Hb(:,ch)./max(total_Hb(:,ch)), 'color', [0 127/255 0]);
    set(handles.checkbox_BOLD_raw, 'value', 1, 'enable', 'on');
    set(handles.checkbox_norm_raw, 'value', 1,'enable', 'on');
    set(handles.push_filter_BOLD, 'enable', 'on');
    set(handles.checkbox_ROI_BOLD, 'enable', 'on', 'value', 0);
    handles.fMRI_data = fMRI_data;
    handles.time_b = time_b;
else % plot the NIRS time series
    plot(time, nirs_data.oxyData(:,ch), 'r');
    hold on;
    plot(time, nirs_data.dxyData(:,ch), 'b');
    plot(time, total_Hb(:, ch), 'color', [0 127/255 0]);
    set(handles.checkbox_BOLD_raw, 'value', 0, 'enable', 'off');
    set(handles.checkbox_norm_raw, 'value', 0, 'enable', 'on');
%     limit_yaxis = [min([min(nirs_data.oxyData(:)) min(nirs_data.dxyData(:)) min(total_Hb(:))]), ...
%         max([max(nirs_data.oxyData(:)) max(nirs_data.dxyData(:)) max(total_Hb(:))])];
end
axes(handles.axes_nirs_timeseries);
default_axis = axis;
axis([min(time) max(time) default_axis(3:4)]);
xlabel('Time (s)');

% set the graphic property
set(handles.axes_nirs_timeseries, 'FontSize', 9);
set(handles.checkbox_HbO, 'value', 1, 'enable', 'on');
set(handles.checkbox_HbR, 'value', 1, 'enable', 'on');
set(handles.checkbox_HbT, 'value', 1, 'enable', 'on');
set(handles.edit_raw_ch, 'string', ['1 / ' num2str(nirs_data.nch)]);
set(handles.slider_raw_ch, 'sliderstep', [1/(nirs_data.nch-1), 1/(nirs_data.nch-1)], 'max', nirs_data.nch, 'min', 1, 'value', 1);
set(handles.slider_raw_ch, 'enable', 'on');
set(handles.slider_process_ch, 'enable', 'off');
set(handles.edit_process_ch, 'string', '');

set(handles.push_filtering,'enable','on');
set(handles.push_model_response, 'enable', 'on')
set(handles.popup_num_model, 'enable', 'off');
set(handles.checkbox_model, 'enable', 'off', 'value', 0);
set(handles.push_ROI_analysis,'enable','on');
set(handles.checkbox_ROI_HbO,'enable','on', 'value', 0);
set(handles.checkbox_ROI_HbR,'enable','on', 'value', 0);
set(handles.checkbox_ROI_HbT,'enable','on', 'value', 0);
set(handles.checkbox_axislimit_raw, 'enable', 'on', 'value', 0);
set(handles.text25, 'enable', 'on');

set(handles.checkbox_process_HbO, 'value', 0, 'enable', 'off');
set(handles.checkbox_process_HbR, 'value', 0, 'enable', 'off');
set(handles.checkbox_process_HbT, 'value', 0, 'enable', 'off');
set(handles.checkbox_BOLD_filt, 'value', 0, 'enable', 'off');
set(handles.checkbox_axislimit_filt, 'value', 0, 'enable','off');
set(handles.listbox_NIRS_info, 'string', {'NIRS processing info.'}, 'value', 1);
set(handles.listbox_fMRI_info, 'string', {'fMRI processing info.'}, 'value', 1);
cla(handles.axes_result, 'reset');

handles.nirs_data = nirs_data;
handles.total_Hb = total_Hb;
handles.time = time;

guidata(hObject, handles);

% --- Executes on slider movement.
function slider_raw_ch_Callback(hObject, eventdata, handles)
if isfield(handles, 'fMRI_data') == 0 && get(handles.checkbox_HbO, 'value') == 0 && get(handles.checkbox_HbR, 'value') == 0 && get(handles.checkbox_HbT, 'value') == 0
    cla(handles.axes_nirs_timeseries, 'reset');
    helpdlg('Please select the hemoglobin type to be displayed.');
    return;
elseif isfield(handles, 'fMRI_data') == 1 && get(handles.checkbox_HbO, 'value') == 0 && get(handles.checkbox_HbR, 'value') == 0 && get(handles.checkbox_HbT, 'value') == 0 && get(handles.checkbox_BOLD_raw,'value') ==0
    cla(handles.axes_nirs_timeseries, 'reset');
    helpdlg('Please select the hemoglobin type or BOLD to be displayed.');
    return;
end


% load the nirs_data
if get(handles.checkbox_HbO, 'value') == 1 || get(handles.checkbox_HbR, 'value') == 1 || get(handles.checkbox_HbT, 'value') == 1
    try % exist?
        nirs_data = handles.nirs_data;
        total_Hb = handles.total_Hb;
    catch % not exist?
        set(handles.checkbox_HbO, 'enable', 'off', 'value', 0);
        set(handles.checkbox_HbR, 'enable', 'off', 'value', 0);
        set(handles.checkbox_HbT, 'enable', 'off', 'value', 0);
        errordlg('NIRS data does not exist. Please load the NIRS data.');
        return;
    end
end

% load the fMRI data
if get(handles.checkbox_BOLD_raw, 'value') == 1
    try % exist?
        fMRI_data = handles.fMRI_data;
    catch
        set(handles.checkbox_BOLD_raw, 'enable', 'off', 'value', 0);
        errordlg('fMRI_data does not exist. Please load the fMRI data.');
        return;
    end
end

% obtain the number of channels from NIRS or BOLD data format
try
    nch = size(nirs_data.oxyData,2);
catch
    nch = size(fMRI_data.bold, 2);
end

h = round(get(handles.slider_raw_ch, 'value'));
set(handles.slider_raw_ch, 'value', h);

time = handles.time;
set(handles.edit_raw_ch, 'string', [num2str(h) ' / ' num2str(nch)]);

axes(handles.axes_nirs_timeseries);
hold off

if get(handles.checkbox_axislimit_raw, 'value') == 1
    limit_xaxis = handles.limit_xaxis;
    limit_yaxis = handles.limit_yaxis;

    axes(handles.axes_nirs_timeseries);
    default_axis = axis;
    if isempty(limit_xaxis) == 0
        default_axis(1:2) = limit_xaxis(1:2);
    end
    if isempty(limit_yaxis) == 0
        default_axis(3:4) = limit_yaxis(1:2);
    end
end

% plot the normalized signals ([0 1])
if get(handles.checkbox_norm_raw, 'value') == 1
    if get(handles.checkbox_HbO, 'value') == 1
        plot(time, nirs_data.oxyData(:,h)./max(nirs_data.oxyData(:,h)),'r');
        hold on
    end
    if get(handles.checkbox_HbR, 'value') == 1
        plot(time, nirs_data.dxyData(:,h)./max(nirs_data.dxyData(:,h)), 'b');
        hold on
    end
    if get(handles.checkbox_HbT, 'value') == 1
        plot(time, total_Hb(:,h)./max(total_Hb(:,h)), 'color', [0 127/255 0]);
        hold on
    end
    if get(handles.checkbox_BOLD_raw, 'value') == 1
        plot(handles.time_b, fMRI_data.bold(:,h)./max(fMRI_data.bold(:,h)), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
elseif get(handles.checkbox_norm_raw, 'value') == 0
    if get(handles.checkbox_HbO, 'value') == 1
        plot(time, nirs_data.oxyData(:,h), 'r');
        hold on
    end
    if get(handles.checkbox_HbR, 'value') == 1
        plot(time, nirs_data.dxyData(:,h), 'b');
        hold on
    end
    if get(handles.checkbox_HbT, 'value') == 1
        plot(time, total_Hb(:,h), 'color', [0 127/255 0]);
        hold on
    end
    if get(handles.checkbox_BOLD_raw, 'value') == 1
        plot(handles.time_b, fMRI_data.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
end
xlabel('Time (s)');
set(handles.axes_nirs_timeseries, 'FontSize', 9);

if get(handles.checkbox_axislimit_raw, 'value') == 0
    axes(handles.axes_nirs_timeseries);
    default_axis = axis;
    if ismember('time', who) == 1
        default_axis(1:2) = [min(time) max(time)];
    end
end
if get(handles.checkbox_model, 'value') == 1
    try
        num_model = get(handles.popup_num_model, 'value');
        scale_model = linspace(default_axis(3), default_axis(4), 6);
        min_model = min(handles.SPM.xX.X(:,num_model));
        max_model = max(handles.SPM.xX.X(:,num_model));
        model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
        hold on
        plot(time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
    catch
        set(handles.checkbox_model, 'value', 0);
        helpdlg('The predicted model response does not exist.');
    end
end
axis(default_axis);
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function slider_raw_ch_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in push_model_response.
function push_model_response_Callback(hObject, eventdata, handles)
% obtain the predicted response
[SPM] = spm_nirs_design_final(handles.fname_nirs, 1, 'model_generation');

for kk = 1:size(SPM.xX.X,2) -1
    str_model{kk} = num2str(kk);
end
% for raw_data
axes(handles.axes_nirs_timeseries);
default_axis = axis;
scale_model = linspace(default_axis(3), default_axis(4), 6);
min_model = min(SPM.xX.X(:,1));
max_model = max(SPM.xX.X(:,1));
model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (SPM.xX.X(:,1) - min_model) + scale_model(3);
hold on
plot(handles.time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
set(handles.checkbox_model, 'enable', 'on', 'value', 1);
set(handles.popup_num_model, 'enable', 'on', 'string', str_model, 'value', 1);

% for filtered_data
if isfield(handles, 'fnirs_data') == 1 || isfield(handles, 'rfMRI_data') == 1 % if the filtered data exists
    axes(handles.axes_result);
    default_axis = axis;
    scale_model = linspace(default_axis(3), default_axis(4), 6);
    model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (SPM.xX.X(:,1) - min_model) + scale_model(3);
    hold on
    plot(handles.time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
end
handles.SPM = SPM;
guidata(hObject, handles);

% --- Executes on button press in checkbox_model.
function checkbox_model_Callback(hObject, eventdata, handles)

if get(handles.checkbox_model, 'value') == 1
    % for raw_data
    num_model = get(handles.popup_num_model, 'value');
    axes(handles.axes_nirs_timeseries);
    default_axis = axis;
    scale_model = linspace(default_axis(3), default_axis(4), 6);

    min_model = min(handles.SPM.xX.X(:,num_model));
    max_model = max(handles.SPM.xX.X(:,num_model));
    model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
    hold on
    plot(handles.time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);

    % for filtered_data
    if isfield(handles, 'fnirs_data') == 1 || isfield(handles, 'rfMRI_data') == 1 % if the filtered data exists
        axes(handles.axes_result);
        default_axis = axis;
        scale_model = linspace(default_axis(3), default_axis(4), 6);
        model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
        hold on
        plot(handles.time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
    end
elseif get(handles.checkbox_model, 'value') == 0
    h = get(handles.slider_raw_ch, 'value');
    time = handles.time;
    nirs_data = handles.nirs_data;
    total_Hb = handles.total_Hb;

    axes(handles.axes_nirs_timeseries);
    hold off

    if get(handles.checkbox_axislimit_raw, 'value') == 1
        limit_xaxis = handles.limit_xaxis;
        limit_yaxis = handles.limit_yaxis;

        axes(handles.axes_nirs_timeseries);
        default_axis = axis;
        if isempty(limit_xaxis) == 0
            default_axis(1:2) = limit_xaxis(1:2);
        end
        if isempty(limit_yaxis) == 0
            default_axis(3:4) = limit_yaxis(1:2);
        end
    end

    % plot the normalized signals ([0 1])
    if get(handles.checkbox_norm_raw, 'value') == 1
        if get(handles.checkbox_HbO, 'value') == 1
            plot(time, nirs_data.oxyData(:,h)./max(nirs_data.oxyData(:,h)),'r');
            hold on
        end
        if get(handles.checkbox_HbR, 'value') == 1
            plot(time, nirs_data.dxyData(:,h)./max(nirs_data.dxyData(:,h)), 'b');
            hold on
        end
        if get(handles.checkbox_HbT, 'value') == 1
            plot(time, total_Hb(:,h)./max(total_Hb(:,h)), 'color', [0 127/255 0]);
            hold on
        end
        if get(handles.checkbox_BOLD_raw, 'value') == 1
            plot(handles.time_b, handles.fMRI_data.bold(:,h)./max(handles.fMRI_data.bold(:,h)), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
            hold on
        end
    elseif get(handles.checkbox_norm_raw, 'value') == 0
        if get(handles.checkbox_HbO, 'value') == 1
            plot(time, nirs_data.oxyData(:,h), 'r');
            hold on
        end
        if get(handles.checkbox_HbR, 'value') == 1
            plot(time, nirs_data.dxyData(:,h), 'b');
            hold on
        end
        if get(handles.checkbox_HbT, 'value') == 1
            plot(time, total_Hb(:,h), 'color', [0 127/255 0]);
            hold on
        end
        if get(handles.checkbox_BOLD_raw, 'value') == 1
            plot(handles.time_b, handles.fMRI_data.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
            hold on
        end
    end
    xlabel('Time (s)');
    set(handles.axes_nirs_timeseries, 'FontSize', 9);

    if get(handles.checkbox_axislimit_raw, 'value') == 0
        axes(handles.axes_nirs_timeseries);
        default_axis = axis;
        if ismember('time', who) == 1
            default_axis(1:2) = [min(time) max(time)];
        end
    end
    axis(default_axis);

    % plot the filtered response
    if isfield(handles, 'fnirs_data') == 1 || isfield(handles, 'ftotal_Hb') == 1 || isfield(handles, 'rfMRI_data') == 1
%         fnirs_data = handles.fnirs_data;
%         ftotal_Hb = handles.ftotal_Hb;

        h = get(handles.slider_process_ch, 'value');
        axes(handles.axes_result);
        hold off

        if get(handles.checkbox_axislimit_filt, 'value') == 1
            limit_xaxis = handles.limit_xaxis_filt;
            limit_yaxis = handles.limit_yaxis_filt;

            axes(handles.axes_result);
            default_axis = axis;

            if isempty(limit_xaxis) == 0
                default_axis(1:2) = limit_xaxis(1:2);
            end
            if isempty(limit_yaxis) == 0
                default_axis(3:4) = limit_yaxis(1:2);
            end
        end

        if get(handles.checkbox_process_HbO, 'value') == 1
            plot(time, handles.fnirs_data.oxyData(:, h), 'r');
            hold on
        end

        if get(handles.checkbox_process_HbR, 'value') == 1
            plot(time, handles.fnirs_data.dxyData(:,h), 'b');
            hold on;
        end

        if get(handles.checkbox_process_HbT, 'value') == 1
            plot(time, handles.ftotal_Hb(:,h), 'color', [0 127/255 0]);
            hold on;
        end

        if get(handles.checkbox_BOLD_filt, 'value') == 1
            plot(handles.time_b, handles.rfMRI_data.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
            hold on
        end
        xlabel('Time (s)');
        set(handles.axes_result, 'FontSize', 9);

        if get(handles.checkbox_axislimit_filt, 'value') == 0
            axes(handles.axes_result);
            default_axis = axis;
            if ismember('time', who) == 1
                default_axis(1:2) = [min(time) max(time)];
            end
        end
        axis(default_axis);
    end
end
% --- Executes on selection change in popup_num_model.
function popup_num_model_Callback(hObject, eventdata, handles)

% for raw data
h = get(handles.slider_raw_ch, 'value');
time = handles.time;
nirs_data = handles.nirs_data;
total_Hb = handles.total_Hb;

axes(handles.axes_nirs_timeseries);
hold off
if get(handles.checkbox_axislimit_raw, 'value') == 1
    limit_xaxis = handles.limit_xaxis;
    limit_yaxis = handles.limit_yaxis;

    axes(handles.axes_nirs_timeseries);
    default_axis = axis;
    if isempty(limit_xaxis) == 0
        default_axis(1:2) = limit_xaxis(1:2);
    end
    if isempty(limit_yaxis) == 0
        default_axis(3:4) = limit_yaxis(1:2);
    end
end

% plot the normalized signals ([0 1])
if get(handles.checkbox_norm_raw, 'value') == 1
    if get(handles.checkbox_HbO, 'value') == 1
        plot(time, nirs_data.oxyData(:,h)./max(nirs_data.oxyData(:,h)),'r');
        hold on
    end
    if get(handles.checkbox_HbR, 'value') == 1
        plot(time, nirs_data.dxyData(:,h)./max(nirs_data.dxyData(:,h)), 'b');
        hold on
    end
    if get(handles.checkbox_HbT, 'value') == 1
        plot(time, total_Hb(:,h)./max(total_Hb(:,h)), 'color', [0 127/255 0]);
        hold on
    end
    if get(handles.checkbox_BOLD_raw, 'value') == 1
        plot(handles.time_b, handles.fMRI_data.bold(:,h)./max(handles.fMRI_data.bold(:,h)), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
elseif get(handles.checkbox_norm_raw, 'value') == 0
    if get(handles.checkbox_HbO, 'value') == 1
        plot(time, nirs_data.oxyData(:,h), 'r');
        hold on
    end
    if get(handles.checkbox_HbR, 'value') == 1
        plot(time, nirs_data.dxyData(:,h), 'b');
        hold on
    end
    if get(handles.checkbox_HbT, 'value') == 1
        plot(time, total_Hb(:,h), 'color', [0 127/255 0]);
        hold on
    end
    if get(handles.checkbox_BOLD_raw, 'value') == 1
        plot(handles.time_b, handles.fMRI_data.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
end
xlabel('Time (s)');
set(handles.axes_nirs_timeseries, 'FontSize', 9);
if get(handles.checkbox_axislimit_raw, 'value') == 0
    axes(handles.axes_nirs_timeseries);
    default_axis = axis;
    if ismember('time', who) == 1
        default_axis(1:2) = [min(time) max(time)];
    end
end
axis(default_axis);

num_model = get(handles.popup_num_model, 'value');
scale_model = linspace(default_axis(3), default_axis(4), 6);
min_model = min(handles.SPM.xX.X(:,num_model));
max_model = max(handles.SPM.xX.X(:,num_model));
model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
hold on
plot(time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);

% plot the filtered response
if isfield(handles, 'fnirs_data') == 1 || isfield(handles, 'ftotal_Hb') == 1 || isfield(handles, 'rfMRI_data') == 1
    fnirs_data = handles.fnirs_data;
    ftotal_Hb = handles.ftotal_Hb;

    h = get(handles.slider_process_ch, 'value');
    axes(handles.axes_result);
    hold off

    if get(handles.checkbox_axislimit_filt, 'value') == 1
        limit_xaxis = handles.limit_xaxis_filt;
        limit_yaxis = handles.limit_yaxis_filt;

        axes(handles.axes_result);
        default_axis = axis;

        if isempty(limit_xaxis) == 0
            default_axis(1:2) = limit_xaxis(1:2);
        end
        if isempty(limit_yaxis) == 0
            default_axis(3:4) = limit_yaxis(1:2);
        end
    end

    if get(handles.checkbox_process_HbO, 'value') == 1
        plot(time, fnirs_data.oxyData(:, h), 'r');
        hold on
    end

    if get(handles.checkbox_process_HbR, 'value') == 1
        plot(time, fnirs_data.dxyData(:,h), 'b');
        hold on;
    end

    if get(handles.checkbox_process_HbT, 'value') == 1
        plot(time, ftotal_Hb(:,h), 'color', [0 127/255 0]);
        hold on;
    end

    if get(handles.checkbox_BOLD_filt, 'value') == 1
        plot(handles.time_b, handles.rfMRI_data.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
    xlabel('Time (s)');
    set(handles.axes_result, 'FontSize', 9);

    if get(handles.checkbox_axislimit_filt, 'value') == 0
        axes(handles.axes_result);
        default_axis = axis;
        if ismember('time', who) == 1
            default_axis(1:2) = [min(time) max(time)];
        end
    end
    scale_model = linspace(default_axis(3), default_axis(4), 6);
    model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
    hold on
    plot(time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
    axis(default_axis);
end

set(handles.checkbox_model,'value', 1);



% --- Executes during object creation, after setting all properties.
function popup_num_model_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in push_ROI_analysis.
function push_ROI_analysis_Callback(hObject, eventdata, handles)
h1 = get(handles.checkbox_ROI_HbO, 'value');
h2 = get(handles.checkbox_ROI_HbR, 'value');
h3 = get(handles.checkbox_ROI_HbT, 'value');
h4 = get(handles.checkbox_ROI_BOLD, 'value');

if h1 == 0 && h2 == 0 && h3 == 0 && h4 == 0
    return;
end

str = 'Input dialog for the ROI analysis';
spm_input(str, 1, 'd');
if isfield(handles, 'fnirs_data') == 1 || isfield(handles, 'ftotal_Hb') == 1 || isfield(handles, 'rfMRI_data') == 1
    % if filtered data exists,
    str = 'specify the data';
    flag_data = spm_input(str, '+1', 'raw|processed');
elseif isfield(handles, 'fnirs_data') == 0
    flag_data = 'raw';
end
str = 'specify channels of interest';
ch_ROI = spm_input(str,2,'r',' ',[Inf 1]);
% str = 'specify design in';
% UNITS = spm_input(str,'+1','scans|secs');
UNITS = 'secs';
str = ['vector of onsets [secs]'];
ons = spm_input(str,4,'r',' ',[Inf 1]);
str = 'average duration [secs]';
input_dur = spm_input(str, 5,'r',' ',[Inf 1]);
tmp_dur = input_dur;
if length(input_dur) == 1
    input_dur = input_dur*ones(size(ons));
end
if sum(diff(input_dur)) ~= 0
    helpdlg('The length of duration should be same.');
    return;
end
switch flag_data
    case 'raw'
        data.oxyData = handles.nirs_data.oxyData;
        data.dxyData = handles.nirs_data.dxyData;
        data.tHbData = handles.total_Hb;
        try
            data.bold = handles.fMRI_data.bold;
            time_b = handles.time_b; % time vector for BOLD
        end
    case 'processed'
        data.oxyData = handles.fnirs_data.oxyData;
        data.dxyData = handles.fnirs_data.dxyData;
        data.tHbData = handles.ftotal_Hb;
        try
            data.bold = handles.rfMRI_data.bold;
            time_b = handles.time_b;
        end
end
time = handles.time; % time vector for NIRS

if max(ons + input_dur) > max(time)
    return;
end

flag_dur = 0;
diff_ons = diff(ons);
if sum(diff_ons - input_dur(1)) == 0
    flag_dur = 1;
    input_dur = input_dur./2;
end

if input_dur(1) > diff_ons(1)
    errordlg('The average duration should be shorter than difference of onsets');
    return;
end
ntask = length(ons);
% generating the boxcar function for BOLD
if h4 == 1
    tSPM.nscan = size(data.bold, 1);
    tSPM.xY.RT = handles.fMRI_data.RT;
    tSPM.xBF.T = 10;
    tSPM.xBF.T0 = 1;
    tSPM.xBF.dt = tSPM.xY.RT/tSPM.xBF.T;
    tSPM.xBF.UNITS = UNITS;
    tSPM.Sess.U.name = {'trial1'};
    tSPM.Sess.U.ons = ons;
    tSPM.Sess.U.dur = input_dur;
    U = nirs_spm_get_ons(tSPM, 1, 1);
    k   = tSPM.nscan(1);
    boxcar_bold = U.u([0:(k - 1)]*tSPM.xBF.T + tSPM.xBF.T0 + 32,:);
    clear tSPM;
    % averating start point & end point
    % for BOLD
    boxcar_bold = [0 boxcar_bold(:)' 0]';
    diff_tmp = diff(boxcar_bold);
    index_start = find(diff_tmp == 1);
    index_end = find(diff_tmp == -1)-1;
    if flag_dur == 1
        index_end = index_start + mean(diff(index_start)) - 1;
    end
    ROI_data.bold = mean(data.bold(:,ch_ROI), 2);
    sum_bold = zeros(index_end(1) - index_start(1) + 1, 1);
    for kk = 1:ntask
        sum_bold = sum_bold + ROI_data.bold(index_start(kk):index_end(kk),1);
    end
    ROI_data.bold = sum_bold./ntask;
    time_roi_b = linspace(0, tmp_dur, size(ROI_data.bold,1));
end
% generating the boxcar function for NIRS
if h1 == 1 || h2 == 1 || h3 == 1
    tSPM.nscan = size(data.oxyData, 1);
    tSPM.xY.RT = 1./handles.nirs_data.fs;
    tSPM.xBF.T = 10;
    tSPM.xBF.T0 = 1;
    tSPM.xBF.dt = tSPM.xY.RT/tSPM.xBF.T;
    tSPM.xBF.UNITS = UNITS;
    tSPM.Sess.U.name = {'trial1'};
    tSPM.Sess.U.ons = ons;
    tSPM.Sess.U.dur = input_dur;
    U = nirs_spm_get_ons(tSPM, 1, 1);
    k   = tSPM.nscan(1);
    boxcar_nirs = U.u([0:(k - 1)]*tSPM.xBF.T + tSPM.xBF.T0 + 32,:);
    clear tSPM;
    % averating start point & end point
    % for NIRS
    boxcar_nirs = [0 boxcar_nirs(:)' 0]';
    diff_tmp = diff(boxcar_nirs);
    index_start = find(diff_tmp == 1);
    index_end = find(diff_tmp == -1)-1;
    if flag_dur == 1
        index_end = index_start + mean(diff(index_start)) - 1;
    end
    ROI_data.oxyData = mean(data.oxyData(:,ch_ROI), 2);
    ROI_data.dxyData = mean(data.dxyData(:,ch_ROI), 2);
    ROI_data.tHbData = mean(data.tHbData(:,ch_ROI), 2);
    sum_oxy = zeros(index_end(1) - index_start(1) + 1, 1);
    sum_dxy = sum_oxy;
    sum_tHb = sum_oxy;
    for kk = 1:ntask
        sum_oxy = sum_oxy + ROI_data.oxyData(index_start(kk):index_end(kk),1);
        sum_dxy = sum_dxy + ROI_data.dxyData(index_start(kk):index_end(kk),1);
        sum_tHb = sum_tHb + ROI_data.tHbData(index_start(kk):index_end(kk),1);
    end
    ROI_data.oxyData = sum_oxy./ntask;
    ROI_data.dxyData = sum_dxy./ntask;
    ROI_data.tHbData = sum_tHb./ntask;
    time_roi_n = linspace(0, tmp_dur, size(ROI_data.oxyData,1));
end

h_ROI = figure('Name', 'Time Series within ROI', 'NumberTitle','off');
hold off;
count = 1;
if h1 == 1
    plot(time_roi_n, ROI_data.oxyData, 'r');
    strhb{count,1} = 'OxyHb';
    count = count + 1;
    hold on;
end
if h2 == 1
    plot(time_roi_n, ROI_data.dxyData, 'b');
    strhb{count,1} = 'DxyHb';
    count = count + 1;
    hold on;
end
if h3 == 1
    plot(time_roi_n, ROI_data.tHbData, 'color', [0 127/255 0]);
    strhb{count,1} = 'TotalHb';
    count = count + 1;
    hold on;
end
if h4 == 1
    plot(time_roi_b, ROI_data.bold, '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
    strhb{count,1} = 'BOLD';
    count = count + 1;
    hold on;
end
xlabel('Time (s)');
legend(strhb);
axis tight;


function edit_ROI_analysis_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_ROI_analysis_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function slider_process_ch_Callback(hObject, eventdata, handles)
h1 = get(handles.checkbox_process_HbO, 'value');
h2 = get(handles.checkbox_process_HbR, 'value');
h3 = get(handles.checkbox_process_HbT, 'value');
h4 = get(handles.checkbox_BOLD_filt, 'value');

if isfield(handles, 'fnirs_data') == 1 && isfield(handles, 'rfMRI_data') == 0 && h1 == 0 && h2 == 0 && h3 == 0
    cla(handles.axes_result, 'reset');
    helpdlg('Please select the hemoglobin type to be displayed.');
    return;
elseif isfield(handles, 'fnirs_data') == 0 && isfield(handles, 'rfMRI_data') == 1 && h4 == 0
    cla(handles.axes_result, 'reset');
    helpdlg('Please select the checkbox for BOLD.');
    return;
elseif isfield(handles, 'fnirs_data') == 1 && isfield(handles, 'rfMRI_data') == 1 && h1 == 0 && h2 == 0 && h3 == 0 && h4 == 0
    cla(handles.axes_result, 'reset');
    helpdlg('Please select the hemoglobin type or BOLD to be displayed.');
    return;
end

% load filtered nirs data
if h1 == 1 || h2 == 1 || h3 == 1
    try
        fnirs_data = handles.fnirs_data;
        ftotal_Hb = handles.ftotal_Hb;
    catch
        errordlg('The filtered NIRS data does not exist. Please perform filtering or baseline correction task first.');
        return;
    end
end

% load filtered fMRI data
if h4 == 1
    try
        rfMRI_data = handles.rfMRI_data;
        time_b = handles.time_b;
    catch
        errordlg('The filtered fMRI data does not exist. Please perform filtering or baseline correction task first.');
        return;
    end
end

% obtain the number of channels from NIRS or BOLD data format
try
    nch = size(fnirs_data.oxyData,2);
catch
    nch = size(rfMRI_data.bold, 2);
end

h = round(get(handles.slider_process_ch, 'value'));
set(handles.slider_process_ch, 'value', h);

time = handles.time;
set(handles.edit_process_ch, 'string', [num2str(h) ' / ' num2str(nch)]);

axes(handles.axes_result);
hold off

if get(handles.checkbox_axislimit_filt, 'value') == 1
    limit_xaxis = handles.limit_xaxis_filt;
    limit_yaxis = handles.limit_yaxis_filt;

    axes(handles.axes_result);
    default_axis = axis;

    if isempty(limit_xaxis) == 0
        default_axis(1:2) = limit_xaxis(1:2);
    end
    if isempty(limit_yaxis) == 0
        default_axis(3:4) = limit_yaxis(1:2);
    end
end

if h1 == 1
   plot(time, fnirs_data.oxyData(:, h), 'r');
   hold on
end

if h2 == 1
    plot(time, fnirs_data.dxyData(:,h), 'b');
    hold on;
end

if h3 == 1
    plot(time, ftotal_Hb(:,h), 'color', [0 127/255 0]);
    hold on;
end

if h4 == 1
    plot(time_b, rfMRI_data.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
    hold on
end
xlabel('Time (s)');
set(handles.axes_result, 'FontSize', 9);

if get(handles.checkbox_axislimit_filt, 'value') == 0
    axes(handles.axes_result);
    default_axis = axis;
    if ismember('time', who) == 1
        default_axis(1:2) = [min(time) max(time)];
    end
end
axis(default_axis);

if get(handles.checkbox_model, 'value') == 1
    try
        num_model = get(handles.popup_num_model, 'value');
        scale_model = linspace(default_axis(3), default_axis(4), 6);
         min_model = min(handles.SPM.xX.X(:,num_model));
         max_model = max(handles.SPM.xX.X(:,num_model));
         model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
        hold on
        plot(time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
    catch
        set(handles.checkbox_model, 'value', 0);
        helpdlg('The predicted model response does not exist.');
    end
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function slider_process_ch_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


function edit_process_ch_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_process_ch_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% --- Executes on button press in checkbox_process_HbO.
function checkbox_process_HbO_Callback(hObject, eventdata, handles)
h1 = get(handles.checkbox_process_HbO, 'value');
h2 = get(handles.checkbox_process_HbR, 'value');
h3 = get(handles.checkbox_process_HbT, 'value');
h4 = get(handles.checkbox_BOLD_filt, 'value');

if isfield(handles, 'fnirs_data') == 1 && isfield(handles, 'rfMRI_data') == 0 && h1 == 0 && h2 == 0 && h3 == 0
    cla(handles.axes_result, 'reset');
    helpdlg('Please select the hemoglobin type to be displayed.');
    return;
elseif isfield(handles, 'fnirs_data') == 0 && isfield(handles, 'rfMRI_data') == 1 && h4 == 0
    cla(handles.axes_result, 'reset');
    helpdlg('Please select the checkbox for BOLD.');
    return;
elseif isfield(handles, 'fnirs_data') == 1 && isfield(handles, 'rfMRI_data') == 1 && h1 == 0 && h2 == 0 && h3 == 0 && h4 == 0
    cla(handles.axes_result, 'reset');
    helpdlg('Please select the hemoglobin type or BOLD to be displayed.');
    return;
end

% load filtered nirs data
if h1 == 1 || h2 == 1 || h3 == 1
    try
        fnirs_data = handles.fnirs_data;
        ftotal_Hb = handles.ftotal_Hb;
    catch
        errordlg('The filtered NIRS data does not exist. Please perform filtering or baseline correction task first.');
        return;
    end
end

% load filtered fMRI data
if h4 == 1
    try
        rfMRI_data = handles.rfMRI_data;
        time_b = handles.time_b;
    catch
        errordlg('The filtered fMRI data does not exist. Please perform filtering or baseline correction task first.');
        return;
    end
end

% obtain the number of channels from NIRS or BOLD data format
try
    nch = size(fnirs_data.oxyData,2);
catch
    nch = size(rfMRI_data.bold, 2);
end

h = get(handles.slider_process_ch, 'value');

time = handles.time;
set(handles.edit_process_ch, 'string', [num2str(h) ' / ' num2str(nch)]);

axes(handles.axes_result);
hold off

if get(handles.checkbox_axislimit_filt, 'value') == 1
    limit_xaxis = handles.limit_xaxis_filt;
    limit_yaxis = handles.limit_yaxis_filt;

    axes(handles.axes_result);
    default_axis = axis;

    if isempty(limit_xaxis) == 0
        default_axis(1:2) = limit_xaxis(1:2);
    end
    if isempty(limit_yaxis) == 0
        default_axis(3:4) = limit_yaxis(1:2);
    end
end

if h1 == 1
   plot(time, fnirs_data.oxyData(:, h), 'r');
   hold on
end

if h2 == 1
    plot(time, fnirs_data.dxyData(:,h), 'b');
    hold on;
end

if h3 == 1
    plot(time, ftotal_Hb(:,h), 'color', [0 127/255 0]);
    hold on;
end

if h4 == 1
    plot(time_b, rfMRI_data.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
    hold on
end
xlabel('Time (s)');
set(handles.axes_result, 'FontSize', 9);

if get(handles.checkbox_axislimit_filt, 'value') == 0
    axes(handles.axes_result);
    default_axis = axis;
    if ismember('time', who) == 1
        default_axis(1:2) = [min(time) max(time)];
    end
end
if get(handles.checkbox_model, 'value') == 1
    try
        num_model = get(handles.popup_num_model, 'value');
        scale_model = linspace(default_axis(3), default_axis(4), 6);
        min_model = min(handles.SPM.xX.X(:,num_model));
        max_model = max(handles.SPM.xX.X(:,num_model));
        model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
        hold on
        plot(time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
    catch
        set(handles.checkbox_model, 'value', 0);
        helpdlg('The predicted model response does not exist.');
    end
end
axis(default_axis);

guidata(hObject, handles);



% --- Executes on button press in checkbox_process_HbR.
function checkbox_process_HbR_Callback(hObject, eventdata, handles)
h1 = get(handles.checkbox_process_HbO, 'value');
h2 = get(handles.checkbox_process_HbR, 'value');
h3 = get(handles.checkbox_process_HbT, 'value');
h4 = get(handles.checkbox_BOLD_filt, 'value');

if isfield(handles, 'fnirs_data') == 1 && isfield(handles, 'rfMRI_data') == 0 && h1 == 0 && h2 == 0 && h3 == 0
    cla(handles.axes_result, 'reset');
    helpdlg('Please select the hemoglobin type to be displayed.');
    return;
elseif isfield(handles, 'fnirs_data') == 0 && isfield(handles, 'rfMRI_data') == 1 && h4 == 0
    cla(handles.axes_result, 'reset');
    helpdlg('Please select the checkbox for BOLD.');
    return;
elseif isfield(handles, 'fnirs_data') == 1 && isfield(handles, 'rfMRI_data') == 1 && h1 == 0 && h2 == 0 && h3 == 0 && h4 == 0
    cla(handles.axes_result, 'reset');
    helpdlg('Please select the hemoglobin type or BOLD to be displayed.');
    return;
end

% load filtered nirs data
if h1 == 1 || h2 == 1 || h3 == 1
    try
        fnirs_data = handles.fnirs_data;
        ftotal_Hb = handles.ftotal_Hb;
    catch
        errordlg('The filtered NIRS data does not exist. Please perform filtering or baseline correction task first.');
        return;
    end
end

% load filtered fMRI data
if h4 == 1
    try
        rfMRI_data = handles.rfMRI_data;
        time_b = handles.time_b;
    catch
        errordlg('The filtered fMRI data does not exist. Please perform filtering or baseline correction task first.');
        return;
    end
end

% obtain the number of channels from NIRS or BOLD data format
try
    nch = size(fnirs_data.oxyData,2);
catch
    nch = size(rfMRI_data.bold, 2);
end

h = get(handles.slider_process_ch, 'value');

time = handles.time;
set(handles.edit_process_ch, 'string', [num2str(h) ' / ' num2str(nch)]);

axes(handles.axes_result);
hold off

if get(handles.checkbox_axislimit_filt, 'value') == 1
    limit_xaxis = handles.limit_xaxis_filt;
    limit_yaxis = handles.limit_yaxis_filt;

    axes(handles.axes_result);
    default_axis = axis;

    if isempty(limit_xaxis) == 0
        default_axis(1:2) = limit_xaxis(1:2);
    end
    if isempty(limit_yaxis) == 0
        default_axis(3:4) = limit_yaxis(1:2);
    end
end

if h1 == 1
   plot(time, fnirs_data.oxyData(:, h), 'r');
   hold on
end

if h2 == 1
    plot(time, fnirs_data.dxyData(:,h), 'b');
    hold on;
end

if h3 == 1
    plot(time, ftotal_Hb(:,h), 'color', [0 127/255 0]);
    hold on;
end

if h4 == 1
    plot(time_b, rfMRI_data.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
    hold on
end
xlabel('Time (s)');
set(handles.axes_result, 'FontSize', 9);

if get(handles.checkbox_axislimit_filt, 'value') == 0
    axes(handles.axes_result);
    default_axis = axis;
    if ismember('time', who) == 1
        default_axis(1:2) = [min(time) max(time)];
    end
end
axis(default_axis);
if get(handles.checkbox_model, 'value') == 1
    try
        num_model = get(handles.popup_num_model, 'value');
        scale_model = linspace(default_axis(3), default_axis(4), 6);
        min_model = min(handles.SPM.xX.X(:,num_model));
        max_model = max(handles.SPM.xX.X(:,num_model));
        model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
        hold on
        plot(time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
    catch
        set(handles.checkbox_model, 'value', 0);
        helpdlg('The predicted model response does not exist.');
    end
end
guidata(hObject, handles);

% --- Executes on button press in checkbox_process_HbT.
function checkbox_process_HbT_Callback(hObject, eventdata, handles)
h1 = get(handles.checkbox_process_HbO, 'value');
h2 = get(handles.checkbox_process_HbR, 'value');
h3 = get(handles.checkbox_process_HbT, 'value');
h4 = get(handles.checkbox_BOLD_filt, 'value');

if isfield(handles, 'fnirs_data') == 1 && isfield(handles, 'rfMRI_data') == 0 && h1 == 0 && h2 == 0 && h3 == 0
    cla(handles.axes_result, 'reset');
    helpdlg('Please select the hemoglobin type to be displayed.');
    return;
elseif isfield(handles, 'fnirs_data') == 0 && isfield(handles, 'rfMRI_data') == 1 && h4 == 0
    cla(handles.axes_result, 'reset');
    helpdlg('Please select the checkbox for BOLD.');
    return;
elseif isfield(handles, 'fnirs_data') == 1 && isfield(handles, 'rfMRI_data') == 1 && h1 == 0 && h2 == 0 && h3 == 0 && h4 == 0
    cla(handles.axes_result, 'reset');
    helpdlg('Please select the hemoglobin type or BOLD to be displayed.');
    return;
end

% load filtered nirs data
if h1 == 1 || h2 == 1 || h3 == 1
    try
        fnirs_data = handles.fnirs_data;
        ftotal_Hb = handles.ftotal_Hb;
    catch
        errordlg('The filtered NIRS data does not exist. Please perform filtering or baseline correction task first.');
        return;
    end
end

% load filtered fMRI data
if h4 == 1
    try
        rfMRI_data = handles.rfMRI_data;
        time_b = handles.time_b;
    catch
        errordlg('The filtered fMRI data does not exist. Please perform filtering or baseline correction task first.');
        return;
    end
end

% obtain the number of channels from NIRS or BOLD data format
try
    nch = size(fnirs_data.oxyData,2);
catch
    nch = size(rfMRI_data.bold, 2);
end

h = get(handles.slider_process_ch, 'value');

time = handles.time;
set(handles.edit_process_ch, 'string', [num2str(h) ' / ' num2str(nch)]);

axes(handles.axes_result);
hold off

if get(handles.checkbox_axislimit_filt, 'value') == 1
    limit_xaxis = handles.limit_xaxis_filt;
    limit_yaxis = handles.limit_yaxis_filt;

    axes(handles.axes_result);

    default_axis = axis;

    if isempty(limit_xaxis) == 0
        default_axis(1:2) = limit_xaxis(1:2);
    end
    if isempty(limit_yaxis) == 0
        default_axis(3:4) = limit_yaxis(1:2);
    end
end

if h1 == 1
   plot(time, fnirs_data.oxyData(:, h), 'r');
   hold on
end

if h2 == 1
    plot(time, fnirs_data.dxyData(:,h), 'b');
    hold on;
end

if h3 == 1
    plot(time, ftotal_Hb(:,h), 'color', [0 127/255 0]);
    hold on;
end

if h4 == 1
    plot(time_b, rfMRI_data.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
    hold on
end
xlabel('Time (s)');
set(handles.axes_result, 'FontSize', 9);

if get(handles.checkbox_axislimit_filt, 'value') == 0
    axes(handles.axes_result);
    default_axis = axis;
    if ismember('time', who) == 1
        default_axis(1:2) = [min(time) max(time)];
    end
end
axis(default_axis);
if get(handles.checkbox_model, 'value') == 1
    try
        num_model = get(handles.popup_num_model, 'value');
        scale_model = linspace(default_axis(3), default_axis(4), 6);
        min_model = min(handles.SPM.xX.X(:,num_model));
        max_model = max(handles.SPM.xX.X(:,num_model));
        model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
        hold on
        plot(time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
    catch
        set(handles.checkbox_model, 'value', 0);
        helpdlg('The predicted model response does not exist.');
    end
end

guidata(hObject, handles);

% --- Executes on button press in push_filtering.
function push_filtering_Callback(hObject, eventdata, handles)
% read the raw data
try
    nirs_data = handles.nirs_data;
    total_Hb = handles.total_Hb;
    time= handles.time;

    % initialize the filtered nirs data as the raw data
    fnirs_data = nirs_data;
    ftotal_Hb = total_Hb;
catch
    errordlg('NIRS data does not exist. Please load the NIRS data first');
    return;
end

spm_input('Abbreviations: F, filtering; B, baseline correction',1, 'd');
% select processing order; baseline correction first? or filtering first?
str = 'Choose the option';

str_order = {'F', 'B', 'B - F', 'F - B'};
str_order = spm_input(str, '+1', 'b', str_order);

switch str_order
    case 'F'
        str = 'Filtering:';
    case 'B'
        str = 'Baseline correction:';
    case 'B - F'
        str = 'Baseline correction and then filtering:';
    case 'F - B'
        str = 'Filtering and then baseline correction:';
end
spm_input(str, '+1', 'd');

str_listbox = {};
% specification of baseline-correction
if strcmp(str_order, 'B') == 1 || strcmp(str_order, 'B - F') == 1 || strcmp(str_order, 'F - B') == 1
    str = 'specify the baseline';
    base = spm_input(str, '+2', 'initial time|time-range');
    switch base
        case 'time-range'
            str = 'start time of baseline [secs]';
            start_time = spm_input(str,'+1','r',' ',1);
            str = 'end time of baseline [secs]';
            end_time = spm_input(str,'+1','r',' ',1);
            diff_tmp = abs(time - start_time);
            index_start = find(diff_tmp == min(diff_tmp));
            diff_tmp = abs(time - end_time);
            index_end = find(diff_tmp == min(diff_tmp));
            str_baseline = [base '[s]:  [' num2str(start_time) '  ' num2str(end_time) ']'];
            % save the start time and end time of baseline correction
            fnirs_data.baseline.start_time = start_time;
            fnirs_data.baseline.end_time = end_time;
            fnirs_data.baseline.index_start = index_start;
            fnirs_data.baseline.index_end = index_end;
        case 'initial time'
            index_start = 1;
            index_end = 1;
            str_baseline = base;
    end
    handles.base = base; % initial time or time-range?
    fnirs_data.baseline.type = base;
    ntime = size(fnirs_data.oxyData, 1);
    str_listbox = [str_listbox {'Baseline correction:'} {str_baseline}];
end

% specification of filtering
if strcmp(str_order, 'F') == 1 || strcmp(str_order, 'F - B') == 1 || strcmp(str_order, 'B - F') == 1
    str = 'Input for filtering';
    spm_input(str, '+2', 'd');
    str = 'Low-pass filter?';
    cL = {'none', 'Gaussian', 'hrf'};
    cL = spm_input(str, '+1', 'b', cL);
    fnirs_data.cL.type = cL;

    if strcmp(cL, 'Gaussian') == 1
        FWHM = spm_input('Gaussian FWHM [seconds]', '+1', 'r', '1.5', 1);
        fnirs_data.cL.FWHM = FWHM;
    end

    str = 'Detrending?';
    cH = {'none', 'Wavelet-MDL', 'DCT'};
    cH = spm_input(str, '+1', 'b', cH);
    fnirs_data.cH.type = cH;
    str_cH = 'Detrending: ';
    if strcmp(cH, 'DCT') == 1
        M = spm_input('High-pass filter cut off [sec]', '+1', 'r', '128', 1);
        str_cH = [str_cH cH '(cutoff freq., 1/' num2str(M) ')'];
        fnirs_data.cH.M = M;
    else % wavelet_MDL or none
        str_cH = [str_cH cH];
    end
    str_cL = 'LPF: ';
    RT = 1./nirs_data.fs;
    row = 1:size(nirs_data.oxyData,1);
    switch cL
        case 'none'
            KL = speye(size(nirs_data.oxyData, 1));
            str_cL = [str_cL cL];
        case 'hrf'
            k = length(row);
            h = spm_hrf(RT);
            h = [h; zeros(size(h))];
            g = abs(fft(h));
            h = real(ifft(g));
            h = fftshift(h)';
            n = length(h);
            d = [1:n] - n/2 -1;
            KL = spdiags(ones(k,1)*h, d, k,k);
            KL = spdiags(1./sum(KL')',0,k,k)*KL;
            str_cL = [str_cL cL];
        case 'Gaussian'
            k = length(row);
            sigma   = FWHM/RT;
            h       = round(4*sigma);
            h       = exp(-[-h:h].^2/(2*sigma^2));
            n       = length(h);
            d       = [1:n] - (n + 1)/2;
            if      n == 1, h = 1; end
            KL = spdiags(ones(k,1)*h, d, k,k);
            KL = spdiags(1./sum(KL')',0,k,k)*KL;
            str_cL = [str_cL cL '(' num2str(FWHM) ' sec)'];
    end

    % for detrending
    switch cH
        case 'none'
            biasM_oxy = 0;
            biasM_dxy = 0;
            biasM_tHb = 0;
        case 'Wavelet-MDL'
            try
                SPM = handles.SPM;
                X = SPM.xX.X;
                fnirs_data.cH.SPM = SPM;
            catch
                helpdlg('To use the wavelet-MDL detrending algorithm, the predictor model should be required. Please enter the model parameters as the following steps.');
                [SPM] = spm_nirs_design_final(handles.fname_nirs, 1, 'model_generation');
                for kk = 1:size(SPM.xX.X,2)-1
                    str_model{kk} = num2str(kk);
                end
                handles.SPM = SPM;
                fnirs_data.cH.SPM = SPM;
                X = SPM.xX.X;
                set(handles.checkbox_model, 'enable', 'on');
                set(handles.popup_num_model, 'enable', 'on', 'string', str_model, 'value', 1);
            end
        case 'DCT'
            k = length(row);
            n       = fix(2*(k*RT)/M + 1);
            X0      = spm_dctmtx(k,n);
            X0 = X0(:,2:end);
    end
    str_listbox = [str_listbox 'Filtering:' {str_cL} {str_cH}];
end
set(handles.listbox_NIRS_info, 'value', 1, 'string', str_listbox);

% application
if strcmp(str_order, 'F') == 1 || strcmp(str_order, 'F - B')
    % smoothing
    fnirs_data.oxyData = KL * nirs_data.oxyData;
    fnirs_data.dxyData = KL * nirs_data.dxyData;
    ftotal_Hb = KL * total_Hb;
    % detrending
    switch cH
        case 'Wavelet-MDL'
            % detrending for wavelet-MDL
            [biasM_oxy] = detrend_wavelet_MDL(fnirs_data.oxyData, X(:,1:end-1));
            [biasM_dxy] = detrend_wavelet_MDL(fnirs_data.dxyData, X(:,1:end-1).*(-1));
            [biasM_tHb] = detrend_wavelet_MDL(ftotal_Hb, X(:,1:end-1));
            fnirs_data.cH.bias_oxy = biasM_oxy;
            fnirs_data.cH.bias_dxy = biasM_dxy;
            fnirs_data.cH.bias_tHb = biasM_tHb;
        case 'DCT'
            % detrending for DCT
            biasM_oxy = X0 * (X0' * fnirs_data.oxyData);
            biasM_dxy = X0 * (X0' * fnirs_data.dxyData);
            biasM_tHb = X0 * (X0' * ftotal_Hb);
    end
    fnirs_data.oxyData = fnirs_data.oxyData - biasM_oxy;
    fnirs_data.dxyData = fnirs_data.dxyData - biasM_dxy;
    ftotal_Hb = ftotal_Hb - biasM_tHb;

    if strcmp(str_order, 'F - B') == 1
        % baseline correction
        fnirs_data.oxyData = fnirs_data.oxyData - ones(ntime,1)*mean(fnirs_data.oxyData(index_start:index_end,:),1);
        fnirs_data.dxyData = fnirs_data.dxyData - ones(ntime,1)*mean(fnirs_data.dxyData(index_start:index_end,:),1);
        ftotal_Hb = ftotal_Hb - ones(ntime,1) * mean(ftotal_Hb(index_start:index_end,:),1);
    end
end

if strcmp(str_order, 'B') == 1 || strcmp(str_order, 'B - F') == 1
    % baseline correction
    fnirs_data.oxyData = fnirs_data.oxyData - ones(ntime,1)*mean(fnirs_data.oxyData(index_start:index_end,:),1);
    fnirs_data.dxyData = fnirs_data.dxyData - ones(ntime,1)*mean(fnirs_data.dxyData(index_start:index_end,:),1);
    ftotal_Hb = ftotal_Hb - ones(ntime,1) * mean(ftotal_Hb(index_start:index_end,:),1);
    if strcmp(str_order, 'B - F') == 1
        % smoothing
        fnirs_data.oxyData = KL * nirs_data.oxyData;
        fnirs_data.dxyData = KL * nirs_data.dxyData;
        ftotal_Hb = KL * total_Hb;
        % detrending
        switch cH
            case 'Wavelet-MDL'
                % detrending for wavelet-MDL
                [biasM_oxy] = detrend_wavelet_MDL(fnirs_data.oxyData, X(:,1:end-1));
                [biasM_dxy] = detrend_wavelet_MDL(fnirs_data.dxyData, X(:,1:end-1).*(-1));
                [biasM_tHb] = detrend_wavelet_MDL(ftotal_Hb, X(:,1:end-1));
                fnirs_data.cH.bias_oxy = biasM_oxy;
                fnirs_data.cH.bias_dxy = biasM_dxy;
                fnirs_data.cH.bias_tHb = biasM_tHb;
            case 'DCT'
                % detrending for DCT
                biasM_oxy = X0 * (X0' * fnirs_data.oxyData);
                biasM_dxy = X0 * (X0' * fnirs_data.dxyData);
                biasM_tHb = X0 * (X0' * ftotal_Hb);
        end
        fnirs_data.oxyData = fnirs_data.oxyData - biasM_oxy;
        fnirs_data.dxyData = fnirs_data.dxyData - biasM_dxy;
        ftotal_Hb = ftotal_Hb - biasM_tHb;
    end
end

% save filtered data in the handle structure
handles.fnirs_data = fnirs_data;
handles.ftotal_Hb = ftotal_Hb;

% plot the time series of BOLD
axes(handles.axes_result);
hold off;
ch = 1;
nch = size(fnirs_data.oxyData,2);
if isfield(handles, 'rfMRI_data') == 1
    % if both filtered NIRS and fMRI exist, default of display mode is
    % normalization
    time_b = linspace(0, size(fnirs_data.oxyData,1)./fnirs_data.fs, size(handles.rfMRI_data.bold, 1));
    plot(time_b, handles.rfMRI_data.bold(:,ch), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
    hold on
    set(handles.checkbox_BOLD_filt, 'value', 1, 'enable', 'on');
end

plot(time, fnirs_data.oxyData(:,ch), 'r');
hold on;
plot(time, fnirs_data.dxyData(:,ch), 'b');
plot(time, ftotal_Hb(:, ch), 'color', [0 127/255 0]);

axes(handles.axes_result);
default_axis = axis;
axis([min(time) max(time) default_axis(3:4)]);
xlabel('Time (s)');

if get(handles.checkbox_model, 'value') == 1
    try
        num_model = get(handles.popup_num_model, 'value');
        scale_model = linspace(default_axis(3), default_axis(4), 6);
        min_model = min(handles.SPM.xX.X(:,num_model));
        max_model = max(handles.SPM.xX.X(:,num_model));
        model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
        hold on
        plot(time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
    catch
        set(handles.checkbox_model, 'value', 0);
        helpdlg('The predicted model response does not exist.');
    end
end

% set the graphic property
set(handles.axes_result, 'FontSize', 9);
set(handles.checkbox_process_HbO, 'enable', 'on', 'value', 1);
set(handles.checkbox_process_HbR, 'enable', 'on', 'value', 1);
set(handles.checkbox_process_HbT, 'enable', 'on', 'value', 1);
set(handles.edit_process_ch, 'string', ['1 / ' num2str(nch)]);
set(handles.slider_process_ch, 'enable', 'on');
set(handles.slider_process_ch, 'sliderstep', [1/(nch-1), 1/(nch-1)], 'max', nch, 'min', 1, 'value', 1);
set(handles.checkbox_axislimit_filt, 'enable', 'on', 'value', 0);
set(handles.push_save,'enable','on');

guidata(hObject, handles);

% --- Executes on button press in checkbox_ROI_HbO.
function checkbox_ROI_HbO_Callback(hObject, eventdata, handles)

% --- Executes on button press in checkbox_ROI_HbR.
function checkbox_ROI_HbR_Callback(hObject, eventdata, handles)

% --- Executes on button press in checkbox_ROI_HbT.
function checkbox_ROI_HbT_Callback(hObject, eventdata, handles)

% --- Executes on button press in push_save.
function push_save_Callback(hObject, eventdata, handles)
try
    nirs_data = handles.fnirs_data;
    nirs_data.tHbData = handles.ftotal_Hb;
    [filen, pathn] = uiputfile('*.mat','Save NIRS data as');
    path_filen = [pathn filen];
    if path_filen == 0
        return;
    end
    save(path_filen, 'nirs_data');
end

try
    fMRI_data = handles.rfMRI_data;
    fMRI_data.raw_data = fMRI_data.bold;
    [filen, pathn] = uiputfile('*.mat', 'Save fMRI data as');
    path_filen = [pathn filen];
    if path_filen == 0
        return;
    end
    save(path_filen, 'fMRI_data');
end


% --- Executes on selection change in listbox_fname.
function listbox_fname_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function listbox_fname_CreateFcn(hObject, eventdata, handles)
% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in checkbox_norm_raw.
function checkbox_norm_raw_Callback(hObject, eventdata, handles)
% manually specifying the limit of x-axis and y-axis

if get(handles.checkbox_axislimit_raw, 'value') == 1
    limit_xaxis = handles.limit_xaxis;
    limit_yaxis = handles.limit_yaxis;

    axes(handles.axes_nirs_timeseries);
    default_axis = axis;
    if isempty(limit_xaxis) == 0
        default_axis(1:2) = limit_xaxis(1:2);
    end
    if isempty(limit_yaxis) == 0
        default_axis(3:4) = limit_yaxis(1:2);
    end
end

% plot the time-series
% load the nirs_data
if get(handles.checkbox_HbO, 'value') == 1 || get(handles.checkbox_HbR, 'value') == 1 || get(handles.checkbox_HbT, 'value') == 1
    nirs_data = handles.nirs_data;
    total_Hb = handles.total_Hb;
end
% load the fMRI data
if get(handles.checkbox_BOLD_raw, 'value') == 1
    fMRI_data = handles.fMRI_data;
end

h = get(handles.slider_raw_ch, 'value');
time = handles.time;

axes(handles.axes_nirs_timeseries);
hold off
% plot the normalized signals ([0 1])
if get(handles.checkbox_norm_raw, 'value') == 1
    if get(handles.checkbox_HbO, 'value') == 1
        plot(time, nirs_data.oxyData(:,h)./max(nirs_data.oxyData(:,h)),'r');
        hold on
    end
    if get(handles.checkbox_HbR, 'value') == 1
        plot(time, nirs_data.dxyData(:,h)./max(nirs_data.dxyData(:,h)), 'b');
        hold on
    end
    if get(handles.checkbox_HbT, 'value') == 1
        plot(time, total_Hb(:,h)./max(total_Hb(:,h)), 'color', [0 127/255 0]);
        hold on
    end
    if get(handles.checkbox_BOLD_raw, 'value') == 1
        plot(handles.time_b, fMRI_data.bold(:,h)./max(fMRI_data.bold(:,h)), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
elseif get(handles.checkbox_norm_raw, 'value') == 0
    if get(handles.checkbox_HbO, 'value') == 1
        plot(time, nirs_data.oxyData(:,h), 'r');
        hold on
    end
    if get(handles.checkbox_HbR, 'value') == 1
        plot(time, nirs_data.dxyData(:,h), 'b');
        hold on
    end
    if get(handles.checkbox_HbT, 'value') == 1
        plot(time, total_Hb(:,h), 'color', [0 127/255 0]);
        hold on
    end
    if get(handles.checkbox_BOLD_raw, 'value') == 1
        plot(handles.time_b, fMRI_data.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
end
xlabel('Time (s)');
set(handles.axes_nirs_timeseries, 'FontSize', 9);

if get(handles.checkbox_axislimit_raw, 'value') == 0
    axes(handles.axes_nirs_timeseries);
    default_axis = axis;
    if ismember('time', who) == 1
        default_axis(1:2) = [min(time) max(time)];
    end
end
axis(default_axis);

if get(handles.checkbox_model, 'value') == 1
    try
        num_model = get(handles.popup_num_model, 'value');
        scale_model = linspace(default_axis(3), default_axis(4), 6);
        min_model = min(handles.SPM.xX.X(:,num_model));
        max_model = max(handles.SPM.xX.X(:,num_model));
        model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
        hold on
        plot(time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
    catch
        set(handles.checkbox_model, 'value', 0);
        helpdlg('The predicted model response does not exist.');
    end
end

guidata(hObject, handles);

% --- Executes on button press in checkbox_BOLD_filt.
function checkbox_BOLD_filt_Callback(hObject, eventdata, handles)
h1 = get(handles.checkbox_process_HbO, 'value');
h2 = get(handles.checkbox_process_HbR, 'value');
h3 = get(handles.checkbox_process_HbT, 'value');
h4 = get(handles.checkbox_BOLD_filt, 'value');

if isfield(handles, 'fnirs_data') == 1 && isfield(handles, 'rfMRI_data') == 0 && h1 == 0 && h2 == 0 && h3 == 0
    cla(handles.axes_result, 'reset');
    helpdlg('Please select the hemoglobin type to be displayed.');
    return;
elseif isfield(handles, 'fnirs_data') == 0 && isfield(handles, 'rfMRI_data') == 1 && h4 == 0
    cla(handles.axes_result, 'reset');
    helpdlg('Please select the checkbox for BOLD.');
    return;
elseif isfield(handles, 'fnirs_data') == 1 && isfield(handles, 'rfMRI_data') == 1 && h1 == 0 && h2 == 0 && h3 == 0 && h4 == 0
    cla(handles.axes_result, 'reset');
    helpdlg('Please select the hemoglobin type or BOLD to be displayed.');
    return;
end

% load filtered nirs data
if h1 == 1 || h2 == 1 || h3 == 1
    try
        fnirs_data = handles.fnirs_data;
        ftotal_Hb = handles.ftotal_Hb;
    catch
        errordlg('The filtered NIRS data does not exist. Please perform filtering or baseline correction task first.');
        return;
    end
end

% load filtered fMRI data
if h4 == 1
    try
        rfMRI_data = handles.rfMRI_data;
        time_b = handles.time_b;
    catch
        errordlg('The filtered fMRI data does not exist. Please perform filtering or baseline correction task first.');
        return;
    end
end

% obtain the number of channels from NIRS or BOLD data format
try
    nch = size(fnirs_data.oxyData,2);
catch
    nch = size(rfMRI_data.bold, 2);
end

h = get(handles.slider_process_ch, 'value');

time = handles.time;
set(handles.edit_process_ch, 'string', [num2str(h) ' / ' num2str(nch)]);

axes(handles.axes_result);
hold off

if get(handles.checkbox_axislimit_filt, 'value') == 1
    limit_xaxis = handles.limit_xaxis_filt;
    limit_yaxis = handles.limit_yaxis_filt;

    axes(handles.axes_result);
    default_axis = axis;

    if isempty(limit_xaxis) == 0
        default_axis(1:2) = limit_xaxis(1:2);
    end
    if isempty(limit_yaxis) == 0
        default_axis(3:4) = limit_yaxis(1:2);
    end
end

if h1 == 1
   plot(time, fnirs_data.oxyData(:, h), 'r');
   hold on
end

if h2 == 1
    plot(time, fnirs_data.dxyData(:,h), 'b');
    hold on;
end

if h3 == 1
    plot(time, ftotal_Hb(:,h), 'color', [0 127/255 0]);
    hold on;
end

if h4 == 1
    plot(time_b, rfMRI_data.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
    hold on
end
xlabel('Time (s)');
set(handles.axes_result, 'FontSize', 9);

if get(handles.checkbox_axislimit_filt, 'value') == 0
    axes(handles.axes_result);
    default_axis = axis;
    if ismember('time', who) == 1
        default_axis(1:2) = [min(time) max(time)];
    end
end
axis(default_axis);
if get(handles.checkbox_model, 'value') == 1
    try
        num_model = get(handles.popup_num_model, 'value');
        scale_model = linspace(default_axis(3), default_axis(4), 6);
        min_model = min(handles.SPM.xX.X(:,num_model));
        max_model = max(handles.SPM.xX.X(:,num_model));
        model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
        hold on
        plot(time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
    catch
        set(handles.checkbox_model, 'value', 0);
        helpdlg('The predicted model response does not exist.');
    end
end
guidata(hObject, handles);


% --- Executes on button press in checkbox_BOLD_raw.
function checkbox_BOLD_raw_Callback(hObject, eventdata, handles)
if isfield(handles, 'fMRI_data') == 0 && get(handles.checkbox_HbO, 'value') == 0 && get(handles.checkbox_HbR, 'value') == 0 && get(handles.checkbox_HbT, 'value') == 0
    cla(handles.axes_nirs_timeseries, 'reset');
    helpdlg('Please select the hemoglobin type to be displayed.');
    return;
elseif isfield(handles, 'fMRI_data') == 1 && get(handles.checkbox_HbO, 'value') == 0 && get(handles.checkbox_HbR, 'value') == 0 && get(handles.checkbox_HbT, 'value') == 0 && get(handles.checkbox_BOLD_raw,'value') ==0
    cla(handles.axes_nirs_timeseries, 'reset');
    helpdlg('Please select the hemoglobin type or BOLD to be displayed.');
    return;
end


% load the nirs_data
if get(handles.checkbox_HbO, 'value') == 1 || get(handles.checkbox_HbR, 'value') == 1 || get(handles.checkbox_HbT, 'value') == 1
    try % exist?
        nirs_data = handles.nirs_data;
        total_Hb = handles.total_Hb;
    catch % not exist?
        set(handles.checkbox_HbO, 'enable', 'off', 'value', 0);
        set(handles.checkbox_HbR, 'enable', 'off', 'value', 0);
        set(handles.checkbox_HbT, 'enable', 'off', 'value', 0);
        errordlg('NIRS data does not exist. Please load the NIRS data.');
        return;
    end
end

% load the fMRI data
if get(handles.checkbox_BOLD_raw, 'value') == 1
    try % exist?
        fMRI_data = handles.fMRI_data;
    catch
        set(handles.checkbox_BOLD_raw, 'enable', 'off', 'value', 0);
        errordlg('fMRI_data does not exist. Please load the fMRI data.');
        return;
    end
end

% obtain the number of channels from NIRS or BOLD data format
try
    nch = size(nirs_data.oxyData,2);
catch
    nch = size(fMRI_data.bold, 2);
end


h = get(handles.slider_raw_ch, 'value');

time = handles.time;
if get(handles.checkbox_model, 'value') == 1
    try
        X = handles.SPM.xX.X;
    catch
        set(handles.checkbox_model, 'value', 0);
        helpdlg('The predicted model response does not exist.');
    end
end

set(handles.edit_raw_ch, 'string', [num2str(h) ' / ' num2str(nch)]);

axes(handles.axes_nirs_timeseries);
hold off

if get(handles.checkbox_axislimit_raw, 'value') == 1
    limit_xaxis = handles.limit_xaxis;
    limit_yaxis = handles.limit_yaxis;

    axes(handles.axes_nirs_timeseries);
    default_axis = axis;
    if isempty(limit_xaxis) == 0
        default_axis(1:2) = limit_xaxis(1:2);
    end
    if isempty(limit_yaxis) == 0
        default_axis(3:4) = limit_yaxis(1:2);
    end
end

% plot the normalized signals ([0 1])
if get(handles.checkbox_norm_raw, 'value') == 1
    if get(handles.checkbox_HbO, 'value') == 1
        plot(time, nirs_data.oxyData(:,h)./max(nirs_data.oxyData(:,h)),'r');
        hold on
    end
    if get(handles.checkbox_HbR, 'value') == 1
        plot(time, nirs_data.dxyData(:,h)./max(nirs_data.dxyData(:,h)), 'b');
        hold on
    end
    if get(handles.checkbox_HbT, 'value') == 1
        plot(time, total_Hb(:,h)./max(total_Hb(:,h)), 'color', [0 127/255 0]);
        hold on
    end
    if get(handles.checkbox_BOLD_raw, 'value') == 1
        plot(handles.time_b, fMRI_data.bold(:,h)./max(fMRI_data.bold(:,h)), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
elseif get(handles.checkbox_norm_raw, 'value') == 0
    if get(handles.checkbox_HbO, 'value') == 1
        plot(time, nirs_data.oxyData(:,h), 'r');
        hold on
    end
    if get(handles.checkbox_HbR, 'value') == 1
        plot(time, nirs_data.dxyData(:,h), 'b');
        hold on
    end
    if get(handles.checkbox_HbT, 'value') == 1
        plot(time, total_Hb(:,h), 'color', [0 127/255 0]);
        hold on
    end
    if get(handles.checkbox_BOLD_raw, 'value') == 1
        plot(handles.time_b, fMRI_data.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
end
xlabel('Time (s)');
set(handles.axes_nirs_timeseries, 'FontSize', 9);

if get(handles.checkbox_axislimit_raw, 'value') == 0
    axes(handles.axes_nirs_timeseries);
    default_axis = axis;
    if ismember('time', who) == 1
        default_axis(1:2) = [min(time) max(time)];
    end
end

if get(handles.checkbox_model, 'value') == 1
    try
        num_model = get(handles.popup_num_model, 'value');
        scale_model = linspace(default_axis(3), default_axis(4), 6);
        min_model = min(handles.SPM.xX.X(:,num_model));
        max_model = max(handles.SPM.xX.X(:,num_model));
        model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
        hold on
        plot(time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
    catch
        set(handles.checkbox_model, 'value', 0);
        helpdlg('The predicted model response does not exist.');
    end
end

axis(default_axis);
guidata(hObject, handles);


% --- Executes on button press in push_filter_BOLD.
function push_filter_BOLD_Callback(hObject, eventdata, handles)
% read the raw data
try
    fMRI_data = handles.fMRI_data;
    time_b = handles.time_b;
    % initialize the filtered nirs data as the raw data
    rfMRI_data = fMRI_data;
    RT = fMRI_data.RT;
catch
    errordlg('fMRI data does not exist. Please load the fMRI data first');
    return;
end

spm_input('Abbreviations: F, filtering; B, fractional BOLD',1, 'd');
% select processing order; estimation of fractional BOLD changes first ? or
% filtering first?
str = 'Choose the option';
str_order = {'F', 'B', 'B - F', 'F - B'};
str_order = spm_input(str, '+1', 'b', str_order);

switch str_order
    case 'F'
        str = 'Filtering:';
    case 'B'
        str = 'Estimating fractional BOLD changes:';
    case 'B - F'
        str = 'Fractional BOLD changes first and then filtering:';
    case 'F - B'
        str = 'Filtering and then fractional BOLD changes:';
end
spm_input(str, '+1', 'd');

time_record = linspace(RT, max(handles.time), size(fMRI_data.bold, 1));

str_listbox = {};
% specification of baseline-correction
if strcmp(str_order, 'B') == 1 || strcmp(str_order, 'B - F') == 1 || strcmp(str_order, 'F - B') == 1
    str = 'specify the baseline';
    base = spm_input(str, '+2', 'initial time|time-range');
    switch base
        case 'time-range'
            str = 'start time of baseline [secs]';
            start_time = spm_input(str,'+1','r',' ',1);
            str = 'end time of baseline [secs]';
            end_time = spm_input(str,'+1','r',' ',1);
            diff_tmp = abs(time_record - start_time);
            index_start = find(diff_tmp == min(diff_tmp));
            diff_tmp = abs(time_record - end_time);
            index_end = find(diff_tmp == min(diff_tmp));
            str_baseline = [base '[s]:  [' num2str(start_time) '  ' num2str(end_time) ']'];
            rfMRI_data.baseline.start_time = start_time;
            rfMRI_data.baseline.end_time = end_time;
            rfMRI_data.baseline.index_start = index_start;
            rfMRI_data.baseline.index_end = index_end;
        case 'initial time'
            index_start = 1;
            index_end = 1;
            str_baseline = base;
    end
    str_listbox = [str_listbox {'Baseline correction:'} {str_baseline}];
    rfMRI_data.baseline.type = base;
end

% specification of filtering
if strcmp(str_order, 'F') == 1 || strcmp(str_order, 'F - B') == 1 || strcmp(str_order, 'B - F') == 1
    str = 'Input for filtering';
    spm_input(str, '+2', 'd');
    str = 'Low-pass filter?';
    cL = {'none', 'Gaussian', 'hrf'};
    cL = spm_input(str, '+1', 'b', cL);
    rfMRI_data.cL.type = cL;

    if strcmp(cL, 'Gaussian') == 1
        FWHM = spm_input('Gaussian FWHM [seconds]', '+1', 'r', '1.5', 1);
        rfMRI_data.cL.FWHM = FWHM;
    end

    str = 'Detrending?';
    cH = {'none', 'DCT'};
    cH = spm_input(str, '+1', 'b', cH);
    rfMRI_data.cH.type = cH;
    str_cH = 'Detrending: ';
    switch cH
        case 'DCT'
            M = spm_input('High-pass filter cut off [sec]', '+1', 'r', '128', 1);
            str_cH = [str_cH cH '(cutoff freq., 1/' num2str(M) ')'];
            rfMRI_data.cH.M = M;
        case 'none'
            str_cH = [str_cH cH];
    end
    str_cL = 'LPF: ';
    row = 1:size(fMRI_data.bold, 1);
    switch cL
        case 'none'
            KL = speye(size(fMRI_data.bold, 1));
            str_cL = [str_cL cL];
        case 'hrf'
            k = length(row);
            h = spm_hrf(RT);
            h = [h; zeros(size(h))];
            g = abs(fft(h));
            h = real(ifft(g));
            h = fftshift(h)';
            n = length(h);
            d = [1:n] - n/2 -1;
            KL = spdiags(ones(k,1)*h, d, k,k);
            KL = spdiags(1./sum(KL')',0,k,k)*KL;
            str_cL = [str_cL cL];
        case 'Gaussian'
            k = length(row);
            sigma   = FWHM/RT;
            h       = round(4*sigma);
            h       = exp(-[-h:h].^2/(2*sigma^2));
            n       = length(h);
            d       = [1:n] - (n + 1)/2;
            if      n == 1, h = 1; end
            KL = spdiags(ones(k,1)*h, d, k,k);
            KL = spdiags(1./sum(KL')',0,k,k)*KL;
            str_cL = [str_cL cL '(' num2str(FWHM) ' sec)'];
    end

    % for detrending
    switch cH
        case 'none'
            biasM_oxy = 0;
            biasM_dxy = 0;
            biasM_tHb = 0;
        case 'DCT'
            k = length(row);
            n       = fix(2*(k*RT)/M + 1);
            X0      = spm_dctmtx(k,n);
            X0 = X0(:,2:end);
    end
    str_listbox = [str_listbox {'Filtering:'} {str_cL} {str_cH}];
end
set(handles.listbox_fMRI_info, 'value', 1, 'string', str_listbox);

if strcmp(str_order, 'F') == 1 || strcmp(str_order, 'F - B')
    % smoothing
    rfMRI_data.bold = KL * fMRI_data.bold;
    % detrending
    if strcmp(cH, 'DCT') == 1
        rfMRI_data.bold = rfMRI_data.bold -  X0 * (X0' * rfMRI_data.bold);
    end
    if strcmp(str_order, 'F - B') == 1
        for kk = 1:size(rfMRI_data.bold, 2)
            rest_bold(kk) = mean(rfMRI_data.bold(index_start:index_end,kk));
            rfMRI_data.bold(:,kk) = (rfMRI_data.bold(:,kk) - rest_bold(kk))./rest_bold(kk);
            % rfMRI_data.bold : fractional BOLD changes
        end
        rfMRI_data.rest_bold = rest_bold;
    end
end

if strcmp(str_order, 'B') == 1 || strcmp(str_order, 'B - F') == 1
    for kk = 1:size(rfMRI_data.bold, 2)
        rest_bold(kk) = mean(rfMRI_data.bold(index_start:index_end,kk));
        rfMRI_data.bold(:,kk) = (rfMRI_data.bold(:,kk) - rest_bold(kk))./rest_bold(kk);
        % rfMRI_data.bold : fractional BOLD changes
    end
    rfMRI_data.rest_bold = rest_bold;
    if strcmp(str_order, 'B - F') == 1
        % smoothing
        rfMRI_data.bold = KL * rfMRI_data.bold;
        % detrending
        if strcmp(cH, 'DCT') == 1
            rfMRI_data.bold = rfMRI_data.bold -  X0 * (X0' * rfMRI_data.bold);
        end
    end
end

% save fractional BOLD changes
handles.rfMRI_data = rfMRI_data;

% plot the time series of BOLD
axes(handles.axes_result);
hold off;
ch = 1;
nch = size(rfMRI_data.bold, 2);
if isfield(handles, 'fnirs_data') == 1 && isfield(handles, 'ftotal_Hb') == 1
    % if both filtered NIRS and fMRI exist, default of display mode is
    % normalization
    fnirs_data  = handles.fnirs_data;
    ftotal_Hb = handles.ftotal_Hb;
    time = handles.time;
    plot(time, fnirs_data.oxyData(:,ch), 'r');
    hold on;
    plot(time, fnirs_data.dxyData(:,ch), 'b');
    plot(time, ftotal_Hb(:,ch), 'color', [0 127/255 0]);
    set(handles.checkbox_process_HbO, 'enable', 'on', 'value', 1);
    set(handles.checkbox_process_HbR, 'enable', 'on', 'value', 1);
    set(handles.checkbox_process_HbT, 'enable', 'on', 'value', 1);
end
plot(time_b, rfMRI_data.bold(:,ch), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);

axes(handles.axes_result);
default_axis = axis;
axis([min(time_b) max(time_b) default_axis(3:4)]);
xlabel('Time (s)');
set(handles.checkbox_BOLD_filt, 'value', 1, 'enable', 'on');

if get(handles.checkbox_model, 'value') == 1
    try
        num_model = get(handles.popup_num_model, 'value');
        scale_model = linspace(default_axis(3), default_axis(4), 6);
        min_model = min(handles.SPM.xX.X(:,num_model));
        max_model = max(handles.SPM.xX.X(:,num_model));
        model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
        hold on
        plot(time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
    catch
        set(handles.checkbox_model, 'value', 0);
        helpdlg('The predicted model response does not exist.');
    end
end
% set the graphic property
set(handles.axes_result, 'FontSize', 9);
set(handles.edit_process_ch, 'string', ['1 / ' num2str(nch)]);
set(handles.slider_process_ch, 'enable', 'on');
set(handles.slider_process_ch, 'sliderstep', [1/(nch-1), 1/(nch-1)], 'max', nch, 'min', 1, 'value', 1);
set(handles.checkbox_axislimit_filt, 'enable', 'on', 'value', 0);
set(handles.push_save,'enable','on');

guidata(hObject, handles);



% --- Executes on button press in checkbox_ROI_BOLD.
function checkbox_ROI_BOLD_Callback(hObject, eventdata, handles)


% --- Executes on selection change in listbox_NIRS_info.
function listbox_NIRS_info_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function listbox_NIRS_info_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox_NIRS_info (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listbox_fMRI_info.
function listbox_fMRI_info_Callback(hObject, eventdata, handles)
% hObject    handle to listbox_fMRI_info (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox_fMRI_info contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox_fMRI_info


% --- Executes during object creation, after setting all properties.
function listbox_fMRI_info_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox_fMRI_info (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in checkbox_axislimit_raw.
function checkbox_axislimit_raw_Callback(hObject, eventdata, handles)
% manually specifying the limit of x-axis and y-axis
% first, obtain the default of axis limit

if get(handles.checkbox_axislimit_raw, 'value') == 1
    title = 'Input for the axis limits';
    prompt = {'Enter the x-axis limit [sec].', 'Enter the y-axis limit.'};
    numlines = 1;
    try
        time = handles.time;
        defaultanswer = {num2str([min(time(:)) max(time(:))]), ''};
    catch
        defaultanswer = {'', ''};
    end
    options.Resize = 'on';
    options.WindowStyle='normal';
    options.Interpreter='tex';
    answer=inputdlg(prompt,title,numlines,defaultanswer,options);
    if isempty(answer)
        set(handles.checkbox_axislimit_raw, 'value', 0);
        return;
    end

    limit_xaxis = str2num(answer{1,1});
    limit_yaxis = str2num(answer{2,1});
    handles.limit_xaxis = limit_xaxis;
    handles.limit_yaxis = limit_yaxis;

    axes(handles.axes_nirs_timeseries);
    default_axis = axis;
    if isempty(limit_xaxis) == 0
        default_axis(1:2) = limit_xaxis(1:2);
    end
    if isempty(limit_yaxis) == 0
        default_axis(3:4) = limit_yaxis(1:2);
    end
end

% plot the time-series
% load the nirs_data
if get(handles.checkbox_HbO, 'value') == 1 || get(handles.checkbox_HbR, 'value') == 1 || get(handles.checkbox_HbT, 'value') == 1
    nirs_data = handles.nirs_data;
    total_Hb = handles.total_Hb;
end
% load the fMRI data
if get(handles.checkbox_BOLD_raw, 'value') == 1
    fMRI_data = handles.fMRI_data;
end

h = get(handles.slider_raw_ch, 'value');
time = handles.time;

axes(handles.axes_nirs_timeseries);
hold off
% plot the normalized signals ([0 1])
if get(handles.checkbox_norm_raw, 'value') == 1
    if get(handles.checkbox_HbO, 'value') == 1
        plot(time, nirs_data.oxyData(:,h)./max(nirs_data.oxyData(:,h)),'r');
        hold on
    end
    if get(handles.checkbox_HbR, 'value') == 1
        plot(time, nirs_data.dxyData(:,h)./max(nirs_data.dxyData(:,h)), 'b');
        hold on
    end
    if get(handles.checkbox_HbT, 'value') == 1
        plot(time, total_Hb(:,h)./max(total_Hb(:,h)), 'color', [0 127/255 0]);
        hold on
    end
    if get(handles.checkbox_BOLD_raw, 'value') == 1
        plot(handles.time_b, fMRI_data.bold(:,h)./max(fMRI_data.bold(:,h)), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
elseif get(handles.checkbox_norm_raw, 'value') == 0
    if get(handles.checkbox_HbO, 'value') == 1
        plot(time, nirs_data.oxyData(:,h), 'r');
        hold on
    end
    if get(handles.checkbox_HbR, 'value') == 1
        plot(time, nirs_data.dxyData(:,h), 'b');
        hold on
    end
    if get(handles.checkbox_HbT, 'value') == 1
        plot(time, total_Hb(:,h), 'color', [0 127/255 0]);
        hold on
    end
    if get(handles.checkbox_BOLD_raw, 'value') == 1
        plot(handles.time_b, fMRI_data.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
end
xlabel('Time (s)');
set(handles.axes_nirs_timeseries, 'FontSize', 9);

if get(handles.checkbox_axislimit_raw, 'value') == 0
    axes(handles.axes_nirs_timeseries);
    default_axis = axis;
    if ismember('time', who) == 1
        default_axis(1:2) = [min(time) max(time)];
    end
end
axis(default_axis);

if get(handles.checkbox_model, 'value') == 1
    try
        num_model = get(handles.popup_num_model, 'value');
        scale_model = linspace(default_axis(3), default_axis(4), 6);
        min_model = min(handles.SPM.xX.X(:,num_model));
        max_model = max(handles.SPM.xX.X(:,num_model));
        model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
        hold on
        plot(time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
    catch
        set(handles.checkbox_model, 'value', 0);
        helpdlg('The predicted model response does not exist.');
    end
end

guidata(hObject, handles);

% --- Executes on button press in checkbox_axislimit_filt.
function checkbox_axislimit_filt_Callback(hObject, eventdata, handles)
% manually specifying the limit of x-axis and y-axis
% first, obtain the default of axis limit

if get(handles.checkbox_axislimit_filt, 'value') == 1
    title = 'Input for the axis limits';
    prompt = {'Enter the x-axis limit [sec].', 'Enter the y-axis limit.'};
    numlines = 1;
    try
        time = handles.time;
        defaultanswer = {num2str([min(time(:)) max(time(:))]), ''};
    catch
        defaultanswer = {'', ''};
    end
    options.Resize = 'on';
    options.WindowStyle='normal';
    options.Interpreter='tex';
    answer=inputdlg(prompt,title,numlines,defaultanswer,options);
    if isempty(answer)
        set(handles.checkbox_axislimit_filt, 'value', 0);
        if isfield(handles, 'limit_xaxis_filt') == 0 && isfield(handles, 'limit_yaxis_filt') == 0
            handles.limit_xaxis_filt = [];
            handles.limit_yaxis_filt = [];
        end
        return;
    end

    limit_xaxis = str2num(answer{1,1});
    limit_yaxis = str2num(answer{2,1});
    handles.limit_xaxis_filt = limit_xaxis;
    handles.limit_yaxis_filt = limit_yaxis;

    axes(handles.axes_result);
    default_axis = axis;
    if isempty(limit_xaxis) == 0
        default_axis(1:2) = limit_xaxis(1:2);
    end
    if isempty(limit_yaxis) == 0
        default_axis(3:4) = limit_yaxis(1:2);
    end
end

h1 = get(handles.checkbox_process_HbO, 'value');
h2 = get(handles.checkbox_process_HbR, 'value');
h3 = get(handles.checkbox_process_HbT, 'value');
h4 = get(handles.checkbox_BOLD_filt, 'value');


% load filtered nirs data
if h1 == 1 || h2 == 1 || h3 == 1
    fnirs_data = handles.fnirs_data;
    ftotal_Hb = handles.ftotal_Hb;
end

% load filtered fMRI data
if h4 == 1
    rfMRI_data = handles.rfMRI_data;
    time_b = handles.time_b;
end

% obtain the number of channels from NIRS or BOLD data format
try
    nch = size(fnirs_data.oxyData,2);
catch
    nch = size(rfMRI_data.bold, 2);
end

h = get(handles.slider_process_ch, 'value');

time = handles.time;
num_model = get(handles.popup_num_model, 'value');
if get(handles.checkbox_model, 'value') == 1
    try
        X = handles.SPM.xX.X;
    catch
        set(handles.checkbox_model, 'value', 0);
        helpdlg('The predicted model response does not exist.');
    end
end

set(handles.edit_process_ch, 'string', [num2str(h) ' / ' num2str(nch)]);

axes(handles.axes_result);
hold off

if h1 == 1
   plot(time, fnirs_data.oxyData(:, h), 'r');
   hold on
end

if h2 == 1
    plot(time, fnirs_data.dxyData(:,h), 'b');
    hold on;
end

if h3 == 1
    plot(time, ftotal_Hb(:,h), 'color', [0 127/255 0]);
    hold on;
end

if h4 == 1
    plot(time_b, rfMRI_data.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
    hold on
end
xlabel('Time (s)');
set(handles.axes_result, 'FontSize', 9);

if get(handles.checkbox_axislimit_filt, 'value') == 0
    axes(handles.axes_result);
    default_axis = axis;
    if ismember('time', who) == 1
        default_axis(1:2) = [min(time) max(time)];
    end
end
axis(default_axis);

if get(handles.checkbox_model, 'value') == 1
    try
        num_model = get(handles.popup_num_model, 'value');
        scale_model = linspace(default_axis(3), default_axis(4), 6);
        min_model = min(handles.SPM.xX.X(:,num_model));
        max_model = max(handles.SPM.xX.X(:,num_model));
        model_raw = (scale_model(4) - scale_model(3))./(max_model - min_model) * (handles.SPM.xX.X(:,num_model) - min_model) + scale_model(3);
        hold on
        plot(time, model_raw, 'color', [0, 0, 0], 'linewidth', 2);
    catch
        set(handles.checkbox_model, 'value', 0);
        helpdlg('The predicted model response does not exist.');
    end
end

guidata(hObject, handles);
