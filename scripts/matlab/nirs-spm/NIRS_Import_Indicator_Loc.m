function varargout = NIRS_Import_Indicator_Loc(varargin)

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @NIRS_Import_Indicator_Loc_OpeningFcn, ...
    'gui_OutputFcn',  @NIRS_Import_Indicator_Loc_OutputFcn, ...
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


% --- Executes just before NIRS_Import_Indicator_Loc is made visible.
function NIRS_Import_Indicator_Loc_OpeningFcn(hObject, eventdata, handles, varargin,data)
handles.output = hObject;

if isempty(eventdata) == 1
    set(handles.push_text, 'enable', 'off');
    set(handles.push_image, 'enable', 'off');
    set(handles.popup_viewBrain, 'enable', 'off');
    set(handles.push_reset, 'enable', 'off');
    
    str_brain{1,1} = 'Ventral view';
    str_brain{2,1} = 'Dorsal view';
    str_brain{3,1} = 'Right lateral view';
    str_brain{4,1} = 'Left lateral view';
    str_brain{5,1} = 'Frontal view';
    str_brain{6,1} = 'Occipital view';
    set(handles.popup_viewBrain, 'string', str_brain, 'value', 2);
elseif eventdata == 1
    string_MNI = get(handles.listbox_posindi_mri, 'string');
    string_MNI{length(string_MNI) + 1,1} = num2str(round(data.*10)./10);
    set(handles.listbox_posindi_mri, 'Value', 1);
    set(handles.listbox_posindi_mri, 'string', string_MNI);
    set(handles.text1, 'string', [num2str(length(string_MNI)) ' References']);
    
elseif eventdata == 2
    string_MNI = get(handles.listbox_posindi_mri, 'string');
    if isempty(string_MNI) == 1
        errordlg('There is no MNI coordinate to be deleted.')
        return;
    end
    
    num_del = ones(length(string_MNI), 1) * (round(data.*10)./10);
    for kk = 1:length(string_MNI)
        num_MNI(kk,:) = str2num(string_MNI{kk,1});
    end
    
    index = find(sum(abs(num_MNI-num_del), 2) == 0);
    if isempty(index) == 1
        errordlg('Please enter the exact coordinate to be deleted.');
        string_new_MNI = string_MNI;
    else
        string_new_MNI = {};
        index_tmp = find(sum(abs(num_MNI-num_del), 2) ~= 0);
        for kk = 1:length(index_tmp)
            string_new_MNI{kk,1} = num2str(num_MNI(index_tmp(kk),:));
        end
    end
    set(handles.listbox_posindi_mri, 'Value', 1);
    set(handles.listbox_posindi_mri,'string', string_new_MNI);
    set(handles.text1, 'string', [num2str(length(string_new_MNI)) ' References']);
end
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = NIRS_Import_Indicator_Loc_OutputFcn(hObject, eventdata, handles)

varargout{1} = handles.output;
varargout{2} = handles;


% --- Executes on selection change in listbox_posindi_mri.
function listbox_posindi_mri_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function listbox_posindi_mri_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in push_save.
function push_save_Callback(hObject, eventdata, handles)
try 
    preproc_info = handles.preproc_info;
    preproc_info.registration = 'NIRS-MR registration';    
    try
        ch_set = handles.ch_set;
        ch = [];
        for kk = 1:size(ch_set,2)
            ch = [ch ch_set(1,kk).ch];
        end
        nch = size(preproc_info.rend_ch_pos{1}.rchn,1);
        if length(ch) ~= length(unique(ch))
            errordlg('Please do not input sets with duplicate elements.');
            return;
        end
        if length(unique(ch)) < nch
            choice = questdlg('All channels are not included in specified sets. Will you continue to save the channel information?', ...
                'Yes', 'No');
            if strcmp(choice, 'No') == 1 || strcmp(choice, 'Cancel') == 1
                return;
            end
        end
        if length(unique(ch)) > nch
            errordlg('Please check the elements within sets.');
            return;
        end
        preproc_info.ch_set = ch_set;
    end
        
    [filen, pathn] = uiputfile('*.mat',['Save the rendered image coordinate of ' num2str(size(preproc_info.ch_MNI_mm,2)) ' channels as']);
    path_file_n = [pathn filen];
    if path_file_n == 0
        return
    end
    save(path_file_n, 'preproc_info');
    clear global wT1_info;
catch
    errordlg('Spatial registration process is not completed.', 'error');
end



% --- Executes on selection change in listbox_posindi_digi.
function listbox_posindi_digi_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function listbox_posindi_digi_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listbox_posopt_mri.
function listbox_posopt_mri_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function listbox_posopt_mri_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listbox_posopt_digi.
function listbox_posopt_digi_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function listbox_posopt_digi_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in push_xls_file.
function push_xls_file_Callback(hObject, eventdata, handles)
% setup for real coordinates
spm_input('Setup for real coordinates', 1, 'd');
str = '# of reference (marker) points:';
num_indi = spm_input(str, '+1', 'r', ' ', 1);
str = '# of optodes:';
num_opt = spm_input(str, '+1', 'r', ' ', 1);

% load the real coordinate file
[filen, pathn] = uigetfile('*.xls;*.txt', 'Select the file that contains real coordinates of optodes.');
path_file_n = [pathn filen];
if path_file_n == 0
    return
end
spm_input(['File name: ' filen], '+1', 'd');

switch filen(end-2:end)
    case 'xls'
        [digit_coordi txt]= xlsread(path_file_n);
    case 'txt'
        fid = fopen(path_file_n);
        for kk = 1:(num_indi+num_opt)
            tmp = fgetl(fid);
            digit_coordi(kk,:) = str2num(tmp);
        end
        fclose(fid);
end
set(handles.listbox_posindi_digi,'string',num2str(digit_coordi(1:num_indi,:)), 'value', 1);
set(handles.listbox_posopt_digi,'string',num2str(digit_coordi(num_indi+1:num_indi+num_opt,:)), 'value', 1);

% load the channel configuration file
[filen, pathn] = uigetfile('*txt; *.csv', 'Select the channel configuration file.');
path_file_n = [pathn filen];
if path_file_n == 0
    return
end
spm_input(['File name: ' filen], '+1', 'd');

fid = fopen(path_file_n);
tline = fgetl(fid);
ch_config.vender = tline;
tline = fgetl(fid);
ch_config.type = tline;
tline = fgetl(fid);
tline = fgetl(fid);
index1 = 1;
while 1
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    index2 = find(tline == ',');
    tilne(index2) =  ' ';    
    spec(index1,:) = str2num(tline);
    index1 = index1 + 1;
end
ch_config.spec = spec;
fclose(fid);
handles.ch_config = ch_config;

set(handles.text2, 'string', [num2str(num_indi) ' References']);
set(handles.text4, 'string', [num2str(num_opt) ' Optodes']);
set(handles.text9, 'string', [num2str(size(ch_config.spec,1)) ' Channels']);
set(handles.text1, 'string', ' References');

set(handles.listbox_posindi_mri, 'string', {}, 'value', 1);
set(handles.popup_spatial_info_NIRS, 'value', 1);
set(handles.listbox_posopt_mri, 'string', {}, 'value', 1);

try
    handles = rmfield(handles, 'str_ch');
end
try
    handles = rmfield(handles, 'preproc_info');
end

guidata(hObject, handles);


% --- Executes on button press in push_getoptpos.
function push_getoptpos_Callback(hObject, eventdata, handles)
global wT1_info;
try
    ch_config = handles.ch_config;
catch
    errordlg('Please specify the channel configuration.');
end

tmp_indi = get(handles.listbox_posindi_digi,'string');
tmp_opt = get(handles.listbox_posopt_digi,'string');
if isempty(tmp_indi) == 1
    errordlg('Please specify the real coordinates of reference points');
    return;
end
pos_indi_digi = str2num(tmp_indi)';
pos_opt_digi = str2num(tmp_opt)';
num_indi = size(pos_indi_digi, 2);

tmp_indi_mri = get(handles.listbox_posindi_mri,'string');
if isempty(tmp_indi_mri) == 1
    errordlg('Please specify the MNI coordinates of reference points');
    return;
end
for kk = 1:num_indi
    pos_indi_mri(:,kk) = str2num(tmp_indi_mri{kk});
end
    
[filen, pathn] = uigetfile('*.mat','Select the normalization transform pararmeter file');
path_file_n = [pathn filen];
if path_file_n == 0
    return
end

load(path_file_n);
[rend rendered_MNI ch_MNI_mm] = get_channelpos(Affine, VF, VG, wT1_info, pos_indi_mri, pos_indi_digi, pos_opt_digi, ch_config.spec);

for kk = 1:6
    rendered_MNI{kk}.ren = rend{kk}.ren;
end

preproc_info.rend_ch_pos = rendered_MNI;
preproc_info.ch_MNI_mm = ch_MNI_mm;
preproc_info.wT1_info = wT1_info;
preproc_info.indi_mri = pos_indi_mri;
preproc_info.indi_digi = pos_indi_digi;
preproc_info.optd_digi = pos_opt_digi;
preproc_info.ch_config = ch_config;

nch = size(ch_MNI_mm, 2);
if nch < 9
    for kk = 1:nch
        str_ch{kk} = ['CH0' num2str(kk)];
    end
else
    for kk = 1:9
        str_ch{kk} = ['CH0' num2str(kk)];
    end
    for kk = 10:nch
        str_ch{kk} = ['CH' num2str(kk)];
    end
end
for kk = 1:nch
    str_list{kk,1} = [str_ch{kk} ':  ' num2str(round(ch_MNI_mm(1:3,kk))')];
end
handles.str_ch = str_ch;
handles.preproc_info = preproc_info;

set(handles.listbox_posopt_mri, 'Value', 1);
set(handles.listbox_posopt_mri, 'string', str_list);
set(handles.popup_spatial_info_NIRS,'value',2);

set(handles.push_text, 'enable', 'on');
set(handles.push_image, 'enable', 'on');
set(handles.popup_viewBrain, 'enable', 'on');
set(handles.push_reset, 'enable', 'on');
    
guidata(hObject, handles);

% --- Executes on button press in push_viewoptpos.
function push_viewoptpos_Callback(hObject, eventdata, handles)
try
    preproc_info = handles.preproc_info;
    NIRS_RegistrationResult_Viewer(preproc_info.rend_ch_pos);
catch
    errordlg('Please perform the NIRS-MRI alignment first.');
    return;
end

% --- Executes on button press in push_MRI.
function push_MRI_Callback(hObject, eventdata, handles)
global handles_registration_NIRS_MRI;
handles_registration_NIRS_MRI{2} = handles;
nirs_spm_image;



% --- Executes on selection change in popup_spatial_info_NIRS.
function popup_spatial_info_NIRS_Callback(hObject, eventdata, handles)
 h = get(handles.popup_spatial_info_NIRS, 'value');
try
    ch_MNI_mm = handles.preproc_info.ch_MNI_mm;
    nch = size(ch_MNI_mm, 2);
    str_ch = handles.str_ch;
    switch h
        case 1
            Nch = size(handles.str_ch, 2);
            tmp_str = [num2str(Nch) ' Channels'];
            str_list{1,1} = tmp_str;
        case 2 %% MNI coordinate
            for kk = 1:nch
                str_list{kk,1} = [str_ch{kk} ':  ' num2str(round(ch_MNI_mm(1:3,kk))')];
            end
        case 3 %% Automatic Anatomical Labeling
            str_list{1,1} = ['         Anatomical label, Percentage of Overlap' ];
            nfri_anatomlabel_modified(ch_MNI_mm(1:3,:)', 'test', 10, 4);
            global nfri_anatomlabel_NIRS_SPM;
            count = 2;
            for kk = 1:nch
                label = nfri_anatomlabel_NIRS_SPM{kk,1};
                prob = nfri_anatomlabel_NIRS_SPM{kk,2};
                for aa = 1:size(label,1)
                    if aa == 1
                        str_list{count,1} = [str_ch{kk} ':' label{aa,1} ',  ' num2str(prob(aa))];
                    else
                        str_list{count,1} = ['         :' label{aa,1} ',  ' num2str(prob(aa))];
                    end
                    count = count + 1;
                end
            end
        case 4 %% Brodmann Area (MRIcro)
            str_list{1,1} = ['         Anatomical label, Percentage of Overlap' ];
            nfri_anatomlabel_modified(ch_MNI_mm(1:3,:)', 'test', 10, 5);
            global nfri_anatomlabel_NIRS_SPM;
            count = 2;
            for kk = 1:nch
                label = nfri_anatomlabel_NIRS_SPM{kk,1};
                prob = nfri_anatomlabel_NIRS_SPM{kk,2};
                for aa = 1:size(label,1)
                    if aa == 1
                        str_list{count,1} = [str_ch{kk} ':' label{aa,1} ',  ' num2str(prob(aa))];
                    else
                        str_list{count,1} = ['         :' label{aa,1} ',  ' num2str(prob(aa))];
                    end
                    count = count + 1;
                end
            end
        case 5 %% LPBA40
            str_list{1,1} = ['         Anatomical label, Percentage of Overlap' ];
            nfri_anatomlabel_modified(ch_MNI_mm(1:3,:)', 'test', 10, 6);
            global nfri_anatomlabel_NIRS_SPM;
            count = 2;
            for kk = 1:nch
                label = nfri_anatomlabel_NIRS_SPM{kk,1};
                prob = nfri_anatomlabel_NIRS_SPM{kk,2};
                for aa = 1:size(label,1)
                    if aa == 1
                        str_list{count,1} = [str_ch{kk} ':' label{aa,1} ',  ' num2str(prob(aa))];
                    else
                        str_list{count,1} = ['         :' label{aa,1} ',  ' num2str(prob(aa))];
                    end
                    count = count + 1;
                end
            end
        case 6 %% Brodmann Area (Talairach Daemon)
            str_list{1,1} = ['         Anatomical label, Percentage of Overlap' ];
            nfri_anatomlabel_modified(ch_MNI_mm(1:3,:)', 'test', 10, 7);
            global nfri_anatomlabel_NIRS_SPM;
            count = 2;
            for kk = 1:nch
                label = nfri_anatomlabel_NIRS_SPM{kk,1};
                prob = nfri_anatomlabel_NIRS_SPM{kk,2};
                for aa = 1:size(label,1)
                    if aa == 1
                        str_list{count,1} = [str_ch{kk} ':' label{aa,1} ',  ' num2str(prob(aa))];
                    else
                        str_list{count,1} = ['         :' label{aa,1} ',  ' num2str(prob(aa))];
                    end
                    count = count + 1;
                end
            end
    end
catch
    str_list = {};
end
set(handles.listbox_posopt_mri, 'string', str_list, 'value', 1);

% --- Executes during object creation, after setting all properties.
function popup_spatial_info_NIRS_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes on button press in push_text.
function push_text_Callback(hObject, eventdata, handles)
[filen, pathn] = uigetfile('*.txt', 'Select the file for specifying the channel-set');
path_file_n = [pathn filen];
if path_file_n == 0
    return;
end

try
    Nch = size(handles.preproc_info.rend_ch_pos{1}.rchn, 1);
    tmp_str = [num2str(Nch) ' Channels'];
    h_str{1,1} = tmp_str;
end

ch_set = [];
count = 1;
fid = fopen(path_file_n);
while 1
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    ch_set(1,count).name = tline(1,1:6);
    ch_set(1,count).ch = unique(sort(str2num(tline(1,9:end))));
    h_str{count+1,1} = [ch_set(1,count).name ': ' num2str(ch_set(1,count).ch(:)')];
    count = count + 1;
end
set(handles.listbox_posopt_mri, 'string', h_str, 'value',1);
set(handles.popup_spatial_info_NIRS, 'value', 1);
handles.ch_set = ch_set;
guidata(hObject, handles);


% --- Executes on button press in push_image.
function push_image_Callback(hObject, eventdata, handles)
view_brain = get(handles.popup_viewBrain, 'string');
h = get(handles.popup_viewBrain, 'value');
try
    preproc_info = handles.preproc_info;
catch
    errordlg('Please perform the NIRS-MRI Alignment first.');
    return;
end

switch view_brain{h,1}
    case 'Ventral view'
        side_hemi = 1;
    case 'Dorsal view'
        side_hemi = 2;
    case 'Right lateral view'
        side_hemi = 3;
    case 'Left lateral view'
        side_hemi = 4;
    case 'Frontal view'
        side_hemi = 5;
    case 'Occipital view'
        side_hemi = 6;
end

rendered_MNI = preproc_info.rend_ch_pos;
rchn = rendered_MNI{side_hemi}.rchn;
cchn = rendered_MNI{side_hemi}.cchn;
Nch = size(rchn,1);

brain = rendered_MNI{side_hemi}.ren;
brain = brain.*0.5;
sbar = linspace(0, 1, 128);
sbrain = ((-sbar(1) + sbar(64))/(0.5)).* brain + sbar(1);
sbrain(1,1) = 1;
mask_disp = ones(size(brain,1), size(brain,2));

for jj = 1:Nch
    if rchn(jj) ~= -1 && cchn(jj) ~= -1
        if rchn(jj) < 6 || cchn(jj) < 6
            sbrain(rchn(jj), cchn(jj)) = 0.9;
            mask_disp(rchn(jj), cchn(jj)) = 0;
        else
            sbrain(rchn(jj)-5:rchn(jj)+5, cchn(jj)-5:cchn(jj)+5) = 0.9;
            mask_disp(rchn(jj)-5:rchn(jj)+5, cchn(jj)-5:cchn(jj)+5) = 0;
        end
    end
end
load Split;
h_ROI_interp = figure('Name', 'Specification of Channel Sets', 'NumberTitle', 'off');
imagesc(sbrain);
colormap(split);
axis image
axis off

for jj = 1:Nch
    if rchn(jj) ~= -1 && cchn(jj) ~= -1 %% updated 2009-02-25
        text(cchn(jj)-5, rchn(jj), num2str(jj), 'color', 'r');
    end
end


mask_interp1 = roipoly; % ROI for 1-set CH
mask_interp = mask_interp1;
mask_disp1 = mask_disp.*mask_interp1;
index_disp = find(mask_disp1 == 1);
clear mask_disp1;
sbrain(index_disp) = 0.67;
figure(h_ROI_interp);
imagesc(sbrain);
colormap(split);
axis image
axis off

ch = [];
for kk = 1:Nch
    if rchn(kk) ~= -1 && cchn(kk) ~= -1
        if mask_interp1(rchn(kk), cchn(kk)) == 1
            ch = [ch kk];
        end
    end
end
ch = sort(ch);

if isempty(ch) == 1
    errordlg('There is no channel within the region of interest selected manually.');
    return;
end

prompt = {['Specified channels are CH #' num2str(ch(:)')  '. Please enter the number of channel-set (min. 1, max. 9, natural number)']};
dlg_title = 'Input for channel-set';
num_lines = 1;
def = {''};
answer = inputdlg(prompt,dlg_title,num_lines,def);
if isempty(answer) == 1 % in case of cancel
    return;
elseif str2num(answer{1,1}) > 9 || str2num(answer{1,1}) < 1
    errordlg('Input number was not correct. Set # should be larger than 0 and smaller than 10');
    return;
end

set(handles.popup_spatial_info_NIRS, 'value', 1);

cur_ch_set.name = ['Set #' answer{1,1}]; % set-ch structure for current input
cur_ch_set.ch = ch;
curset_num = str2num(answer{1,1});
try
    ch_set = handles.ch_set;
    for kk = 1:size(ch_set,2)
        set_num(kk,1) = str2num(ch_set(kk).name(6));
    end
    set_index = find(set_num == curset_num);
    if isempty(set_index) == 1
        diff_set = set_num - curset_num;
        set_index = find(diff_set < 0);
        if isempty(set_index) == 1
            ch_set = [cur_ch_set ch_set];
        else
            set_index = find(abs(diff_set(set_index)) == min(abs(diff_set(set_index))));
            ch_set = [ch_set(1, 1:set_index) cur_ch_set ch_set(1,set_index+1:end)];
        end
    else
        tmp_ch = ch_set(1,set_index).ch;
        tmp_ch = [tmp_ch ch];
        tmp_ch = sort(unique(tmp_ch));
        ch_set(1,set_index).ch = tmp_ch;
    end
catch
    ch_set = cur_ch_set;
    clear cur_ch_set;
end

handles.ch_set = ch_set;
% update the listbox

tmp_str = [num2str(Nch) ' Channels'];
h_str{1,1} = tmp_str;

for kk = 1:size(ch_set,2)
    h_str{kk+1,1} = [ch_set(1,kk).name ': ' num2str(ch_set(1,kk).ch(:)')];
end
set(handles.listbox_posopt_mri, 'string', h_str, 'value',1);
guidata(hObject, handles);


% --- Executes on selection change in popup_viewBrain.
function popup_viewBrain_Callback(hObject, eventdata, handles)
% hObject    handle to popup_viewBrain (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popup_viewBrain contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popup_viewBrain


% --- Executes during object creation, after setting all properties.
function popup_viewBrain_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popup_viewBrain (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in push_reset.
function push_reset_Callback(hObject, eventdata, handles)
% update the handle structure
try
    handles = rmfield(handles, 'ch_set');
end

h_str = {};
try
    Nch = size(handles.preproc_info.rend_ch_pos{1}.rchn, 1);
    tmp_str = [num2str(Nch) ' Channels'];
    h_str{1,1} = tmp_str;    
end
set(handles.listbox_posopt_mri, 'string', h_str, 'value', 1);
set(handles.popup_spatial_info_NIRS, 'value', 1);

guidata(hObject, handles);




% --- Executes on button press in push_export.
function push_export_Callback(hObject, eventdata, handles)
h = get(handles.popup_spatial_info_NIRS,'value');
str_type = get(handles.popup_spatial_info_NIRS, 'string');
str_type = str_type{h};
if h ~= 1
    str_ch = get(handles.listbox_posopt_mri, 'string');
    [filen, pathn] = uiputfile('*.txt', ['Save ' str_type 'as ']);
    path_file_n = [pathn filen];
    if path_file_n == 0
        return
    end
    try
        fid = fopen(path_file_n, 'w');
        for kk = 1:size(str_ch, 1)
            fprintf(fid, '%s\r\n', str_ch{kk,1});
        end
        fclose(fid);
    end
end


