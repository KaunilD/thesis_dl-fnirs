function varargout = NIRS_Registration_Standalone(varargin)

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @NIRS_Registration_Standalone_OpeningFcn, ...
    'gui_OutputFcn',  @NIRS_Registration_Standalone_OutputFcn, ...
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


% --- Executes just before NIRS_Registration_Standalone is made visible.
function NIRS_Registration_Standalone_OpeningFcn(hObject, eventdata, handles, varargin, data)
handles.output = hObject;
h_w1 = get(handles.radio_with_digitizer,'value');
h_w2 = get(handles.radio_without_digitizer, 'value');
if h_w1 == 0 && h_w2 == 0
    set(handles.radio_with_digitizer, 'value', 0);
    set(handles.radio_without_digitizer, 'value', 1);
    set(handles.radio_optodes, 'enable','on','value',1);
    set(handles.radio_channels, 'enable', 'on','value',0);
    set(handles.push_MNIcoordinate, 'string', 'Select the file to contain MNI coordinates of NIRS optodes');
    set(handles.text1, 'string', 'MNI coordinate of specific optodes:[ Optd #, x, y, z ]');
    
    set(handles.push_ch_config, 'enable','on');
    set(handles.push_MNIcoordinate, 'enable','on');
    set(handles.push_MNI_template, 'enable','on');
    set(handles.edit_input_MNI, 'enable','on','string','');
    set(handles.push_add, 'enable','on');
    set(handles.push_del, 'enable','on');
    set(handles.push_clear, 'enable','on');
    set(handles.push_viewOptd, 'enable', 'on');
    set(handles.push_real_reference, 'enable', 'off');
    set(handles.push_real_optode_channel, 'enable','off');
    set(handles.edit_real_reference, 'enable', 'off','string','');
    set(handles.edit_real_optode_channel, 'enable', 'off','string','');
    set(handles.push_registration_NIRS,'enable','off');
    set(handles.text1, 'enable', 'on');
    set(handles.text6, 'enable', 'on');
    str_list{1,1} = '';
    str_list{2,1} = ' If you don''t have the 3D digitizing system,';
    str_list{3,1} = ' NIRS-SPM provides two methods for receiving the MNI coor-';
    str_list{4,1} = ' dinates:';
    str_list{5,1} = ' 1) maual input of MNI coordinates,';
    str_list{6,1} = ' 2) choice of MNI coordinates from SPM template images.';
    set(handles.listbox_MNI_coordinate,'value', 2);
    set(handles.listbox_MNI_coordinate,'string',str_list);
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
    % initialize the handles structure
    try
        handles = rmfield(handles, 'ch_set');
    end
    try
        handles = rmfield(handles, 'preproc_info');
    end
end
if eventdata == 1
    num_add = round(data.*10./10);
    set(handles.edit_input_MNI, 'string', num2str(num_add));    
elseif eventdata == 2    
end

guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = NIRS_Registration_Standalone_OutputFcn(hObject, eventdata, handles)
% Get default command line output from handles structure
varargout{1} = handles.output;
varargout{2} = handles;

% --- Executes on selection change in listbox_MNI_coordinate.
function listbox_MNI_coordinate_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function listbox_MNI_coordinate_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in push_MNIcoordinate.
function push_MNIcoordinate_Callback(hObject, eventdata, handles)
[filen, pathn] = uigetfile('*.csv;*.txt', 'Specify the MNI coordinate file');
path_file_n = [pathn filen];
if path_file_n == 0
    return
end

% initialize
try
    handles = rmfield(handles, 'preproc_info');
end
try
    handles = rmfield(handles, 'ch_set');
end

h = get(handles.radio_optodes, 'value');
index = 0;
fid = fopen(path_file_n);
if h == 1
    while 1
        tline = fgetl(fid);
        if ~ischar(tline), break, end
        index2 = find(tline == ',');
        tline(index2) = ' ';
        index = index + 1;
        optd_MNI_mm(1:3, index) = str2num(tline);
        str_optd{index,1} = ['Optd' num2str(index)];
    end
    optd_MNI_mm(4, :) = 1;
    handles.str_optd = str_optd;
    handles.optd_MNI_mm = optd_MNI_mm;
    try
        handles = rmfield(handles, 'str_ch');
    end
elseif h == 0
    while 1
        tline = fgetl(fid);
        if ~ischar(tline), break, end
        index2 = find(tline == ',');
        tline(index2) = ' ';
        index = index + 1;
        ch_MNI_mm(1:3, index) = str2num(tline);
        str_ch{index,1} = ['CH' num2str(index)];
    end
    ch_MNI_mm(4, :) = 1;
    handles.str_ch = str_ch;
    handles.preproc_info.ch_MNI_mm = ch_MNI_mm;
    try
        handles = rmfield(handles, 'str_optd');
        handles = rmfield(handles, 'optd_MNI_mm');
    end
end
fclose(fid);

set(handles.popup_spatial_info_NIRS, 'value', 2);
count = 4;
if h == 1
    str_list{1,1} = [num2str(size(str_optd, 1)) ' Optodes'];
    noptd = size(str_optd, 1);
    for kk = 1:noptd
        str_list{count,1} = [str_optd{kk} ' : ' num2str(round(optd_MNI_mm(1:3, kk)'))];
        count = count + 1;
    end
elseif h == 0
    str_list{1,1} = [num2str(size(str_ch, 1)) ' Channels'];
    nch = size(str_ch, 1);
    for kk = 1:nch
        str_list{count,1} = [str_ch{kk} ' : ' num2str(round(ch_MNI_mm(1:3, kk)'))];
        count = count + 1;
    end
end
str_list{3,1} = [' # , x, y, z'];
set(handles.listbox_MNI_coordinate, 'value', 1);
set(handles.listbox_MNI_coordinate, 'string', str_list);
guidata(hObject, handles);

function edit_input_MNI_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function edit_input_MNI_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in push_add.
function push_add_Callback(hObject, eventdata, handles)
string_add = get(handles.edit_input_MNI, 'string');
index = find(string_add == ',');
num_add = str2num(string_add);
h = get(handles.radio_optodes, 'value');
set(handles.popup_spatial_info_NIRS,'value',2);
if h == 1 %% add the specific optode position
    try
        optd_MNI_mm = handles.optd_MNI_mm;
        str_optd = handles.str_optd;
        count = size(optd_MNI_mm, 2)+1;
        if size(num_add, 2) == 3
            num_add(2:4) = num_add(1:3);
            num_add(1) = count;
        end
    catch
        count = 1;
        if size(num_add, 2) == 4
            num_add(1) = 1;
        elseif size(num_add, 2) == 3
            num_add(2:4) = num_add(1:3);
            num_add(1) = 1;
        end
    end
    optd_MNI_mm(1:3,num_add(1)) = num_add(2:4);
    optd_MNI_mm(4,:) = 1;
    str_optd{num_add(1),1} = ['Optd' num2str(num_add(1))];
    try
        preproc_info = handles.preproc_info;
        preproc_info = rmfield(preproc_info, 'ch_MNI_mm');
        handles.preproc_info = preproc_info;
        handles = rmfield(handles, 'str_ch');
    end
    str_list{1,1} = [num2str(count) ' Optodes'];
    count = 4;
    for kk = 1:size(str_optd, 1)
        str_list{count, 1} = [str_optd{kk,1} ' : ' num2str(round(optd_MNI_mm(1:3, kk)'))];
        count = count + 1;
    end
    handles.str_optd = str_optd;
    handles.optd_MNI_mm = optd_MNI_mm;
elseif h == 0 %% add the specific channel position
    try
        ch_MNI_mm = handles.preproc_info.ch_MNI_mm;
        str_ch = handles.str_ch;
        count = size(ch_MNI_mm, 2)+1;
        if size(num_add, 2) == 3
            num_add(2:4) = num_add(1:3);
            num_add(1) = count;
        end
    catch
        count = 1;
        if size(num_add, 2) == 4
            num_add(1) = 1;
        elseif size(num_add, 2) == 3
            num_add(2:4) = num_add(1:3);
            num_add(1) = 1;
        end
    end
    ch_MNI_mm(1:3,num_add(1)) = num_add(2:4);
    ch_MNI_mm(4,:) = 1;
    str_ch{num_add(1),1} = ['CH' num2str(num_add(1))];
    str_list{1,1} = [num2str(count) ' Channels'];
    count = 4;
    for kk = 1:size(str_ch, 1)
        str_list{count, 1} = [str_ch{kk,1}, ' : ' num2str(round(ch_MNI_mm(1:3, kk)'))];
        count = count + 1;
    end
    handles.str_ch = str_ch;
    handles.preproc_info.ch_MNI_mm = ch_MNI_mm;
end
str_list{3,1} = [' # , x, y, z'];
set(handles.listbox_MNI_coordinate, 'value', 1);
set(handles.listbox_MNI_coordinate, 'string', str_list);
guidata(hObject, handles);

% --- Executes on button press in push_del.
function push_del_Callback(hObject, eventdata, handles)
string_del = get(handles.edit_input_MNI, 'string');
index = find(string_del == ',');
num_del = str2num(string_del);
h = get(handles.radio_optodes, 'value');
set(handles.popup_spatial_info_NIRS,'value',2);
if h == 1 %% del the specific optode position
    try
        optd_MNI_mm = handles.optd_MNI_mm;
    catch
        helpdlg('There is no file to be deleted.');
        return;
    end
    if size(num_del, 2) == 1
        try
            optd_MNI_mm(:, num_del(1)) = [];
        end
    end
    if size(num_del, 2) == 3
        tmp = round(optd_MNI_mm(1:3,:));
        tmp = abs(tmp' - ones(size(tmp,2),1) * num_del);
        [r c] = find(sum(tmp,2) == 0);
        optd_MNI_mm(:, r) = [];
    end
    
    if size(num_del, 2) == 4
        try
            tmp = round(optd_MNI_mm(1:3, num_del(1)));
            if sum(abs(tmp' - num_del(2:4))) == 0
                optd_MNI_mm(:, num_del(1)) = [];
            end
        end
    end
    
    if isempty(optd_MNI_mm) ~= 1
        for kk = 1:size(optd_MNI_mm,2)
            str_optd{kk,1} = ['Optd' num2str(kk)];
        end
        str_list{1,1} = [num2str(size(str_optd, 1)) ' Optodes'];
        count = 4;
        for kk = 1:size(str_optd, 1)
            str_list{count, 1} = [str_optd{kk,1} ' : ' num2str(round(optd_MNI_mm(1:3, kk)'))];
            count = count + 1;
        end
        handles.str_optd = str_optd;
        handles.optd_MNI_mm = optd_MNI_mm;
    else
        str_list{1,1} = [num2str(0) ' Optodes'];
        try
            handles = rmfield(handles, 'str_optd');
            handles = rmfield(handles, 'optd_MNI_mm');
        end
    end
    try
        preproc_info = handles.preproc_info;
        preproc_info = rmfield(preproc_info, 'ch_MNI_mm');
        handles.preproc_info = preproc_info;
        handles = rmfield(handles, 'str_ch');
    end
elseif h == 0
    try
        ch_MNI_mm = handles.preproc_info.ch_MNI_mm;
    catch
        helpdlg('There is no file to be deleted.');
        return;
    end
    if size(num_del, 2) == 1
        try
            ch_MNI_mm(:, num_del(1)) = [];
        end
    end
    if size(num_del, 2) == 4
        try
            tmp = round(ch_MNI_mm(1:3, num_del(1)));
            if sum(abs(tmp' - num_del(2:4))) == 0
                ch_MNI_mm(:, num_del(1)) = [];
            end
        end
    end
    
    if isempty(ch_MNI_mm) ~= 1
        for kk = 1:size(ch_MNI_mm,2)
            str_ch{kk,1} = ['CH' num2str(kk)];
        end
        str_list{1,1} = [num2str(size(str_ch, 1)) ' Channels'];
        count = 4;
        for kk = 1:size(str_ch, 1)
            str_list{count, 1} = [str_ch{kk,1} ' : ' num2str(round(ch_MNI_mm(1:3, kk)'))];
            count = count + 1;
        end
        handles.str_ch = str_ch;
        handles.ch_MNI_mm = ch_MNI_mm;
    else
        str_list{1,1} = [num2str(0) ' Channels'];
        try
            handles = rmfield(handles, 'str_ch');
            handles = rmfield(handles, 'ch_MNI_mm');
        end
    end
    
end

str_list{3,1} = [' # , x, y, z'];
set(handles.listbox_MNI_coordinate, 'value', 1);
set(handles.listbox_MNI_coordinate, 'string', str_list);
guidata(hObject, handles);

% --- Executes on button press in push_MNI_template.
function push_MNI_template_Callback(hObject, eventdata, handles)
global handles_standalone;
handles_standalone{2} = handles;
nirs_spm_image('standalone');


% --- Executes on button press in push_viewCh.
function push_viewCh_Callback(hObject, eventdata, handles)
h_registration = get(handles.radio_with_digitizer, 'value');
if h_registration == 1
    try
        global NFRI_result;
        WShatC = NFRI_result.OtherC;
        nch = size(handles.str_ch, 1);
        noptd = size(handles.str_optd, 1);
        ch_MNI_mm = ones(4, nch);
        ch_MNI_mm(1:3,:) = WShatC(noptd+1:end,1:3)';
        optd_MNI_mm = ones(4, noptd);
        optd_MNI_mm(1:3,:) = WShatC(1:noptd, 1:3)';
        handles.optd_MNI_mm = optd_MNI_mm;
    catch
        helpdlg('Registration process was not performed. Please do the registration of NIRS channel using ''Registration(use the NFRI function)'' button.');
        return;
    end
elseif h_registration == 0
    button = questdlg('Do you want to project the input positions to the cortical surface?', 'Question Dialog', 'Yes', 'No', 'Yes');
    h1 = get(handles.radio_optodes, 'value');
    h2 = get(handles.radio_channels, 'value');
    try
        preproc_info = handles.preproc_info;
    end
    if h1 == 1 && h2 == 0 %%% spatial registration of optodes
        try
            ch_config = handles.ch_config.spec;
        catch
            errordlg('Please specify the channel configuration.');
            return;
        end
        try
            optd_MNI_mm = handles.optd_MNI_mm;
        catch
            helpdlg('Please specify the MNI coordinates of optodes.');
            return;
        end
        if strcmp(button, 'Yes') == 1
            %             load surf_raw;
            %             [ch_MNI_mm] = optd2ch(optd_MNI_mm(1:3,:), ch_config, surf_raw);
            ch_HS = zeros(3,length(ch_config));
            for i = 1:length(ch_config)
                ch_HS(:,i) = (optd_MNI_mm(1:3,ch_config(i,1))+optd_MNI_mm(1:3,ch_config(i,2)))/2;
            end
            ch_HS = [ch_HS; ones(1, size(ch_HS,2))];
            [ch_MNI_mm] = projection_CS(ch_HS);
        elseif strcmp(button, 'No') == 1
            [ch_MNI_mm] = optd2ch(optd_MNI_mm(1:3,:), ch_config);
        end
        for kk = 1:size(ch_MNI_mm,2)
            str_ch{kk,1} = ['CH' num2str(kk)];
        end
        handles.str_ch = str_ch;
    elseif h1 == 0 & h2 == 1 %%% spatial registration of channels        
        try
            ch_MNI_mm = handles.preproc_info.ch_MNI_mm;
            MNI_mm = ch_MNI_mm(1:3,:);
        catch
            helpdlg('Please specify the MNI coordinates of channels.');
            return
        end
        if strcmp(button, 'Yes') == 1
            MNI_mm = [MNI_mm; ones(1, size(MNI_mm,2))];
            [ch_MNI_mm] = projection_CS(MNI_mm);
        end        
    end
end
template_info = spm_vol([spm('dir') filesep 'templates' filesep 'T1.nii']); %%% T1, T2, EPI template .mat .dim parameter are same
ch_MNI_vx = inv(template_info.mat) * ch_MNI_mm;
[rend, rendered_MNI] = render_MNI_coordinates(ch_MNI_vx, template_info);

for kk = 1:6
    rendered_MNI{kk}.ren = rend{kk}.ren;
end

% updates (2010. 9. 3) - registration viewer
NIRS_RegistrationResult_Viewer(rendered_MNI);

% NIRS_Rendered_MNI_Viewer(rendered_MNI);
preproc_info.rend_ch_pos = rendered_MNI;
preproc_info.ch_MNI_mm = ch_MNI_mm;

clear h_str
tmp_str = [num2str(size(handles.str_ch, 1)) ' Channels'];
try
    tmp_str = [tmp_str ' and ' num2str(size(handles.str_optd,1)) ' Optodes'];
end
h_str{1,1} = tmp_str;
set(handles.popup_spatial_info_NIRS, 'value', 1);
set(handles.listbox_MNI_coordinate, 'string', h_str, 'value', 1);

set(handles.push_text, 'enable', 'on');
set(handles.push_image, 'enable', 'on');
set(handles.popup_viewBrain, 'enable', 'on');
set(handles.push_reset, 'enable', 'on');
handles.preproc_info = preproc_info;

try
    handles = rmfield(handles, 'ch_set');
end
guidata(hObject, handles);

% --- Executes on button press in push_clear.
function push_clear_Callback(hObject, eventdata, handles)
str_list{1,1} = '';
str_list{2,1} = ' If you don''t have the 3D digitizing system,';
str_list{3,1} = ' NIRS-SPM provides two methods for receiving the MNI coor-';
str_list{4,1} = ' dinates:';
str_list{5,1} = ' 1) maual input of MNI coordinates,';
str_list{6,1} = ' 2) choice of MNI coordinates from SPM template images.';

set(handles.listbox_MNI_coordinate,'value', 2);
set(handles.listbox_MNI_coordinate,'string',str_list);
set(handles.popup_spatial_info_NIRS, 'value', 1);

try
    handles = rmfield(handles, 'str_ch');
    handles = rmfield(handles, 'preproc_info');
end
try
    handles = rmfield(handles, 'str_optd');
    handles = rmfield(handles, 'optd_MNI_mm');
end
guidata(hObject, handles);


% --- Executes on button press in push_save.
function push_save_Callback(hObject, eventdata, handles)
try
    preproc_info = handles.preproc_info;
    preproc_info.template_info = spm_vol([spm('dir') filesep 'templates' filesep 'T1.nii']);
    preproc_info.registration = 'standalone';
    
    % save the channel-set information
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
    if isfield(preproc_info, 'rend_ch_pos') == 1
        [filen, pathn] = uiputfile('*.mat',['Save the rendered image coordinate of ' num2str(size(preproc_info.ch_MNI_mm,2)) ' channels as']);
        path_file_n = [pathn filen];
        if path_file_n == 0
            return
        end
        save(path_file_n, 'preproc_info');
    else
        errordlg('Please obtain the MNI coordinates of the rendered brain. Select the View Ch. button.','Error');
        return;
    end
catch
    errordlg('Please obtain the MNI coordinates of the rendered brain. Select the View Ch. button.','Error');
    return;
end


% --- Executes on button press in radio_optodes.
function radio_optodes_Callback(hObject, eventdata, handles)
h = get(handles.radio_optodes, 'value');
if h == 0
    set(handles.radio_channels, 'value', 1);
    set(handles.push_ch_config,'enable', 'off');
    set(handles.push_MNIcoordinate, 'string', 'Select the file to contain MNI coordinates of NIRS channels');
    set(handles.text1, 'string', 'MNI coordinate of specific channel:[ Ch #, x, y, z ]');
    set(handles.push_viewOptd, 'enable', 'off');
elseif h == 1
    set(handles.radio_channels, 'value', 0);
    set(handles.push_ch_config,'enable', 'on');
    set(handles.push_MNIcoordinate, 'string', 'Select the file to contain MNI coordinates of NIRS optodes');
    set(handles.text1, 'string', 'MNI coordinate of specific optode:[ Optd #, x, y, z ]');
    set(handles.push_viewOptd, 'enable', 'on');
end

try
    handles = rmfield(handles, 'str_ch');
    handles = rmfield(handles, 'preproc_info');
end
try
    handles = rmfield(handles, 'str_optd');
    handles = rmfield(handles, 'optd_MNI_mm');
end
try
    handles = rmfield(handles, 'ch_set');
end

str_list{1,1} = '';
str_list{2,1} = ' If you don''t have the 3D digitizing system,';
str_list{3,1} = ' NIRS-SPM provides two methods for receiving the MNI coor-';
str_list{4,1} = ' dinates:';
str_list{5,1} = ' 1) maual input of MNI coordinates,';
str_list{6,1} = ' 2) choice of MNI coordinates from SPM template images.';
set(handles.popup_spatial_info_NIRS,'value', 1);
set(handles.listbox_MNI_coordinate,'value',2);
set(handles.listbox_MNI_coordinate,'string',str_list);
guidata(hObject, handles);

% --- Executes on button press in radio_channels.
function radio_channels_Callback(hObject, eventdata, handles)
h = get(handles.radio_channels, 'value');
if h == 0
    set(handles.radio_optodes, 'value', 1);
    set(handles.push_ch_config,'enable', 'on');
    set(handles.push_MNIcoordinate, 'string', 'Select the file to contain MNI coordinates of NIRS optodes');
    set(handles.text1, 'string', 'MNI coordinate of specific optode:[ Optd #, x, y, z ]');
    set(handles.push_viewOptd, 'enable', 'on');
elseif h == 1
    set(handles.radio_optodes, 'value', 0);
    set(handles.push_ch_config,'enable', 'off');
    set(handles.push_MNIcoordinate, 'string', 'Select the file to contain MNI coordinates of NIRS channels');
    set(handles.text1, 'string', 'MNI coordinate of specific channel:[ Ch #, x, y, z ]');
    set(handles.push_viewOptd, 'enable', 'off');
end

try
    handles = rmfield(handles, 'str_ch');
    handles = rmfield(handles, 'preproc_info');
end
try
    handles = rmfield(handles, 'str_optd');
    handles = rmfield(handles, 'optd_MNI_mm');
end
try
    handles = rmfield(handles, 'ch_set');
end

str_list{1,1} = '';
str_list{2,1} = ' If you don''t have the 3D digitizing system,';
str_list{3,1} = ' NIRS-SPM provides two methods for receiving the MNI coor-';
str_list{4,1} = ' dinates:';
str_list{5,1} = ' 1) maual input of MNI coordinates,';
str_list{6,1} = ' 2) choice of MNI coordinates from SPM template images.';
set(handles.popup_spatial_info_NIRS,'value', 1);
set(handles.listbox_MNI_coordinate,'value',2);
set(handles.listbox_MNI_coordinate,'string',str_list);

guidata(hObject, handles);

% --- Executes on button press in push_ch_config.
function push_ch_config_Callback(hObject, eventdata, handles)
[filen, pathn] = uigetfile('*txt; *.csv', 'Select the channel configuration file.');
path_file_n = [pathn filen];
if path_file_n == 0
    return
end
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
guidata(hObject, handles);


% --- Executes on button press in push_viewOptd.
function push_viewOptd_Callback(hObject, eventdata, handles)
try
    optd_MNI_mm = handles.optd_MNI_mm;
catch
    helpdlg('Registration process was not performed. Please do the registration of NIRS channel using ''Registration(use the NFRI function)'' button.');
    return;
end

template_info = spm_vol([spm('dir') filesep 'templates' filesep 'T1.nii']); %%% T1, T2, EPI template .mat .dim parameter are same
optd_MNI_vx = inv(template_info.mat) * optd_MNI_mm;
[rend, rendered_MNI] = render_MNI_coordinates(optd_MNI_vx, template_info);
for kk = 1:6
    rendered_MNI{kk}.ren = rend{kk}.ren;
end
NIRS_Rendered_MNI_Viewer(rendered_MNI);



% --- Executes on button press in push_real_reference.
function push_real_reference_Callback(hObject, eventdata, handles)
[filen, pathn] = uigetfile('*.csv', 'Specify the file of REFERENCE position in REAL space.');
path_file_n = [pathn filen];
if path_file_n == 0
    return;
end
handles.origin_fname = path_file_n;
set(handles.edit_real_reference,'enable','inactive', 'string', filen);
guidata(hObject, handles);

% --- Executes on button press in push_real_optode_channel.
function push_real_optode_channel_Callback(hObject, eventdata, handles)
[filen, pathn] = uigetfile('*.csv', 'Specify the file of Optode and Channel position in REAL space.');
path_file_n = [pathn filen];
if path_file_n == 0
    return;
end
handles.others_fname = path_file_n;

fid = fopen(path_file_n);
count = 1;
while 1
    tline = fgetl(fid);
    index = find(tline == ',');
    str_tmp = tline(1:index(1)-1);
    if strcmp(str_tmp, 'CH01') == 1
        str_ch{1,1} = str_tmp;
        break,
    else
        str_optd{count,1} = str_tmp;
        count = count + 1;
    end
end
count = 2;
while 1
    tline = fgetl(fid);
    if~ischar(tline), break, end
    index = find(tline == ',');
    str_ch{count,1} = tline(1:index(1)-1);
    count = count + 1;
end

handles.str_optd = str_optd;
handles.str_ch = str_ch;
str_list{1,1} = [num2str(size(handles.str_ch, 1)) ' Channels and ' num2str(size(handles.str_optd,1)) ' Optodes'];
set(handles.listbox_MNI_coordinate, 'Value', 1);
set(handles.listbox_MNI_coordinate, 'string', str_list);
set(handles.edit_real_optode_channel,'enable','inactive','string',filen);
guidata(hObject, handles);


function edit_real_reference_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_real_reference_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_real_optode_channel_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_real_optode_channel_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in push_registration_NIRS.
function push_registration_NIRS_Callback(hObject, eventdata, handles)
try
    origin_fname = handles.origin_fname;
    others_fname = handles.others_fname;
catch
    helpdlg('Please specify the file which contains the reference and channel positions in real coordinate.');
    return;
end
try
    nfri_mni_estimation_modified(origin_fname, others_fname);
catch
    helpdlg('nfri_mni_estimation function does not exist. Please add the path of NFRI function folders.');
    return;
end

% initialization
try
    handles = rmfield(handles, 'preproc_info');
end
try
    handles = rmfield(handles, 'ch_set');
end
try
    handles = rmfield(handles, 'optd_MNI_mm');
end

guidata(hObject, handles);

% --- Executes on button press in radio_with_digitizer.
function radio_with_digitizer_Callback(hObject, eventdata, handles)
h = get(handles.radio_with_digitizer, 'value');
if h == 1
    set(handles.radio_without_digitizer, 'value', 0);
    set(handles.radio_optodes, 'enable','off');
    set(handles.radio_channels, 'enable', 'off');
    set(handles.push_ch_config, 'enable','off');
    set(handles.push_MNIcoordinate, 'enable','off');
    set(handles.push_MNI_template, 'enable','off');
    set(handles.edit_input_MNI, 'enable','off','string','');
    set(handles.push_add, 'enable','off');
    set(handles.push_del, 'enable','off');
    set(handles.push_clear, 'enable','off');
    set(handles.text1, 'enable', 'off');
    set(handles.text6, 'enable', 'off');
    set(handles.push_viewOptd, 'enable', 'on');
    set(handles.push_real_reference, 'enable', 'on');
    set(handles.push_real_optode_channel, 'enable','on');
    set(handles.edit_real_reference, 'enable', 'inactive', 'string', '');
    set(handles.edit_real_optode_channel, 'enable', 'inactive', 'string', '');
    set(handles.push_registration_NIRS,'enable','on');
    str_list{1,1} = '';
    str_list{2,1} = ' If you have the 3D digitizing system,';
    str_list{3,1} = ' NFRI fNIRS tools(Singh et al.,2005) incorporated in NIRS-SPM';
    str_list{4,1} = ' allow spatial registration of NIRS channels from real space to';
    str_list{5,1} = ' MNI space.';
    str_list{6,1} = ' In order to produce results using this function, you are also ';
    str_list{7,1} = ' required to cite the following paper in addition to NIRS-SPM ';
    str_list{8,1} = ' papers (Ye et al., 2009; Jang et al., 2009): ';
    str_list{9,1} = '';
    str_list{10,1} = 'A.K. Singh et al. / NeuroImage 27 (2005) 842-851';
elseif h == 0
    set(handles.radio_without_digitizer, 'value', 1);
    set(handles.radio_optodes, 'enable','on','value',1);
    set(handles.radio_channels, 'enable', 'on','value',0);
    set(handles.push_MNIcoordinate, 'string', 'Select the file to contain MNI coordinates of NIRS optodes');
    set(handles.text1, 'string', 'MNI coordinate of specific optodes:[ Optd #, x, y, z ]');
    set(handles.push_ch_config, 'enable','on');
    set(handles.push_MNIcoordinate, 'enable','on');
    set(handles.push_MNI_template, 'enable','on');
    set(handles.edit_input_MNI, 'enable','o n','string','');
    set(handles.push_add, 'enable','on');
    set(handles.push_del, 'enable','on');
    set(handles.push_clear, 'enable','on');
    set(handles.push_real_reference, 'enable', 'off');
    set(handles.push_real_optode_channel, 'enable','off');
    set(handles.edit_real_reference, 'enable', 'off','string','');
    set(handles.edit_real_optode_channel, 'enable', 'off','string','');
    set(handles.push_registration_NIRS,'enable','off');
    set(handles.text1, 'enable', 'on');
    set(handles.text6, 'enable', 'on');
    set(handles.push_viewOptd, 'enable', 'on');
    str_list{1,1} = '';
    str_list{2,1} = ' If you don''t have the 3D digitizing system,';
    str_list{3,1} = ' NIRS-SPM provides two methods for receiving the MNI coor-';
    str_list{4,1} = ' dinates:';
    str_list{5,1} = ' 1) maual input of MNI coordinates,';
    str_list{6,1} = ' 2) choice of MNI coordinates from SPM template images.';
end

try
    handles = rmfield(handles, 'str_ch');
    handles = rmfield(handles, 'preproc_info');
end
try
    handles = rmfield(handles, 'str_optd');
    handles = rmfield(handles, 'optd_MNI_mm');
end
try
    handles = rmfield(handles, 'ch_set');
end

try
    clear global nfri_anatomlabel_NIRS_SPM;
end
try
    clear global NFRI_result;
end

set(handles.popup_spatial_info_NIRS,'value', 1);
set(handles.listbox_MNI_coordinate,'value',2);
set(handles.listbox_MNI_coordinate,'string',str_list);
set(handles.push_text, 'enable', 'off');
set(handles.push_image, 'enable', 'off');
set(handles.popup_viewBrain, 'enable', 'off');
set(handles.push_reset, 'enable', 'off');
guidata(hObject, handles);


% --- Executes off buttoff press in radio_without_digitizer.

function radio_without_digitizer_Callback(hObject, eventdata, handles)
h = get(handles.radio_without_digitizer, 'value');
if h == 1
    set(handles.radio_with_digitizer, 'value', 0);
    set(handles.radio_without_digitizer, 'value', 1);
    set(handles.radio_optodes, 'enable','on','value',1);
    set(handles.radio_channels, 'enable', 'on','value',0);
    set(handles.push_MNIcoordinate, 'string', 'Select the file to contain MNI coordinates of NIRS optodes');
    set(handles.text1, 'string', 'MNI coordinate of specific optodes:[ Optd #, x, y, z ]');
    
    set(handles.push_ch_config, 'enable','on');
    set(handles.push_MNIcoordinate, 'enable','on');
    set(handles.push_MNI_template, 'enable','on');
    set(handles.edit_input_MNI, 'enable','on','string','');
    set(handles.push_add, 'enable','on');
    set(handles.push_del, 'enable','on');
    set(handles.push_clear, 'enable','on');
    set(handles.push_viewOptd, 'enable', 'on');
    set(handles.push_real_reference, 'enable', 'off');
    set(handles.push_real_optode_channel, 'enable','off');
    set(handles.edit_real_reference, 'enable', 'off','string','');
    set(handles.edit_real_optode_channel, 'enable', 'off','string','');
    set(handles.push_registration_NIRS,'enable','off');
    set(handles.text1, 'enable', 'on');
    set(handles.text6, 'enable', 'on');
    str_list{1,1} = '';
    str_list{2,1} = ' If you don''t have the 3D digitizing system,';
    str_list{3,1} = ' NIRS-SPM provides two methods for receiving the MNI coor-';
    str_list{4,1} = ' dinates:';
    str_list{5,1} = ' 1) maual input of MNI coordinates,';
    str_list{6,1} = ' 2) choice of MNI coordinates from SPM template images.';
elseif h == 0
    set(handles.radio_with_digitizer, 'value', 1);
    set(handles.radio_optodes, 'enable','off');
    set(handles.radio_channels, 'enable', 'off');
    set(handles.push_ch_config, 'enable','off');
    set(handles.push_MNIcoordinate, 'enable','off');
    set(handles.push_MNI_template, 'enable','off');
    set(handles.edit_input_MNI, 'enable','off','string','');
    set(handles.push_add, 'enable','off');
    set(handles.push_del, 'enable','off');
    set(handles.push_clear, 'enable','off');
    set(handles.push_viewOptd, 'enable', 'on');
    set(handles.push_real_reference, 'enable', 'on');
    set(handles.push_real_optode_channel, 'enable','on');
    set(handles.edit_real_reference, 'enable', 'inactive','string','');
    set(handles.edit_real_optode_channel, 'enable', 'inactive','string','');
    set(handles.push_registration_NIRS,'enable','on');
    set(handles.text1, 'enable', 'off');
    set(handles.text6, 'enable', 'off');
    str_list{1,1} = '';
    str_list{2,1} = ' If you have the 3D digitizing system,';
    str_list{3,1} = ' NFRI fNIRS tools(Singh et al.,2005) incorporated in NIRS-SPM';
    str_list{4,1} = ' allow spatial registration of NIRS channels from real space to';
    str_list{5,1} = ' MNI space.';
    str_list{6,1} = ' In order to produce results using this function, you are also ';
    str_list{7,1} = ' required to cite the following paper in addition to NIRS-SPM ';
    str_list{8,1} = ' papers (Ye et al., 2009; Jang et al., 2009): ';
    str_list{9,1} = '';
    str_list{10,1} = 'A.K. Singh et al. / NeuroImage 27 (2005) 842-851';
end

try
    handles = rmfield(handles, 'str_ch');
    handles = rmfield(handles, 'preproc_info');
end
try
    handles = rmfield(handles, 'str_optd');
    handles = rmfield(handles, 'optd_MNI_mm');
end
try
    handles = rmfield(handles, 'ch_set');
end

try
    clear global nfri_anatomlabel_NIRS_SPM;
end
try
    clear global NFRI_result;
end

set(handles.popup_spatial_info_NIRS,'value', 1);
set(handles.listbox_MNI_coordinate,'value',2);
set(handles.listbox_MNI_coordinate,'string',str_list);
set(handles.push_text, 'enable', 'off');
set(handles.push_image, 'enable', 'off');
set(handles.popup_viewBrain, 'enable', 'off');
set(handles.push_reset, 'enable', 'off');
guidata(hObject, handles);



% --- Executes on selection change in popup_spatial_info_NIRS.
function popup_spatial_info_NIRS_Callback(hObject, eventdata, handles)
h = get(handles.popup_spatial_info_NIRS, 'value');
if h == 1 % spatial information of NIRS channels and optodes
    if isfield(handles, 'str_ch') == 1
        tmp_str = [num2str(size(handles.str_ch, 1)) ' Channels'];
        try
            tmp_str = [tmp_str ' and ' num2str(size(handles.str_optd,1)) ' Optodes'];
        end
        str_list{1,1} = tmp_str;
        try
            ch_set = handles.ch_set;
            for kk = 1:size(ch_set,2)
                str_list{kk+1,1} = [ch_set(1,kk).name ': ' num2str(ch_set(1,kk).ch(:)')];
            end
        end
    elseif isfield(handles, 'str_optd') == 1 && isfield(handles, 'str_ch') == 0
        str_list{1,1} = [num2str(size(handles.str_optd, 1)) ' Optodes'];
    elseif isfield(handles, 'str_optd') == 0 && isfield(handles, 'str_ch') == 0
        h2 = get(handles.radio_with_digitizer, 'value');
        if h2 == 1 % with 3D digitizer
            str_list{1,1} = '';
            str_list{2,1} = ' If you have the 3D digitizing system,';
            str_list{3,1} = ' NFRI fNIRS tools(Singh et al.,2005) incorporated in NIRS-SPM';
            str_list{4,1} = ' allow spatial registration of NIRS channels from real space to';
            str_list{5,1} = ' MNI space.';
            str_list{6,1} = ' In order to produce results using this function, you are also ';
            str_list{7,1} = ' required to cite the following paper in addition to NIRS-SPM ';
            str_list{8,1} = ' papers (Ye et al., 2009; Jang et al., 2009): ';
            str_list{9,1} = '';
            str_list{10,1} = 'A.K. Singh et al. / NeuroImage 27 (2005) 842-851';
        elseif h2 == 0 % without 3D digitizer
            str_list{1,1} = '';
            str_list{2,1} = ' If you don''t have the 3D digitizing system,';
            str_list{3,1} = ' NIRS-SPM provides two methods for receiving the MNI coor-';
            str_list{4,1} = ' dinates:';
            str_list{5,1} = ' 1) maual input of MNI coordinates,';
            str_list{6,1} = ' 2) choice of MNI coordinates from SPM template images.';
        end
    end
    %     set(handles.listbox_MNI_coordinate,'value', 2);
else
    WShatC = [];
    if isfield(handles, 'str_optd') == 1 && isfield(handles, 'str_ch') == 1
        str_optd = handles.str_optd;
        str_ch = handles.str_ch;
        str_list{1,1} = [num2str(size(handles.str_ch, 1)) ' Channels and ' num2str(size(handles.str_optd,1)) ' Optodes'];
        nch = size(str_ch, 1);
        noptd = size(str_optd, 1);
        if get(handles.radio_with_digitizer, 'value') == 1
            try
                global NFRI_result;
                WShatC = NFRI_result.OtherC;
            end
        elseif get(handles.radio_with_digitizer, 'value') == 0
            WShatC = handles.optd_MNI_mm(1:3,:)';
            WShatC(noptd+1:nch+noptd,1:3) = handles.preproc_info.ch_MNI_mm(1:3,:)';
        end
    elseif isfield(handles, 'str_optd') == 1 && isfield(handles, 'str_ch') == 0
        str_optd = handles.str_optd;
        str_list{1,1} = [num2str(size(handles.str_optd,1)) ' Optodes'];
        noptd = size(str_optd, 1);
        nch = 0;
        WShatC = handles.optd_MNI_mm(1:3,:)';
    elseif isfield(handles, 'str_optd') == 0 && isfield(handles, 'str_ch') == 1
        str_ch = handles.str_ch;
        str_list{1,1} = [num2str(size(handles.str_ch, 1)) ' Channels'];
        noptd = 0;
        nch = size(str_ch, 1);
        WShatC = handles.preproc_info.ch_MNI_mm(1:3,:)';
    else
        return;
    end
    if isempty(WShatC) ~= 1
        switch h
            case 2 %% MNI Coordinates
                str_list{3,1} = [' # , x, y, z'];
                count = 4;
                for kk = 1:nch
                    str_list{count,1} = [str_ch{kk,1} ' : ' num2str(round(WShatC(noptd+kk,:)))];
                    count = count + 1;
                end
                for kk = 1:noptd
                    str_list{count,1} = [str_optd{kk,1} ' : ' num2str(round(WShatC(kk,:)))];
                    count = count + 1;
                end
            case 3 %% Automatic Anatomical Labeling
                str_list{3,1} = [' # , Anatomical label, Percentage of Overlap'];
                nfri_anatomlabel_modified(WShatC, 'test', 10, 4);
                global nfri_anatomlabel_NIRS_SPM;
                count = 4;
                for kk = 1:nch
                    label = nfri_anatomlabel_NIRS_SPM{noptd+kk,1};
                    prob = nfri_anatomlabel_NIRS_SPM{noptd+kk,2};
                    for aa = 1:size(label,1)
                        str_list{count, 1} = [str_ch{kk,1} ' : ' label{aa,1} ',  ' num2str(prob(aa))];
                        count = count + 1;
                    end
                    count = count + 1;
                end
                for kk = 1:noptd
                    label = nfri_anatomlabel_NIRS_SPM{kk,1};
                    prob = nfri_anatomlabel_NIRS_SPM{kk,2};
                    for aa = 1:size(label,1)
                        str_list{count, 1} = [str_optd{kk,1} ' : ' label{aa,1} ',  ' num2str(prob(aa))];
                        count = count + 1;
                    end
                    count = count + 1;
                end
            case 4 %% Brodmann Area (MRIcro)
                str_list{3,1} = [' # , Anatomical label, Percentage of Overlap'];
                nfri_anatomlabel_modified(WShatC, 'test', 10, 5);
                global nfri_anatomlabel_NIRS_SPM;
                count = 4;
                for kk = 1:nch
                    label = nfri_anatomlabel_NIRS_SPM{noptd+kk,1};
                    prob = nfri_anatomlabel_NIRS_SPM{noptd+kk,2};
                    for aa = 1:size(label,1)
                        str_list{count, 1} = [str_ch{kk,1} ' : ' label{aa,1} ',  ' num2str(prob(aa))];
                        count = count + 1;
                    end
                    count = count + 1;
                end
                for kk = 1:noptd
                    label = nfri_anatomlabel_NIRS_SPM{kk,1};
                    prob = nfri_anatomlabel_NIRS_SPM{kk,2};
                    for aa = 1:size(label,1)
                        str_list{count, 1} = [str_optd{kk,1} ' : ' label{aa,1} ',  ' num2str(prob(aa))];
                        count = count + 1;
                    end
                    count = count + 1;
                end
            case 5 %% LPBA40
                str_list{3,1} = [' # , Anatomical label, Percentage of Overlap'];
                nfri_anatomlabel_modified(WShatC, 'test', 10, 6);
                global nfri_anatomlabel_NIRS_SPM;
                count = 4;
                for kk = 1:nch
                    label = nfri_anatomlabel_NIRS_SPM{noptd+kk,1};
                    prob = nfri_anatomlabel_NIRS_SPM{noptd+kk,2};
                    for aa = 1:size(label,1)
                        str_list{count, 1} = [str_ch{kk,1} ' : ' label{aa,1} ',  ' num2str(prob(aa))];
                        count = count + 1;
                    end
                    count = count + 1;
                end
                for kk = 1:noptd
                    label = nfri_anatomlabel_NIRS_SPM{kk,1};
                    prob = nfri_anatomlabel_NIRS_SPM{kk,2};
                    for aa = 1:size(label,1)
                        str_list{count, 1} = [str_optd{kk,1} ' : ' label{aa,1} ',  ' num2str(prob(aa))];
                        count = count + 1;
                    end
                    count = count + 1;
                end
            case 6 %% Brodmann Area (Talairach Daemon)
                str_list{3,1} = [' # , Anatomical label, Percentage of Overlap'];
                nfri_anatomlabel_modified(WShatC, 'test', 10, 7);
                global nfri_anatomlabel_NIRS_SPM;
                count = 4;
                for kk = 1:nch
                    label = nfri_anatomlabel_NIRS_SPM{noptd+kk,1};
                    prob = nfri_anatomlabel_NIRS_SPM{noptd+kk,2};
                    for aa = 1:size(label,1)
                        str_list{count, 1} = [str_ch{kk,1} ' : ' label{aa,1} ',  ' num2str(prob(aa))];
                        count = count + 1;
                    end
                    count = count + 1;
                end
                for kk = 1:noptd
                    label = nfri_anatomlabel_NIRS_SPM{kk,1};
                    prob = nfri_anatomlabel_NIRS_SPM{kk,2};
                    for aa = 1:size(label,1)
                        str_list{count, 1} = [str_optd{kk,1} ' : ' label{aa,1} ',  ' num2str(prob(aa))];
                        count = count + 1;
                    end
                    count = count + 1;
                end
        end
        %         set(handles.listbox_MNI_coordinate,'value', 1);
    else
        %         set(handles.listbox_MNI_coordinate,'value', 1);
        str_list{1,1} = '';
    end
end
set(handles.listbox_MNI_coordinate,'string',str_list,'value',1);

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

tmp_str = [num2str(size(handles.str_ch, 1)) ' Channels'];
try
    tmp_str = [tmp_str ' and ' num2str(size(handles.str_optd,1)) ' Optodes'];
end
h_str{1,1} = tmp_str;

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
set(handles.listbox_MNI_coordinate, 'string', h_str, 'value',1);
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
    errordlg('Please project MNI coordinates to rendered brain first.');
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

tmp_str = [num2str(size(handles.str_ch, 1)) ' Channels'];
try
    tmp_str = [tmp_str ' and ' num2str(size(handles.str_optd,1)) ' Optodes'];
end
h_str{1,1} = tmp_str;

for kk = 1:size(ch_set,2)
    h_str{kk+1,1} = [ch_set(1,kk).name ': ' num2str(ch_set(1,kk).ch(:)')];
end
set(handles.listbox_MNI_coordinate, 'string', h_str, 'value',1);
guidata(hObject, handles);

% --- Executes on selection change in popup_viewBrain.
function popup_viewBrain_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function popup_viewBrain_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in push_reset.
function push_reset_Callback(hObject, eventdata, handles)
% update the handle structure
try
    handles = rmfield(handles, 'ch_set');
end

tmp_str = [num2str(size(handles.str_ch, 1)) ' Channels'];
try
    tmp_str = [tmp_str ' and ' num2str(size(handles.str_optd,1)) ' Optodes'];
end
h_str{1,1} = tmp_str;
set(handles.popup_spatial_info_NIRS, 'value', 1);
set(handles.listbox_MNI_coordinate, 'string', h_str, 'value', 1);

guidata(hObject, handles);



% --- Executes on button press in push_export.
function push_export_Callback(hObject, eventdata, handles)
h = get(handles.popup_spatial_info_NIRS,'value');
str_type = get(handles.popup_spatial_info_NIRS, 'string');
str_type = str_type{h};
if h ~= 1
    str_ch = get(handles.listbox_MNI_coordinate, 'string');
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


