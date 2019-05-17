function varargout = NIRS_Estimation(varargin)

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @NIRS_Estimation_OpeningFcn, ...
    'gui_OutputFcn',  @NIRS_Estimation_OutputFcn, ...
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


% --- Executes just before NIRS_Estimation is made visible.
function NIRS_Estimation_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;

set(handles.checkbox_individual, 'value', 1);
set(handles.checkbox_group, 'value', 0);
set(handles.edit_spmdir, 'enable', 'inactive');
set(handles.push_spmdir, 'string', 'Specify SPM.mat file');
set(handles.push_spec_indiv, 'enable', 'off');
set(handles.checkbox_HbO, 'enable', 'inactive');
set(handles.checkbox_HbR, 'enable', 'inactive');
set(handles.checkbox_HbT, 'enable', 'inactive');
set(handles.text7, 'enable', 'off');
set(handles.checkbox_HbO, 'value', 1);
set(handles.checkbox_HbR, 'value', 0);
set(handles.checkbox_HbT, 'value', 0);
set(handles.edit_nsubj, 'enable','off');
set(handles.edit_min_subj,'enable', 'off');
set(handles.text8, 'enable', 'off');

% Update handles structure
guidata(hObject, handles);


% --- Outputs from this function are returned to the command line.
function varargout = NIRS_Estimation_OutputFcn(hObject, eventdata, handles)
% Get default command line output from handles structure
varargout{1} = handles.output;



function edit_nirsfname_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_nirsfname_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_spmdir_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_spmdir_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in push_est.
function push_est_Callback(hObject, eventdata, handles)
h = 1-get(handles.checkbox_individual, 'value');
switch h
    case 0 %% individual analysis
        fname = get(handles.edit_spmdir,'string');
        if isempty(fname) == 1
            errordlg('Please specify SPM.mat file');
            return;
        end
        load(fname);
        switch SPM_nirs.nirs.step
            case 'specification'
                % load the measurements
                load(SPM_nirs.nirs.fname);                
                switch SPM_nirs.nirs.Hb
                    case 'HbO'
                        Y = nirs_data.oxyData;
                    case 'HbR'
                        Y = nirs_data.dxyData;
                    case 'HbT'
                        try
                            Y = nirs_data.tHbData;
                        catch
                            Y = nirs_data.oxyData + nirs_data.dxyData;
                        end
                end
                % load the GLM parameter file (SPM) after 'specification'
                if isfield(SPM_nirs.xVi,'V') == 1 % precoloring method
                    SPM_nirs = rmfield(SPM_nirs, 'xVi');
                    [SPM_nirs] = precoloring(SPM_nirs, Y);
                elseif isfield(SPM_nirs.xVi, 'V') == 0 % prewhitening method which needs the ReML
                    [SPM_nirs] = prewhitening(SPM_nirs, Y, handles.pathn);
                end
                save([handles.pathn 'SPM_indiv_' SPM_nirs.nirs.Hb '.mat'], 'SPM_nirs');
                % delete precalculated files (e.g. interpolated beta, its
                % covariance and t- or F-statistics)
                fname_others = cellstr(spm_select('FPList', handles.pathn, ['^interp.*\' SPM_nirs.nirs.Hb '.mat$']));
                if strcmp(fname_others{1}, filesep) ~= 1
                    delete(fname_others{:});
                end
                fname_others = cellstr(spm_select('FPList', handles.pathn, '^interp_matrix.*\.mat$'));
                if strcmp(fname_others{1}, filesep) ~= 1
                    choice = questdlg('Interpolating matrix files previously calculated were detected. These files may have an effect on the result. It is recommended to delete relevant files. Would you like to remove files?', ...
                        'NIRS-SPM', 'Yes', 'No', 'Yes');
                    switch choice 
                        case 'Yes' 
                            delete(fname_others{:});
                    end
                end
                msgbox('Estimation of model parameters has been completed.','NIRS-SPM Individual Analysis');
            case 'estimation'
                errordlg('Model parameter estimation has already been completed. Please perform the model specification step again.');
        end
    case 1 %% group analysis
        try 
            fname_ginterp_betas = handles.fname_ginterp_betas;
        catch
            errordlg('Please specify the individual-level interpolated betas for group analysis.');
            return
        end
        nsubj = str2num(get(handles.edit_nsubj, 'string'));
        min_subj = str2num(get(handles.edit_min_subj, 'string'));
        if isempty(min_subj) == 1
            min_subj = nsubj; %%% default of the min_subj : nsubj
            set(handles.edit_min_subj, 'string', num2str(min_subj));
        elseif min_subj < 2
            min_subj = 2;
            set(handles.edit_min_subj, 'string', num2str(min_subj));
            warndlg('The minimum number of overlapped individual subjects should be over 1.');
            return;
        end
        [gavg_beta, group_beta, xX, index_group, nsubj_mask, brain_view, fname_interp_cov] = nirs_spm_group(fname_ginterp_betas, min_subj);
        % gavg_beta: for group t-stat
        % group_beta: for group F-stat 
        % common variable: index_group, nsubj_mask, brain_view, fname_interp_var
        dir_spm = get(handles.edit_spmdir, 'string');
        SPM_nirs.nirs.level = 'group';
        SPM_nirs.nirs.nsubj = nsubj;
        SPM_nirs.nirs.min_subj = min_subj;
        if get(handles.checkbox_HbO, 'value') == 1
            SPM_nirs.nirs.Hb = 'HbO';
        elseif get(handles.checkbox_HbR, 'value') == 1
            SPM_nirs.nirs.Hb = 'HbR';
        elseif get(handles.checkbox_HbT, 'value') == 1
            SPM_nirs.nirs.Hb = 'HbT';
        end
        SPM_nirs.xX = xX;
        SPM_nirs.nirs.brain_view = brain_view; %name,index,size
        SPM_nirs.nirs.index_group = index_group;
        SPM_nirs.nirs.nsubj_mask = nsubj_mask;
        SPM_nirs.nirs.fname_ginterp_cov = fname_interp_cov;
        SPM_nirs.nirs.fname_ginterp_beta = [dir_spm filesep 'ginterp_beta_' SPM_nirs.nirs.brain_view.name SPM_nirs.nirs.Hb '.mat'];
        SPM_nirs.nirs.fname_ginterp_avg_beta = [dir_spm filesep 'ginterp_avgbeta_' SPM_nirs.nirs.brain_view.name SPM_nirs.nirs.Hb '.mat'];
        
        fname_others = cellstr(spm_select('FPList', dir_spm, ['^ginterp_T.*\' SPM_nirs.nirs.Hb '.mat$']));
        if strcmp(fname_others{1}, filesep) ~= 1
            delete(fname_others{:});
        end
        fname_others = cellstr(spm_select('FPList', dir_spm, ['^ginterp_F.*\' SPM_nirs.nirs.Hb '.mat$']));
        if strcmp(fname_others{1}, filesep) ~= 1
            delete(fname_others{:});
        end
        
        % save a SPM_nirs_group_brainview_HbX.mat in specified dir. 
        save([dir_spm filesep 'SPM_group_' SPM_nirs.nirs.brain_view.name SPM_nirs.nirs.Hb '.mat'], 'SPM_nirs');
        % save a group-average beta 
        save(SPM_nirs.nirs.fname_ginterp_avg_beta, 'gavg_beta');
        % save matrix containing individual interpolated betas 
        save(SPM_nirs.nirs.fname_ginterp_beta, 'group_beta');
        msgbox('Estimation of model parameters has been completed.','NIRS-SPM Group Analysis');
        handles = rmfield(handles, 'fname_ginterp_betas');
end


% --- Executes on button press in push_spmdir.
function push_spmdir_Callback(hObject, eventdata, handles)
h = get(handles.checkbox_individual, 'value');
if h == 1
    [filen, pathn] = uigetfile('*.mat', 'Specify SPM.mat file');
    fname = [pathn filen];
    if fname == 0
        return
    end
    load(fname);
    switch SPM_nirs.nirs.Hb
        case 'HbO'
            set(handles.checkbox_HbO, 'value', 1);
            set(handles.checkbox_HbR, 'value', 0);
            set(handles.checkbox_HbT, 'value', 0);
        case 'HbR'
            set(handles.checkbox_HbO, 'value', 0);
            set(handles.checkbox_HbR, 'value', 1);
            set(handles.checkbox_HbT, 'value', 0);
        case 'HbT'
            set(handles.checkbox_HbO, 'value', 0);
            set(handles.checkbox_HbR, 'value', 0);
            set(handles.checkbox_HbT, 'value', 1);
    end
    handles.pathn = pathn;
elseif h == 0
    cur_dir = cd;
    fname = uigetdir(cur_dir,'Select SPM directory to be estimated and saved');
    if fname == 0
        return;
    end
end
set(handles.edit_spmdir, 'string', fname);
guidata(hObject, handles);



% --- Executes on button press in checkbox_HbO.
function checkbox_HbO_Callback(hObject, eventdata, handles)

% --- Executes on button press in checkbox_HbR.
function checkbox_HbR_Callback(hObject, eventdata, handles)


% --- Executes on button press in checkbox_HbT.
function checkbox_HbT_Callback(hObject, eventdata, handles)


% --- Executes on button press in checkbox_individual.
function checkbox_individual_Callback(hObject, eventdata, handles)
h = get(handles.checkbox_individual, 'value');
if h == 1
    set(handles.checkbox_group, 'value', 0);
    set(handles.push_spec_indiv, 'enable','off');
    set(handles.push_spmdir, 'string', 'Specify SPM.mat file');
    set(handles.edit_nsubj, 'enable','off');
    set(handles.text7, 'enable', 'off');
    set(handles.edit_min_subj, 'enable', 'off');
    set(handles.text8, 'enable', 'off');
end


% --- Executes on button press in checkbox_group.
function checkbox_group_Callback(hObject, eventdata, handles)
h = get(handles.checkbox_group, 'value');
if h == 1
    set(handles.checkbox_individual, 'value', 0);
    set(handles.push_spec_indiv, 'enable', 'on');
    set(handles.push_spmdir, 'string', 'Select SPM directory to be estimated & saved');
    set(handles.edit_nsubj, 'enable', 'on');
    set(handles.text7, 'enable', 'on');
    set(handles.text8, 'enable', 'on');
    set(handles.edit_min_subj, 'enable', 'on');
end


% --- Executes on button press in push_spec_indiv.
function push_spec_indiv_Callback(hObject, eventdata, handles)
nsubj = str2num(get(handles.edit_nsubj, 'string'));
fname_ginterp_betas = {};
count = 0;
while count < nsubj
    [filen, pathn] = uigetfile('interp_beta*.mat', ['Total(' num2str(count) '/' num2str(nsubj) ') Specify individual-level interpolated betas (interp_beta_*.mat)'], 'MultiSelect','on');
    if pathn == 0 
        return;
    end
    if iscell(filen) == 1
        nfile = length(filen);
        for kk = 1:nfile 
            fname_ginterp_betas = [fname_ginterp_betas, fullfile(pathn, filen{kk})];
        end     
        count = count + nfile;
    else
        fname_ginterp_betas{count+1} = fullfile(pathn, filen);
        count = count + 1;
    end
end

if count > nsubj
    errordlg('Please select individual files less than total number of subjects for group analysis.');
    return;
end

switch fname_ginterp_betas{1}(end-6:end-4)
    case 'HbO'
        set(handles.checkbox_HbO, 'value', 1);
        set(handles.checkbox_HbR, 'value', 0);
        set(handles.checkbox_HbT, 'value', 0);
    case 'HbR'
        set(handles.checkbox_HbO, 'value', 0);
        set(handles.checkbox_HbR, 'value', 1);
        set(handles.checkbox_HbT, 'value', 0);
    case 'HbT'
        set(handles.checkbox_HbO, 'value', 0);
        set(handles.checkbox_HbR, 'value', 0);
        set(handles.checkbox_HbT, 'value', 1);
end
handles.fname_ginterp_betas = fname_ginterp_betas;
guidata(hObject, handles);


function edit_nsubj_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_nsubj_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function edit_min_subj_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function edit_min_subj_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


