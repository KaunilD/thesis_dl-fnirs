function varargout = NIRS_Results_Viewer(varargin)
% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @NIRS_Results_Viewer_OpeningFcn, ...
    'gui_OutputFcn',  @NIRS_Results_Viewer_OutputFcn, ...
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


% --- Executes just before NIRS_Results_Viewer is made visible.
function NIRS_Results_Viewer_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;

cla(handles.nirs_result, 'reset');
axis off

set(handles.checkbox_HbO, 'enable', 'inactive', 'value', 1);
set(handles.checkbox_HbR, 'enable', 'inactive', 'value', 0);
set(handles.checkbox_HbT, 'enable', 'inactive', 'value', 0);

set(handles.checkbox_EC, 'value', 1);
set(handles.checkbox_tube, 'value', 0, 'enable', 'on');
set(handles.checkbox_none, 'value', 0);
set(handles.edit_pvalue, 'string', num2str(0.05));
set(handles.edit_spmnirsdir, 'enable', 'inactive','string', '');
set(handles.edit_chpos, 'enable', 'inactive', 'string', '');

string_view{1,1} = '';
string_view{2,1} = 'Ventral view';
string_view{3,1} = 'Dorsal view';
string_view{4,1} = 'Right Lateral view';
string_view{5,1} = 'Left Lateral view';
string_view{6,1} = 'Frontal view';
string_view{7,1} = 'Occipital view';
set(handles.popup_viewBrain,'string', string_view,'value',1);

set(handles.push_export, 'enable', 'off');

% Update handles structure
guidata(hObject, handles);



% --- Outputs from this function are returned to the command line.
function varargout = NIRS_Results_Viewer_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;


% --- Executes on button press in checkbox_fMRI.
function checkbox_fMRI_Callback(hObject, eventdata, handles)


% --- Executes on button press in push_spmnirsdir.
function push_spmnirsdir_Callback(hObject, eventdata, handles)
[filen, pathn] = uigetfile('*.mat', 'Specify SPM.mat file');
fname = [pathn filen];
if fname == 0
    return
end
if strcmp(filen(1:9), 'SPM_indiv') ~=1 && strcmp(filen(1:9), 'SPM_group') ~= 1
    errordlg('Please select the SPM_indiv_HbX.mat or SPM_group_HbX.mat file');
    return;
end

load(fname);
switch SPM_nirs.nirs.Hb
    case 'HbO'
        set(handles.checkbox_HbO, 'value', 1, 'enable', 'inactive');
        set(handles.checkbox_HbR, 'value', 0, 'enable', 'off');
        set(handles.checkbox_HbT, 'value', 0, 'enable', 'off');
    case 'HbR'
        set(handles.checkbox_HbO, 'value', 0, 'enable', 'off');
        set(handles.checkbox_HbR, 'value', 1, 'enable', 'inactive');
        set(handles.checkbox_HbT, 'value', 0, 'enable', 'off');
    case 'HbT'
        set(handles.checkbox_HbO, 'value', 0, 'enable', 'off');
        set(handles.checkbox_HbR, 'value', 0, 'enable', 'off');
        set(handles.checkbox_HbT, 'value', 1, 'enable', 'inactive');
end

switch SPM_nirs.nirs.level
    case 'individual'
        set(handles.push_contrast, 'enable', 'on');
        set(handles.checkbox_tube, 'enable', 'on');
        set(handles.push_chpos, 'enable', 'on');
        set(handles.edit_chpos, 'enable', 'on');
        set(handles.popup_viewBrain, 'enable', 'on');
        try
            load(get(handles.edit_chpos,'string'));
            count = 2;
            string_view{1} = '';
            for kk = 1:6
                if isempty(find(preproc_info.rend_ch_pos{kk}.rchn ~= -1)) == 0
                    switch kk
                        case 1
                            string_view{count,1} = 'Ventral view';
                        case 2
                            string_view{count,1} = 'Dorsal view';
                        case 3
                            string_view{count,1} = 'Right Lateral view';
                        case 4
                            string_view{count,1} = 'Left Lateral view';
                        case 5
                            string_view{count,1} = 'Frontal view';
                        case 6
                            string_view{count,1} = 'Occipital view';
                    end
                    count = count + 1;
                end
            end
            set(handles.popup_viewBrain,'string', string_view,'value',1);
        end
    case 'group'
        % set(handles.push_contrast, 'enable', 'off');
        set(handles.checkbox_tube, 'enable', 'off', 'value', 0);
        set(handles.push_chpos, 'enable', 'off');
        set(handles.edit_chpos, 'enable', 'off');
        set(handles.checkbox_none, 'value', 0);
        set(handles.checkbox_EC, 'value', 1);
        switch SPM_nirs.nirs.brain_view.name
            case 'ventral'
                string_view{1,1} = 'Ventral view';
            case 'dorsal'
                string_view{1,1} = 'Dorsal view';
            case 'right'
                string_view{1,1} = 'Right Lateral view';
            case 'left'
                string_view{1,1} = 'Left Lateral view';
            case 'frontal'
                string_view{1,1} = 'Frontal view';
            case 'occipital'
                string_view{1,1} = 'Occipital view';
        end
        set(handles.popup_viewBrain,'string', string_view, 'enable', 'inactive','value',1);
end
set(handles.edit_spmnirsdir,'string', fname);
handles.pathn = pathn; % directory containing SPM_nirs.mat file

try
    handles = rmfield(handles, 'Ic');
end

guidata(hObject, handles);


% --- Executes on button press in push_chpos.
function push_chpos_Callback(hObject, eventdata, handles)
[filen, pathn] = uigetfile('*.mat','Select channel and optode position .mat file');
path_file_n = [pathn filen];
if path_file_n == 0
    return
end

set(handles.edit_chpos,'string',path_file_n);
set(handles.push_contrast,'enable','on');
load(path_file_n);

flag = 0;
count = 2;
string_view{1} = '';
for kk = 1:6
    rchn = preproc_info.rend_ch_pos{kk}.rchn;
    if isfield(preproc_info, 'ch_set') == 1
        for aa = 1:length(preproc_info.ch_set)
            tmp_index = find(rchn(preproc_info.ch_set(aa).ch) ~= -1);
            if length(tmp_index) > 2
                flag = 1;
            end
        end
    elseif isfield(preproc_info, 'ch_set') == 0 % ch-set was not defined
        tmp_index = find(rchn ~= -1);
        if length(tmp_index) > 2
            flag = 1;
        end
    end
    
    if flag == 1
        switch kk
            case 1
                string_view{count,1} = 'Ventral view';
            case 2
                string_view{count,1} = 'Dorsal view';
            case 3
                string_view{count,1} = 'Right Lateral view';
            case 4
                string_view{count,1} = 'Left Lateral view';
            case 5
                string_view{count,1} = 'Frontal view';
            case 6
                string_view{count,1} = 'Occipital view';
        end
        count = count + 1;
        flag = 0;
    end
end

try
    handles = rmfield(handles, 'Ic');
end

set(handles.popup_viewBrain,'string', string_view,'value',1);


% --- Executes on button press in push_contrast.
function push_contrast_Callback(hObject, eventdata, handles)
% contrast step:
% SPM load, selection of contrast vector

% get the SPM
filen_SPM = get(handles.edit_spmnirsdir, 'string');
if isempty(filen_SPM) == 1
    errordlg('Please specify the SPM_indiv_HbX.mat file');
    return;
end
load(filen_SPM);
dir_spm = handles.pathn; % SPM_nirs.mat directory


% specification of contrast vector
[Ic, xCon] = nirs_spm_conman(SPM_nirs, 'T&F', Inf, 'Select contrasts...', 'for conjunction', 1);
SPM_nirs.xCon = xCon;
set(handles.checkbox_tube, 'enable', 'off', 'value', 0); % Tube formula is not available in F-stat and group analysis

switch SPM_nirs.nirs.level
    case 'individual'% individual analysis
        % receive channel information
        path_filen = get(handles.edit_chpos, 'string');
        if isempty(path_filen) == 1
            errordlg('Please specify the file to contain the information of the channel locations');
            return;
        end
        load(path_filen);
        
        % specify the view of rendered brain
        view_brain = get(handles.popup_viewBrain, 'value');
        string_brain = get(handles.popup_viewBrain, 'string');
        
        switch string_brain{view_brain,1}
            case ''
                errordlg('Please specify the view of the brain.');
                return;
            case 'Ventral view'
                spec_hemi = 'ventral';
                side_hemi = 1;
            case 'Dorsal view'
                spec_hemi = 'dorsal';
                side_hemi = 2;
            case 'Right Lateral view'
                spec_hemi = 'right';
                side_hemi = 3;
            case 'Left Lateral view'
                spec_hemi = 'left';
                side_hemi = 4;
            case 'Frontal view'
                spec_hemi = 'frontal';
                side_hemi = 5;
            case 'Occipital view'
                spec_hemi = 'occipital';
                side_hemi = 6;
        end
        
        % all channel information on the specific view of rendered brain
        rchn = preproc_info.rend_ch_pos{side_hemi}.rchn;
        cchn = preproc_info.rend_ch_pos{side_hemi}.cchn;
        s1 = size(preproc_info.rend_ch_pos{side_hemi}.ren, 1);
        s2 = size(preproc_info.rend_ch_pos{side_hemi}.ren, 2);
        [x, y] = meshgrid(1:s2, 1:s1); %s1, s2: size of specific view of rendered brain
        
        % channel information within specific set
        if isfield(preproc_info, 'ch_set') == 1
            total_set = size(preproc_info.ch_set,2);
            for kk = 1:total_set 
                index = unique([find(rchn(preproc_info.ch_set(kk).ch) ~= -1) find(cchn(preproc_info.ch_set(kk).ch) ~= -1)]);
                chs{kk} = preproc_info.ch_set(kk).ch(index); % channel number within specific set
                name{kk} = preproc_info.ch_set(kk).name;
            end
        elseif isfield(preproc_info, 'ch_set') == 0
            total_set = 1;
            chs{1} = find(rchn ~= -1);
            name{1} = 'Set #1';
        end
        
        %------------------------------------------------------------------
        % 1. loading interpolating matrix and calculating LKC values
        %------------------------------------------------------------------
        B = {};
        Bx = {};
        By = {};
        
        index_mask = [];
        index_mask_set = {};
        
        filen_interp_matrix = [dir_spm 'interp_matrix_' spec_hemi '.mat'];
        fid = fopen(filen_interp_matrix);
        
        if fid ~= -1 % if interpolating matrix exists,
            fclose(fid);
            try
                load(filen_interp_matrix);
            end
        end
        % if interpolating matrix does not exist,
        if isempty(B) == 1 || isempty(index_mask_set) == 1
            % extraction of interpolating kernels
            L2_set = [];
            
            for kk = 1:total_set
                nch = length(chs{kk}); % # of channels
                if nch > 2 %minimum # of channels for interpolation should be more than 2
                    % we consider NIRS signals from different set of channel are
                    % independent
                    rch_set = rchn(chs{kk});
                    cch_set = cchn(chs{kk});
                    mtx_eye = eye(nch);
                    B_set = [];
                    Bx_set = [];
                    By_set = [];
                    
                    % B_volume: only used for expected-EC
                    B_volume = zeros(s1, s2, nch);
                    for aa = 1:nch
                        disp(['Extracting interpolating kernels depending on Ch#' num2str(aa) ' (' name{kk} ')']);
                        grid_eye = griddata(cch_set, rch_set, (mtx_eye(:,aa))', x, y, 'cubic');
                        if aa == 1
                            mask = 1 - isnan(grid_eye);
                            index_mask_set{kk,1} = find(mask == 1);
                        end
                        B_set(aa, :) = grid_eye(index_mask_set{kk})';
                        grid_eye(find(mask == 0)) = 0;
                        B_volume(:,:,aa) = grid_eye;
                        
                        [Bx_ch By_ch] = gradient(grid_eye);
                        Bx_set(aa,:) = Bx_ch(index_mask_set{kk})';
                        By_set(aa,:) = By_ch(index_mask_set{kk})';
                    end
                    disp('Completed.');
                    B{kk} = B_set;
                    Bx{kk} = Bx_set;
                    By{kk} = By_set;
                    clear B_set;
                    clear Bx_set;
                    clear By_set;
                    
                    %%% calculation of LKC for expected EC-based p-value correction
                    res = SPM_nirs.nirs.res(:,chs{kk});
                    disp(['Calculating Lipschitz-Killing curvatures (' name{kk} ')...']);
                    [L2_set(kk)] = calc_LKC(B_volume, mask, res, SPM_nirs.nirs.level);
                    disp('Completed.');
                else
                    disp('The number of channels should be more than 2 for generating the interpolated statistical maps');
                    return;
                end
            end
            % save interpolating kernels
            save([dir_spm 'interp_matrix_' spec_hemi '.mat'], 'B', 'index_mask_set');
            % save gradient of interpolating kernels in x and y direction
            save([dir_spm 'interp_matrix_grad_' spec_hemi '.mat'], 'Bx', 'By', 'index_mask_set');
            
            % calculating other Lipschitz-Killing curvature
            L2 = sum(L2_set);
            r = sqrt(L2./pi);
            L1 = pi * r;
            L0 = 1;
            % save LKC values on the SPM_nirs structures
            SPM_nirs.nirs.LKC{side_hemi} = [L0 L1 L2];
        else
            flag = 0;
            try
                LKC = SPM_nirs.nirs.LKC{side_hemi};
                if isempty(LKC) == 0 && size(LKC,2) == 3
                    % both interpolating matrix and LKC values exist
                    flag = 1;
                end
            end
            if flag == 0 % interpolating matrix exists, but LKC does not exist
                L2_set = [];
                for kk = 1:total_set
                    if isempty(B{kk}) == 0
                        nch = length(chs{kk});
                        B_volume = zeros(s1, s2, nch);
                        for aa = 1:nch
                            B_ch = zeros(s1, s2);
                            B_ch(index_mask_set{kk}) = B{kk}(aa,:);
                            B_volume(:,:,aa) = B_ch;
                        end
                        clear B_ch;
                        mask = zeros(s1, s2);
                        mask(index_mask_set{kk}) = 1;
                        res = SPM_nirs.nirs.res(:,chs{kk});
                        disp(['Calculating Lipschitz-Killing curvatures (' name{kk} ')...']);
                        [L2_set(kk)] = calc_LKC(B_volume, mask, res, SPM_nirs.nirs.level);
                        disp('completed.');
                    end
                end
                L2 = sum(L2_set);
                r = sqrt(L2./pi);
                L1 = pi * r;
                L0 = 1;
                % save LKC values on the SPM_nirs structures
                SPM_nirs.nirs.LKC{side_hemi} = [L0 L1 L2];
            end
        end
        % finish (interpolating kernel, LKC matrix)
        
        %------------------------------------------------------------------
        % 2. calculating individual-level interpolated betas and its
        % covariances (before multiplicatioin with contrast vector)
        %------------------------------------------------------------------
        interp_beta = [];
        interp_var = [];
        index_mask = [];
        df = [];
        % try loading a individual-level interpolated beta file
        filename_beta = [dir_spm 'interp_beta_' spec_hemi SPM_nirs.nirs.Hb '.mat'];
        fid = fopen(filename_beta);
        if fid ~= -1 % if exists
            fclose(fid);
            try
                load(filename_beta);
            end
        end
        filename_cov = [dir_spm 'interp_cov_' spec_hemi SPM_nirs.nirs.Hb '.mat'];
        fid = fopen(filename_cov);
        if fid ~= -1
            fclose(fid);
            try
                load(filename_cov);
            end
        end
        
        if isempty(interp_beta) == 1 || isempty(interp_var) == 1 || isempty(index_mask) == 1
            index_mask = cell2mat(index_mask_set);
            interp_beta = []; % interpolated betas
            interp_var = [];
            cov_beta_r = {};
            var = SPM_nirs.nirs.ResSS./SPM_nirs.xX.trRV;
            xCor = SPM_nirs.xX.Bcov;
            for kk = 1:total_set
                if isempty(B{kk}) == 0 % interpolating matrix exists
                    interp_beta = [interp_beta SPM_nirs.nirs.beta(:,chs{kk}) * B{kk}];
                    nch = length(chs{kk}); % # of channels
                    mtx_var = zeros(nch);
                    for aa = 1:nch
                        for bb = 1:nch
                            mtx_var(aa,bb) = var(chs{kk}(aa), chs{kk}(bb));
                        end
                    end
                    [V_X D_X] = eig(mtx_var);
                    tmp = D_X.^(1/2) * V_X' * B{kk};
                    interp_var = [interp_var sum(tmp.^2,1)];
                    
                    % calculation of kappa value (only used in tube)
                    cov_beta = kron(mtx_var, xCor);
                    [U,S,V] = svd(cov_beta);
                    cov_beta_r{kk} = U*(S.^(0.5))*V';
                end
            end
            df(2) = SPM_nirs.xX.erdf;
            save(filename_cov, 'interp_var', 'index_mask', 'xCor', 'df', 'cov_beta_r'); clear xCor;
            % save interpolated betas and variance for group analysis
            xX = SPM_nirs.xX; save(filename_beta, 'interp_beta', 'index_mask', 'xX'); clear xX;
            % xX only used for contrast manager of group analysis
        end
        % finish (individual-level interpolated betas and its covariance
        
        %------------------------------------------------------------------
        % 3. calculation of individual-level T- or F- stat
        %------------------------------------------------------------------
        stat = []; % T- or F-statistic matrix
        % loading a file (interpolated T- or F-statistics)
        filename = [dir_spm 'interp_' xCon(Ic).STAT '_' num2str(Ic) '_' spec_hemi SPM_nirs.nirs.Hb '.mat'];
        fid = fopen(filename);
        if fid ~= -1 % if exists
            fclose(fid);
            try
                load(filename);
            end
        end
        if isempty(stat) == 1 || isempty(index_mask) == 1
            switch xCon(Ic).STAT
                case 'T' % t-statistic calculation
                    % load gradient of interpolation matrix in x,y direction
                    if isempty(Bx) == 1 || isempty(By) == 1
                        load([dir_spm 'interp_matrix_grad_' spec_hemi '.mat']);
                    end
                    % covariance of interpolated beta
                    cxCor = xCon(Ic).c' * SPM_nirs.xX.Bcov * xCon(Ic).c;
                    kappa_set = [];
                    for kk = 1:total_set
                        if isempty(B{kk}) == 0 % interpolating matrix exists,
                            disp(['Calculating kappa values for tube formula correction (' name{kk} ')...']);
                            kappa_set(kk) = calc_kappa(B{kk},Bx{kk},By{kk},cov_beta_r{kk}, xCon(Ic).c);
                            disp('Completed.');
                        end
                    end
                    kappa = sum(kappa_set); % summation of kappa values
                    SPM_nirs.nirs.kappa(Ic, side_hemi) = kappa;
                    stat = (xCon(Ic).c' * interp_beta)./sqrt(cxCor .* interp_var);
                    % save interpolated t-statistics
                    filename = [dir_spm 'interp_T_' num2str(Ic) '_' spec_hemi SPM_nirs.nirs.Hb '.mat'];
                case 'F'
                    [stat df] = calc_F_stat(SPM_nirs, B, chs, xCon(Ic).c);
                    % save interpolated F-statistics
                    filename = [dir_spm 'interp_F_' num2str(Ic) '_' spec_hemi SPM_nirs.nirs.Hb '.mat'];
            end
            stat = stat(:);
            save(filename, 'stat', 'df', 'index_mask');
        end
        % end of calculation of individual statistics
        clear interp_beta;
        brain = preproc_info.rend_ch_pos{side_hemi}.ren;
        brain = brain * 0.5;
        set(handles.push_export, 'enable', 'on');
        if strcmp(xCon(Ic).STAT, 'T') 
            set(handles.checkbox_tube, 'enable','on');
        end            
    case 'group'
        %------------------------------------------------------------------
        % calculation of group-level T- or F-statistics
        %------------------------------------------------------------------
        stat = []; % T- or F-statistic matrix
        % loading a file (interpolated T- or F-statistics)
        filename = [dir_spm 'ginterp_' xCon(Ic).STAT '_' num2str(Ic) '_' SPM_nirs.nirs.brain_view.name SPM_nirs.nirs.Hb '.mat'];
        fid = fopen(filename);
        if fid ~= -1 % if exists
            fclose(fid);
            try
                load(filename);
            end
        end
        if isempty(stat) == 1 || isempty(index_mask) == 1
            index_group = SPM_nirs.nirs.index_group;
            nvox = length(index_group);
            nvox_brain = SPM_nirs.nirs.brain_view.size(1) * SPM_nirs.nirs.brain_view.size(2);
            nsubj = SPM_nirs.nirs.nsubj;
            L = SPM_nirs.nirs.nsubj_mask(index_group); % # of subject on each voxel
            
            % contrast * group-level interp beta
            load(SPM_nirs.nirs.fname_ginterp_avg_beta);
            cgavg_beta = xCon(Ic).c' * gavg_beta;
            load(SPM_nirs.nirs.fname_ginterp_beta);
            
            % group-level residuals
            Xg = ones(nsubj, 1);
            Yg = xCon(Ic).c' * [group_beta{:}];
            Yg = reshape(Yg', [nvox, nsubj])';
            res = Yg - Xg * cgavg_beta;
            
            % LKC calculation
            disp('Calculating Lipschitz-Killing curvatures ...');
            [L2] = calc_LKC(index_group, SPM_nirs.nirs.brain_view.size, res, SPM_nirs.nirs.level);
            disp('Completed.');
            r = sqrt(L2./pi);
            L1 = pi * r;
            L0 = 1;
            SPM_nirs.nirs.LKC{SPM_nirs.nirs.brain_view.index} = [L0 L1 L2];
            
            df = []; % degree of freedom
            switch xCon(Ic).STAT
                case 'T' % t-statistic calculation
                    var_s = zeros(1, nvox);
                    for kk = 1:nsubj
                        var_s = var_s + (xCon(Ic).c' * group_beta{kk} - cgavg_beta).^2;
                    end
                    var_s = var_s ./ (L-1);
                    gsum_var = zeros(1, nvox_brain);
                    term_df = zeros(1, nvox_brain);
                    
                    % calculation group-level variance
                    for kk = 1:nsubj
                        load(SPM_nirs.nirs.fname_ginterp_cov{kk});
                        indiv_cov = interp_var .* (xCon(Ic).c' * xCor * xCon(Ic).c);
                        gsum_var(index_mask) = gsum_var(index_mask) + indiv_cov;
                        term_df(index_mask) = term_df(index_mask) + (indiv_cov.^2)./df(2);
                    end
                    gsum_var = gsum_var(index_group);
                    term_df = term_df(index_group);
                    term = var_s .* L + gsum_var;
                    
                    stat = (xCon(Ic).c' * gavg_beta)./(sqrt(term)./L);
                    term2 = ((L.^2)./(L-1)).* (var_s.^2);
                    df(2) = sum((term.^2) ./ (term2 + term_df))./nvox; % degree of freedom
                case 'F' % F-statistic calculation
                    X1o = pinv(Xg');
                    [trMV, trMVMV] = spm_SpUtil('trMV', X1o, 1);
                    df(1) = trMV^2/trMVMV;
                    R = eye(nsubj) - Xg*pinv(Xg);
                    RVR = sum(res.^2)./trace(R);
                    MVM = (inv(sum(X1o.^2)).* (cgavg_beta.^2))./trMV;
                    stat = MVM./RVR;
                    df(2) = trace(R)^2./trace(R*R);
            end
            stat = stat(:);
            index_mask = index_group;
            save(filename, 'stat', 'df', 'index_mask');
        end
        side_hemi = SPM_nirs.nirs.brain_view.index;
        load([spm('dir') filesep 'rend' filesep 'render_single_subj.mat']);
        %%% modified (2008. 10. 13) %%%
        brain = rend{side_hemi}.ren;
        if issparse(brain),
            d = size(brain);
            B1 = spm_dctmtx(d(1),d(1));
            B2 = spm_dctmtx(d(2),d(2));
            brain = B1*brain*B2';
        end
        msk = find(brain>1);brain(msk)=1;
        msk = find(brain<0);brain(msk)=0;
        brain = brain(end:-1:1,:);
        brain = brain * 0.5;
        set(handles.push_export, 'enable', 'off');
end

% min and max values of statistics for colormap control
min_stat = min(stat);
max_stat = max(stat);
smin_stat = max_stat - ((max_stat - min_stat)./63) * 127;
sbar = linspace(smin_stat, max_stat, 128);

stat_brain = ((-sbar(1) + sbar(64))/(0.5)).*brain + sbar(1);
stat_brain(index_mask) = stat;

axes(handles.nirs_result);
imagesc(stat_brain);
load Split
colormap(split)
axis off
axis image
hc = colorbar;
set(hc, 'YLim', [sbar(65) sbar(128)]);
y_tick = linspace(sbar(65), sbar(128), 5)';
set(hc, 'YTick', y_tick);
set(hc, 'FontSize', 8);

handles.Ic = Ic;

% update & save the SPM_indiv_HbX.mat file
save(filen_SPM, 'SPM_nirs');
guidata(hObject, handles);


function edit_spmnirsdir_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_spmnirsdir_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_chpos_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_chpos_CreateFcn(hObject, eventdata, handles)
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in checkbox_HbO.
function checkbox_HbO_Callback(hObject, eventdata, handles)

% --- Executes on button press in checkbox_HbR.
function checkbox_HbR_Callback(hObject, eventdata, handles)

% --- Executes on button press in checkbox_HbT.
function checkbox_HbT_Callback(hObject, eventdata, handles)

% --- Executes on button press in checkbox_tube.
function checkbox_tube_Callback(hObject, eventdata, handles)
h = get(handles.checkbox_tube, 'value');
if h == 1
    set(handles.checkbox_none, 'value', 0);
    set(handles.checkbox_EC, 'value', 0);
end

% --- Executes on button press in checkbox_none.
function checkbox_none_Callback(hObject, eventdata, handles)
h = get(handles.checkbox_none, 'value');
if h == 1
    set(handles.checkbox_tube, 'value', 0);
    set(handles.checkbox_EC, 'value', 0);
end


function edit_pvalue_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_pvalue_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in push_view.
function push_view_Callback(hObject, eventdata, handles)
% in this routine, inference about brain activation will be performed.
% to achieve that, threshold value and interpolated T- or F-statistic
% should be calculated.

% load SPM_nirs file
filen_SPM = get(handles.edit_spmnirsdir, 'string');
if isempty(filen_SPM) == 1
    errordlg('Please specify a SPM_indiv_HbX.mat or SPM_group_HbX.mat file.');
    return;
end
load(filen_SPM);
dir_spm = handles.pathn; % SPM directory

% specification of p-value correction method (EC, tube, or none).
h(1) = get(handles.checkbox_EC, 'value');
h(2) = get(handles.checkbox_tube, 'value');
h(3) = get(handles.checkbox_none, 'value');

if sum(h) == 0
    errordlg('Please specify a method for p-value correction.');
    return;
end

try
    Ic = handles.Ic;
catch
    errordlg('Please specify the contrast vector. It is recommended to perform CONTRAST routine first.');
    return;
end

switch SPM_nirs.nirs.level
    case 'individual' % individual inference
        % specification of brain view
        view_brain = get(handles.popup_viewBrain, 'value');
        string_brain = get(handles.popup_viewBrain, 'string');
        switch string_brain{view_brain,1}
            case ''
                errordlg('Please specify the view of the brain.');
                return;
            case 'Ventral view'
                spec_hemi = 'ventral';
                side_hemi = 1;
            case 'Dorsal view'
                spec_hemi = 'dorsal';
                side_hemi = 2;
            case 'Right Lateral view'
                spec_hemi = 'right';
                side_hemi = 3;
            case 'Left Lateral view'
                spec_hemi = 'left';
                side_hemi = 4;
            case 'Frontal view'
                spec_hemi = 'frontal';
                side_hemi = 5;
            case 'Occipital view'
                spec_hemi = 'occipital';
                side_hemi = 6;
        end
        % specification of channel positions
        filen = get(handles.edit_chpos, 'string');
        if isempty(filen) == 1
            errordlg('Please specify a file to contain information of channel positions.');
            return;
        end
        load(filen);
        brain = preproc_info.rend_ch_pos{side_hemi}.ren;
        brain = brain .* 0.5;
        filen = [dir_spm 'interp_' SPM_nirs.xCon(Ic).STAT '_' num2str(Ic) '_' spec_hemi SPM_nirs.nirs.Hb '.mat'];
    case 'group'
        side_hemi = SPM_nirs.nirs.brain_view.index;
        load([spm('dir') filesep 'rend' filesep 'render_single_subj.mat']);
        %%% modified (2008. 10. 13) %%%
        brain = rend{side_hemi}.ren;
        if issparse(brain),
            d = size(brain);
            B1 = spm_dctmtx(d(1),d(1));
            B2 = spm_dctmtx(d(2),d(2));
            brain = B1*brain*B2';
        end
        msk = find(brain>1);brain(msk)=1;
        msk = find(brain<0);brain(msk)=0;
        brain = brain(end:-1:1,:);
        brain = brain * 0.5;
        filen = [dir_spm 'ginterp_' SPM_nirs.xCon(Ic).STAT '_' num2str(Ic) '_' SPM_nirs.nirs.brain_view.name SPM_nirs.nirs.Hb '.mat'];
end

% load interpolated T- or F- statistics
fid = fopen(filen);
if fid == -1 % specified file does not exist,
    errordlg('Please calculate interpolated T- or F-statistics. It is recommended to perform CONTRAST routine first.');
    return;
else % if exists,
    fclose(fid);
    load(filen); % stat and df matrix will be loaded.
end

p_value = str2num(get(handles.edit_pvalue, 'string'));

% calculation of threshold value, depending on a correction method
if h(1) == 1 && h(2) == 0 && h(3) == 0 % EC-based correction
    threshold = calc_EC(SPM_nirs.nirs.LKC{side_hemi}, p_value, SPM_nirs.xCon(Ic).STAT, df);
elseif h(1) == 0 && h(2) == 1 && h(3) == 0 % tube formula-based correction
    threshold = calc_tube(SPM_nirs.nirs.kappa(Ic, side_hemi), p_value);
elseif h(1) == 0 && h(2) == 0 && h(3) == 1 % none correction
    threshold = spm_u(p_value, df, SPM_nirs.xCon(Ic).STAT);
else
    return;
end

index_over = find(stat > threshold);
if isempty(index_over) == 1
    warndlg('There is no significant voxel.');
    return;
else
    index_mask = index_mask(index_over);
    stat = stat(index_over);
    
    % min and max values of statistics for colormap control
    min_stat = min(stat);
    max_stat = max(stat);
    smin_stat = max_stat - ((max_stat - min_stat)./63) * 127;
    sbar = linspace(smin_stat, max_stat, 128);
    
    stat_brain = ((-sbar(1) + sbar(64))/(0.5)).*brain + sbar(1);
    stat_brain(index_mask) = stat;
    
    flag_fmri = get(handles.checkbox_fMRI, 'value');
    if flag_fmri == 1
        global fmri_result_path;
        cur_dir = cd;
        if isempty(fmri_result_path)
            errordlg('Please make inference of fMRI activation using Results fMRI button.');
            set(handles.checkbox_fMRI, 'value', 0);
            return;
        end
        cd(fmri_result_path);
        load fmri_result;
        cd(cur_dir);
        index_edge_fmri = find(edge(fmri_result{side_hemi}.data, 'sobel') == 1);
        stat_brain(index_edge_fmri) = sbar(85);
    end
    
    axes(handles.nirs_result);
    imagesc(stat_brain);
    load Split
    colormap(split)
    axis off
    axis image
    hc = colorbar;
    set(hc, 'YLim', [sbar(65) sbar(128)]);
    y_tick = linspace(sbar(65), sbar(128), 5)';
    set(hc, 'YTick', y_tick);
    set(hc, 'FontSize', 8);
end


handles.th_z = threshold;
guidata(hObject, handles);



% --- Executes on selection change in popup_viewBrain.
function popup_viewBrain_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function popup_viewBrain_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in push_save_image.
function push_save_image_Callback(hObject, eventdata, handles)
% save the current axes as png file format
[filen pathn] = uiputfile('*.png', 'Save the current image as');
path_filen = [pathn filen];
if path_filen == 0
    return;
end
F = getframe(gca);
imwrite(F.cdata, path_filen);

filen_gui = [filen(1:end-4) '_with_GUI.png'];
path_filen2 = [pathn filen_gui];
% save the current image with GUI pannel as png file format
F = getframe(gcf);
imwrite(F.cdata, path_filen2);

% --- Executes on button press in push_export.
function push_export_Callback(hObject, eventdata, handles)
% channe-wise statistics
% exporting the 1) beta, 2) t- or F-statistics
filen_SPM = get(handles.edit_spmnirsdir, 'string');
if isempty(filen_SPM) == 1
    errordlg('Please specify the SPM_indiv_HbX.mat or SPM_group_HbX.mat file');
    return;
end
load(filen_SPM);
dir_spm = handles.pathn; % SPM directory

view_brain = get(handles.popup_viewBrain, 'value');
string_brain = get(handles.popup_viewBrain, 'string');
switch string_brain{view_brain,1}
    case ''
        errordlg('Please specify the view of the brain.');
        return;
    case 'Ventral view'
        spec_hemi = 'ventral';
        side_hemi = 1;
    case 'Dorsal view'
        spec_hemi = 'dorsal';
        side_hemi = 2;
    case 'Right Lateral view'
        spec_hemi = 'right';
        side_hemi = 3;
    case 'Left Lateral view'
        spec_hemi = 'left';
        side_hemi = 4;
    case 'Frontal view'
        spec_hemi = 'frontal';
        side_hemi = 5;
    case 'Occipital view'
        spec_hemi = 'occipital';
        side_hemi = 6;
end

filen_ch = get(handles.edit_chpos, 'string');
try
    load(filen_ch);
catch
    errordlg('Problem in loading the channel file.');
    return;
end
nch = size(preproc_info.ch_MNI_mm, 2);
bs = size(preproc_info.rend_ch_pos{side_hemi}.ren);
MNI_xyz = round(preproc_info.ch_MNI_mm);
rchn = preproc_info.rend_ch_pos{side_hemi}.rchn;
cchn = preproc_info.rend_ch_pos{side_hemi}.cchn;

p_value = get(handles.edit_pvalue, 'string');
if get(handles.checkbox_tube, 'value') == 1
    str_correction = 'tube formula correction';
end
if get(handles.checkbox_none, 'value') == 1
    str_correction = 'uncorrected';
end
if get(handles.checkbox_EC, 'value') == 1
    str_correction = 'LKC-based EC correction';
end

th_z = handles.th_z;
str_th_z = num2str(th_z);

switch SPM_nirs.nirs.level
    case 'individual'
        try
            Ic = handles.Ic;
        catch
            errordlg('Please specify the contrast vector first.');
            return;
        end
        
        % load interpolated betas
        fname_beta = [dir_spm 'interp_beta_' spec_hemi SPM_nirs.nirs.Hb '.mat'];
        try
            load(fname_beta);
        catch
            errordlg('Problem in loading the interpolated beta file.');
            return;
        end
        cbeta = zeros(bs);
        cbeta(index_mask) = SPM_nirs.xCon(Ic).c' * interp_beta;
        cbeta = reshape(cbeta, [bs(1) bs(2)]);
        
        cbeta_ch = {};
        for kk = 1:nch
            if rchn(kk) ~= -1 && cchn(kk) ~= -1
                cbeta_ch{kk} = num2str(cbeta(rchn(kk), cchn(kk)));
            else
                cbeta_ch{kk} = 'N/A';
            end
        end
        % load T- or F-stat
        fname_stat = [dir_spm 'interp_' SPM_nirs.xCon(Ic).STAT '_' num2str(Ic) '_' spec_hemi SPM_nirs.nirs.Hb '.mat'];
        try
            load(fname_stat);
        catch
            errordlg('Problem in loading the t- or F-stat file.');
            return;
        end
        stat_mtx = zeros(bs);
        stat_mtx(index_mask) = stat;
        stat_mtx = reshape(stat_mtx, [bs(1) bs(2)]);
        stat_ch = {};
        for kk = 1:nch
            if rchn(kk) ~= -1 && cchn(kk) ~= -1
                stat_ch{kk} = num2str(stat_mtx(rchn(kk), cchn(kk)));
            else
                stat_ch{kk} = 'N/A';
            end
        end
        
        [filen, pathn] = uiputfile('*.txt', ['Save the beta and t- or F-statistics on channels as ']);
        path_file_n = [pathn filen];
        if path_file_n == 0
            return;
        end
        
        try
            fid = fopen(path_file_n, 'w');
            fprintf(fid, '%s\r\n', ['File name (SPM): ' filen_SPM]);
            fprintf(fid, '%s\r\n', ['File name (Channel): ' filen_ch]);
            fprintf(fid, '%s\r\n', ['Brain view: ' spec_hemi]);
            fprintf(fid, '%s\r\n', ['Contrast vector: ' num2str(SPM_nirs.xCon(Ic).c')]);
            fprintf(fid, '%s\r\n', ['p-value: ' p_value ' (' str_correction ')']);
            fprintf(fid, '%s\r\n', ['its corresponding z-threshold: ' str_th_z]);
            fprintf(fid, '%s\r\n', []);
            fprintf(fid, '%s\r\n', ['CH # (x,y,z [MNI,mm]), Beta, ' SPM_nirs.xCon(Ic).STAT '-statistics']);
            
            for kk = 1:nch
                fprintf(fid, '%s\r\n', ['CH ' num2str(kk) ' (' num2str(MNI_xyz(1,kk)) ', ' num2str(MNI_xyz(2,kk)) ', ' num2str(MNI_xyz(3,kk)) '): ' cbeta_ch{kk} ', ' stat_ch{kk}]);
            end
            fclose(fid);
        end
    case 'group'
        errordlg('This function does not support for group analysis');
        return;
end



% --- Executes on button press in checkbox_EC.
function checkbox_EC_Callback(hObject, eventdata, handles)
h = get(handles.checkbox_EC, 'value');
if h == 1
    set(handles.checkbox_none, 'value', 0);
    set(handles.checkbox_tube, 'value', 0);
end

