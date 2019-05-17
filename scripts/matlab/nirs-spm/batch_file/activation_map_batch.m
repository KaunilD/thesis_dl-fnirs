function [stat_brain, act_brain, threshold] = activation_map_batch(filen_SPM, filen_ch, con_name, con_vec, STAT, spec_hemi, p_value, correct_p, disp_fig)
% NIRS-SPM batch script for 'Results NIRS' routine, which calculates the
% activation map over the threshold
%
% Input variables
% 1. filen_SPM : the name of file which results from model estimation,
% e.g.,'...\NIRS_SPM_v3_3\Sample_data\categorical_indiv_HbO\SPM_indiv_HbO.mat';
% 2. filen_ch : the name of nirs channel file,
% e.g.,'...\NIRS_SPM_v3_3\Sample_data\Registration\channel_NIRS_fMRI.mat';
% 3. con_name : contrast name, e.g., 'right finger tapping task'
% 4. con_vec: contrast vector, e.g., [1 0 0 0]'
% 5. STAT: either T- or F- statistics, e.g., 'T'
% 5. spec_hemi : specific view of the rendered brain, e.g.,
% 1) 'ventral' : ventral view,
% 2) 'dorsal' : dorsal view,
% 3) 'right' : right lateral view
% 4) 'left' : left lateral view
% 5) 'frontal' : frontal view
% 6) 'occipital' : occipital view
% 5. p_value : p-value, e.g., 0.05
% 6. correct_p : p-value correction method,
% e.g., 'EC': Lipschitz-Killing curvature based expected Euler characteristics
% 'tube': Sun's tube formula, 'none': uncorrected
% 7. disp_fig : 1 : show interpolated t- or F-map and activation map over
% the specific threshold.
% 0 : do not show the result.

% output variables
% stat_brain : interpolated t- or F-map on the rendered brain 
% act_brain : activation map over the threshold
% threshold: threshold value 
%
% example usage
% >> filen_SPM =
% 'C:\NIRS_SPM_v3_3\Sample_data\categorical_indiv_HbO\SPM_indiv_HbO.mat';
% >> filen_ch =
% 'C:\NIRS_SPM_v3_3\Sample_data\Registration\channel_NIRS_fMRI.mat';
% >> con_name = 'right finger tapping';
% >> con_vec = [1 0 0 0]';
% >> STAT = 'T';
% >> spec_hemi = 'left';
% >> p_value = 0.05;
% >> correct_p = 'EC';
% >> disp_fig = 1;
% >> [stat_brain, act_brain, threshold] = activation_map_batch(filen_SPM,
% filen_ch, con_name, con_vec, STAT, spec_hemi, p_value, correct_p,
% disp_fig);



if isempty(filen_SPM) == 1
disp('Please specify the SPM_indiv_HbX.mat file as an input variable.');
return;
end
load(filen_SPM);
dir_spm = [fileparts(filen_SPM) filesep];


% specification of contrast vector
%[Ic, xCon] = nirs_spm_conman(SPM_nirs, 'T&F', Inf, 'Select contrasts...', 'for conjunction', 1);
if isfield(SPM_nirs, 'xCon') == 1
    xCon = SPM_nirs.xCon;
    Ic = size(SPM_nirs.xCon, 2) + 1;
else
    Ic = 1;
end


con_vec = con_vec(:);
xCon(1,Ic) = struct('name', con_name, 'STAT', STAT, 'c', con_vec, 'X0', [], 'iX0', [], 'X1o', [], 'eidf', [], 'Vcon', [], 'Vspm', []);
SPM_nirs.xCon = xCon;

% receive channel information
if isempty(filen_ch) == 1
    disp('Please specify the file to contain the information of the channel locations');
    return;
end
load(filen_ch);

% specify the view of rendered brain

switch spec_hemi
    case ''
        disp('Please specify the view of the brain.');
        return;
    case 'ventral'
        side_hemi = 1;
    case 'dorsal'
        side_hemi = 2;
    case 'right'
        side_hemi = 3;
    case 'left'
        side_hemi = 4;
    case 'frontal'
        side_hemi = 5;
    case 'occipital'
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

% min and max values of statistics for colormap control
min_stat = min(stat);
max_stat = max(stat);
smin_stat = max_stat - ((max_stat - min_stat)./63) * 127;
sbar = linspace(smin_stat, max_stat, 128);

stat_brain = ((-sbar(1) + sbar(64))/(0.5)).*brain + sbar(1);
stat_brain(index_mask) = stat;

if disp_fig
    figure('Name', ['Interpolated ' STAT '-statistics map'], 'NumberTitle', 'off');
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

% update & save the SPM_indiv_HbX.mat file
save(filen_SPM, 'SPM_nirs');

% inference of brain activation

% calculation of threshold value, depending on a correction method
switch correct_p
    case 'EC'
        threshold = calc_EC(SPM_nirs.nirs.LKC{side_hemi}, p_value, SPM_nirs.xCon(Ic).STAT, df);
    case 'tube'
        threshold = calc_tube(SPM_nirs.nirs.kappa(Ic, side_hemi), p_value);
    case 'none'
        threshold = spm_u(p_value, df, SPM_nirs.xCon(Ic).STAT);
end

index_over = find(stat > threshold);
load Split;
if isempty(index_over) == 1
    disp('There is no significant voxel.');
    act_brain = brain;
    split = split(1:64,:);
    sbar = [];
else
    index_mask = index_mask(index_over);
    stat = stat(index_over);
    % min and max values of statistics for colormap control
    min_stat = min(stat);
    max_stat = max(stat);
    smin_stat = max_stat - ((max_stat - min_stat)./63) * 127;
    sbar = linspace(smin_stat, max_stat, 128);
    act_brain = ((-sbar(1) + sbar(64))/(0.5)).*brain + sbar(1);
    act_brain(index_mask) = stat;
end

if disp_fig
    figure('Name', ['Activated region of brain (p<' num2str(p_value) ', ' correct_p ' correction)'], 'NumberTitle', 'off');
    imagesc(act_brain);
    colormap(split)
    axis off
    axis image
    try
        hc = colorbar;
        set(hc, 'YLim', [sbar(65) sbar(128)]);
        y_tick = linspace(sbar(65), sbar(128), 5)';
        set(hc, 'YTick', y_tick);
        set(hc, 'FontSize', 8);
    end
end
