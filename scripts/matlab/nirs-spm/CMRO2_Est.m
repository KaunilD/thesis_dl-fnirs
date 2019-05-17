function varargout = CMRO2_Est(varargin)

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @CMRO2_Est_OpeningFcn, ...
    'gui_OutputFcn',  @CMRO2_Est_OutputFcn, ...
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


% --- Executes just before CMRO2_Est is made visible.
function CMRO2_Est_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;


set(handles.checkbox_HbORaw,'enable', 'off', 'value', 0);
set(handles.checkbox_HbRRaw,'enable', 'off', 'value', 0);
set(handles.checkbox_HbTRaw,'enable', 'off', 'value', 0);
set(handles.checkbox_BOLDRaw, 'enable', 'off', 'value', 0);
set(handles.checkbox_normalizeRaw,'enable','off','value', 0);
set(handles.edit_numChRaw, 'enable', 'inactive', 'string', '');
set(handles.checkbox_CBFResult, 'enable', 'off', 'value', 0);
set(handles.checkbox_CMRO2Result, 'enable', 'off', 'value', 0);
set(handles.checkbox_normailzeResult,'enable','off','value', 0);
set(handles.edit_numChResult, 'enable', 'inactive', 'string', '');
set(handles.edit_corrcoef, 'enable', 'inactive', 'string', '');
set(handles.edit_numChFitting, 'enable','inactive', 'string','');
set(handles.push_save, 'enable', 'off');

set(handles.slider_numChRaw, 'enable', 'inactive', 'max', 1, 'min', 0,'value', 1);
set(handles.slider_numChResult, 'enable', 'inactive', 'max', 1, 'min', 0,'value', 1);
set(handles.slider_mfFitting, 'enable', 'inactive', 'max', 1,'min', 0, 'value', 1);

cla(handles.axes_rawData, 'reset');
cla(handles.axes_resultData, 'reset');
cla(handles.axes_mfFitting, 'reset');
str_list{1} = '';
set(handles.listbox_info, 'string', str_list, 'value', 1);

% reset the handles structure
try
    handles = rmfield(handles, 'mean_nirs');
    handles = rmfield(handles, 'mean_fMRI');
    handles = rmfield(handles, 'time_n');
    handles = rmfield(handles, 'time_b');
    handles = rmfield(handles, 'ch_ROI');
    handles = rmfield(handles, 'CBF');
    handles = rmfield(handles, 'CMRO2');
    handles = rmfield(handles, 'mf_nirs');
    handles = rmfield(handles, 'mf_bold');   
    handles = rmfield(handles, 'search_region');
    handles = rmfield(handles, 'f_v_model');
    handles = rmfield(handles, 'opt_param');
end
% Update handles structure
guidata(hObject, handles);



% --- Outputs from this function are returned to the command line.
function varargout = CMRO2_Est_OutputFcn(hObject, eventdata, handles)

varargout{1} = handles.output;


% --- Executes on button press in push_setup.
function push_setup_Callback(hObject, eventdata, handles)
% load NIRS and display its filtering parameters
disp('Setup for NIRS data...');
[fname_nirs, sts] = spm_select(1, 'mat', 'Select NIRS data');
if sts == 0
    return;
end

spm_input('Setup for NIRS data', 1, 'd');
[tmp, disp_fname] = fileparts(fname_nirs);
spm_input(['File name: ' disp_fname], '+1', 'd');
load(fname_nirs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% unit : mM -> uM
nirs_data.oxyData = nirs_data.oxyData .* 1000;
nirs_data.dxyData = nirs_data.dxyData .* 1000;

try
    nirs_data.tHbData = nirs_data.tHbData .* 1000;
catch
    nirs_data.tHbData = nirs_data.oxyData + nirs_data.dxyData;
end

mean_nirs.unit = 'uM';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% disply the filtering parameter
str = 'Detrending?';
str = [str ' ' nirs_data.cH.type];
if strcmp(nirs_data.cH.type, 'DCT') == 1 % DCT-based detrending
    str = [str ',  cut-off ' num2str(nirs_data.cH.M) '{s}'];
end
spm_input(str, '+1', 'd');
mean_nirs.cH = nirs_data.cH;

str = 'Smoothing?';
str = [str ' ' nirs_data.cL.type];
if strcmp(nirs_data.cL.type, 'Gaussian') == 1
    str = [str ', FWHM ' num2str(nirs_data.cL.FWHM) '{s}'];
end
spm_input(str, '+1', 'd');
mean_nirs.cL = nirs_data.cL;
mean_nirs.fname = fname_nirs;

disp('Complete.');
disp('Setup for fMRI data...');
% load BOLD and display its filtering parameters
[fname_fMRI, sts] = spm_select(1, 'mat', 'Select fMRI data');
if sts == 0
    return;
end

spm_input('Setup for fMRI data', '+2', 'd');
[dir_fMRI, disp_fname] = fileparts(fname_fMRI);
spm_input(['File name: ' disp_fname], '+1', 'd');
load(fname_fMRI);

% disply the filtering parameter
str = 'Detrending?';
str = [str ' ' fMRI_data.cH.type];
if strcmp(fMRI_data.cH.type, 'DCT') == 1 % DCT-based detrending
    str = [str ',  cut-off ' num2str(fMRI_data.cH.M) '{s}'];
end
spm_input(str, '+1', 'd');
mean_fMRI.cH = fMRI_data.cH;

str = 'Smoothing?';
str = [str ' ' fMRI_data.cL.type];
if strcmp(fMRI_data.cL.type, 'Gaussian') == 1
    str = [str ', FWHM ' num2str(fMRI_data.cL.FWHM) '{s}'];
end
spm_input(str, '+1', 'd');
mean_fMRI.cL = fMRI_data.cL;
mean_fMRI.fname = fname_fMRI;

disp('Complete.');

try
    if strcmp(fMRI_data.flag, 'sample_data') == 1 && strcmp(nirs_data.flag, 'sample_data') == 1
        fMRI_data.fname_ch = [dir_fMRI filesep 'channel_positions.mat'];
    end
end

% channel file name
spm_input('NIRS-fMRI alignment', '+2', 'd');
[tmp, disp_fname] = fileparts(fMRI_data.fname_ch);
spm_input(['File name (Ch.):' disp_fname], '+1', 'd');

% specification of channels within ROI
load(fMRI_data.fname_ch);
NIRS_RegistrationResult_Viewer(preproc_info.rend_ch_pos);
ch_ROI = spm_input('Channels within ROI:', '+1', 'r', ' ', [Inf 1]);

% experimental protocol (onset, duration)
spm_input('Setup for experimental protocol', 1, 'd');
UNITS = 'secs';
str = ['vector of onsets [secs]'];
ons = spm_input(str, '+1','r',' ',[Inf 1]); % onset vector
str = 'stimulus duration [secs]';
input_dur = spm_input(str, '+1','r',' ',[Inf 1]); % stimulus duration
if length(input_dur) == 1
    input_dur = input_dur*ones(size(ons));
end
if sum(diff(input_dur)) ~= 0
    helpdlg('The length of stimulus duration should be same.');
    return;
end
% boxcar function for BOLD
tSPM.nscan = size(fMRI_data.bold, 1);
tSPM.xY.RT = fMRI_data.RT;
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
% boxcar_bold(1) = 0;

% start & end index for averaging BOLD signals
clear tSPM;
diff_tmp = diff(boxcar_bold);
index_start = find(diff_tmp == 1) + 1;
block_scan = index_start(2) - index_start(1);
Hb = convmtx(boxcar_bold, block_scan);
Hb = Hb(1:size(fMRI_data.bold,1),:);

% wavelet-MDL based averaging
for kk = 1:length(ch_ROI)
    [Bout, theta_bold, nbasis_boldt] = hrf_est_wave_lasso_MDL(fMRI_data.bold(:,ch_ROI(kk)), Hb, 1);
    mean_fMRI.bold(:,kk) = Hb(index_start(1):index_start(1) + block_scan -1,:) * Bout * theta_bold(:,end);
end

% boxcar function for NIRS
tSPM.nscan = size(nirs_data.oxyData,1);
tSPM.xY.RT = 1./nirs_data.fs;
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

% start & end index for averaging NIRS signals
diff_tmp = diff(boxcar_nirs);
index_start = find(diff_tmp == 1) + 1;
block_scan = index_start(2) - index_start(1);
H = convmtx(boxcar_nirs, block_scan);
H = H(1:size(nirs_data.oxyData,1),:);

disp('Wavelet-MDL based averaging starts...');
% wavelet-MDL based averaging
nch = length(ch_ROI);
for kk = 1:nch
    [Bout, theta_oxy, nbasis_oxyt] = hrf_est_wave_lasso_MDL(nirs_data.oxyData(:, ch_ROI(kk)), H, 0);
    mean_nirs.oxyData(:,kk) = H(index_start(1):index_start(1)+block_scan-1,:) * Bout * theta_oxy(:,end);
    [Bout, theta_dxy, nbasis_dxyt] = hrf_est_wave_lasso_MDL(nirs_data.dxyData(:, ch_ROI(kk)), H, 0);
    mean_nirs.dxyData(:,kk) = H(index_start(1):index_start(1)+block_scan-1,:) * Bout * theta_dxy(:,end);
    [Bout, theta_tHb, nbasis_tHbt] = hrf_est_wave_lasso_MDL(nirs_data.tHbData(:, ch_ROI(kk)), H, 0);
    mean_nirs.tHbData(:,kk) = H(index_start(1):index_start(1)+block_scan-1,:) * Bout * theta_tHb(:,end);
end
disp('Complete.');
time_n = linspace(0, ons(2) - ons(1), size(mean_nirs.oxyData,1));
time_b = linspace(0, ons(2) - ons(1), size(mean_fMRI.bold, 1));

if nch == 1
    set(handles.slider_numChRaw, 'enable', 'inactive');    
    set(handles.slider_mfFitting, 'enable', 'inactive');    
    set(handles.slider_numChResult, 'enable', 'inactive');    
else
    set(handles.slider_numChRaw, 'sliderstep', [1/(nch-1), 1/(nch-1)], 'max', nch, 'min', 1, 'value', 1, 'enable', 'on');
    set(handles.slider_mfFitting, 'sliderstep', [1/(nch-1), 1/(nch-1)], 'max', nch, 'min', 1, 'value', 1, 'enable', 'on');
    set(handles.slider_numChResult, 'sliderstep', [1/(nch-1), 1/(nch-1)], 'max', nch, 'min', 1, 'value', 1, 'enable', 'on');
end
    

set(handles.edit_numChFitting, 'enable', 'inactive', 'string', num2str(ch_ROI(1)));
set(handles.edit_numChResult, 'enable', 'inactive', 'string', num2str(ch_ROI(1)));
set(handles.edit_numChRaw, 'enable', 'inactive', 'string', num2str(ch_ROI(1)));

% plot the results
axes(handles.axes_rawData);
hold off;
plot(time_b, mean_fMRI.bold(:,1), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
hold on
plot(time_n, mean_nirs.oxyData(:,1), 'r');
plot(time_n, mean_nirs.dxyData(:,1), 'b');
plot(time_n, mean_nirs.tHbData(:,1), 'color', [0 127/255 0]);
default_axis = axis;
axis([min(time_n) max(time_n) default_axis(3:4)]);
xlabel('Time (s)');
% done

for kk = 1:nch
    mean_fMRI.interp_bold(:,kk) = interp1(time_b, mean_fMRI.bold(:,kk), time_n);
end

disp('Setup for search range from model parameters...');

spm_input('Search range from model parameters', 1, 'd');
str = 'Hypercapnia(%)';
H = spm_input(str, '+1', 'r', '0 10', 2);
H = H.*0.01;
str = 'Beta';
beta = spm_input(str, '+1', 'r', '1 2' ,2);
str = 'CBV_0';
CBV_0 = spm_input(str, '+1', 'r', '40 140' ,2);
str = 'SO_2 (%)';
SO_2 = spm_input(str, '+1', 'r', '55 80' ,2);
SO_2 = SO_2.*0.01;
str = 'Venous deoxy-Hb ratio';
gamma_r = spm_input(str, '+1', 'r', '0.5 1.5', 2);
str = 'Venous total-Hb ratio';
gamma_v = spm_input(str, '+1', 'r', '0.5 1.5', 2);
str = 'Partial volume factor';
p_factor = spm_input(str, '+1', 'r', '6.2', 1);

% CBF-CBV model
spm_input('CBF-CBV relationship', 1, 'd');
str = 'Alpha (CBF=CBV^alpha)';
alpha = spm_input(str, '+1', 'r', '0.38', 1);
%
%
% % CBF-CBV model
% spm_input('CBF-CBV relationship', 1, 'd');
% str = 'specify the model';
% model_fv = spm_input(str, '+1', 'Grubb|Buxton');
% switch model_fv
%     case 'Grubb'
%         str = 'Alpha (CBF=CBV^alpha)';
%         alpha = spm_input(str, '+1', 'r', '0.38', 1);
%     case 'Buxton'
%         str = 'MTT+viscoelastic time';
%         tau = spm_input(str, '+1', 'r', '5', 1);
% end

% estimate the CBF and CMRO2 using deoxy, total-Hb, and bold response

in_H = linspace(min(H), max(H), 50);
in_beta = linspace(min(beta), max(beta), 30);
p_factor = 1./p_factor;

cmin_etar = (1-max(SO_2)).*min(CBV_0)./max(gamma_r);
cmax_etar = (1-min(SO_2)).*max(CBV_0)./min(gamma_r);

cmin_etav = min(CBV_0)./(max(gamma_v));
cmax_etav = max(CBV_0)./(min(gamma_v));
disp('Complete.');
disp('Optimizing the model parameters & Estimating CBF and CMRO2...');

for kk = 1:nch
    hrf_dxy_tmp = mean_nirs.dxyData(:,kk);
    hrf_tHb_tmp = mean_nirs.tHbData(:,kk);
    hrf_bold_tmp = mean_fMRI.interp_bold(:,kk);
    
    min_dxy = min(hrf_dxy_tmp);
    max_dxy = max(hrf_dxy_tmp);
    min_tHb = min(hrf_tHb_tmp);
    max_tHb = max(hrf_tHb_tmp);
    max_BOLD = max(hrf_bold_tmp);
    
    if max_BOLD > min(H)
        in_H = linspace(max_BOLD, max(H), 50);
    end
    min_Q0 = max((-1)*min_dxy, max_dxy./(2^alpha - 1));
    min_V0 = max((-1)*min_tHb, max_tHb./(2^alpha - 1));
    min_Q0 = max(min_Q0, cmin_etar * p_factor);
    min_V0 = max(min_V0, cmin_etav * p_factor);
    in_eta_r = linspace(min_Q0, cmax_etar * p_factor, 50);
    in_eta_v = linspace(min_V0, cmax_etav * p_factor, 50);
    in_beta0 = 1;
    
    [opt_param] = calc_costfun_mindiff_ver2(in_eta_r, in_eta_v, in_beta0, in_H, hrf_bold_tmp, hrf_dxy_tmp, hrf_tHb_tmp);
    [opt_param2] = calc_costfun_mindiff_ver2(opt_param(1), in_eta_v, in_beta, opt_param(4), hrf_bold_tmp, hrf_dxy_tmp, hrf_tHb_tmp);
   
    opt_H(kk) = opt_param(4);
    opt_eta_r(kk) = opt_param(1);
    opt_eta_v(kk) = opt_param2(2);
    opt_beta(kk) = opt_param2(3);
    
    rdxy = 1 + hrf_dxy_tmp./opt_eta_r(kk);
    rtHb = 1 + hrf_tHb_tmp./opt_eta_v(kk);
    
    eq_x = opt_H(kk).*(1-rtHb.*((rdxy./rtHb).^opt_beta(kk)));
    eq_y = hrf_bold_tmp;
    
    var_bold = 1-hrf_bold_tmp./opt_H(kk);
    
    % calculate the oxygen extraction fraction from NIRS and BOLD
    % biophysical model
    mf_bold(:,kk) = sign(rtHb).*(abs(rtHb).^(-1./opt_beta(kk))).*sign(var_bold).*(abs(var_bold).^(1/opt_beta(kk)));
    mf_nirs(:,kk) = rdxy./rtHb;
    
    % calculate the CBF, CBV, and CMRO2
    CBV(:,kk) = rtHb;
    CBF(:,kk) = sign(CBV(:,kk)).*(abs(CBV(:,kk)).^(1/alpha));
    CMRO2(:,kk) = mf_nirs(:,kk).*CBF(:,kk);
    
    % calculate the flow-metabolism coupling ratio
    [tmp_n] = polyfit(CMRO2(1:round(input_dur(1)*nirs_data.fs),kk)-1, CBF(1:round(input_dur(1) * nirs_data.fs),kk)-1,1);
    fit_n(kk) = tmp_n(1);
    
    % calculate the correlation coefficient between mf_nirs and mf_bold
    tmp = corrcoef(mf_nirs(:,kk), mf_bold(:,kk));    
    R(kk) = tmp(1,2);    
end
disp('Complete.');

% plot the m/f from NIRS and BOLD model for evaluating the results
axes(handles.axes_mfFitting);
hold off;
plot(time_n, mf_nirs(:,1), 'r');
hold on
plot(time_n, mf_bold(:,1));
default_axis = axis;
axis([min(time_n) max(time_n) default_axis(3:4)]);
xlabel('Time (s)');


% plot the CBF and CMRO2 on the same figure
axes(handles.axes_resultData);
hold off;
plot(time_n, CBF(:,1),'r');
hold on
plot(time_n, CMRO2(:,1));
default_axis = axis;
axis([min(time_n) max(time_n) default_axis(3:4)]);
xlabel('Time (s)');

% reset the handles structure
try
    handles = rmfield(handles, 'mean_nirs');
    handles = rmfield(handles, 'mean_fMRI');
    handles = rmfield(handles, 'time_n');
    handles = rmfield(handles, 'time_b');
    handles = rmfield(handles, 'ch_ROI');
end

handles.mean_nirs = mean_nirs;
handles.mean_fMRI = mean_fMRI;
handles.time_n = time_n;
handles.time_b = time_b;

handles.ch_ROI = ch_ROI; % channels within ROI
% handles.R = R; % correlation coefficient between mf_nirs and mf_bold
handles.CMRO2 = CMRO2;
handles.CBF = CBF;

handles.mf_nirs = mf_nirs;
handles.mf_bold = mf_bold;

% write the summary of results on the listbox
str_list{1} = 'Input information:';
str_list{2} = '';
str_list{3} = 'Search range from model parameters';
str_list{4} = ['H (Hypercapnia calibration): [' num2str(H(1).*100) ' ' num2str(H(2).*100) '] (%)'];
str_list{5} = ['Beta (Davis`s model parameter): [' num2str(beta(1)) ' ' num2str(beta(2)) ']'];
str_list{6} = ['CBV_0 (Total baseline blood volume): [' num2str(CBV_0(1)) ' ' num2str(CBV_0(2)) '] (uM)'];
str_list{7} = ['SO_2 (Baseline oxygenation saturation): [' num2str(SO_2(1)*100) ' ' num2str(SO_2(2)*100) '] (%)'];
str_list{8} = ['gamma_r (Venous deoxy-hemoglobin ratio): [' num2str(gamma_r(1)) ' ' num2str(gamma_r(2)) ']'];
str_list{9} = ['gamma_v (Venous total-hemoglobin ratio): [' num2str(gamma_v(1)) ' ' num2str(gamma_v(2)) ']'];
str_list{10} = ['CBF-CBV relationship: Grubb`s model (alpha = ' num2str(alpha) ')'];
str_list{11} = '';
str_list{12} = 'Output information:';
str_list{13} = '';
str_list{14} = 'Optimized parameter values';
str_list{15} = ['Channels within region of interest:     ' num2str(ch_ROI(:)')];
str_list{16} = ['H (Hypercapnia calibration):               ' num2str(opt_H(:)'*100) '(%)'];
str_list{17} = ['Beta (Davis`s model parameter):        ' num2str(opt_beta(:)')];
str_list{18} = ['Eta_r = (1-SO2)*CBV_0/gamma_r:      ' num2str(opt_eta_r(:)')];
str_list{19} = ['Eta_v = CBV_0/gamma_v:                   ' num2str(opt_eta_v(:)')];
str_list{20} = ['Correlation coefficients:                      ' num2str(R(:)')];
str_list{21} = ['Coupling ratio =(rCBF-1)/(rCMRO2-1): ' num2str(fit_n(:)')];

set(handles.listbox_info, 'string', str_list)

% save the search region from model parameters as the handle structure
search_region.H = H;
search_region.beta = beta;
search_region.CBV_0 = CBV_0;
search_region.SO_2 = SO_2;
search_region.gamma_r = gamma_r;
search_region.gamma_v = gamma_v;
search_region.p_factor = p_factor;

f_v_model.type = 'Grubb`s model';
f_v_model.alpha = alpha;

clear opt_param;
opt_param.H = opt_H;
opt_param.beta = opt_beta;
opt_param.eta_r = opt_eta_r;
opt_param.eta_v = opt_eta_v;
opt_param.R = R; % correlation coefficient

handles.search_region = search_region;
handles.f_v_model = f_v_model;
handles.opt_param = opt_param;

% update the graphic property
set(handles.checkbox_HbORaw, 'enable', 'on', 'value', 1);
set(handles.checkbox_HbRRaw, 'enable', 'on', 'value', 1);
set(handles.checkbox_HbTRaw, 'enable', 'on', 'value', 1);
set(handles.checkbox_BOLDRaw, 'enable', 'on', 'value', 1);
set(handles.checkbox_normalizeRaw, 'enable', 'on', 'value', 0);
set(handles.checkbox_CBFResult,'enable', 'on',  'value', 1);
set(handles.checkbox_CMRO2Result,'enable', 'on',  'value', 1);
set(handles.checkbox_normailzeResult, 'enable', 'on','value', 0);
set(handles.edit_corrcoef, 'string', num2str(R(1)));
set(handles.push_save, 'enable', 'on');

guidata(hObject, handles);

% --- Executes on button press in checkbox_HbORaw.
function checkbox_HbORaw_Callback(hObject, eventdata, handles)
% load the raw data from the handles
mean_nirs = handles.mean_nirs;
mean_fMRI = handles.mean_fMRI;
time_n = handles.time_n;
time_b = handles.time_b;
h = get(handles.slider_numChRaw, 'value');
h2 = get(handles.checkbox_normalizeRaw, 'value');

c1 = get(handles.checkbox_HbORaw, 'value');
c2 = get(handles.checkbox_HbRRaw, 'value');
c3 = get(handles.checkbox_HbTRaw, 'value');
c4 = get(handles.checkbox_BOLDRaw, 'value');

% plot the resutls
axes(handles.axes_rawData);
hold off;
if h2 == 1
    if c4 == 1
        plot(time_b, mean_fMRI.bold(:,h)./max(mean_fMRI.bold(:,h)), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
    if c1 == 1
        plot(time_n, mean_nirs.oxyData(:,h)./max(mean_nirs.oxyData(:,h)), 'r');
        hold on
    end
    if c2 == 1
        plot(time_n, mean_nirs.dxyData(:,h)./max(mean_nirs.dxyData(:,h)), 'b');
        hold on
    end
    if c3 == 1
        plot(time_n, mean_nirs.tHbData(:,h)./max(mean_nirs.tHbData(:,h)),'color', [0 127/255 0]);
        hold on
    end
elseif     h2== 0
    if c4 == 1
        plot(time_b, mean_fMRI.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
    if c1 == 1
        plot(time_n, mean_nirs.oxyData(:,h), 'r');
        hold on
    end
    if c2 == 1
        plot(time_n, mean_nirs.dxyData(:,h), 'b');
        hold on;
    end
    if c3 == 1
        plot(time_n, mean_nirs.tHbData(:,h),'color', [0 127/255 0]);
        hold on
    end
end
default_axis = axis;
axis([min(time_n) max(time_n) default_axis(3:4)]);
xlabel('Time (s)');



% --- Executes on button press in checkbox_HbRRaw.
function checkbox_HbRRaw_Callback(hObject, eventdata, handles)
% load the raw data from the handles
mean_nirs = handles.mean_nirs;
mean_fMRI = handles.mean_fMRI;
time_n = handles.time_n;
time_b = handles.time_b;
h = get(handles.slider_numChRaw, 'value');
h2 = get(handles.checkbox_normalizeRaw, 'value');

c1 = get(handles.checkbox_HbORaw, 'value');
c2 = get(handles.checkbox_HbRRaw, 'value');
c3 = get(handles.checkbox_HbTRaw, 'value');
c4 = get(handles.checkbox_BOLDRaw, 'value');

% plot the resutls
axes(handles.axes_rawData);
hold off;
if h2 == 1
    if c4 == 1
        plot(time_b, mean_fMRI.bold(:,h)./max(mean_fMRI.bold(:,h)), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
    if c1 == 1
        plot(time_n, mean_nirs.oxyData(:,h)./max(mean_nirs.oxyData(:,h)), 'r');
        hold on
    end
    if c2 == 1
        plot(time_n, mean_nirs.dxyData(:,h)./max(mean_nirs.dxyData(:,h)), 'b');
        hold on
    end
    if c3 == 1
        plot(time_n, mean_nirs.tHbData(:,h)./max(mean_nirs.tHbData(:,h)),'color', [0 127/255 0]);
        hold on
    end
elseif     h2== 0
    if c4 == 1
        plot(time_b, mean_fMRI.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
    if c1 == 1
        plot(time_n, mean_nirs.oxyData(:,h), 'r');
        hold on
    end
    if c2 == 1
        plot(time_n, mean_nirs.dxyData(:,h), 'b');
        hold on;
    end
    if c3 == 1
        plot(time_n, mean_nirs.tHbData(:,h),'color', [0 127/255 0]);
        hold on
    end
end

default_axis = axis;
axis([min(time_n) max(time_n) default_axis(3:4)]);
xlabel('Time (s)');



% --- Executes on button press in checkbox_HbTRaw.
function checkbox_HbTRaw_Callback(hObject, eventdata, handles)
% load the raw data from the handles
mean_nirs = handles.mean_nirs;
mean_fMRI = handles.mean_fMRI;
time_n = handles.time_n;
time_b = handles.time_b;
h = get(handles.slider_numChRaw, 'value');
h2 = get(handles.checkbox_normalizeRaw, 'value');

c1 = get(handles.checkbox_HbORaw, 'value');
c2 = get(handles.checkbox_HbRRaw, 'value');
c3 = get(handles.checkbox_HbTRaw, 'value');
c4 = get(handles.checkbox_BOLDRaw, 'value');

% plot the resutls
axes(handles.axes_rawData);
hold off;
if h2 == 1
    if c4 == 1
        plot(time_b, mean_fMRI.bold(:,h)./max(mean_fMRI.bold(:,h)), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
    if c1 == 1
        plot(time_n, mean_nirs.oxyData(:,h)./max(mean_nirs.oxyData(:,h)), 'r');
        hold on
    end
    if c2 == 1
        plot(time_n, mean_nirs.dxyData(:,h)./max(mean_nirs.dxyData(:,h)), 'b');
        hold on
    end
    if c3 == 1
        plot(time_n, mean_nirs.tHbData(:,h)./max(mean_nirs.tHbData(:,h)),'color', [0 127/255 0]);
        hold on
    end
elseif     h2== 0
    if c4 == 1
        plot(time_b, mean_fMRI.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
    if c1 == 1
        plot(time_n, mean_nirs.oxyData(:,h), 'r');
        hold on
    end
    if c2 == 1
        plot(time_n, mean_nirs.dxyData(:,h), 'b');
        hold on;
    end
    if c3 == 1
        plot(time_n, mean_nirs.tHbData(:,h),'color', [0 127/255 0]);
        hold on
    end
end

default_axis = axis;
axis([min(time_n) max(time_n) default_axis(3:4)]);
xlabel('Time (s)');



% --- Executes on button press in checkbox_BOLDRaw.
function checkbox_BOLDRaw_Callback(hObject, eventdata, handles)
% load the raw data from the handles
mean_nirs = handles.mean_nirs;
mean_fMRI = handles.mean_fMRI;
time_n = handles.time_n;
time_b = handles.time_b;
h = get(handles.slider_numChRaw, 'value');
h2 = get(handles.checkbox_normalizeRaw, 'value');

c1 = get(handles.checkbox_HbORaw, 'value');
c2 = get(handles.checkbox_HbRRaw, 'value');
c3 = get(handles.checkbox_HbTRaw, 'value');
c4 = get(handles.checkbox_BOLDRaw, 'value');

% plot the resutls
axes(handles.axes_rawData);
hold off;
if h2 == 1
    if c4 == 1
        plot(time_b, mean_fMRI.bold(:,h)./max(mean_fMRI.bold(:,h)), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
    if c1 == 1
        plot(time_n, mean_nirs.oxyData(:,h)./max(mean_nirs.oxyData(:,h)), 'r');
        hold on
    end
    if c2 == 1
        plot(time_n, mean_nirs.dxyData(:,h)./max(mean_nirs.dxyData(:,h)), 'b');
        hold on
    end
    if c3 == 1
        plot(time_n, mean_nirs.tHbData(:,h)./max(mean_nirs.tHbData(:,h)),'color', [0 127/255 0]);
        hold on
    end
elseif     h2== 0
    if c4 == 1
        plot(time_b, mean_fMRI.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
    if c1 == 1
        plot(time_n, mean_nirs.oxyData(:,h), 'r');
        hold on
    end
    if c2 == 1
        plot(time_n, mean_nirs.dxyData(:,h), 'b');
        hold on;
    end
    if c3 == 1
        plot(time_n, mean_nirs.tHbData(:,h),'color', [0 127/255 0]);
        hold on
    end
end

default_axis = axis;
axis([min(time_n) max(time_n) default_axis(3:4)]);
xlabel('Time (s)');



% --- Executes on button press in checkbox_normalizeRaw.
function checkbox_normalizeRaw_Callback(hObject, eventdata, handles)
% load the raw data from the handles
mean_nirs = handles.mean_nirs;
mean_fMRI = handles.mean_fMRI;
time_n = handles.time_n;
time_b = handles.time_b;
h = get(handles.slider_numChRaw, 'value');
h2 = get(handles.checkbox_normalizeRaw, 'value');

c1 = get(handles.checkbox_HbORaw, 'value');
c2 = get(handles.checkbox_HbRRaw, 'value');
c3 = get(handles.checkbox_HbTRaw, 'value');
c4 = get(handles.checkbox_BOLDRaw, 'value');

% plot the resutls
axes(handles.axes_rawData);
hold off;
if h2 == 1
    if c4 == 1
        plot(time_b, mean_fMRI.bold(:,h)./max(mean_fMRI.bold(:,h)), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
    if c1 == 1
        plot(time_n, mean_nirs.oxyData(:,h)./max(mean_nirs.oxyData(:,h)), 'r');
        hold on
    end
    if c2 == 1
        plot(time_n, mean_nirs.dxyData(:,h)./max(mean_nirs.dxyData(:,h)), 'b');
        hold on
    end
    if c3 == 1
        plot(time_n, mean_nirs.tHbData(:,h)./max(mean_nirs.tHbData(:,h)),'color', [0 127/255 0]);
        hold on
    end
elseif     h2== 0
    if c4 == 1
        plot(time_b, mean_fMRI.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
    if c1 == 1
        plot(time_n, mean_nirs.oxyData(:,h), 'r');
        hold on
    end
    if c2 == 1
        plot(time_n, mean_nirs.dxyData(:,h), 'b');
        hold on;
    end
    if c3 == 1
        plot(time_n, mean_nirs.tHbData(:,h),'color', [0 127/255 0]);
        hold on
    end
end

default_axis = axis;
axis([min(time_n) max(time_n) default_axis(3:4)]);
xlabel('Time (s)');


% --- Executes on slider movement.
function slider_numChRaw_Callback(hObject, eventdata, handles)
mean_nirs = handles.mean_nirs;
mean_fMRI = handles.mean_fMRI;
time_n = handles.time_n;
time_b = handles.time_b;
% for slider control
h = get(handles.slider_numChRaw, 'value');
h = round(h);
set(handles.slider_numChRaw, 'value', h);
h2 = get(handles.checkbox_normalizeRaw, 'value');
c1 = get(handles.checkbox_HbORaw, 'value');
c2 = get(handles.checkbox_HbRRaw, 'value');
c3 = get(handles.checkbox_HbTRaw, 'value');
c4 = get(handles.checkbox_BOLDRaw, 'value');


% plot the resutls
axes(handles.axes_rawData);
hold off;
if h2 == 1
    if c4 == 1
        plot(time_b, mean_fMRI.bold(:,h)./max(mean_fMRI.bold(:,h)), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
    if c1 == 1
        plot(time_n, mean_nirs.oxyData(:,h)./max(mean_nirs.oxyData(:,h)), 'r');
        hold on
    end
    if c2 == 1
        plot(time_n, mean_nirs.dxyData(:,h)./max(mean_nirs.dxyData(:,h)), 'b');
        hold on
    end
    if c3 == 1
        plot(time_n, mean_nirs.tHbData(:,h)./max(mean_nirs.tHbData(:,h)),'color', [0 127/255 0]);
        hold on
    end
elseif h2== 0
    if c4 == 1
        plot(time_b, mean_fMRI.bold(:,h), '-bs', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'g', 'MarkerSize', 3);
        hold on
    end
    if c1 == 1
        plot(time_n, mean_nirs.oxyData(:,h), 'r');
        hold on
    end
    if c2 == 1
        plot(time_n, mean_nirs.dxyData(:,h), 'b');
        hold on;
    end
    if c3 == 1
        plot(time_n, mean_nirs.tHbData(:,h),'color', [0 127/255 0]);
        hold on
    end
end

default_axis = axis;
axis([min(time_n) max(time_n) default_axis(3:4)]);
xlabel('Time (s)');
set(handles.edit_numChRaw, 'string', num2str(handles.ch_ROI(h)));

% --- Executes during object creation, after setting all properties.
function slider_numChRaw_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function edit_numChRaw_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_numChRaw_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_numChResult_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_numChResult_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function slider_numChResult_Callback(hObject, eventdata, handles)
% load the result data from the handles
% load the raw data from the handles
CBF = handles.CBF;
CMRO2 = handles.CMRO2;
time_n = handles.time_n;
ch_ROI = handles.ch_ROI;

h = get(handles.slider_numChResult, 'value');
h = round(h);
set(handles.slider_numChResult, 'value', h);
set(handles.edit_numChResult,'string', num2str(ch_ROI(h)));

h2 = get(handles.checkbox_normailzeResult, 'value');
c1 = get(handles.checkbox_CBFResult, 'value');
c2 = get(handles.checkbox_CMRO2Result, 'value');

% plot the resutls
axes(handles.axes_resultData);
hold off;
if h2 == 1 % normalize the result
    if c1 == 1 % CBF;
        plot(time_n, CBF(:,h)./max(CBF(:,h)),'r');
        hold on
    end
    if c2 == 1
        plot(time_n, CMRO2(:,h)./max(CMRO2(:,h)));
        hold on
    end
elseif h2 == 0 % without normalization
    if c1 == 1 % CBF
        plot(time_n, CBF(:,h), 'r');
        hold on
    end
    if c2 == 1 % CMRO2
        plot(time_n, CMRO2(:,h));
        hold on;
    end    
end

default_axis = axis;
axis([min(time_n) max(time_n) default_axis(3:4)]);
xlabel('Time (s)');

% --- Executes during object creation, after setting all properties.
function slider_numChResult_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in checkbox_normailzeResult.
function checkbox_normailzeResult_Callback(hObject, eventdata, handles)
% load the result data from the handles
% load the raw data from the handles
CBF = handles.CBF;
CMRO2 = handles.CMRO2;
time_n = handles.time_n;
h = get(handles.slider_numChResult, 'value');
h2 = get(handles.checkbox_normailzeResult, 'value');
c1 = get(handles.checkbox_CBFResult, 'value');
c2 = get(handles.checkbox_CMRO2Result, 'value');

% plot the resutls
axes(handles.axes_resultData);
hold off;
if h2 == 1 % normalize the result
    if c1 == 1 % CBF;
        plot(time_n, CBF(:,h)./max(CBF(:,h)),'r');
        hold on
    end
    if c2 == 1
        plot(time_n, CMRO2(:,h)./max(CMRO2(:,h)));
        hold on
    end
elseif h2 == 0 % without normalization
    if c1 == 1 % CBF
        plot(time_n, CBF(:,h), 'r');
        hold on
    end
    if c2 == 1 % CMRO2
        plot(time_n, CMRO2(:,h));
        hold on;
    end    
end

default_axis = axis;
axis([min(time_n) max(time_n) default_axis(3:4)]);
xlabel('Time (s)');


% --- Executes on button press in checkbox_BOLDResult.
function checkbox_BOLDResult_Callback(hObject, eventdata, handles)


% --- Executes on button press in checkbox_HbTResult.
function checkbox_HbTResult_Callback(hObject, eventdata, handles)


% --- Executes on button press in checkbox_CMRO2Result.
function checkbox_CMRO2Result_Callback(hObject, eventdata, handles)
% load the result data from the handles
% load the raw data from the handles
CBF = handles.CBF;
CMRO2 = handles.CMRO2;
time_n = handles.time_n;
h = get(handles.slider_numChResult, 'value');
h2 = get(handles.checkbox_normailzeResult, 'value');
c1 = get(handles.checkbox_CBFResult, 'value');
c2 = get(handles.checkbox_CMRO2Result, 'value');

% plot the resutls
axes(handles.axes_resultData);
hold off;
if h2 == 1 % normalize the result
    if c1 == 1 % CBF;
        plot(time_n, CBF(:,h)./max(CBF(:,h)),'r');
        hold on
    end
    if c2 == 1
        plot(time_n, CMRO2(:,h)./max(CMRO2(:,h)));
        hold on
    end
elseif h2 == 0 % without normalization
    if c1 == 1 % CBF
        plot(time_n, CBF(:,h), 'r');
        hold on
    end
    if c2 == 1 % CMRO2
        plot(time_n, CMRO2(:,h));
        hold on;
    end    
end

default_axis = axis;
axis([min(time_n) max(time_n) default_axis(3:4)]);
xlabel('Time (s)');

% --- Executes on button press in checkbox_CBFResult.
function checkbox_CBFResult_Callback(hObject, eventdata, handles)
% load the result data from the handles
% load the raw data from the handles
CBF = handles.CBF;
CMRO2 = handles.CMRO2;
time_n = handles.time_n;
h = get(handles.slider_numChResult, 'value');
h2 = get(handles.checkbox_normailzeResult, 'value');
c1 = get(handles.checkbox_CBFResult, 'value');
c2 = get(handles.checkbox_CMRO2Result, 'value');

% plot the resutls
axes(handles.axes_resultData);
hold off;
if h2 == 1 % normalize the result
    if c1 == 1 % CBF;
        plot(time_n, CBF(:,h)./max(CBF(:,h)),'r');
        hold on
    end
    if c2 == 1
        plot(time_n, CMRO2(:,h)./max(CMRO2(:,h)));
        hold on
    end
elseif h2 == 0 % without normalization
    if c1 == 1 % CBF
        plot(time_n, CBF(:,h), 'r');
        hold on
    end
    if c2 == 1 % CMRO2
        plot(time_n, CMRO2(:,h));
        hold on;
    end    
end
default_axis = axis;
axis([min(time_n) max(time_n) default_axis(3:4)]);
xlabel('Time (s)');

% --- Executes on slider movement.
function slider_mfFitting_Callback(hObject, eventdata, handles)
mf_bold = handles.mf_bold;
mf_nirs = handles.mf_nirs;
time_n = handles.time_n;
ch_ROI = handles.ch_ROI;
R = handles.opt_param.R;

h = get(handles.slider_mfFitting, 'value');
h = round(h);
set(handles.slider_mfFitting, 'value', h);
set(handles.edit_numChFitting,'string', num2str(ch_ROI(h)));
set(handles.edit_corrcoef, 'string', num2str(R(h)));

axes(handles.axes_mfFitting);
hold off;
plot(time_n, mf_nirs(:,h),'r');
hold on
plot(time_n, mf_bold(:,h));
default_axis = axis;
axis([min(time_n) max(time_n) default_axis(3:4)]);
xlabel('Time (s)');


% --- Executes during object creation, after setting all properties.
function slider_mfFitting_CreateFcn(hObject, eventdata, handles)
% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end




function edit_numChFitting_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_numChFitting_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_corrcoef_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_corrcoef_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listbox_info.
function listbox_info_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function listbox_info_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in push_save.
function push_save_Callback(hObject, eventdata, handles)
try
    perfusion.mean_nirs = handles.mean_nirs;
    perfusion.mean_fMRI = handles.mean_fMRI;
    perfusion.time_n = handles.time_n;
    perfusion.time_b = handles.time_b;
    perfusion.ch_ROI = handles.ch_ROI;
    perfusion.rCBF = handles.CBF;
    perfusion.rCMRO2 = handles.CMRO2;
    perfusion.mf_nirs = handles.mf_nirs;
    perfusion.mf_bold = handles.mf_bold;
    perfusion.search_region = handles.search_region;
    perfusion.f_v_model = handles.f_v_model;
    perfusion.opt_param = handles.opt_param;
    [filen, pathn]= uiputfile('*.mat', 'Save the results as');
    path_filen = [pathn filen];
    if path_filen == 0
        return
    end
    save(path_filen, 'perfusion');
catch
    return;
end
