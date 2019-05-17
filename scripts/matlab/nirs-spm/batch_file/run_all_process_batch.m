% step 1: data conversion (Oxymon MKIII)
fname_nirs = 'C:\NIRS_SPM_v4\Sample_data\NIRS_data_file\Artinis_OXYMON_SampleData.nir';
system = 'oxymon';
fs = 0.975;
dist = 3.5;
wavelength = [856 781];
DPF = 4;
flag_DPF_correction = 1;
[nirs_data] = data_conversion_batch(fname_nirs, system, fs, dist, wavelength, DPF, flag_DPF_correction, [], [], []);
save('C:\NIRS_SPM_v4\Sample_data\NIRS_data_file\converted_NIRS.mat', 'nirs_data');

% step 2: model specification
fname_nirs = 'C:\NIRS_SPM_v4\Sample_data\NIRS_data_file\converted_NIRS.mat';
hb = 'hbo';
HPF = 'wavelet';
LPF = 'hrf';
method_cor = 0;
dir_save = 'C:\NIRS_SPM_v4\Sample_data\NIRS_data_file\categorical_HbO\';
mkdir(dir_save);
flag_window = 1;
hrf_type = 2;
units = 1;
names{1} = 'right finger tapping';
onsets{1} = 42:51:501;
durations{1} = 21 * ones(10,1);
[SPM_nirs] = specification_batch(fname_nirs, hb, HPF, LPF, method_cor, dir_save, flag_window, hrf_type, units,  names, onsets, durations);

% step 3: model estimation
fname_nirs = 'C:\NIRS_SPM_v4\Sample_data\NIRS_data_file\converted_NIRS.mat';
fname_SPM = 'C:\NIRS_SPM_v4\Sample_data\NIRS_data_file\categorical_HbO\SPM_indiv_HbO.mat';
[SPM_nirs] = estimation_batch(fname_SPM, fname_nirs);

% step 4: activation map (T-statistic, EC-corrected p-value < 0.05)
fname_SPM = 'C:\NIRS_SPM_v4\Sample_data\NIRS_data_file\categorical_HbO\SPM_indiv_HbO.mat';
fname_ch = 'C:\NIRS_SPM_v4\Sample_data\Registration\channel_NIRS_fMRI.mat';
con_name = 'Right finger tapping';
con_vec = [1 0 0 0]';
STAT = 'T';
spec_hemi = 'dorsal';
p_value = 0.05;
correct_p = 'EC';
disp_fig = 1;
[stat_brain, act_brain, threshold] = activation_map_batch(fname_SPM, fname_ch, con_name, con_vec, STAT, spec_hemi, p_value, correct_p, disp_fig);