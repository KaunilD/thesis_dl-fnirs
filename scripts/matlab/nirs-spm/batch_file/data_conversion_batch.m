function [nirs_data] = data_conversion_batch(fname_nirs, system, fs, dist, wavelength, DPF, flag_DPF_correction, total_ch, ext_coef, ch_config)

% NIRS-SPM batch script for 'Convert' routine, which reads the optical
% density or hemoglobin concentration changes from the raw data and
% converts it to NIRS-SPM format.
%
% Input variables
% 1. fname_nirs : the name of the nirs file,
% e.g.,...\NIRS_SPM_v3_2\Sample_data\Artinis_OXYMON_SampleData.nir
% Note that the input variable for multiple nirs files should be cell
% array
% 2. system : specific nirs system, e.g., 'oxymon'
% 1) 'oxymon' - OXYMON MK III
% 2) 'hitachi_1set_OD' - Hitachi ETG-4000 (1 set, Optical Density)
% 3) 'hitachi_2set_OD' - Hitachi ETG-4000 (2 set, Optical Density)
% 4) 'hitachi_hb' - Hitachi ETG-4000 (HbO, HbR, HbT)
% 5) 'iss' - ISS Imagent
% 6) 'hamamatsu' - Hamamatsu NIRO-200
% 7) 'nirx' - NIRX DYNOT-232
% 8) 'spectratech' - Spectratech OEG-16
% 9) 'shimadzu' - Shimadzu OMM FOIRE-3000
% 10) 'biopac' - BIOPAC fNIR
% 11) 'manual_OD'- Manual input mode for converting optical density changes
% 12) 'manual_hb' - Manual Input mode for reading HbO/HbR changes
% 3. fs : sampling frequency(Hz), e.g., 10
% 4. dist : distance between source and detector(cm), e.g., 3.5
% 5. wavelength : wavelength of the light source (nm), e.g.. [856 781]
% 6. DPF : differential pathlength factor, e.g., 4
% 7. flag_DPF_correction : 0 - without DPF correction, 1 - with DPF correction
% e.g., 1
% 8. total_ch : total number of channels, e.g., 24
% 9. ext_coef : extinction coefficient, e.g. [1.4866 3.8437 2.2314 1.7917]
% 10. ch_config:  the name of the channel configuratio file,
% e.g.,...\NIRS_SPM_v3_2_r3\Sample_data\Ch_config\NIRX_DYNOT232_20x32_80ch.txt
%
% example usages :
% if you want to convert the Hitachi ETG-4000 (1 set, Optical Density)
% data, enter the command as follows;
% >> [nirs_data] =
% data_conversion_batch('C:\NIRS_SPM_v3_2_r3\Sample_data\NIRS_data_file\Hitach
% i_ETG4000_SampleData.csv', 'hitachi_1set_OD')
%
% if you want to convert the Hitachi ETG-4000 (2 set, Optical Density)
% data, enter the command as follows;
% >> fname_nirs{1} =
% 'C:\NIRS_SPM_v3_2_r3\Sample_data\NIRS_data_file\Hitachi_ETG4000_Set1.csv';
% >> fname_nirs{2} =
% 'C:\NIRS_SPM_v3_2_r3\Sample_data\NIRS_data_file\Hitachi_ETG4000_Set2.csv';
% >> [nirs_data] = data_conversion_batch(fname_nirs, 'hitachi_2set_OD');
%
% if you want to convert the Hitachi ETG-4000 (HbO, HbR, HbT)
% data, enter the command as follows;
% >> fname_nirs{1} =
% 'C:\NIRS_SPM_v3_2_r3\Sample_data\NIRS_data_file\Hitachi_ETG4000_24Ch_Oxy.csv';
% >> fname_nirs{2} =
% 'C:\NIRS_SPM_v3_2_r3\Sample_data\NIRS_data_file\Hitachi_ETG4000_24Ch_Deoxy.c
% sv';
% >> fname_nirs{3} =
% 'C:\NIRS_SPM_v3_2_r3\Sample_data\NIRS_data_file\Hitachi_ETG4000_24Ch_Total.c
% sv';
% [nirs_data] = data_conversion_batch(fname_nirs, 'hitachi_hb');
%
% if you want to convert the ISS Imagent, enter the command as follows:
% >> fname_nirs =
% 'C:\NIRS_SPM_v3_2_r3\Sample_data\NIRS_data_file\ISS_Imagent_SampleData.log';
% >> [nirs_data] = data_conversion_batch(fname_nirs, 'iss');
%
% If you want to convert the NIRX DYNOT-232, enter the command as follows:
% >> fname_nirs{1} =
% 'C:\NIRS_SPM_v3_2_r3\Sample_data\NIRS_data_file\NIRX_DYNOT232_Set1.wl1';
% >> fname_nirs{2} =
% 'C:\NIRS_SPM_v3_2_r3\Sample_data\NIRS_data_file\NIRX_DYNOT232_Set2.wl2';
% >> ch_config =
% 'C:\NIRS_SPM_v3_2_r3\Sample_data\Ch_config\NIRX_DYNOT232_20x32_80ch.txt';
% >> [nirs_data] = data_conversion_batch(fname_nirs, 'nirx', 2.44, 2.5,
% [760 830], [7.15 5.98], 0, 80, [1.4866 3.8437 2.2314 1.7917], ch_config);
%
% % If you want to convert the Spectratech OEG-16, enter the command as follows:
% >> fname_nirs =
% 'C:\NIRS_SPM_v3_2_r3\Sample_data\NIRS_data_file\Spectratech_OEG16_SampleData
% .csv';
% >> [nirs_data] = data_conversion_batch(fname_nirs, 'spectratech', fs);
%
% If you want to convert the Shimadzu OMM FOIRE-3000, enter the command as follows:
% >> fname_nirs = 'C:\NIRS_SPM_v3_2_r3\Sample_data\NIRS_data_file\Shimadzu_FT_Right_4x4x2.TXT';
% >> [nirs_data] = data_conversion_batch(fname_nirs, 'shimadzu');
%
% If you want to convert the BIOPAC fNIR, enter the command as follows:
% >> fname_nirs = 'C:\NIRS_SPM_v3_2_r3\Sample_data\NIRS_data_file\BIOPAC_fNIR_SampleData.oxy';
% >> [nirs_data] = data_conversion_batch(fname_nirs, 'biopac');
%
%--------------------------------------------------------------------------
% if you want to convert the optical density changes as for manual input of
% the modified Beer-Lambert law's parameters.
% >> fname_nirs =
% 'C:\NIRS_SPM_v3_2_r3\Sample_data\NIRS_data_file\OpticalDensity_SampleData
% .csv';
% >> [nirs_data] = data_conversion_batch(fname_nirs, 'manual_OD', 9.75,
% 3.5, [856 781], 4, 1, 24);
%
% if you have an input for extinction coefficients (ext_coef),
% e.g. for 856 nm wavelength, ext_coef of HbO is 1.1885 and HbR is 0.7923, 
% for 781nm wavelength, ext_coef of HbO is 0.7422 and HbR is 1.0803.
% >> ext_coef = [1.1885 0.7923 0.7422 1.0803];
% >> [nirs_data] = data_conversion_batch(fname_nirs, 'manual_OD', 9.75,
% 3.5, [856 781], 4, 0, 24, ext_coef);
%
% NIRS-SPM batch function allows channel wise input of source-detector,
% DPF, wavelength, and extinction coefficients. 
% for example, 
% (1st channel) fs:9, wavelength:856 & 781nm, ext_coef:[1.2 0.8 0.7 1.1]
% (2nd channel) fs:8, wavelength:853 & 779nm, ext_coef:[1.1 0.9 0.5 1.3]
% >> fs = [9 8];
% >> wavelength = [856 781 853 779];
% >> ext_coef = [1.2 0.8 0.7 1.1 1.1 0.9 0.5 1.3];
%
% If you have an *.txt or *.csv file which contain the parameters, 
% the type of 'fname_nirs' variable should be defined as cell array.
% 1st element of fname_nirs : the name of optical density file 
% 2nd element of fname_nirs : the name of model parameter file 
% >> fname_nirs{1} = 'C:\NIRS_SPM_v3_2_r3\Sample_data\NIRS_data_file\OpticalDensity_SampleData
% .csv';
% >> fname_nirs{2} =
% 'C:\NIRS_SPM_v3_2_r3\Sample_data\NIRS_data_file\Sample_Parameters.txt';
% >> [nirs_data] = data_conversion_batch(fname_nirs, 'manual_OD');
%
% If you want to read the hemoglobin changes from specific file,
% >> fname_nirs =
% 'C:\NIRS_SPM_v3_2_r3\Sample_data\NIRS_data_file\HbO_HbR_SampleData.csv';
% >> [nirs_data] = data_conversion_batch(fname_nirs, 'manual_hb', 9.75);
%--------------------------------------------------------------------------

switch system
    case 'oxymon' % OXYMON MK III
        if nargin < 7
            disp('ERROR: Not enough input arguments');
            nirs_data = [];
            return;
        end
        rwavelength = round(wavelength);
        if flag_DPF_correction == 1
            if (sum(rwavelength >= 704) + sum(rwavelength <= 972)) ~= 4
                disp('ERROR: NIRS-SPM allows the wavelength range of DPF correction between 704nm and 972nm');
                nirs_data = [];
                return;
            end
        end
        nirs_data = nir_to_mat(fname_nirs, fs, dist, rwavelength, DPF, flag_DPF_correction);
        nirs_data.wavelength = wavelength;
        nirs_data.distance = dist;
        nirs_data.DPF = DPF;
        nirs_data.nch = size(nirs_data.oxyData, 2);
        if flag_DPF_correction == 1
            nirs_data.DPF_correction = 'Charite correction';
        elseif flag_DPF_correction == 0
            nirs_data.DPF_correction = 'none';
        end
    case 'hitachi_1set_OD' % Hitachi ETG-4000 (1 set, Optical Density)
        if nargin < 2
            disp('ERROR: Not enough input arguments');
            nirs_data = [];
            return;
        end
        fid = fopen(fname_nirs);
        disp('Please wait...');
        while 1
            tline = fgetl(fid);
            nindex = find(tline == ',');
            if isempty(nindex) == 0
                switch tline(1:nindex(1)-1)
                    case 'Wave Length'
                        txt_wavelength = tline(nindex(1)+1:end);
                        nch = length(nindex)./2;
                        disp('Data reading (1/3) has been completed.');
                    case 'Sampling Period[s]'
                        txt_fs = tline(nindex(1)+1:end);
                        fs = 1./str2num(txt_fs);
                        disp('Data reading (2/3) has been completed.');
                end
            elseif isempty(strfind(tline, 'Data')) == 0
                tline = fgetl(fid);
                nindex = find(tline == ',');
                if isempty(strfind(tline, 'Probe1')) == 1
                    disp('ERROR: Please select the optical density file which was measured from the first set of probes (Probe 1).');
                    nirs_data = [];
                    return;
                end
                txt_probe = tline(nindex(1)+1:end);
                index_tmp = find(txt_probe == ',');
                txt_probe(index_tmp) = ' ';
                [token remain] = strtok(txt_probe);
                count = 3;
                while isempty(token) ~= 1
                    [token remain] = strtok(remain);
                    if strcmp(token, 'Mark') == 1
                        col_mark = count;
                    end
                    if strcmp(token, 'PreScan') == 1
                        col_prescan = count;
                    end
                    count = count + 1;
                end
                count = 1;
                while 1
                    tline = fgetl(fid);
                    if ischar(tline) == 0, break, end,
                    nindex = find(tline == ',');
                    tline2 = tline;
                    tline2(nindex) = ' ';
                    tline2 = str2num(tline2);
                    mes(tline2(1),:) = tline2(2:2+2*nch-1);
                    try
                        baseline(tline2(1)) = str2num(tline(nindex(col_prescan-1)+1:nindex(col_prescan)-1));
                    catch
                        baseline(tline2(1)) = str2num(tline(nindex(col_prescan-1)+1:end));
                    end
                    try
                        vector_onset(tline2(1)) = str2num(tline(nindex(col_mark-1)+1:nindex(col_mark)-1));
                    end
                    count = count + 1;
                end
                disp('Data reading (3/3) has been completed.');
                break,
            end
        end
        fclose(fid);
        nindex = find(txt_wavelength == ',');
        index_base = find(baseline == 1);
        disp('Data conversion starts...');
        for kk = 1:nch-1
            wav1 = txt_wavelength(nindex(2*kk-1)-6:nindex(2*kk-1)-2);
            wav2 = txt_wavelength(nindex(2*kk)-6:nindex(2*kk)-2);
            [hb_tmp, hbo_tmp, hbt_tmp] = mes2hb(mes(:,2*kk-1:2*kk), [str2num(wav1) str2num(wav2)], [index_base(1) index_base(end)]);
            nirs_data.oxyData(:,kk) = hbo_tmp;
            nirs_data.dxyData(:,kk) = hb_tmp;
        end
        wav1 = txt_wavelength(nindex(2*nch-1)-6:nindex(2*nch-1)-2);
        wav2 = txt_wavelength(end-5:end-1);
        [hb_tmp, hbo_tmp, hbt_tmp] = mes2hb(mes(:,2*nch-1:2*nch), [str2num(wav1) str2num(wav2)], [index_base(1) index_base(end)]);
        disp('Data conversion has been completed.');
        try
            vector_onset(index_base(1):index_base(end)) = [];
            nirs_data.vector_onset = vector_onset(:);
        end
        nirs_data.oxyData(:,nch) = hbo_tmp;
        nirs_data.dxyData(:,nch) = hb_tmp;
        nirs_data.fs = fs;
        nirs_data.nch = nch;
    case 'hitachi_2set_OD' % Hitachi ETG-4000 (2 set, Optical Density)
        if nargin < 2
            disp('ERROR: Not enough input arguments');
            nirs_data = [];
            return;
        end
        if iscell(fname_nirs) == 0
            disp('ERROR: The name of nirs file should be a cell array.');
            nirs_data = [];
            return;
        end
        fid = fopen(fname_nirs{1});
        disp('Reading the nirs data from the first set of probes starts...');
        while 1
            tline = fgetl(fid);
            nindex = find(tline == ',');
            if isempty(nindex) == 0
                switch tline(1:nindex(1)-1)
                    case 'Wave Length'
                        txt_wavelength = tline(nindex(1)+1:end);
                        nch = length(nindex)./2;
                    case 'Sampling Period[s]'
                        txt_fs = tline(nindex(1)+1:end);
                        fs = 1./str2num(txt_fs);
                end
            elseif isempty(strfind(tline, 'Data')) == 0
                tline = fgetl(fid);
                nindex = find(tline == ',');
                if isempty(strfind(tline, 'Probe1')) == 1
                    disp('ERROR: Please select the optical density file which was measured from the first set of probes (Probe 1).');
                    nirs_data = [];
                    return;
                end
                txt_probe = tline(nindex(1)+1:end);
                index_tmp = find(txt_probe == ',');
                txt_probe(index_tmp) = ' ';
                [token remain] = strtok(txt_probe);
                count = 3;
                while isempty(token) ~= 1
                    [token remain] = strtok(remain);
                    if strcmp(token, 'Mark') == 1
                        col_mark = count;
                    end
                    if strcmp(token, 'PreScan') == 1
                        col_prescan = count;
                    end
                    count = count + 1;
                end
                count = 1;
                while 1
                    tline = fgetl(fid);
                    if ischar(tline) == 0, break, end,
                    nindex = find(tline == ',');
                    tline2 = tline;
                    tline2(nindex) = ' ';
                    tline2 = str2num(tline2);
                    mes(tline2(1),:) = tline2(2:2+2*nch-1);
                    try
                        baseline(tline2(1)) = str2num(tline(nindex(col_prescan-1)+1:nindex(col_prescan)-1));
                    catch
                        baseline(tline2(1)) = str2num(tline(nindex(col_prescan-1)+1:end));
                    end
                    try
                        vector_onset1(tline2(1)) = str2num(tline(nindex(col_mark-1)+1:nindex(col_mark)-1));
                    end
                    count = count + 1;
                end
                break,
            end
        end
        fclose(fid);

        nindex = find(txt_wavelength == ',');
        index_base = find(baseline == 1);
        for kk = 1:nch-1
            wav1 = txt_wavelength(nindex(2*kk-1)-6:nindex(2*kk-1)-2);
            wav2 = txt_wavelength(nindex(2*kk)-6:nindex(2*kk)-2);
            [hb_tmp, hbo_tmp, hbt_tmp] = mes2hb(mes(:,2*kk-1:2*kk), [str2num(wav1) str2num(wav2)], [index_base(1) index_base(end)]);
            oxyData1(:,kk) = hbo_tmp;
            dxyData1(:,kk) = hb_tmp;
        end
        wav1 = txt_wavelength(nindex(2*nch-1)-6:nindex(2*nch-1)-2);
        wav2 = txt_wavelength(end-5:end-1);
        [hb_tmp, hbo_tmp, hbt_tmp] = mes2hb(mes(:,2*nch-1:2*nch), [str2num(wav1) str2num(wav2)], [index_base(1) index_base(end)]);
        try
            vector_onset1(index_base(1):index_base(end)) = [];
        end
        oxyData1(:,nch) = hbo_tmp;
        dxyData1(:,nch) = hb_tmp; % end of reading the left side data
        clear mes
        clear baseline
        disp('Completed.');
        disp('Reading the nirs data from the second set of probes starts...');
        try
            fid = fopen(fname_nirs{2});
        catch
            disp('ERROR: The second file does not exist.');
            nirs_data = [];
            return;
        end
        while 1
            tline = fgetl(fid);
            nindex = find(tline == ',');
            if isempty(nindex) == 0
                switch tline(1:nindex(1)-1)
                    case 'Wave Length'
                        txt_wavelength = tline(nindex(1)+1:end);
                        nch = length(nindex)./2;
                    case 'Sampling Period[s]'
                        txt_fs = tline(nindex(1)+1:end);
                        fs = 1./str2num(txt_fs);
                end
            elseif isempty(strfind(tline, 'Data')) == 0
                tline = fgetl(fid);
                nindex = find(tline == ',');
                if isempty(strfind(tline, 'Probe2')) == 1
                    disp('ERROR: Please select the optical density file which was measured from the second set of probes (Probe 2).');
                    nirs_data = [];
                    return;
                end
                txt_probe = tline(nindex(1)+1:end);
                index_tmp = find(txt_probe == ',');
                txt_probe(index_tmp) = ' ';
                [token remain] = strtok(txt_probe);
                count = 3;
                while isempty(token) ~= 1
                    [token remain] = strtok(remain);
                    if strcmp(token, 'Mark') == 1
                        col_mark = count;
                    end
                    if strcmp(token, 'PreScan') == 1
                        col_prescan = count;
                    end
                    count = count + 1;
                end
                count = 1;
                while 1
                    tline = fgetl(fid);
                    if ischar(tline) == 0, break, end,
                    nindex = find(tline == ',');
                    tline2 = tline;
                    tline2(nindex) = ' ';
                    tline2 = str2num(tline2);
                    mes(tline2(1),:) = tline2(2:2+2*nch-1);
                    try
                        baseline(tline2(1)) = str2num(tline(nindex(col_prescan-1)+1:nindex(col_prescan)-1));
                    catch
                        baseline(tline2(1)) = str2num(tline(nindex(col_prescan-1)+1:end));
                    end
                    try
                        vector_onset2(tline2(1)) = str2num(tline(nindex(col_mark-1)+1:nindex(col_mark)-1));
                    end
                    count = count + 1;
                end
                break,
            end
        end
        fclose(fid);

        nindex = find(txt_wavelength == ',');
        index_base = find(baseline == 1);
        for kk = 1:nch-1
            wav1 = txt_wavelength(nindex(2*kk-1)-6:nindex(2*kk-1)-2);
            wav2 = txt_wavelength(nindex(2*kk)-6:nindex(2*kk)-2);
            [hb_tmp, hbo_tmp, hbt_tmp] = mes2hb(mes(:,2*kk-1:2*kk), [str2num(wav1) str2num(wav2)], [index_base(1) index_base(end)]);
            oxyData2(:,kk) = hbo_tmp;
            dxyData2(:,kk) = hb_tmp;
        end
        wav1 = txt_wavelength(nindex(2*nch-1)-6:nindex(2*nch-1)-2);
        wav2 = txt_wavelength(end-5:end-1);
        [hb_tmp, hbo_tmp, hbt_tmp] = mes2hb(mes(:,2*nch-1:2*nch), [str2num(wav1) str2num(wav2)], [index_base(1) index_base(end)]);
        oxyData2(:,nch) = hbo_tmp;
        dxyData2(:,nch) = hb_tmp;
        try
            vector_onset2(index_base(1):index_base(end)) = [];
            index_onset = find(vector_onset1 ~= vector_onset2);
            if isempty(index_onset) == 1
                nirs_data.vector_onset = vector_onset1(:);
            end
        end
        nch = size(oxyData1, 2) + size(oxyData2, 2);
        nirs_data.oxyData = zeros(size(oxyData1,1), nch);
        nirs_data.dxyData = zeros(size(dxyData1,1), nch);

        nirs_data.oxyData(:, 1:size(oxyData1,2)) = oxyData1(:,:);
        nirs_data.oxyData(:, size(oxyData1,2)+1:end) = oxyData2(:,:);
        nirs_data.dxyData(:, 1:size(dxyData1,2)) = dxyData1(:,:);
        nirs_data.dxyData(:, size(dxyData1,2)+1:end) = dxyData2(:,:);
        nirs_data.fs = fs;
        nirs_data.nch = nch;
        disp('Completed.');
    case 'hitachi_hb' % Hitachi ETG-4000 (HbO, HbR, HbT)
        if nargin < 2
            disp('ERROR: Not enough input arguments');
            nirs_data = [];
            return;
        end
        if iscell(fname_nirs) == 0
            disp('ERROR: The name of nirs file should be a cell array.');
            nirs_data = [];
            return;
        end
        fid = fopen(fname_nirs{1});
        disp('Loading the OxyHb data starts...');
        while 1
            tline = fgetl(fid);
            if isempty(strfind(tline, 'Sampling Period[s]')) == 0
                nindex = find(tline == ',');
                txt_fs = tline(nindex(1)+1:end);
                fs = 1./str2num(txt_fs);
            end
            if isempty(strfind(tline, 'Data')) == 0
                tline = fgetl(fid);
                nch = length(strfind(tline, 'CH'));
                nindex = find(tline == ',');
                if isempty(strfind(tline, 'Oxy')) == 1
                    disp('ERROR: Please select the Oxy-Hb file.');
                    nirs_data = [];
                    return;
                end
                try
                    col_mark = strfind(tline, 'Mark');
                    col_mark = col_mark(1);
                    col_mark = find(nindex == col_mark - 1) + 1;
                end
                try
                    col_prescan = strfind(tline, 'PreScan');
                    col_prescan = col_prescan(1);
                    col_prescan = find(nindex == col_prescan - 1)+1;
                end
                while 1
                    tline = fgetl(fid);
                    if ischar(tline) == 0, break, end,
                    nindex = find(tline == ',');
                    tline2 = tline;
                    tline2(nindex) = ' ';
                    tline2 = str2num(tline2);
                    nirs_data.oxyData(tline2(1), :) = tline2(2:nch+1);
                    try
                        vector_onset(tline2(1)) = tline2(col_mark);
                    end
                    try
                        baseline(tline2(1)) = tline2(col_prescan);
                    end
                end
                break;
            end
        end
        disp('HbO data loading has been finished.');

        %%% read the deoxy-hb data
        fid = fopen(fname_nirs{2});
        disp('Loading the DeoxyHb data starts...');
        while 1
            tline = fgetl(fid);
            if isempty(strfind(tline, 'Sampling Period[s]')) == 0
                nindex = find(tline == ',');
                txt_fs = tline(nindex(1)+1:end);
                fs = 1./str2num(txt_fs);
            end
            if isempty(strfind(tline, 'Data')) == 0
                tline = fgetl(fid);
                nch = length(strfind(tline, 'CH'));
                nindex = find(tline == ',');
                if isempty(strfind(tline, 'Deoxy')) == 1
                    disp('ERROR: Please select the Deoxy-Hb file.');
                    nirs_data = [];
                    return;
                end
                while 1
                    tline = fgetl(fid);
                    if ischar(tline) == 0, break, end,
                    nindex = find(tline == ',');
                    tline2 = tline;
                    tline2(nindex) = ' ';
                    tline2 = str2num(tline2);
                    nirs_data.dxyData(tline2(1), :) = tline2(2:nch+1);
                end
                break;
            end
        end
        disp('HbR data loading has been finished.');

        %%% read the total-hb data
        fid = fopen(fname_nirs{3});
        disp('Loading the TotalHb data starts...');
        while 1
            tline = fgetl(fid);
            if isempty(strfind(tline, 'Sampling Period[s]')) == 0
                nindex = find(tline == ',');
                txt_fs = tline(nindex(1)+1:end);
                fs = 1./str2num(txt_fs);
            end
            if isempty(strfind(tline, 'Data')) == 0
                tline = fgetl(fid);
                nch = length(strfind(tline, 'CH'));
                nindex = find(tline == ',');
                if isempty(strfind(tline, 'Total')) == 1
                    disp('ERROR: Please select the Total-Hb file.');
                    nirs_data = [];
                    return;
                end
                while 1
                    tline = fgetl(fid);
                    if ischar(tline) == 0, break, end,
                    nindex = find(tline == ',');
                    tline2 = tline;
                    tline2(nindex) = ' ';
                    tline2 = str2num(tline2);
                    nirs_data.tHbData(tline2(1), :) = tline2(2:nch+1);
                end
                break;
            end
        end
        disp('HbT data loading has been finished.');
        try
            nirs_data.vector_onset = vector_onset(:);
        end
        try
            index_base = find(baseline == 1);
            nirs_data.vector_onset(index_base(1):index_base(end)) = [];
            nirs_data.oxyData(index_base(1):index_base(end)) = [];
            nirs_data.dxyData(index_base(1):index_base(end)) = [];
            nirs_data.tHbData(index_base(1):index_base(end)) = [];
        end
        nirs_data.fs = fs;
        nirs_data.nch = nch;
    case 'iss' % ISS Imagent
        if nargin < 2
            disp('ERROR: Not enough input arguments');
            nirs_data = [];
            return;
        end
        fid = fopen(fname_nirs);
        tline = fgetl(fid);
        index = 1;
        while 1
            tline = fgetl(fid);
            if ~ischar(tline), break, end;
            tline = str2num(tline);
            nirs_data.oxyData(index, :) = tline(1,6:2:end-2);
            nirs_data.dxyData(index, :) = tline(1,7:2:end-1);
            nirs_data.vector_onset(index,1) = tline(1,end);
            time(index,1) = tline(1,1);
            index = index + 1;
        end
        fclose(fid);
        nirs_data.vector_onset = nirs_data.vector_onset(:);
        fs = 1./(mean(diff(time)));
        nirs_data.nch = size(nirs_data.oxyData,2);
    case 'hamamatsu' % Hamamatsu NIRO-200
        if nargin < 2
            disp('ERROR: Not enough input arguments');
            nirs_data = [];
            return;
        end
        fid = fopen(fname_nirs);
        count = 1;
        while 1
            tline = fgetl(fid);
            nindex= find(tline == ',');
            if isempty(nindex) == 0 & isempty(str2num(tline(2:nindex(1)-2))) == 0
                elpsec(count,1) = str2num(tline(nindex(1)+1:nindex(2)-1));
                nirs_data.oxyData(count,1) = str2num(tline(nindex(2)+1:nindex(3)-1));
                nirs_data.dxyData(count,1) = str2num(tline(nindex(3)+1:nindex(4)-1));
                nirs_data.oxyData(count,2) = str2num(tline(nindex(6)+1:nindex(7)-1));
                nirs_data.dxyData(count,2) = str2num(tline(nindex(7)+1:nindex(8)-1));
                count = count + 1;
            elseif ischar(tline) == 0, break, end,
        end
        fclose(fid);
        fs = 1./mean(diff(elpsec));
        nirs_data.nch = size(nirs_data.oxyData,2);
    case 'shimadzu' %Shimadzu OMM FOIRE-3000
        if nargin < 2
            disp('ERROR: Not enough input arguments');
            nirs_data = [];
            return;
        end
        fid = fopen(fname_nirs);
        disp('Data loading...');
        while 1
            tline = fgetl(fid);
            nindex = find(tline == ',');
            tline(nindex) = ' ';
            [token, remain] = strtok(tline);
            if strncmp(tline, 'Time Range',10) == 1
                [token remain] = strtok(remain);
                [token remain] = strtok(remain);
                stt = str2num(token);
                [token remain] = strtok(remain);
                stp = str2num(token);
            end
            %%%%'Time(sec)'
            if strcmp(token, 'Time(sec)') == 1
                index = 1;
                while 1
                    tline2 = fgetl(fid);
                    if ischar(tline2) == 0, break, end,
                    newlabel = strrep(tline2, 'Z', '');
                    nindex = find(newlabel == ',');
                    newlabel = str2num(newlabel);
                    nirs_data.oxyData(index, :) = newlabel(1,5:3:end-2);
                    nirs_data.dxyData(index, :) = newlabel(1,6:3:end-1);
                    nirs_data.tHbData(index, :) = newlabel(1,7:3:end);
                    time(index,1) = newlabel(1,1);
                    index = index + 1;
                end
                break
            end
        end
        fclose(fid);
        fs = 1./(mean(diff(time)));
        nirs_data.fs = fs;
        nirs_data.nch = size(nirs_data.oxyData,2);
        disp('Complete.');
    case 'spectratech' % Spectratech OEG-16
        if nargin < 3
            disp('ERROR: Not enough input arguments');
            nirs_data = [];
            return;
        end
        fid = fopen(fname_nirs);
        while 1
            tline = fgetl(fid);
            nindex = find(tline == ',');
            tline(nindex) = ' ';
            [token, remain] = strtok(tline);
            if strcmp(token, 'evt') == 1
                count = 1;
                while isempty(remain) ~= 1
                    [token remain] = strtok(remain);
                    index1 = find(token == '(');
                    index2 = find(token == ')');
                    switch token(index1+1:index2-1)
                        case 'O'
                            try
                                col_oxy(end+1) = count + 1;
                            catch
                                col_oxy(1) = count + 1;
                            end
                        case 'D'
                            try
                                col_dxy(end+1) = count + 1;
                            catch
                                col_dxy(1) = count + 1;
                            end
                        case 'O+D'
                            try
                                col_tHb(end+1) = count + 1;
                            catch
                                col_tHb(1) = count + 1;
                            end
                    end
                    count = count + 1;
                end

                count = 1;
                while 1
                    tline = fgetl(fid);
                    if ischar(tline) == 0, break, end,
                    nindex = find(tline == ',');
                    tline(nindex) = ' ';
                    mes(count,:) = str2num(tline);
                    count = count + 1;
                end
                break,
            end
        end
        nirs_data.oxyData = mes(:, col_oxy);
        nirs_data.dxyData = mes(:, col_dxy);
        nirs_data.tHbData = mes(:, col_tHb);
        nirs_data.fs = fs;
        nirs_data.nch = size(nirs_data.oxyData, 2);
    case 'nirx' %NIRX DYNOT-232
        if nargin < 10
            disp('ERROR: Not enough input arguments');
            nirs_data = [];
            return;
        end

        switch ch_config(end-2:end)
            case 'mat'
                load(ch_config);
                IMGlabel = ni.IMGlabel;
                for kk = 1:total_ch
                    tmp = IMGlabel{kk};
                    index = find(tmp == '-');
                    config(kk,1) = str2num(tmp(1:index-1));
                    config(kk,2) = str2num(tmp(index+1:end));
                end
            case 'txt'
                fid = fopen(ch_config);
                for kk = 1:4
                    tline = fgetl(fid);
                end
                count = 1;
                while 1
                    tline = fgetl(fid);
                    if ~ischar(tline); break, end;
                    tline = str2num(tline);
                    config(count,:) = tline;
                    count = count + 1;
                end
                fclose(fid);
        end
        disp('Reading the channel configuration file has been completed.');

        fid = fopen(fname_nirs{1});
        count = 1;
        while 1
            tline = fgetl(fid);
            if ~ischar(tline), break, end
            tline = str2num(tline);
            mes1(count,:) = tline;
            count = count + 1;
        end
        disp('Reading the nirs data from the first set of probes has been completed.');
        fclose(fid);
        fid = fopen(fname_nirs{2});
        count = 1;
        while 1
            tline = fgetl(fid);
            if ~ischar(tline); break, end;
            tline = str2num(tline);
            mes2(count,:) = tline;
            count = count + 1;
        end
        fclose(fid);
        disp('Reading the nirs data from the second set of probes has been completed.');
        nTx = max(config(:,1));
        nRx = max(config(:,2));
        %         if size(mes1, 2) ~= nTx*nRx || size(mes2, 2) ~= nTx*nRx
        %             nTx = inputdlg('Please enter the number of sources');
        %             nTx = str2num(cell2mat(nTx));
        %             nRx = inputdlg('Please enter the number of detectors');
        %             nRx = str2num(cell2mat(nRx));
        %         end
        s1 = count-1;
        
        mes2_log = real(-log10( (mes2  )./ ...
            ( repmat(mean(mes2,1), [s1,1]))   ))    ;
        mes1_log = real(-log10( (mes1  )./ ...
            ( repmat(mean(mes1,1), [s1,1]))   ))    ;
        
        coefMat = dist.*(diag(DPF) * [ext_coef(1,1:2); ext_coef(1,3:4)]);
        coefMat = pinv(coefMat);
        
        for kk = 1:total_ch
            index = (config(kk, 1)-1)* nRx + config(kk,2);
            oxydxy = coefMat * [mes1_log(:, index)'; mes2_log(:, index)'];
            oxyData(:, kk) = oxydxy(1,:)';
            dxyData(:, kk) = oxydxy(2,:)';
        end
        disp('Completed.');
        
        nirs_data.oxyData = oxyData;
        nirs_data.dxyData = dxyData;
        nirs_data.nch = total_ch;
        nirs_data.fs = fs;
        nirs_data.wavelength = wavelength;
        nirs_data.distance = dist;
        nirs_data.DPF = DPF;
    case 'biopac' %BIOPAC fNIR
        if nargin < 2
            disp('ERROR: Not enough input arguments');
            nirs_data = [];
            return;
        end
        fid = fopen(fname_nirs);
        index = 1;
        disp('Reading the data of BIOPAC system starts...');
        while 1
            tline = fgetl(fid);
            tline = str2num(tline);
            if isempty(tline) == 0
                time(index) = tline(1);
                nirs_data.dxyData(index, :) = tline(1,2:2:end);
                nirs_data.oxyData(index, :) = tline(1,3:2:end);
                index = index + 1;
                while 1
                    tline = fgetl(fid);
                    if isempty(strfind(tline, 'Device Stopped')) == 0
                        break;
                    end
                    tline = str2num(tline);
                    time(index) = tline(1);
                    nirs_data.dxyData(index, :) = tline(1,2:2:end);
                    nirs_data.oxyData(index, :) = tline(1,3:2:end);
                    index = index + 1;
                end
                break;
            end
        end
        fclose(fid);
        fs = 1./(mean(diff(time)));
        nirs_data.fs = fs;
        nirs_data.nch = size(nirs_data.oxyData,2);
        disp('Completed.');
    case 'manual_OD'
        if length(fname_nirs) == 1            
            if nargin < 8
                disp('ERROR: Not enough input arguments');
                nirs_data = [];
                return;
            elseif nargin > 9
                disp('WARNNING: Too many inputs! Please confirm your inputs!');
                nirs_data = [];
                return;
            elseif nargin == 8
                ext_coef = [];            
            end
        elseif length(fname_nirs) == 2 % text file input mode
            if nargin < 2
                disp('ERROR: Not enough input arguments');
                nirs_data = [];
                return;
            end
            ext_coef = [];
            fname_param = fname_nirs{2}; % parameter file name            
            fname_nirs = fname_nirs{1}; % change the type of variable from cell to char
            fid = fopen(fname_param);        
            switch fname_param(end-2:end)
                case 'txt' % .txt file format
                    disp('Reading the parameter values from *.txt file...');
                    while 1
                        tline = fgetl(fid);
                        if ~ischar(tline), break, end;
                        index = find(tline == ' ');
                        % reading the parameter values from text file
                        if strcmpi(tline(1:index(1)-1), 'Total_number_of_Ch.') == 1
                            total_ch = str2num(tline(index(1)+1:end));
                        elseif strcmpi(tline(1:index(1)-1), 'Sampling_freq.[Hz]') == 1
                            fs = str2num(tline(index(1)+1:end));
                        elseif strcmpi(tline(1:index(1)-1), 'Distance[cm]') == 1
                            dist = str2num(tline(index(1)+1:end));
                        elseif strcmpi(tline(1:index(1)-1), 'Wave_length[nm]') == 1
                            wavelength = str2num(tline(index(1)+1:end));
                        elseif strcmpi(tline(1:index(1)-1), 'DPF') == 1
                            DPF = str2num(tline(index(1)+1:end));
                        elseif strcmpi(tline(1:index(1)-1), 'Correction') == 1
                            if strcmpi(tline(index(1)+1:end), 'yes') == 1
                                flag_DPF_correction = 1;
                            elseif strcmpi(tline(index(1)+1:end), 'no') == 1
                                flag_DPF_correction = 0;
                            end
                        elseif strcmpi(tline(1:index(1)-1),'Extinction_coefficient') == 1
                            ext_coef = str2num(tline(index(1)+1:end));
                        end
                    end                    
                case 'csv' % .csv file format
                    disp('Reading the parameter values from *.csv file...');
                    fid = fopen(fname_param);            
                    while 1
                        tline = fgetl(fid);
                        if ~ischar(tline), break, end;
                        index = find(tline == ',');
                        tline(index) = ' ';
                        % reading the parameter values from text file
                        if strcmpi(tline(1:index(1)-1), 'Total_number_of_Ch.') == 1
                            total_ch = str2num(tline(index(1)+1:end));
                        elseif strcmpi(tline(1:index(1)-1), 'Sampling_freq.[Hz]') == 1
                            fs = str2num(tline(index(1)+1:end));
                        elseif strcmpi(tline(1:index(1)-1), 'Distance[cm]') == 1
                            dist = str2num(tline(index(1)+1:end));
                        elseif strcmpi(tline(1:index(1)-1), 'Wave_length[nm]') == 1
                            wavelength = str2num(tline(index(1)+1:end));
                        elseif strcmpi(tline(1:index(1)-1), 'DPF') == 1
                            DPF = str2num(tline(index(1)+1:end));
                        elseif strcmpi(tline(1:index(1)-1), 'Correction') == 1
                            if strcmpi(tline(index(1)+1:end), 'yes') == 1
                                flag_DPF_correction = 1;
                            elseif strcmpi(tline(index(1)+1:end), 'no') == 1
                                flag_DPF_correction = 0;
                            end
                        elseif strcmpi(tline(1:index(1)-1),'Extinction_coefficient') == 1
                            ext_coef = str2num(tline(index(1)+1:end));
                        end
                    end                
            end
        end 
        fclose(fid);
        
        rwavelength = round(wavelength);
        rwavelength = reshape(rwavelength, [2 size(rwavelength,2)/2])';
        
        %% reshape the input vectors
        fs = fs(:);
        dist = dist(:);
        DPF = DPF(:);
        ext_coef = ext_coef(:);
                
        %% check if source-detector distance, wavelength, and DPF depend on
        %% specific channels or not.
        flag_param = 0;
        if length(dist) == 1
            dist = dist * ones(total_ch,1);
            flag_param = flag_param + 1;
        end
        if length(rwavelength) == 2
            rwavelength = ones(total_ch,1) * rwavelength;
            flag_param = flag_param + 1;
        end
        if length(DPF) == 1
            DPF = DPF * ones(total_ch,1);
            flag_param = flag_param + 1;
        end        
        if length(ext_coef) == 4 || isempty(ext_coef) == 1
            try
                ext_coef = ext_coef * ones(1, total_ch);
                ext_coef = ext_coef(:);
                flag_param = flag_param + 1;
            end
        end   
        disp('Generating a coefficient matrix from input variables...');
        if flag_param == 4 % same parameters will be applied to all channels
            if isempty(ext_coef) == 1 
                load COPE_e_coef; % load the extinction coefficient file
                index_wav1 = find(e_coef(:,1) == rwavelength(1,1));
                index_wav2 = find(e_coef(:,1) == rwavelength(1,2));
                wav1_ecoef = e_coef(index_wav1,2:3);
                wav2_ecoef = e_coef(index_wav2,2:3);
            elseif isempty(ext_coef) == 0
                wav1_ecoef = ext_coef(1:2,1)';
                wav2_ecoef = ext_coef(3:4,1)';
            end
            if flag_DPF_correction == 1
                if (sum(rwavelength(1,:) >= 704) + sum(rwavelength(1,:) <= 972)) == 4
                    load Charite_DPF_correction
                    index_DPF1 = find(DPF_correction(:,1) == rwavelength(1,1));
                    index_DPF2 = find(DPF_correction(:,1) == rwavelength(1,2));
                    wav1_ecoef = wav1_ecoef .* DPF_correction(index_DPF1, 2);
                    wav2_ecoef = wav2_ecoef .* DPF_correction(index_DPF2, 2);
                else
                    disp('WARRING: NIRS-SPM corrects DPF in the range of wavelength ([704nm 972nm]).');
                    flag_DPF_correction = 0;
                    return;
                end
            end
            tot_ecoef = [wav1_ecoef; wav2_ecoef];
            tot_ecoef = tot_ecoef .* DPF(1,1) .* dist(1,1);
            coefMat = pinv(tot_ecoef);
            coefMat = reshape(coefMat(:) * ones(1, total_ch), [2 2 total_ch]);
        else %% channel-wise parameters (DPF, extinction coefficient, distance)
            coefMat = zeros(2, 2, total_ch);
            for kk = 1:total_ch
                if isempty(ext_coef) == 1
                    load COPE_e_coef; % load the extinction coefficient file 
                    index_wav1 = find(e_coef(:,1) == rwavelength(kk,1));
                    index_wav2 = find(e_coef(:,1) == rwavelength(kk,2));
                    wav1_ecoef = e_coef(index_wav1,2:3);
                    wav2_ecoef = e_coef(index_wav2,2:3);
                elseif isempty(ext_coef) == 0
                    wav1_ecoef = ext_coef(4*kk-3:4*kk-2,1)';
                    wav2_ecoef = ext_coef(4*kk-1:4*kk,1)';
                end
                if flag_DPF_correction == 1
                    if (sum(rwavelength(kk,:) >= 704) + sum(rwavelength(kk,:) <= 972)) == 4
                        load Charite_DPF_correction
                        index_DPF1 = find(DPF_correction(:,1) == rwavelength(kk,1));
                        index_DPF2 = find(DPF_correction(:,1) == rwavelength(kk,2));
                        wav1_ecoef = wav1_ecoef .* DPF_correction(index_DPF1, 2);
                        wav2_ecoef = wav2_ecoef .* DPF_correction(index_DPF2, 2);
                    else
                        disp(['DPF correction was not applied to ' num2str(kk) 'th channel, wavelength range of ' num2str(kk) 'th channel is not between 704nm and 972nm.']);
                    end
                end
                tot_ecoef = [wav1_ecoef; wav2_ecoef];
                tot_ecoef = tot_ecoef .* DPF(kk,1) .* dist(kk,1);
                coefMat(:,:, kk) = pinv(tot_ecoef);
            end
        end % reading the coefficient matrix
        
        disp('Reading optical density changes from an input file...');
        %% read the optical density changes
        try
            fid = fopen(fname_nirs);
        catch
            disp('ERROR: unable to read the nirs file!!');
            nirs_data = [];
            return;
        end
        index = 1;
        switch fname_nirs(end-2:end)
            case 'csv'
                while 1
                    tline = fgetl(fid);
                    if ~ischar(tline), break, end;
                    index_comma = find(tline == ',');
                    tline(index_comma) = ' ';
                    tline = str2num(tline);
                    mes(index,:) = tline;
                    index = index + 1;
                end
            case 'txt'
                while 1
                    tline = fgetl(fid);
                    if ~ischar(tline), break, end;
                    tline = str2num(tline);
                    mes(index,:) = tline;
                    index = index + 1;
                end
        end 
        fclose(fid);        
        disp('Calculating oxy- and deoxy-hemoglobin changes using modified Beer-Lambert law...');
        %% calculate the oxy- and deoxy-hemoglobin changes using modified
        %% Beer-Lambert law
        for kk = 1:total_ch
            oxydxy = (coefMat(:,:,kk) * [mes(:,2*(kk-1)+1)'; mes(:, 2*kk)'])';
            oxyData(:,kk) = oxydxy(:,1);
            dxyData(:,kk) = oxydxy(:,2);
        end
        nirs_data.oxyData = oxyData;
        nirs_data.dxyData = dxyData;
        nirs_data.nch = total_ch;
        nirs_data.fs = fs;
        nirs_data.wavelength = wavelength;
        nirs_data.distance = dist;
        nirs_data.DPF = DPF;
        if flag_DPF_correction == 0
            nirs_data.DPF_correction = 'none';
        elseif flag_DPF_correction ==  1
            nirs_data.DPF_correction = 'Charite correction';
        end
        disp('Completed.');
    case 'manual_hb' % read the hemoglobin concentration changes from *.csv or *.txt file
        if nargin < 3 
            disp('ERROR: Not enough input arguments');            
            nirs_data = [];
            return;
        end        
        disp('Reading the hemoglobin concentration changes from an input file...');
        fid = fopen(fname_nirs);
        index = 1;
        switch fname_nirs(end-2:end)
            case 'csv'
                while 1
                    tline = fgetl(fid);
                    if ~ischar(tline), break, end;
                    index_comma = find(tline == ',');
                    tline(index_comma) = ' ';
                    tline = str2num(tline);
                    nirs_data.oxyData(index, :) = tline(1,1:2:end-1);
                    nirs_data.dxyData(index, :) = tline(1,2:2:end);
                    index = index + 1;
                end
            case 'txt'
                while 1
                    tline = fgetl(fid);
                    if ~ischar(tline), break, end;
                    tline = str2num(tline);
                    nirs_data.oxyData(index, :) = tline(1,1:2:end-1);
                    nirs_data.dxyData(index, :) = tline(1,2:2:end);
                    index = index + 1;
                end
        end
        fclose(fid);        
        nch = size(nirs_data.oxyData,2);
        nirs_data.nch = nch;
        nirs_data.fs = fs;
        disp('Completed.');
end