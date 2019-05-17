function varargout = NIRS_Data_Conversion(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @NIRS_Data_Conversion_OpeningFcn, ...
    'gui_OutputFcn',  @NIRS_Data_Conversion_OutputFcn, ...
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


function NIRS_Data_Conversion_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;

%%%% initilization (OXYMON MK III)
set(handles.Pop_SystemConfig,'value',1);
set(handles.checkbox_system, 'value',1);
set(handles.checkbox_manual, 'value',0);
set(handles.edit_nch, 'enable', 'off', 'string', '24');
set(handles.edit_fs, 'enable', 'on');
set(handles.edit_distance, 'enable', 'on', 'string', '3.5');
set(handles.edit_wavelength, 'enable', 'on', 'string', '856 781');
set(handles.edit_DPF, 'enable', 'on', 'string', '4');
set(handles.checkbox_DPF_correction, 'value', 1, 'enable', 'on');
set(handles.push_load_param, 'enable', 'off');
set(handles.edit_extinc, 'string', ' ', 'enable', 'off');
set(handles.text9, 'enable', 'off');
set(handles.text10, 'enable', 'off');
guidata(hObject, handles);


% --- Outputs from this function are returned to the command line.
function varargout = NIRS_Data_Conversion_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;


% --- Executes on button press in push_nirfile.
function push_nirfile_Callback(hObject, eventdata, handles)

h = get(handles.Pop_SystemConfig, 'value');
h2 = get(handles.checkbox_system, 'value');

if h2 == 1 %%% system configuration
    if h == 1 %%% OXYMON MK III
        [filen, pathn] = uigetfile('*.nir; *.csv; *.xls', 'Select a data file measured by Oxymon MKIII system');
        path_file_n = [pathn filen];
        if filen(1) == 0 || pathn(1) == 0
            return;
        end
        [tmp1, tmp2, ext] = fileparts(path_file_n);
        if strcmpi(ext, '.nir') == 1 % optical density changes
            fs = str2num(get(handles.edit_fs, 'string'));
            dist = str2num(get(handles.edit_distance, 'string'));
            wavelength = str2num(get(handles.edit_wavelength, 'string'));
            rwavelength = round(wavelength);
            DPF = str2num(get(handles.edit_DPF, 'string'));
            if isempty(fs) == 1 || isempty(dist) == 1 || isempty(wavelength) == 1 || isempty(DPF) == 1
                errordlg('Please enter all required parameter values.');
                return;
            end
            flag_DPF_correction = get(handles.checkbox_DPF_correction, 'value');
            if flag_DPF_correction == 1
                if (sum(rwavelength >= 704) + sum(rwavelength <= 972)) ~= 4
                    errordlg('NIRS-SPM allows the wavelength range of DPF correction between 704nm and 972nm');
                    flag_DPF_correction = 0;
                    set(handles.checkbox_DPF_correction, 'value', 0);
                    return;
                end
            end
            nirs_data = nir_to_mat(path_file_n, fs, dist, rwavelength, DPF, flag_DPF_correction);
            nirs_data.wavelength = wavelength;
            nirs_data.distance = dist;
            nirs_data.DPF = DPF;
            if flag_DPF_correction == 1
                nirs_data.DPF_correction = 'Charite correction';
            elseif flag_DPF_correction == 0
                nirs_data.DPF_correction = 'none';
            end
        elseif strcmpi(ext, '.csv') == 1 || strcmpi(ext, '.txt') == 1
            fid = fopen(path_file_n);
            while 1
                tline = fgetl(fid);
                nindex = find(tline == ',');
                tline(nindex) =  ' ';
                [token, remain] = strtok(tline);
                switch token
                    case 'Export'
                        if isempty(strfind(remain, 'sample rate')) == 0
                            idx1 = strfind(remain, 'rate');
                            idx2 = strfind(remain, 'Hz');
                            fs = str2num(remain(idx1+5:idx2-1));                            
                        end
                    case 'Column'
                        oxy_col = [];
                        deoxy_col = [];
                        while 1
                            tline = fgetl(fid);
                            nindex = find(tline == ',');
                            tline(nindex) = ' ';
                            [token, remain] = strtok(tline);
                            if isempty(strfind(remain, 'Event')) == 0, break, end,                            
                            if isempty(strfind(remain, 'O2Hb')) == 0
                                oxy_col = [oxy_col str2num(token)];
                            elseif isempty(strfind(remain, 'HHb')) == 0
                                deoxy_col = [deoxy_col str2num(token)];
                            end
                        end
                        tmp = fgetl(fid); % blank line;
                        tmp = fgetl(fid); % column number
                        count = 1;
                        while 1 % reading the data
                            tline = fgetl(fid);
                            if ischar(tline) == 0, break, end,
                            nindex = find(tline == ',');
                            tline(nindex) = ' ';
                            tmp = str2num(tline);
                            oxyData(count,:) = tmp(oxy_col);
                            dxyData(count,:) = tmp(deoxy_col);
                            count = count + 1;
                        end
                        break;
                end
            end
            nirs_data.oxyData = oxyData;
            nirs_data.dxyData = dxyData;
            nirs_data.fs = fs;
        elseif strcmpi(ext, '.xls') == 1
            fs = str2num(get(handles.edit_fs, 'string'));
            tmp_data = xlsread(path_file_n, -1);
            nirs_data.oxyData = tmp_data(:,1:3:end);
            nirs_data.dxyData = tmp_data(:,2:3:end);
            nirs_data.fs = fs;
        end
        prompt = {'Please enter the channels which will be used for further analysis.'};
        dlg_title = 'Input dialog';
        num_lines = 1;
        tmp = 1:size(nirs_data.oxyData,2);
        def = {num2str(tmp)};
        answer = inputdlg(prompt, dlg_title, num_lines, def);
        nirs_data.oxyData = nirs_data.oxyData(:,str2num(answer{1}));
        nirs_data.dxyData = nirs_data.dxyData(:,str2num(answer{1}));
        nirs_data.nch = size(nirs_data.oxyData,2);
    elseif h == 2 %%% Hitachi ETG4000 (1set)
        [filen, pathn] = uigetfile('*.csv','Select File to Convert');
        path_file_n = [pathn filen];
        if filen(1) == 0 | pathn(1) == 0
            return;
        end
        fid = fopen(path_file_n);
        h_wait = waitbar(0, 'Please wait... ');
        while 1
            tline = fgetl(fid);
            % update, May 6, 2012
            if isempty(strfind(tline, 'Wave Length')) == 0 %reading wave lengths on each probe
                nindex = find(tline == ',');
                tline(nindex) = ' ';
                txt_wavelength = tline(nindex(1)+1:end);
                error_index = strfind(txt_wavelength, '..');
                txt_wavelength(error_index) =  [];
                
                remain = txt_wavelength;
                ndata = 0;
                wav_mat = [];
                while 1
                    [token remain] = strtok(remain);
                    if isempty(token) == 1
                        break;
                    end
                    sind = find(token == '(')+1;
                    eind = find(token == ')')-1;
                    wav_mat = [wav_mat str2num(token(sind:eind))];
                    ndata = ndata + 1;
                end
                waitbar(1/3, h_wait, 'Data reading (1/3) has been completed.');
            elseif isempty(strfind(tline, 'Sampling Period[s]')) == 0 % reading sampling period
                nindex = find(tline ==  ',');
                tline(nindex) = ' ';
                txt_fs = tline(nindex(1)+1:end);
                try
                    fs = 1./mean(str2num(txt_fs));
                catch
                    error_index = strfind(txt_fs, '..');
                    txt_fs(error_index) = [];
                    fs = 1./mean(str2num(txt_fs));
                end
                waitbar(2/3, h_wait, 'Data reading (2/3) has been completed.');
            elseif isempty(strfind(tline, 'Probe1')) == 0
                disp('Reading the data from Probe1 ...');
                nindex = find(tline ==  ',');
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
                while 1
                    tline = fgetl(fid);
                    if ischar(tline) == 0, break, end,
                    nindex = find(tline == ',');
                    try
                        count = str2num(tline(1:nindex(1)-1));
                        tline(nindex) = ' ';
                        try
                            mes(count, :) = str2num(tline(nindex(1)+1:nindex(ndata+1)-1));
                        catch
                            str = tline(nindex(1)+1:nindex(ndata+1)-1);
                            error_index = strfind(str, '..');
                            str(error_index) = [];
                            mes(count, :) = str2num(str);
                        end
                        try
                            baseline(count) = str2num(tline(nindex(col_prescan-1)+1:nindex(col_prescan)-1));
                        catch
                            baseline(count) = str2num(tline(nindex(col_prescan-1)+1:end));
                        end
                        try
                            vector_onset(count) = str2num(tline(nindex(col_mark-1)+1:nindex(col_mark)-1));
                        end
                    end
                end
                waitbar(3/3, h_wait,'Data reading (3/3) has been completed.');
                disp('Completed.');
                break,
            end
        end
        close(h_wait);
        fclose(fid);
        
        index_base = find(baseline == 1);
        h_wait = waitbar(0, 'Data conversion processing... ');
        nch = ndata./2;
        for kk = 1:nch
            waitbar(kk/(nch-1), h_wait);
            [hb_tmp, hbo_tmp, hbt_tmp] = mes2hb(mes(:,2*kk-1:2*kk), [wav_mat(1,2*kk-1) wav_mat(1,2*kk)], [index_base(1) index_base(end)]);
            nirs_data.oxyData(:,kk) = hbo_tmp;
            nirs_data.dxyData(:,kk) = hb_tmp;
        end
        close(h_wait);
        try
            vector_onset(index_base(1):index_base(end)) = [];
            nirs_data.vector_onset = vector_onset(:);
        end
        nirs_data.fs = fs;
        nirs_data.nch = nch;
        
        set(handles.edit_fs,'string', num2str(fs));
        set(handles.edit_nch, 'string', num2str(nch));
        
    elseif h == 3 %%% Hitachi ETG4000 (2set)
        [filen, pathn] = uigetfile('*.csv','Select File (1st set of optodes) to Convert');
        path_file_n = [pathn filen];
        if filen(1) == 0 | pathn(1) == 0
            return;
        end
        fid = fopen(path_file_n);
        h_wait = waitbar(0, 'Please wait... ');
        
        % start reading data
        while 1
            tline = fgetl(fid);
            % update, May 6, 2012
            if isempty(strfind(tline, 'Wave Length')) == 0 %reading wave lengths on each probe
                nindex = find(tline == ',');
                tline(nindex) = ' ';
                txt_wavelength = tline(nindex(1)+1:end);
                error_index = strfind(txt_wavelength, '..');
                txt_wavelength(error_index) =  [];
                
                remain = txt_wavelength;
                ndata = 0;
                wav_mat = [];
                while 1
                    [token remain] = strtok(remain);
                    if isempty(token) == 1
                        break;
                    end
                    sind = find(token == '(')+1;
                    eind = find(token == ')')-1;
                    wav_mat = [wav_mat str2num(token(sind:eind))];
                    ndata = ndata + 1;
                end
                waitbar(1/3, h_wait, 'Data reading (1/3) has been completed.');
            elseif isempty(strfind(tline, 'Sampling Period[s]')) == 0 % reading sampling period
                nindex = find(tline ==  ',');
                tline(nindex) = ' ';
                txt_fs = tline(nindex(1)+1:end);
                try
                    fs = 1./mean(str2num(txt_fs));
                catch
                    error_index = strfind(txt_fs, '..');
                    txt_fs(error_index) = [];
                    fs = 1./mean(str2num(txt_fs));
                end
                waitbar(2/3, h_wait, 'Data reading (2/3) has been completed.');
            elseif isempty(strfind(tline, 'Probe1')) == 0
                disp('Reading the data from Probe1 ...');
                nindex = find(tline ==  ',');
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
                while 1
                    tline = fgetl(fid);
                    if ischar(tline) == 0, break, end,
                    nindex = find(tline == ',');
                    try
                        count = str2num(tline(1:nindex(1)-1));
                        tline(nindex) = ' ';
                        try
                            mes(count, :) = str2num(tline(nindex(1)+1:nindex(ndata+1)-1));
                        catch
                            str = tline(nindex(1)+1:nindex(ndata+1)-1);
                            error_index = strfind(str, '..');
                            str(error_index) = [];
                            mes(count, :) = str2num(str);
                        end
                        try
                            baseline(count) = str2num(tline(nindex(col_prescan-1)+1:nindex(col_prescan)-1));
                        catch
                            baseline(count) = str2num(tline(nindex(col_prescan-1)+1:end));
                        end
                        try
                            vector_onset1(count) = str2num(tline(nindex(col_mark-1)+1:nindex(col_mark)-1));
                        end
                    end
                end
                waitbar(3/3, h_wait,'Data reading (3/3) has been completed.');
                disp('Completed.');
                break,
            end
        end
        close(h_wait);
        fclose(fid);
        
        index_base = find(baseline == 1);
        h_wait = waitbar(0, 'Data conversion processing... ');
        nch = ndata./2;
        for kk = 1:nch
            waitbar(kk/(nch-1), h_wait);
            [hb_tmp, hbo_tmp, hbt_tmp] = mes2hb(mes(:,2*kk-1:2*kk), [wav_mat(1,2*kk-1) wav_mat(1,2*kk)], [index_base(1) index_base(end)]);
            oxyData1(:,kk) = hbo_tmp;
            dxyData1(:,kk) = hb_tmp;
        end
        close(h_wait);
        try
            vector_onset1(index_base(1):index_base(end)) = [];
        end
        clear mes
        clear baseline
        
        % reading the dataset from the 2nd probe 
        [filen, pathn] = uigetfile('*.csv','Select File (2nd set of optodes) to Convert');
        path_file_n = [pathn filen];
        if filen(1) == 0 || pathn(1) == 0
            return;
        end
        fid = fopen(path_file_n);
        h_wait = waitbar(0, 'Please wait... ');
        while 1
            tline = fgetl(fid);
            % update, May 6, 2012
            if isempty(strfind(tline, 'Wave Length')) == 0 %reading wave lengths on each probe
                nindex = find(tline == ',');
                tline(nindex) = ' ';
                txt_wavelength = tline(nindex(1)+1:end);
                error_index = strfind(txt_wavelength, '..');
                txt_wavelength(error_index) =  [];
                
                remain = txt_wavelength;
                ndata = 0;
                wav_mat = [];
                while 1
                    [token remain] = strtok(remain);
                    if isempty(token) == 1
                        break;
                    end
                    sind = find(token == '(')+1;
                    eind = find(token == ')')-1;
                    wav_mat = [wav_mat str2num(token(sind:eind))];
                    ndata = ndata + 1;
                end
                waitbar(1/2, h_wait, 'Data reading (1/2) has been completed.');
            elseif isempty(strfind(tline, 'Probe2')) == 0
                disp('Reading the data from Probe2 ...');
                nindex = find(tline ==  ',');
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
                while 1
                    tline = fgetl(fid);
                    if ischar(tline) == 0, break, end,
                    nindex = find(tline == ',');
                    try
                        count = str2num(tline(1:nindex(1)-1));
                        tline(nindex) = ' ';
                        try
                            mes(count, :) = str2num(tline(nindex(1)+1:nindex(ndata+1)-1));
                        catch
                            str = tline(nindex(1)+1:nindex(ndata+1)-1);
                            error_index = strfind(str, '..');
                            str(error_index) = [];
                            mes(count, :) = str2num(str);
                        end
                        try
                            baseline(count) = str2num(tline(nindex(col_prescan-1)+1:nindex(col_prescan)-1));
                        catch
                            baseline(count) = str2num(tline(nindex(col_prescan-1)+1:end));
                        end
                        try
                            vector_onset2(count) = str2num(tline(nindex(col_mark-1)+1:nindex(col_mark)-1));
                        end
                    end
                end
                waitbar(2/2, h_wait,'Data reading (2/2) has been completed.');
                disp('Completed.');
                break,
            end
        end
        close(h_wait);
        fclose(fid);
        
        index_base = find(baseline == 1);
        h_wait = waitbar(0, 'Data conversion processing... ');
        for kk = 1:nch
            waitbar(kk/(nch-1), h_wait);
            [hb_tmp, hbo_tmp, hbt_tmp] = mes2hb(mes(:,2*kk-1:2*kk), [wav_mat(1,2*kk-1) wav_mat(1,2*kk)], [index_base(1) index_base(end)]);
            oxyData2(:,kk) = hbo_tmp;
            dxyData2(:,kk) = hb_tmp;
        end
        close(h_wait);
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
        set(handles.edit_fs,'string', num2str(fs));
        set(handles.edit_nch, 'string', num2str(nch));
    elseif h == 4 %%% Hitachi ETG-4000 (HbO, HbR, HbT)
        %%% read the oxy-hb data
        [filen, pathn] = uigetfile('*.csv','Select the OxyHb (HbO) file');
        path_file_n = [pathn filen];
        if filen(1) == 0 | pathn(1) == 0
            return;
        end
        fid = fopen(path_file_n);
        disp('Loading the OxyHb data starts...');
        while 1
            tline = fgetl(fid);
            if isempty(strfind(tline, 'Sampling Period[s]')) == 0
                nindex = find(tline == ',');
                tline(nindex) = ' ';
                txt_fs = tline(nindex(1)+1:end);
                fs = 1./mean(str2num(txt_fs));
            end
            if isempty(strfind(tline, 'Data')) == 0
                tline = fgetl(fid);
                nch = length(strfind(tline, 'CH'));
                nindex = find(tline == ',');
                if isempty(strfind(tline, 'Oxy')) == 1
                    errordlg('Please select the Oxy-Hb file.');
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
                    tline_data = tline(nindex(1)+1:nindex(nch+1)-1);
                    nindex_d = find(tline_data == ',');
                    tline_data(nindex_d) = ' ';
                    tline_data = str2num(tline_data);
                    count = str2num(tline(1:nindex(1)-1));
                    nirs_data.oxyData(count, :) = tline_data;
                    try
                        vector_onset(count) = str2num(tline(nindex(col_mark-1)+1:nindex(col_mark)-1));
                    end
                    try
                        baseline(count) = str2num(tline(nindex(col_prescan-1)+1:nindex(col_prescan)-1));
                    end
                end
                break;
            end
        end
        disp('HbO data loading has been finished.');
        
        %%% read the deoxy-hb data
        [filen, pathn] = uigetfile('*.csv','Select the DeoxyHb (HbR) file');
        path_file_n = [pathn filen];
        if filen(1) == 0 | pathn(1) == 0
            return;
        end
        fid = fopen(path_file_n);
        disp('Loading the DeoxyHb data starts...');
        while 1
            tline = fgetl(fid);
            if isempty(strfind(tline, 'Sampling Period[s]')) == 0
                if ismember('fs', who) == 0
                    nindex = find(tline == ',');
                    tline(nindex) = ' ';
                    txt_fs = tline(nindex(1)+1:end);
                    fs = 1./mean(str2num(txt_fs));
                end
            end
            if isempty(strfind(tline, 'Data')) == 0
                tline = fgetl(fid);
                nch = length(strfind(tline, 'CH'));
                nindex = find(tline == ',');
                if isempty(strfind(tline, 'Deoxy')) == 1
                    errordlg('Please select the Deoxy-Hb file.');
                    return;
                end
                while 1
                    tline = fgetl(fid);
                    if ischar(tline) == 0, break, end,
                    nindex = find(tline == ',');
                    tline_data = tline(nindex(1)+1:nindex(nch+1)-1);
                    nindex_d = find(tline_data == ',');
                    tline_data(nindex_d) = ' ';
                    tline_data = str2num(tline_data);
                    count = str2num(tline(1:nindex(1)-1));
                    nirs_data.dxyData(count, :) = tline_data;
                end
                break;
            end
        end
        disp('HbR data loading has been finished.');
        
        %%% read the total-hb data
        [filen, pathn] = uigetfile('*.csv','Select the TotalHb (HbT) file');
        path_file_n = [pathn filen];
        if filen(1) == 0 | pathn(1) == 0
            return;
        end
        fid = fopen(path_file_n);
        disp('Loading the TotalHb data starts...');
        while 1
            tline = fgetl(fid);
            if isempty(strfind(tline, 'Sampling Period[s]')) == 0
                if ismember('fs', who) == 0
                    nindex = find(tline == ',');
                    tline(nindex) = ' ';
                    txt_fs = tline(nindex(1)+1:end);
                    fs = 1./mean(str2num(txt_fs));
                end
            end
            if isempty(strfind(tline, 'Data')) == 0
                tline = fgetl(fid);
                nch = length(strfind(tline, 'CH'));
                nindex = find(tline == ',');
                if isempty(strfind(tline, 'Total')) == 1
                    errordlg('Please select the Total-Hb file.');
                    return;
                end
                while 1
                    tline = fgetl(fid);
                    if ischar(tline) == 0, break, end,
                    nindex = find(tline == ',');
                    tline_data = tline(nindex(1)+1:nindex(nch+1)-1);
                    nindex_d = find(tline_data == ',');
                    tline_data(nindex_d) = ' ';
                    tline_data = str2num(tline_data);
                    count = str2num(tline(1:nindex(1)-1));
                    nirs_data.tHbData(count, :) = tline_data;
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
        set(handles.edit_fs,'string', num2str(fs));
        set(handles.edit_nch, 'string', num2str(nch));
        
    elseif h == 5 %%% ISS Imagent
        [filen, pathn] = uigetfile('*.log; *.txt','Select File to Read the HbR/HbT Concentration Changes');
        path_file_n = [pathn filen];
        if filen(1) == 0 | pathn(1) == 0
            return;
        end
        fid = fopen(path_file_n);
        tline = fgetl(fid);
        index = 1;
        while 1
            tline = fgetl(fid);
            if ~ischar(tline), break, end;
            tline = str2num(tline);
            if size(tline, 1) == 0
                continue
            end
            nirs_data.oxyData(index, :) = tline(1,6:2:end-2);
            nirs_data.dxyData(index, :) = tline(1,7:2:end-1);
            nirs_data.vector_onset(index,1) = tline(1,end);
            time(index,1) = tline(1,1);
            index = index + 1;
        end
        fclose(fid);
        nirs_data.vector_onset = nirs_data.vector_onset(:);
        fs = 1./(mean(diff(time)));
        set(handles.edit_fs, 'string', num2str(fs));
        nirs_data.nch = size(nirs_data.oxyData,2);
        set(handles.edit_nch, 'string', num2str(nirs_data.nch));
        set(handles.edit_fs, 'enable', 'on');
        
    elseif h == 6 %%% Hamamatsu NIRO-200
        [filen, pathn] = uigetfile('*.NI2','Select File to Read the HbO/HbR Concentration Changes');
        path_file_n = [pathn filen];
        if filen(1) == 0 | pathn(1) == 0
            return;
        end
        fid = fopen(path_file_n);
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
        set(handles.edit_fs, 'string', num2str(fs));
        nirs_data.nch = size(nirs_data.oxyData,2);
        set(handles.edit_nch, 'string', num2str(nirs_data.nch));
        set(handles.edit_fs, 'enable', 'on');
    elseif h == 7 %%% NIRX DYNOT-232
        %%% read the parameters
        disp('Reading the parameters for modified Beer-Lambert law starts...');
        nch = str2num(get(handles.edit_nch,'string'));
        fs = str2num(get(handles.edit_fs, 'string'))';
        dist = str2num(get(handles.edit_distance, 'string'))';
        wavelength = str2num(get(handles.edit_wavelength, 'string'));
        DPF = str2num(get(handles.edit_DPF, 'string'));
        ecoef = str2num(get(handles.edit_extinc, 'string'));
        if isempty(fs) == 1 || isempty(dist) ==1 || isempty(wavelength) == 1 || isempty(DPF) == 1 || isempty(ecoef) == 1
            errordlg('Please enter all required parameter values.');
            return;
        end
        try
            config = handles.config;
        catch
            errordlg('Please load the channel configuration');
            return;
        end
        disp('Completed.');
        
        [filen, pathn] = uigetfile('*.wl1','Select the First File to Read the Optical Density Changes'); % load the *.wl1 file
        path_file_n = [pathn filen];
        if filen(1) == 0 || pathn(1) == 0
            return;
        end
        fid = fopen(path_file_n);
        count = 1;
        disp(['Reading optical density changes from ' filen ' starts...']);
        while 1
            tline = fgetl(fid);
            if ~ischar(tline), break, end
            tline = str2num(tline);
            mes1(count,:) = tline;
            count = count + 1;
        end
        fclose(fid);
        disp('Completed.');
        
        [filen, pathn] = uigetfile('*.wl2', 'Select the Second File to Read the Optical Density Changes'); %% load the *.wl2 file
        path_file_n = [pathn filen];
        if filen(1) == 0 | pathn(1) == 0
            return;
        end
        fid = fopen(path_file_n);
        count = 1;
        
        disp(['Reading optical density changes from ' filen ' starts...']);
        while 1
            tline = fgetl(fid);
            if ~ischar(tline); break, end;
            tline = str2num(tline);
            mes2(count,:) = tline;
            count = count + 1;
        end
        fclose(fid);
        disp('Completed.');
        
        disp('Converting optical density changes to hemoglobin changes starts...');
        nTx = max(config(:,1));
        nRx = max(config(:,2));
        if size(mes1, 2) ~= nTx*nRx || size(mes2, 2) ~= nTx*nRx
            nTx = inputdlg('Please enter the number of sources');
            nTx = str2num(cell2mat(nTx));
            nRx = inputdlg('Please enter the number of detectors');
            nRx = str2num(cell2mat(nRx));
        end
        
        %eingef?t analog nilab2 nur der Fall "mean of whole timecourse"
        
        s1 = count-1;
        
        mes2_log = real(-log10( (mes2  )./ ...
            ( repmat(mean(mes2,1), [s1,1]))   ))    ;        
        mes1_log = real(-log10( (mes1  )./ ...
            ( repmat(mean(mes1,1), [s1,1]))   ))    ;
        
        coefMat = dist.*(diag(DPF) * [ecoef(1,1:2); ecoef(1,3:4)]);
        coefMat = pinv(coefMat);
        
        for kk = 1:nch
            index = (config(kk, 1)-1)* nRx + config(kk,2);
            oxydxy = coefMat * [mes1_log(:, index)'; mes2_log(:, index)'];
            oxyData(:, kk) = oxydxy(1,:)';
            dxyData(:, kk) = oxydxy(2,:)';
        end
        disp('Completed.');
        
        nirs_data.oxyData = oxyData;
        nirs_data.dxyData = dxyData;
        nirs_data.nch = nch;
        nirs_data.fs = fs;
        nirs_data.wavelength = wavelength;
        nirs_data.distance = dist;
        nirs_data.DPF = DPF;
        
    elseif h == 8 %%%% Spectratech OEG-16
        [filen, pathn] = uigetfile('*.csv', 'Select File to Convert');
        path_file_n = [pathn filen];
        if filen(1) == 0 | pathn(1) == 0
            return;
        end
        fid = fopen(path_file_n);
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
        nirs_data.fs = str2num(get(handles.edit_fs, 'string'));
        nirs_data.nch = size(nirs_data.oxyData, 2);
    elseif h == 9 %%% Shimadzu OMM,FOIRE-3000
        [filen, pathn] = uigetfile('*.txt','Select File to Read OMM TXT file');
        path_file_n = [pathn filen];
        if filen(1) == 0 | pathn(1) == 0
            return;
        end
        fid = fopen(path_file_n);
        
        h_wait = waitbar(0, 'Data loading... ');
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
                    waitbar((newlabel(1,1)-stt)/(stp-stt), h_wait);
                    time(index,1) = newlabel(1,1);
                    index = index + 1;
                end
                break
            end
        end
        close(h_wait);
        fclose(fid);
        fs = 1./(mean(diff(time)));
        nirs_data.fs = fs;
        set(handles.edit_fs, 'string', num2str(fs));
        nirs_data.nch = size(nirs_data.oxyData,2);
        set(handles.edit_nch, 'string', num2str(nirs_data.nch));
        set(handles.edit_fs, 'enable', 'on');
    elseif h == 10 % BIOPAC fNIR
        [filen, pathn] = uigetfile('*.oxy','Select File to Read HbO/HbR Chanes');
        path_file_n = [pathn filen];
        if filen(1) == 0 | pathn(1) == 0
            return;
        end
        fid = fopen(path_file_n);
        if filen(1) == 0 | pathn(1) == 0
            return;
        end
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
        set(handles.edit_fs, 'string', num2str(fs));
        nirs_data.nch = size(nirs_data.oxyData,2);
        set(handles.edit_nch, 'string', num2str(nirs_data.nch));
        set(handles.edit_fs, 'enable', 'on');
        disp('Finished.');
    elseif h == 11 % HomER Software
        % first, load the raw data '*.nir' to read the model parameter and channel
        % configuration.
        
        [filen, pathn] = uigetfile('*.nirs', 'Select ''*.nirs'' File to Read the Model Parameters');
        path_file_n = [pathn filen];
        if filen(1) == 0 | pathn(1) == 0
            return;
        end
        disp('Reading the HomER data starts...');
        load(path_file_n, '-mat');
        
        %second, load the *.mat file to read the Hb concentration changes
        [filen, pathn] = uigetfile('*.mat', 'Select ''*.mat'' File to Read HbO/HbR/HbT Changes');
        path_file_n = [pathn filen];
        if filen(1) == 0 | pathn(1) == 0
            return;
        end
        load(path_file_n);
        
        fs = 1./mean(diff(t));% sampling frequency
        
        set(handles.edit_fs, 'string', num2str(fs));
        set(handles.edit_wavelength, 'string', num2str(SD.Lambda));
        set(handles.edit_nch, 'string', num2str(size(dataSave.HbO, 2)));
        
        nirs_data.oxyData = dataSave.HbO;
        nirs_data.dxyData = dataSave.HbR;
        try
            nirs_data.tHbData = dataSave.HbT;
        catch
            nirs_data.tHbData = nirs_data.oxyData + nirs_data.dxyData;
        end
        nirs_data.fs = fs;
        nirs_data.wavelength = SD.Lambda;
        nirs_data.nch = size(dataSave.HbO,2);
        nirs_data.ch_config = SD.MeasList;
        disp('Finished.');
    end
elseif h2 == 0 %% manual input
    if h == 1 %%% optical density changes
        nch = str2num(get(handles.edit_nch, 'string'));
        fs = str2num(get(handles.edit_fs, 'string'))';
        dist = str2num(get(handles.edit_distance, 'string'))';
        wavelength = str2num(get(handles.edit_wavelength, 'string'));
        rwavelength = round(wavelength);
        rwavelength = reshape(rwavelength, [2 size(rwavelength,2)/2])';
        DPF = str2num(get(handles.edit_DPF,'string'))';
        flag_DPF_correction = get(handles.checkbox_DPF_correction,'value');
        
        try
            ext_coef = str2num(get(handles.edit_extinc, 'string'))';
        catch
            ext_coef = [];
        end
        
        if isempty(fs) == 1 || isempty(dist) == 1 || isempty(wavelength) == 1 || isempty(DPF) == 1 || isempty(nch) == 1
            errordlg('Please enter all required parameter values.');
            return;
        end
        
        %% check if source-detector distance, wavelength, and DPF depend on
        %% specific channels or not.
        flag_param = 0;
        if length(dist) == 1
            dist = dist * ones(nch,1);
            flag_param = flag_param + 1;
        end
        if length(rwavelength) == 2
            rwavelength = ones(nch,1) * rwavelength;
            flag_param = flag_param + 1;
        end
        if length(DPF) == 1
            DPF = DPF * ones(nch,1);
            flag_param = flag_param + 1;
        end
        if length(ext_coef) == 4 || isempty(ext_coef) == 1
            try
                ext_coef = ext_coef * ones(1, nch);
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
                    errordlg('NIRS-SPM allows the wavelength range of DPF correction between 704nm and 972nm');
                    flag_DPF_correction = 0;
                    set(handles.checkbox_DPF_correction, 'value', 0);
                    return;
                end
            end
            tot_ecoef = [wav1_ecoef; wav2_ecoef];
            tot_ecoef = tot_ecoef .* DPF(1,1) .* dist(1,1);
            coefMat = pinv(tot_ecoef);
            coefMat = reshape(coefMat(:) * ones(1, nch), [2 2 nch]);
        else %% channel-wise parameters (DPF, extinction coefficient, distance)
            coefMat = zeros(2, 2, nch);
            for kk = 1:nch
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
        [filen, pathn] = uigetfile('*.csv; *.txt','Select File to Read the Optical Density Changes');
        path_file_n = [pathn filen];
        if filen(1) == 0 || pathn(1) == 0
            return;
        end
        fid = fopen(path_file_n);
        index = 1;
        switch filen(end-2:end)
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
        for kk = 1:nch
            oxydxy = (coefMat(:,:,kk) * [mes(:,2*(kk-1)+1)'; mes(:, 2*kk)'])';
            oxyData(:,kk) = oxydxy(:,1);
            dxyData(:,kk) = oxydxy(:,2);
        end
        nirs_data.oxyData = oxyData;
        nirs_data.dxyData = dxyData;
        nirs_data.nch = nch;
        nirs_data.fs = fs;
        nirs_data.wavelength = wavelength;
        nirs_data.distance = dist;
        nirs_data.DPF = DPF;
        if flag_DPF_correction == 0
            nirs_data.DPF_correction = 'none';
        elseif flag_DPF_correction ==  1
            nirs_data.DPF_correction = 'Charite correction';
        end
    elseif h == 2 %%%% converted HbO and HbR
        fs = str2num(get(handles.edit_fs, 'string'));
        if isempty(fs) == 1
            errordlg('Sampling frequency is not specified.');
            return;
        end
        [filen, pathn] = uigetfile('*.csv; *.txt','Select File to Read the HbR/HbT Concentration Changes');
        path_file_n = [pathn filen];
        if filen(1) == 0 | pathn(1) == 0
            return;
        end
        fid = fopen(path_file_n);
        index = 1;
        switch filen(end-2:end)
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
        set(handles.edit_nch, 'string', num2str(nch));
        nirs_data.nch = nch;
        nirs_data.fs = fs;
    end
end
handles.nirs_data = nirs_data;
guidata(hObject, handles);

% --- Executes on button press in push_save.
function push_save_Callback(hObject, eventdata, handles)
nirs_data = handles.nirs_data;
if get(handles.checkbox_system, 'value') == 1 && (get(handles.Pop_SystemConfig, 'value') == 5 || get(handles.Pop_SystemConfig, 'value') == 6 || get(handles.Pop_SystemConfig, 'value') == 9 || get(handles.Pop_SystemConfig, 'value') == 10)%% 5: ISS imagent, 6: Hamamatsu NIRO-200, 9 : Shimadzu FOIRE-3000, 10: BIOPAC fNIR
    nirs_data.fs = str2num(get(handles.edit_fs, 'string'));
end

% for HomER software
if get(handles.checkbox_system, 'value') == 1 && get(handles.Pop_SystemConfig, 'value') == 11
    try
        [filen, pathn] = uiputfile('*.txt', 'Select File to Save the Channel Configuration');
        path_filen = [pathn filen];
        if path_filen == 0
            return;
        end
        fid = fopen(path_filen, 'w');
        fprintf(fid, '%s\r\n', 'Channel Configuration adapted from the HomER software');
        fprintf(fid, '%s\r\n', [num2str(nirs_data.nch) 'ch']);
        fprintf(fid, '%s\r\n', ' set');
        fprintf(fid, '%s\r\n', '');
        
        [r c] = find(nirs_data.ch_config(:,3) == 1 & nirs_data.ch_config(:,4) == 1);
        for kk = 1:length(r)
            fprintf(fid, '%s\r\n', num2str(nirs_data.ch_config(r(kk),1:2)));
        end
        fclose(fid);
        nirs_data = rmfield(nirs_data, 'ch_config');
    catch
        errordlg('Failed to save the channel configuration automatically!! Please manually write the channel configuration and save it as txt format.');
    end
end

if isempty(nirs_data) == 0
    [filen, pathn] = uiputfile('*.mat');
    path_filen = [pathn filen];
    if path_filen == 0
        return;
    end
    save(path_filen, 'nirs_data');
else
    errordlg('Optical density data is not converted. Please execute the converting routing again','Error');
    return;
end


function edit_fs_Callback(hObject, eventdata, handles)

function edit_fs_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_nch_Callback(hObject, eventdata, handles)

function edit_nch_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on selection change in Pop_SystemConfig.
function Pop_SystemConfig_Callback(hObject, eventdata, handles)
h = get(handles.Pop_SystemConfig, 'value');
h2 = get(handles.checkbox_system, 'value');
set(handles.edit_extinc, 'enable', 'off', 'string', ' ');
set(handles.text9, 'enable','off');
set(handles.text10, 'enable', 'off');

if h2 == 1 %%% System Configuration
    if h == 1 %%%OXYMON MK3
        set(handles.edit_nch,'enable', 'off', 'string', '24');
        set(handles.edit_fs, 'enable', 'on');
        set(handles.edit_distance, 'enable', 'on', 'string', '3.5');
        set(handles.edit_wavelength, 'enable', 'on', 'string', '856 781');
        set(handles.checkbox_DPF_correction, 'value', 1, 'enable', 'on');
        set(handles.edit_DPF, 'enable', 'on', 'string', '4');
        set(handles.push_load_param, 'enable', 'off');
    elseif h == 2  || h == 3 || h == 4 || h == 5 || h == 6 || h == 9 || h == 10 || h == 11%%%% Hitachi ETG 4000 (1set) || Hitachi ETG4000 (2set) || Hitachi ETG4000 (HbO, HbR, HbT) || ISS Imagent || Hamamatsu NIRO-200 || Shimadzu OMM,FOIRE-3000 || BIOPAC fNIR || HomER software
        set(handles.edit_nch,'enable', 'off', 'string', '');
        set(handles.edit_fs, 'enable', 'off', 'string', '');
        set(handles.edit_distance, 'enable', 'off', 'string', '');
        set(handles.edit_wavelength, 'enable', 'off', 'string', '');
        set(handles.checkbox_DPF_correction, 'value', 0, 'enable', 'off');
        set(handles.edit_DPF, 'enable', 'off', 'string', '');
        set(handles.push_load_param, 'enable', 'off');
    elseif h == 7 %%%% DYNOT 232 System (NIRX)
        %% under construction
        set(handles.edit_extinc, 'enable', 'on', 'string', '1.4866 3.8437 2.2314 1.7917');
        set(handles.text9, 'enable','on');
        set(handles.text10, 'enable', 'on');
        set(handles.edit_nch,'enable', 'on','string','80');
        set(handles.edit_fs, 'enable', 'on');
        set(handles.edit_distance, 'enable', 'on', 'string', '2.5');
        set(handles.edit_wavelength, 'enable', 'on','string', '760 830');
        set(handles.edit_DPF, 'enable', 'on','string', '7.15 5.98');
        set(handles.push_load_param, 'enable', 'on', 'string', 'Load Ch. configuration');
        set(handles.checkbox_DPF_correction, 'value', 0, 'enable', 'off');
    elseif h == 8 %%%%% Spectratech OEG-16
        set(handles.edit_nch,'enable', 'off', 'string', '');
        set(handles.edit_fs, 'enable', 'on', 'string', '');
        set(handles.edit_distance, 'enable', 'off', 'string', '');
        set(handles.edit_wavelength, 'enable', 'off', 'string', '');
        set(handles.checkbox_DPF_correction, 'value', 0, 'enable', 'off');
        set(handles.edit_DPF, 'enable', 'off', 'string', '');
        set(handles.push_load_param, 'enable', 'off');
    end
elseif h2 == 0 %%% Manual configuration
    if h == 1 %%% optical density changes
        set(handles.edit_nch, 'enable', 'on', 'string', '');
        set(handles.edit_fs, 'enable', 'on');
        set(handles.edit_distance, 'enable', 'on');
        set(handles.edit_wavelength, 'enable', 'on');
        set(handles.checkbox_DPF_correction, 'value', 1, 'enable', 'on');
        set(handles.edit_DPF, 'enable', 'on');
        set(handles.push_load_param, 'enable', 'on', 'string', 'Load parameters');
        set(handles.text9, 'enable', 'on');
        set(handles.text10, 'enable', 'on');
        set(handles.edit_extinc,'enable','on');
    elseif h == 2 %%% converted HbO and HbR concentration changes
        set(handles.edit_nch, 'enable', 'off', 'string', '');
        set(handles.edit_fs, 'enable', 'on');
        set(handles.edit_distance, 'enable', 'off', 'string', '');
        set(handles.edit_wavelength, 'enable', 'off', 'string', '');
        set(handles.checkbox_DPF_correction, 'value', 0, 'enable', 'off');
        set(handles.edit_DPF, 'enable', 'off','string', '');
        set(handles.push_load_param, 'enable', 'off');
    end
end


% --- Executes during object creation, after setting all properties.
function Pop_SystemConfig_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in checkbox_manual.
function checkbox_manual_Callback(hObject, eventdata, handles)
h = get(handles.checkbox_manual, 'value');
set(handles.checkbox_system, 'value', 1-h);
if h == 1
    str{1} = 'Optical density changes';
    str{2} = 'Converted HbO and HbR changes';
    set(handles.edit_nch, 'enable', 'on', 'string', '');
    set(handles.edit_distance, 'enable', 'on', 'string', '');
    set(handles.edit_wavelength, 'enable', 'on', 'string', '');
    set(handles.edit_DPF, 'enable', 'on', 'string', '');
    set(handles.push_load_param, 'enable', 'on', 'string', 'Load parameters');
    set(handles.edit_extinc, 'enable', 'on', 'string', ' ');
    set(handles.text9, 'enable', 'on');
    set(handles.text10, 'enable', 'on');
else
    str{1} = 'OXYMON MK III';
    str{2} = 'Hitachi ETG-4000 (1 set, Optical density)';
    str{3} = 'Hitachi ETG-4000 (2 set, Optical density)';
    str{4} = 'Hitachi ETG-4000 (HbO, HbR, HbT)';
    str{5} = 'ISS Imagent';
    str{6} = 'Hamamatsu NIRO-200';
    str{7} = 'NIRX DYNOT-232';
    str{8} = 'Spectratech OEG-16';
    str{9} = 'Shimadzu OMM,FOIRE-3000';
    str{10} = 'BIOPAC fNIR';
    str{11} = 'HomER Software';
    set(handles.edit_nch, 'enable', 'off', 'string', '24');
    set(handles.edit_distance, 'enable', 'on', 'string', '3.5');
    set(handles.edit_wavelength, 'enable', 'on', 'string', '856 781');
    set(handles.edit_DPF, 'enable', 'on', 'string', '4');
    set(handles.push_load_param, 'enable', 'off');
    set(handles.edit_extinc, 'enable', 'off', 'string', ' ');
    set(handles.text9, 'enable','off');
    set(handles.text10, 'enable', 'off');
end
set(handles.edit_fs, 'enable', 'on');
set(handles.checkbox_DPF_correction, 'value', 1, 'enable', 'on');
set(handles.Pop_SystemConfig, 'string',str);
set(handles.Pop_SystemConfig, 'value',1);

% --- Executes on button press in checkbox_system.
function checkbox_system_Callback(hObject, eventdata, handles)
h = get(handles.checkbox_system, 'value');
set(handles.checkbox_manual, 'value', 1-h);
if h == 1
    str{1} = 'OXYMON MK III';
    str{2} = 'Hitachi ETG-4000 (1 set, Optical density)';
    str{3} = 'Hitachi ETG-4000 (2 set, Optical density)';
    str{4} = 'Hitachi ETG-4000 (HbO, HbR, HbT)';
    str{5} = 'ISS Imagent';
    str{6} = 'Hamamatsu NIRO-200';
    str{7} = 'NIRX DYNOT-232';
    str{8} = 'Spectratech OEG-16';
    str{9} = 'Shimadzu OMM,FOIRE-3000';
    str{10} = 'BIOPAC fNIR';
    str{11} = 'HomER Software';
    set(handles.edit_nch, 'enable', 'off', 'string', '24');
    set(handles.edit_distance, 'enable', 'on', 'string', '3.5');
    set(handles.edit_wavelength, 'enable', 'on', 'string', '856 781');
    set(handles.edit_DPF, 'enable', 'on', 'string', '4');
    set(handles.push_load_param, 'enable', 'off');
    set(handles.edit_extinc, 'enable', 'off', 'string', ' ');
    set(handles.text9, 'enable','off');
    set(handles.text10, 'enable', 'off');
else
    str{1} = 'Optical density changes';
    str{2} = 'Converted HbO and HbR changes';
    set(handles.edit_nch, 'enable', 'on', 'string', '');
    set(handles.edit_distance, 'enable', 'on', 'string', '');
    set(handles.edit_wavelength, 'enable', 'on', 'string', '');
    set(handles.edit_DPF, 'enable', 'on', 'string', '');
    set(handles.push_load_param, 'enable', 'on', 'string', 'Load parameters');
    set(handles.edit_extinc, 'enable', 'on', 'string', ' ');
    set(handles.text9, 'enable','on');
    set(handles.text10, 'enable', 'on');
end
set(handles.text9, 'enable','off');
set(handles.text10, 'enable', 'off');
set(handles.edit_fs, 'enable', 'on');
set(handles.checkbox_DPF_correction, 'value', 1, 'enable', 'on');
set(handles.Pop_SystemConfig, 'string',str);
set(handles.Pop_SystemConfig, 'value',1);


% --- Executes on button press in checkbox_DPF_correction.
function checkbox_DPF_correction_Callback(hObject, eventdata, handles)

function edit_wavelength_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function edit_wavelength_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function edit_DPF_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function edit_DPF_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function edit_distance_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function edit_distance_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in push_load_param.
function push_load_param_Callback(hObject, eventdata, handles)
h2 = get(handles.checkbox_system, 'value');
nch = str2num(get(handles.edit_nch, 'string'));
if h2 == 1
    [filen, pathn] = uigetfile('*.txt; *.mat','Select the file to contain the channel configuration.');
    path_file_n = [pathn filen];
    if filen(1) == 0 | pathn(1) == 0
        return;
    end
    switch filen(end-2:end)
        case 'mat'
            load(path_file_n);
            IMGlabel = ni.IMGlabel;
            for kk = 1:nch
                tmp = IMGlabel{kk};
                index = find(tmp == '-');
                config(kk,1) = str2num(tmp(1:index-1));
                config(kk,2) = str2num(tmp(index+1:end));
            end
        case 'txt'
            fid = fopen(path_file_n);
            
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
    handles.config = config;
    guidata(hObject, handles);
elseif h2 == 0
    [filen, pathn] = uigetfile('*.txt; *.csv','Select the parameter file used in modified Beer-lambert law.');
    path_file_n = [pathn filen];
    if filen(1) == 0 | pathn(1) == 0
        return;
    end
    switch filen(end-2:end)
        case 'txt' % text file format
            fid = fopen(path_file_n);
            while 1
                tline = fgetl(fid);
                if ~ischar(tline), break, end;
                index = find(tline == ' ');
                % reading the parameter values from text file
                if strcmpi(tline(1:index(1)-1), 'Total_number_of_Ch.') == 1
                    set(handles.edit_nch, 'string', tline(index(1)+1:end));
                elseif strcmpi(tline(1:index(1)-1), 'Sampling_freq.[Hz]') == 1
                    set(handles.edit_fs, 'string', tline(index(1)+1:end));
                elseif strcmpi(tline(1:index(1)-1), 'Distance[cm]') == 1
                    set(handles.edit_distance, 'string', tline(index(1)+1:end));
                elseif strcmpi(tline(1:index(1)-1), 'Wave_length[nm]') == 1
                    set(handles.edit_wavelength, 'string', tline(index(1)+1:end));
                elseif strcmpi(tline(1:index(1)-1), 'DPF') == 1
                    set(handles.edit_DPF, 'string',  tline(index(1)+1:end));
                elseif strcmpi(tline(1:index(1)-1), 'Correction') == 1
                    if strcmpi(tline(index(1)+1:end), 'yes') == 1
                        set(handles.checkbox_DPF_correction,'value',1);
                    elseif strcmpi(tline(index(1)+1:end), 'no') == 1
                        set(handles.checkbox_DPF_correction,'value',0);
                    end
                elseif strcmpi(tline(1:index(1)-1),'Extinction_coefficient') == 1
                    set(handles.edit_extinc, 'string', tline(index(1)+1:end));
                end
            end
            fclose(fid);
        case 'csv' % .csv file format
            disp('Reading the parameter values from *.csv file...');
            fid = fopen(path_file_n);
            while 1
                tline = fgetl(fid);
                if ~ischar(tline), break, end;
                index = find(tline == ',');
                tline(index) = ' ';
                % reading the parameter values from text file
                if strcmpi(tline(1:index(1)-1), 'Total_number_of_Ch.') == 1
                    set(handles.edit_nch, 'string', tline(index(1)+1:end));
                elseif strcmpi(tline(1:index(1)-1), 'Sampling_freq.[Hz]') == 1
                    set(handles.edit_fs, 'string', tline(index(1)+1:end));
                elseif strcmpi(tline(1:index(1)-1), 'Distance[cm]') == 1
                    set(handles.edit_distance, 'string', tline(index(1)+1:end));
                elseif strcmpi(tline(1:index(1)-1), 'Wave_length[nm]') == 1
                    set(handles.edit_wavelength, 'string', tline(index(1)+1:end));
                elseif strcmpi(tline(1:index(1)-1), 'DPF') == 1
                    set(handles.edit_DPF, 'string',  tline(index(1)+1:end));
                elseif strcmpi(tline(1:index(1)-1), 'Correction') == 1
                    if strcmpi(tline(index(1)+1:end), 'yes') == 1
                        set(handles.checkbox_DPF_correction,'value',1);
                    elseif strcmpi(tline(index(1)+1:end), 'no') == 1
                        set(handles.checkbox_DPF_correction,'value',0);
                    end
                elseif strcmpi(tline(1:index(1)-1),'Extinction_coefficient') == 1
                    set(handles.edit_extinc, 'string', tline(index(1)+1:end));
                end
            end
            fclose(fid);
            %         case 'mat' % .mat file format
            %             load(path_file_n);
            %             try
            %                 set(handles.edit_nch, 'string', num2str(nch));
            %             end
            %             try
            %                 set(handles.edit_fs, 'string', num2str(fs));
            %             end
            %             try
            %                 set(handles.edit_distance, 'string', num2str(distance));
            %             end
            %             try
            %                 set(handles.edit_wavelength, 'string', num2str(wavelength));
            %             end
            %             try
            %                 set(handles.edit_DPF, 'string',  num2str(DPF));
            %             end
            %             try
            %                 switch DPF_correction
            %                     case 'yes'
            %                         set(handles.checkbox_DPF_correction, 'value',1);
            %                     case 'no'
            %                         set(handles.checkbox_DPF_correction, 'value',0);
            %                 end
            %             end
            %             try
            %                 set(handles.edit_extinc, 'string', num2str(ext_coef));
            %             end
    end
end


function edit_extinc_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_extinc_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


