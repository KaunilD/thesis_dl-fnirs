function [argout] = data_conversion(path_file_n)
    disp(['Reading: ', path_file_n])
    
    fid = fopen(path_file_n);
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
            disp('steps 1 of 3 completed')
    %         waitbar(1/3, h_wait, 'Data reading (1/3) has been completed.');
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
            disp('steps 2 of 3 completed')

    %         waitbar(2/3, h_wait, 'Data reading (2/3) has been completed.');
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
    %         waitbar(3/3, h_wait,'Data reading (3/3) has been completed.');
            disp('Reading Completed.');
            break,
        end
    end
    fclose(fid);

    index_base = find(baseline == 1);
    disp('Converting data...');
    nch = ndata./2;
    for kk = 1:nch
        [hb_tmp, hbo_tmp, hbt_tmp] = mes2hb(mes(:,2*kk-1:2*kk), [wav_mat(1,2*kk-1) wav_mat(1,2*kk)], [index_base(1) index_base(end)]);
        nirs_data.oxyData(:,kk) = hbo_tmp;
        nirs_data.dxyData(:,kk) = hb_tmp;
    end
    try
        vector_onset(index_base(1):index_base(end)) = [];
        nirs_data.vector_onset = vector_onset(:);
    end
    nirs_data.fs = fs;
    nirs_data.nch = nch;
    disp('Conversion complete!');
    argout = nirs_data;      
end