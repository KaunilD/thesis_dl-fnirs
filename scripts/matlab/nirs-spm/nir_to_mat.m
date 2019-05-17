function [converted_data] = nir_to_mat(filen,fs, dist, wavelength, DPF, flag_DPF_correction)

% parameter setup ----------------------------- %
timeLength = 14;            % 14 char
channelNum = 24;
samplingFreq = fs;
lightNum = 2;               % # of light source (856/781)
disp('Parameter setup is completed');

fid = fopen(filen,'r');
disp('Data input is completed');

% File Input --------------------------------- %
% First line
iterLine = 1;
flagOn = -1;
tmpLine = fgetl(fid);
inputLine = str2num(tmpLine(timeLength+1:end));
% rawData = zeros(10000,length(inputLine));
iterData = 1;
iterLine = 2;
rawData = [];
while(1) % if 'start A' and 'end B' events do exist,
    % line input
    tmpLine = fgetl(fid);
    
    % end of file
    if(tmpLine == -1)
        break;
    end;    
    % A start / B end -> will be blank array
    inputLine = str2num(tmpLine(timeLength+1:end));
    if(isempty(inputLine))
        flagOn = 1;
    end;
    
    % recording..
    if(flagOn > 0)
        if(isempty(inputLine))  % start & end line
            seFlag = isspace(tmpLine(end-3));
            if(seFlag)          % end line
                inputLine = str2num(tmpLine(timeLength+1:end-6));
                rawData(iterData,:) = inputLine;
                flagOn = 0;
            else                % start line
                inputLine = str2num(tmpLine(timeLength+1:end-8));
                rawData(iterData,:) = inputLine;
                iterData = iterData + 1;
            end;
        else
            rawData(iterData,:) = inputLine;
            iterData = iterData + 1;
        end;
    elseif(flagOn == 0)
        break;
    end;
    iterLine = iterLine + 1;
end;
fclose(fid);

% if events do not exist,
if isempty(rawData) == 1
    fid = fopen(filen, 'r');
    count = 1;
    while 1
        tline = fgetl(fid);
        if ischar(tline) == 0, break, end,
        rawData(count, :) = str2num(tline(timeLength+1:end));
        count = count + 1;
    end
    fclose(fid);
elseif isempty(rawData) == 0
    rawData = rawData(1:iterData,:);    
end
rawdataLength = size(rawData,1);    
disp('Data input is completed');

% Raw data :
% Rx1 - 16 column, Rx2 - 16, Rx3 - 16, Rx4 - 16
% -> (N by 72)
% 16 column:
% (Tx1_856 / Tx1_781), (Tx2_856 / Tx2_781), ..., (Tx8_856 / Tx8_781)



% Data Alignment -------------------------------------------- %
% Rx1	Tx1	  Rx2	 Tx2   |	Tx3 (1) Rx3 (2)	Tx4 (3)	Rx4
% 			               |    (4)     (5)     (6)     (7)
%                          |
% Tx3	Rx3	  Tx4	 Rx4   |	Tx3 (8)	Rx3 (9)	Tx4 (10)Rx4
%                          |
%                          |
% Rx4	Tx5	  Rx1	 Tx6   |	...(24)
%                          |
%                          |
% Tx7	Rx2	  Tx8	 Rx3   |

% Channel configuration (Convolved Tx#)
% Rx1 : 1 3 4 5 6 8
% Rx2 : 1 2 4 5 7 8
% Rx3 : 1 3 4 5 6 8
% Rx4 : 2 3 4 5 6 7
% Ch N : (Receiver, Output)
% Ch01 : 1,1    % Ch02 : 2,1    % Ch03 : 2,2    % Ch04 : 1,3    
% Ch05 : 3,1    % Ch06 : 2,4    % Ch07 : 4,2    % Ch08 : 3,3    
% Ch09 : 3,4    % Ch10 : 4,4    % Ch11 : 4,3    % Ch12 : 3,5    
% Ch13 : 1,4    % Ch14 : 4,6    % Ch15 : 4,5    % Ch16 : 1,5    
% Ch17 : 1,6    % Ch18 : 4,7    % Ch19 : 2,5    % Ch20 : 1,8    
% Ch21 : 3,6    % Ch22 : 2,7    % Ch23 : 2,8    % Ch24 : 3,8    

channelConfig = [1 2 2 1 3 2 4 3 3 4 4 3 1 4 4 1 1 4 2 1 3 2 2 3;
                 1 1 2 3 1 4 2 3 4 4 3 5 4 6 5 5 6 7 5 8 6 7 8 8];
alignData = zeros(rawdataLength,channelNum*lightNum);               % channel number : 24, light source number : 2

for(iterChannel = 1:channelNum)
    columnNum = 16*(channelConfig(1,iterChannel)-1) + 2*(channelConfig(2,iterChannel)-1) + 1;
    alignData(:,2*(iterChannel-1)+1) = rawData(:,columnNum);
    alignData(:,2*iterChannel) = rawData(:,columnNum+1);
end;
disp('Data alignment is completed');

% Take delta --------------------------------------------- %
% set first-element is zero..
deltaData = zeros(size(alignData));
for(iterCol = 1:channelNum*lightNum)
    deltaData(:,iterCol) = alignData(:,iterCol)-alignData(1,iterCol);
end;

% Transform --------------------------------------------- %
% from 2 light source
% to   oxy / deoxy

if flag_DPF_correction == 1
    load Charite_DPF_correction
    index_DPF1 = find(DPF_correction(:,1) == wavelength(1));
    index_DPF2 = find(DPF_correction(:,1) == wavelength(2));
end
load COPE_e_coef; %% load the extinction coefficient file
index_wav1 = find(e_coef(:,1) == wavelength(1));
index_wav2 = find(e_coef(:,1) == wavelength(2));
wav1_ecoef = e_coef(index_wav1,2:3);
wav2_ecoef = e_coef(index_wav2,2:3);
if flag_DPF_correction == 1
    wav1_ecoef = wav1_ecoef .* DPF_correction(index_DPF1, 2);
    wav2_ecoef = wav2_ecoef .* DPF_correction(index_DPF2, 2);
end
tot_ecoef = [wav1_ecoef; wav2_ecoef];
tot_ecoef = tot_ecoef .* DPF .* dist;
coefMat = pinv(tot_ecoef);

oxyData = zeros(rawdataLength,channelNum);
dxyData = zeros(rawdataLength,channelNum);
for(iterCol = 1:channelNum)
    oxydxy = (coefMat * [deltaData(:,2*(iterCol-1)+1)';deltaData(:,2*iterCol)'])';
    oxyData(:,iterCol) = oxydxy(:,1);
    dxyData(:,iterCol) = oxydxy(:,2);
end


% save the converted data and paramter
converted_data.oxyData = oxyData;
converted_data.dxyData = dxyData;
converted_data.fs = samplingFreq;
% converted_data.nch = channelNum;
disp('Delta & Transform is completed');

