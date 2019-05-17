function anatomlabel_final( ID, saveFileName, rn, col )

% ANATOMLABEL_FINAL - Automated anatomical labeling without SPM functions.
%
% USAGE.
%    anatomlabel_final( ID, saveFileName, rn, col )
%
% DESCRIPTION.
%   Function will perform anatomical labeling of spherical region around
%   projected point([x,y,z] - center of sphere). In the default radius is
%   set to 10mm. Function returns list of anatomical label and percentage
%   of overlap. This function is based on 'nfri_anatomlabel'.
%
% INPUTS.
%   ID is [n,3] matrix of vectors. ID can be mat file or excel file.
%   saveFileName is string, it's your desire name for exported csv file.
%   rn is value which define radius of spherical region around
%   point which should be labeled.
%   col is 4-7 values.
%    4 ... AAL
%    5 ... Brodmann area (Chris rorden' MRIcro)
%    6 ... LPBA40
%    7 ... Brodmann area (Talairach daemon)
%
% OUTPUT.
%   Is named csv file writen to the same folder.
%
% EXAMPLE.
%   >> xyz = randn(10,3) * 50;
%   >> anatomlabel_final(xyz, 'justtest', 10, 4);
%
% REFERENCES.
%   AAL files downloaded from http://www.cyceron.fr/freeware/
%    Anatomical Automatic Labeling (AAL).
%    Author: Tzourio-Mazoyer, N. at.all
%    Title: Automated anatomical labeling of activations in SPM using a
%    macroscopic anatomical parcellation of the MNI MRI 	single-subject brain;
%    Journal: Neuroimage 15 273-289 2002 "http://www.sciencedirect.com/"
%   MRIcro
%   LPBA40
%   Talairach daemon

%   Brain Research Group
%   Food Physics Laboratory, Division of Food Function,
%   National Food Research Institute,  Tsukuba,  Japan
%   WEB: http://brain.job.affrc.go.jp,  EMAIL: dan@affrc.go.jp
%   AUTHOR:  Valer Jurcak,    DATE: 12.jan.2006,    VERSION: 1.1
%   REVISED: TSUZUKI Daisuke, DATE: 13.sep.2006,    VERSION: 1.2
%   REVISED: TSUZUKI Daisuke, DATE: 02.jul.2007,    VERSION: 1.3
%   REVISED: TSUZUKI Daisuke, DATE: 13.nov.2008,    VERSION: 1.4
%-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

% -------------------
% Load reference data
% -------------------

ThisScriptPath = fileparts(mfilename('fullpath'));

MatFilePath = [ThisScriptPath filesep 'mat' filesep 'XYZABLT.mat']; % XYZABLT
% MatFilePath = [ThisScriptPath filesep 'mat' filesep 'nfri_a.mat']; % XYZI
load (MatFilePath); % 'nfri_anatomlabel.mat' is the target.

switch col
    case 4
        MatFilePath = [ThisScriptPath filesep 'mat' filesep 'label_a.mat']; % label_a
        load (MatFilePath);
        Label = label_a;
    case 5
        MatFilePath = [ThisScriptPath filesep 'mat' filesep 'label_b.mat']; % label_b
        load (MatFilePath);
        Label = label_b;
    case 6
        MatFilePath = [ThisScriptPath filesep 'mat' filesep 'label_l.mat']; % label_l
        load (MatFilePath);
        Label = label_l;
    case 7
        MatFilePath = [ThisScriptPath filesep 'mat' filesep 'label_b.mat']; % label_b
        load (MatFilePath);
        Label = label_b;
end

% CHECKING INPUTS
% ..............................................................  ^..^ )~

% IRadius is not defined
if nargin == 2
    rn = 10;
end

% Input ID is already loaded variable
if isnumeric (ID)
    MAT = ID;
end

%  Input ID is excel file
if isstr (ID)
    E = [ ID, sprintf('%s', '.xls')]; % check existance
    if exist( E, 'file')
        MAT = xlsread(ID); % excel file was found
        clear E
        MAT = MAT(1:end, 1:3); % dress for excel
    end
end

% Still don't have MAT file, try to search among variables
if ~exist( 'MAT', 'var')

    E = [ ID '.mat' ]; % are stored in mat file
    if exist( E, 'file') % checks existence of file
        load(ID)
        magic_str = [ 'MAT = ' ID ';']; % was saved as "save ID ID"
        eval(magic_str)
        clear E magic_str
    end
end

% Input was not analyzed -> return function
if ~exist( 'MAT' ,'var')
    error('Can not analyze input (points which should be labeled).');
end


% SET SAVING
% ..............................................................  ^..^ )~
Nemo = [ saveFileName '.csv' ]; % save output as csv file
ResultString = [];


% START BATCH PROCESS
% ..............................................................  ^..^ )~
for count = 1 : size(MAT, 1)
    fprintf('%d/%d\n', count, size(MAT, 1));
    DataRow = MAT(count, :);

    % Get all points within radius rn from given point, defauls is 10mm
    [SP, I] = SphPoints ( DataRow, rn, XYZABLT(:, 1:3));

    data = XYZABLT(I, col); % The 4th column of XYZABLT is the label of AAL.
    udata = unique(data);

    nos = [];
    ROIlist = {};

    for i = 1 : size( udata, 1 )
        c = find( data == udata(i) );
        nos = [ nos size(c, 1) ];
        ROIlist{end+1} = Label{udata(i)};
    end

    ROIlist = ROIlist';
    clear i j

    Perc = nos/sum(nos);
    Perc = Perc';

    % Checking sizes of outputs
    if size(ROIlist, 1) ~= size(Perc, 1)
        error('Failed to getlabels.');
    end

    ResultString = sprintf('%s\n%d,,%f,%f,%f,%f,\n', ...
        ResultString, count, DataRow(1,1), DataRow(1,2), DataRow(1,3), rn);
    
    global nfri_anatomlabel_NIRS_SPM;
    if count == 1 & isempty(nfri_anatomlabel_NIRS_SPM) ~= 1
        clear gloabl nfri_anatomlabel_NIRS_SPM;
        global nfri_anatomlabel_NIRS_SPM;
    end
    nfri_anatomlabel_NIRS_SPM{count,1} = ROIlist;
    nfri_anatomlabel_NIRS_SPM{count,2} = Perc;
    for i = 1 : size(ROIlist, 1)
        ResultString = sprintf('%s,,,,,"%s",%f\n', ResultString, ROIlist{i}, Perc(i));
        % I quote ROIlist{i} by "(double quotation) for escape comma character
        % in CSV format.
    end
end

fprintf('%s', ResultString);


fidSave = fopen( Nemo, 'w' );
fprintf(fidSave, '%s', ResultString);
fclose(fidSave);