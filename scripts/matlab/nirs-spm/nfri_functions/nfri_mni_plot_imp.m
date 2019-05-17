function nfri_mni_plot(CM, DataFile, SimpleFlag)

% NFRI_MNI_PLOT - Cortical activation viewer.
%
% USAGE.
%   nfri_mni_plot;
%   nfri_mni_plot(CM);
%   nfri_mni_plot(CM, DataFile);
%   nfri_mni_plot(CM, DataFile, 'Simple');
%
% DESCRIPTION.
%   This function will load co-ordinates of points stored in *.xls file.
%   You can plot circle of activations around this points; Radius will
%   be composite SD. It will find all points from xall, yall, zall
%   around these points. It will plot them and color them according to
%   Activation values (Va) which are stored in 5th column in DataFile.
%
% INPUTS.
%   CM is the Color Scale pattern. Default is 'jet'.
%   DataFile is the Excel file which stores x, y, z, SD, Va.

% -------------------
% Datafile definition
% -------------------

if nargin <= 1,
  [FileNamOrigin, PathNamOrigin] = ...
    uigetfile('*.xls', 'Select the xls file');

  if(PathNamOrigin == 0),
    error('File selection is cancelled.');
  else
    DataFile = fullfile(PathNamOrigin, FileNamOrigin);
  end
end

% -------------------
% ColorMap definition
% -------------------

if nargin == 0,
  CM = 'jet';
end


% -----------------------------
% Open the template head figure
% -----------------------------

FigFile = [fileparts(mfilename('fullpath')) filesep 'fig/nfri_mni_plot/Brain.fig'];
% MP = open('Brain.fig'); hold on;
MP = open(FigFile); hold on;

Handles = struct('NaN', [], 'X', []); % Object handles ... this variable will be used as "set(MP, 'UserData', handles);"


% ----------------------
% How to make 'Brain.fig'
% ----------------------

%     load Brain;
%     MakePlot3D([xall', yall', zall'], 7, 'bone', 'FlipCM');
%     set(gcf, ...
%         'Name', 'MNI plot', ...
%         'NumberTitle', 'off', ...
%         'Color', [0.2 0.2 0.2]);
%     axis equal;
%     saveas(gcf, 'Brain.fig', 'fig');
set(MP, 'Name', 'NFRI MNI plot');


% ------------------
% Load cortical data
% ------------------

MatFile = [fileparts(mfilename('fullpath')) filesep 'mat/nfri_mni_plot/Brain.mat'];
load (MatFile);


% ------------
% -- Define constant value for coloring
% -- I guess I should calculate appropriate value for visualize
% --  dynamically by using Brain coordinate.
% -----

% MarkerSize
MS = 7;

% BrainCentroid = mean([xall' yall' zall'], 1);
% BrainColorMap = flipud(bone(69));

% We load excel data
LoadedData= xlsread(DataFile);

% We'll load values which determin colorrange Va
% We always store these values in excel file
% min value is stored in last row
% max value is stored as last but one

Ans = questdlg('Apply automatic scale? (Ignore the max and min value in the sheet?)', 'Automatic scale?');
DummyScale = 100;

if isequal(Ans, 'Yes')
  MaxVa = round(max(LoadedData(1:end-2,5)) * DummyScale);
  MinVa = round(min(LoadedData(1:end-2,5)) * DummyScale);

  %   prompt={'Enter the max value:','Enter the min value:'};
  %   name='Input for max and min values';
  %   numlines=1;
  %   defaultanswer={'10', '-10'};
  %
  %   answer=inputdlg(prompt,name,numlines,defaultanswer);
  %   MaxVa = str2num(answer{1});
  %   MaxVa = round(MaxVa * DummyScale);
  %   MinVa = str2num(answer{2});
  %   MinVa = round(MinVa * DummyScale);

elseif isequal(Ans, 'No')
  MaxVa = round(LoadedData(end - 1, 5)*DummyScale);
  MinVa = round(LoadedData(end, 5)*DummyScale);

  ColorAns = questdlg('Do you plot outlier points(less than minimum value, more than maximum value) with white and black color?', 'Color choice');
  if isequal(ColorAns, 'Yes')
    WBFlag = 1;
    %   elseif isequal(ColorAns, 'No')
    %     WBFlag = 0;
    %   else
    %     return;
  end

else
  return;
end



% We'll cut last 2 rows
LL = size(LoadedData, 1);
Data = LoadedData(1:(LL - 2), :);

% We define Va - Activation (color intensity value for every point)
Va=Data(:, 5)';

% We'll make length of range bigger
Va = round(Va * 100);

% We generate equally incremented vector CMR from min to max
CMR = MinVa : MaxVa;
% CMR=round(min(Va):max(Va));
% if CMR(1) > 100
%     CMR(1:end) = CMR - 100;
% end

% We get length of the vector
Cnp=length(CMR);


Str = ['Cmp = flipud(' CM '(Cnp));'];
eval(Str);

% Cmp = flipud(jet(Cnp));
% Cmp = flipud(hot(Cnp)); % Define colormap of circles
% This can by jet, hot, cool, autumn, colorcube,
% copper, pink, spring, prism, summer, winter

% Here we wanna find all xall, yall, zall points round our points
% Pcol1 is (x) value of points, Pcol2 is (y) and Pcol2 s (z) value
Pcol1=Data(:,1);
Pcol2=Data(:,2);
Pcol3=Data(:,3);

% Here we start script. Script will run m times, depends how many points
% we have in Data.xls file stored in rows. First it will find all xall,
% yall, zall points round points co-ordinates. We'll get cube. Than
% we'll find all lengths from our point to each point inside cube. Than
% we'll cut all lengths longer than Radius. Now we have circle. We can plot
% them.

Radius = Data(:, 4)'; % load Composit SD from xls file
Diameter = Radius * 2;

% 2008/12/22
NanI = isnan(Va);
if sum(NanI) > 0
  NanFlag = 1;
  Va(NanI) = MaxVa; % Temp value
end
if exist('WBFlag', 'var')
  BlackPointsI = Va > MaxVa;
  WhitePointsI = Va < MinVa;
  Va(BlackPointsI | WhitePointsI) = MaxVa; % Temp value
end
if ~exist('WBFlag', 'var')
  LessThanMinI = Va <= MinVa;
  Va(LessThanMinI) = MinVa;
  MoreThanMaxI = Va >= MaxVa;
  Va(MoreThanMaxI) = MaxVa;
end
CenterColors = Cmp((MaxVa - Va) + 1, :);
if exist('WBFlag', 'var')
  CenterColors(BlackPointsI, :) = repmat([0 0 0], sum(BlackPointsI), 1);
  CenterColors(WhitePointsI, :) = repmat([1 1 1], sum(WhitePointsI), 1);
end
if exist('NanFlag', 'var')
  CenterColors(NanI, :) = repmat([0.5 0.5 0.5], sum(NanI), 1);
end
ChangeFlag = true(length(Pcol1));


DrawMain(Data, Pcol1, Pcol2, Pcol3, Radius, Diameter, xall, yall, zall, Va, MaxVa, MinVa, CenterColors, nargin);

% for m = 1:length(Pcol1)
%   I=(((Pcol1(m)+Diameter(m))>=xall & ...
%     xall >=(Pcol1(m)-Diameter(m))) & ...
%     ((Pcol2(m)+Diameter(m))>=yall & ...
%     yall >=(Pcol2(m)-Diameter(m))) & ...
%     ((Pcol3(m)+Diameter(m))>=zall & ...
%     zall >=(Pcol3(m)-Diameter(m))));
%   XX=xall(I);
%   YY=yall(I);
%   ZZ=zall(I);
% 
%   R=[];
%   for n = 1:length(XX)
%     R(n) = [sqrt((Data((m),1)-XX(n))^2 + ...
%       (Data((m),2)-YY(n))^2 + ...
%       (Data((m),3)-ZZ(n))^2)];
%   end
% 
%   Rcut = R <= Radius(m);
%   Rcircle=R(Rcut);
% 
%   Pxall=XX(Rcut);
%   Pyall=YY(Rcut);
%   Pzall=ZZ(Rcut);
% 
% 
%   % -------------------------
%   % -- Plot color sampling --
%   % -------------------------
% 
%   % Rcut = (R >= Radius(m) -1 & R <= Radius(m) + 1);
%   Samples = [XX(Rcut)' YY(Rcut)' ZZ(Rcut)'];
% 
%   if isempty(Samples)
%     % printf('%s\n', [int2str(m) ': empty.']);
%     fprintf('%d: empty.\n', m);
%     continue;
%   end
% 
%   D = round(DistBtw(BrainCentroid, Samples)) - 48;
%   % EdgeColor = mean(BrainColorMap(D, :));
% 
%   % Clearing
%   % clear I  R  XX  YY  ZZ  n  Rcut Rcircle
% 
%   % We'll plot circle of points round our points colored according color
%   % intensity value (Va) - Activation; colorrange has range MinVa to MaxVa
%   % - check last 2 row in excel file
% 
%   hold on;
% 
%   % -------------------------------------
%   % Labeling for each estimated positions
%   % -------------------------------------
% 
%   TextXYZ = [Pcol1(m), Pcol2(m), Pcol3(m)] + ...
%     [Pcol1(m), Pcol2(m), Pcol3(m)] / ...
%     norm([Pcol1(m), Pcol2(m), Pcol3(m)]) * 10;
%   TH = text(TextXYZ(1), TextXYZ(2), TextXYZ(3), int2str(m), ...
%     'Color', 'black', ...
%     'FontName', 'FixedWidth', ...
%     'Visible', 'Off', ...
%     'BackgroundColor', [.7 .9 .7]);
% 
%   % 2008/12/22(Mon) I commented out this routine for Dan san's request.
%   %{
%   if isnan(Va(m))
%     NanXYZ = [Pcol1(m), Pcol2(m), Pcol3(m)] + ...
%       [Pcol1(m), Pcol2(m), Pcol3(m)] / ...
%       norm([Pcol1(m), Pcol2(m), Pcol3(m)]) * 5;
%     % plot3(Pcol1(1), Pcol2(m), Pcol3(m), 'x', 'MarkerSize', MS, 'LineWidth', 10, 'Color', [1 0 0]);
% 
%     Handles.NaN = [Handles.NaN, plot3(Pxall, Pyall, Pzall, 'w.', 'MarkerSize', MS)];
%     % set(MP, 'UserData', .xhandles = 1;
%     Handles.X = [Handles.X, plot3(NanXYZ(1), NanXYZ(2), NanXYZ(3), 'x', 'MarkerSize', MS*2, 'LineWidth', 3, 'Color', [1 0 0])];
%     continue;
%   end
%   
%   % If Va > max value of our colorrange MaxVa - cirle will be black
%   if ((Va(m) > MaxVa) && exist('WBFlag', 'var'))
%     plot3(Pxall, Pyall, Pzall, 'k.', 'MarkerSize', MS); % hb ... handle of black points
%   end
% 
%   % If Va < min value of our colorrange MinVa - cirle will be white
%   if ((Va(m) < MinVa) && exist('WBFlag', 'var'))
%     plot3(Pxall, Pyall, Pzall, 'w.', 'MarkerSize', MS); % hw ... handle of white points
%   end
%   
%   % True color plot for outlier not black and white
%   if ~exist('WBFlag', 'var')
%     if Va(m) <= MinVa
%       Va(m) = MinVa;
%     end
%     if Va(m) >= MaxVa
%       Va(m) = MaxVa;
%     end
%   end
%   %}
%   
%   % Here we'll plot points according colorrange
%   if (Va(m) <= MaxVa) && (Va(m) >= MinVa)
%     Circle = [Pxall' Pyall' Pzall'];
% 
%     Center = [Pcol1(m) Pcol2(m) Pcol3(m)];
%     CenterColor = CenterColors(m, :);
%     % CenterColor = Cmp((MaxVa-Va(m))+1, :); % 2008/12/21 Commented out.
% 
%     D = round(DistBtw(Center, Circle));
%     MaxD = max(D);
%     MinD = min(D);
% 
%     if nargin < 3 % If 'SimpleFlag' is not active: normal drawing
% 
%       for j = 1 : size(Circle, 1)
%         LocalD = DistBtw(Center, Circle(j, :));
% 
%         D = DistBtw(BrainCentroid, Circle(j, :));
%         SurfaceColor = BrainColorMap(round(D)-48, :);
% 
%         SurfaceColorRatio = LocalD / Radius(m);
%         PatchColorRatio = 1 - SurfaceColorRatio;
% 
%         KeepRatio = 0.7;
% 
%         KeepRatio = 1 - KeepRatio;
%         if PatchColorRatio > KeepRatio;
%           PlotColor = CenterColor;
%         else
%           PatchColorRatio = PatchColorRatio .* (1 / KeepRatio);
%           SurfaceColorRatio = 1 - PatchColorRatio;
%           PlotColor = SurfaceColor .* SurfaceColorRatio + ...
%             CenterColor .* PatchColorRatio;
%         end
% 
%         % ---------------------------------------------
%         % Correction of color value.
%         % More than 1 will be 1, less than 0 will be 0.
%         % ---------------------------------------------
% 
%         Sel = PlotColor(:, :) > 1;
%         PlotColor(Sel) = 1;
% 
%         Sel = PlotColor(:, :) < 0;
%         PlotColor(Sel) = 0;
% 
% 
%         % --------
%         % Plotting
%         % --------
% 
%         plot3(Circle(j,1), Circle(j,2), Circle(j,3), ...
%           '.', 'Color', PlotColor, 'MarkerSize', MS);
%       end
%       clear j;
% 
%     else % If 'SimpleFlag' is active: simple drawing
% 
%       PlotColor = CenterColor;
%       plot3(Circle(:,1), Circle(:,2), Circle(:,3), ...
%         '.', 'Color', PlotColor, 'MarkerSize', MS);
% 
%     end
%   end
% end

set(Handles.NaN, 'Visible', 'Off');
set(MP, 'UserData', Handles);

% Set plot properties
view(270,0)

colormap(CM);
colorbarh = colorbar;
newtick = linspace(MinVa, MaxVa, length(get(colorbarh, 'YTick'))+2);
newtick = newtick(2:end-1) / DummyScale;
newtickc = num2str(newtick');
set(colorbarh, 'YTickLabel', newtickc);

FigPos = get(gcf, 'Position');

uicontrol('Style', 'PushButton', ...
  'String', 'Toggle Label', ...
  'Unit', 'Pixel', ...
  'FontName', 'FixedWidth', ...
  'CallBack', @ToggleLabel, ...
  'Position', [ FigPos(3) - 100 - 10, 10, 100, 20]);

%{
uicontrol('Style', 'PushButton', ...
  'String', 'Change NaN symbol', ...
  'Unit', 'Pixel', ...
  'FontName', 'FixedWidth', ...
  'CallBack', @ChangeColor, ...
  'Position', [ FigPos(3) - 250 - 20, 10, 150, 20]);
%}

%%

function DrawMain(Data, Pcol1, Pcol2, Pcol3, Radius, Diameter, xall, yall, zall, Va, MaxVa, MinVa, CenterColors, nargin_origin)
for m = 1:length(Pcol1)
  I=(((Pcol1(m)+Diameter(m))>=xall & ...
    xall >=(Pcol1(m)-Diameter(m))) & ...
    ((Pcol2(m)+Diameter(m))>=yall & ...
    yall >=(Pcol2(m)-Diameter(m))) & ...
    ((Pcol3(m)+Diameter(m))>=zall & ...
    zall >=(Pcol3(m)-Diameter(m))));
  XX=xall(I);
  YY=yall(I);
  ZZ=zall(I);

  R=[];
  for n = 1:length(XX)
    R(n) = [sqrt((Data((m),1)-XX(n))^2 + ...
      (Data((m),2)-YY(n))^2 + ...
      (Data((m),3)-ZZ(n))^2)];
  end

  Rcut = R <= Radius(m);
  Rcircle=R(Rcut);

  Pxall=XX(Rcut);
  Pyall=YY(Rcut);
  Pzall=ZZ(Rcut);


  % -------------------------
  % -- Plot color sampling --
  % -------------------------

  % Rcut = (R >= Radius(m) -1 & R <= Radius(m) + 1);
  Samples = [XX(Rcut)' YY(Rcut)' ZZ(Rcut)'];

  if isempty(Samples)
    % printf('%s\n', [int2str(m) ': empty.']);
    fprintf('%d: empty.\n', m);
    continue;
  end
  
  BrainColorMap = flipud(bone(69));
  BrainCentroid = mean([xall' yall' zall'], 1);
  D = round(DistBtw(BrainCentroid, Samples)) - 48;
  % EdgeColor = mean(BrainColorMap(D, :));

  % Clearing
  % clear I  R  XX  YY  ZZ  n  Rcut Rcircle

  % We'll plot circle of points round our points colored according color
  % intensity value (Va) - Activation; colorrange has range MinVa to MaxVa
  % - check last 2 row in excel file

  hold on;

  % -------------------------------------
  % Labeling for each estimated positions
  % -------------------------------------

  TextXYZ = [Pcol1(m), Pcol2(m), Pcol3(m)] + ...
    [Pcol1(m), Pcol2(m), Pcol3(m)] / ...
    norm([Pcol1(m), Pcol2(m), Pcol3(m)]) * 10;
  TH = text(TextXYZ(1), TextXYZ(2), TextXYZ(3), int2str(m), ...
    'Color', 'black', ...
    'FontName', 'FixedWidth', ...
    'Visible', 'Off', ...
    'BackgroundColor', [.7 .9 .7]);

  % 2008/12/22(Mon) I commented out this routine for Dan san's request.
  %{
  if isnan(Va(m))
    NanXYZ = [Pcol1(m), Pcol2(m), Pcol3(m)] + ...
      [Pcol1(m), Pcol2(m), Pcol3(m)] / ...
      norm([Pcol1(m), Pcol2(m), Pcol3(m)]) * 5;
    % plot3(Pcol1(1), Pcol2(m), Pcol3(m), 'x', 'MarkerSize', MS, 'LineWidth', 10, 'Color', [1 0 0]);

    Handles.NaN = [Handles.NaN, plot3(Pxall, Pyall, Pzall, 'w.', 'MarkerSize', MS)];
    % set(MP, 'UserData', .xhandles = 1;
    Handles.X = [Handles.X, plot3(NanXYZ(1), NanXYZ(2), NanXYZ(3), 'x', 'MarkerSize', MS*2, 'LineWidth', 3, 'Color', [1 0 0])];
    continue;
  end
  
  % If Va > max value of our colorrange MaxVa - cirle will be black
  if ((Va(m) > MaxVa) && exist('WBFlag', 'var'))
    plot3(Pxall, Pyall, Pzall, 'k.', 'MarkerSize', MS); % hb ... handle of black points
  end

  % If Va < min value of our colorrange MinVa - cirle will be white
  if ((Va(m) < MinVa) && exist('WBFlag', 'var'))
    plot3(Pxall, Pyall, Pzall, 'w.', 'MarkerSize', MS); % hw ... handle of white points
  end
  
  % True color plot for outlier not black and white
  if ~exist('WBFlag', 'var')
    if Va(m) <= MinVa
      Va(m) = MinVa;
    end
    if Va(m) >= MaxVa
      Va(m) = MaxVa;
    end
  end
  %}
  
  % Here we'll plot points according colorrange
  if (Va(m) <= MaxVa) && (Va(m) >= MinVa)
    Circle = [Pxall' Pyall' Pzall'];

    Center = [Pcol1(m) Pcol2(m) Pcol3(m)];
    CenterColor = CenterColors(m, :);
    % CenterColor = Cmp((MaxVa-Va(m))+1, :); % 2008/12/21 Commented out.

    D = round(DistBtw(Center, Circle));
    MaxD = max(D);
    MinD = min(D);

    MS = 7;
  
    if nargin_origin < 3 % If 'SimpleFlag' is not active: normal drawing

      for j = 1 : size(Circle, 1)
        LocalD = DistBtw(Center, Circle(j, :));

        D = DistBtw(BrainCentroid, Circle(j, :));
        SurfaceColor = BrainColorMap(round(D)-48, :);

        SurfaceColorRatio = LocalD / Radius(m);
        PatchColorRatio = 1 - SurfaceColorRatio;

        KeepRatio = 0.7;

        KeepRatio = 1 - KeepRatio;
        if PatchColorRatio > KeepRatio;
          PlotColor = CenterColor;
        else
          PatchColorRatio = PatchColorRatio .* (1 / KeepRatio);
          SurfaceColorRatio = 1 - PatchColorRatio;
          PlotColor = SurfaceColor .* SurfaceColorRatio + ...
            CenterColor .* PatchColorRatio;
        end

        % ---------------------------------------------
        % Correction of color value.
        % More than 1 will be 1, less than 0 will be 0.
        % ---------------------------------------------

        Sel = PlotColor(:, :) > 1;
        PlotColor(Sel) = 1;

        Sel = PlotColor(:, :) < 0;
        PlotColor(Sel) = 0;


        % --------
        % Plotting
        % --------

        plot3(Circle(j,1), Circle(j,2), Circle(j,3), ...
          '.', 'Color', PlotColor, 'MarkerSize', MS);
      end
      clear j;

    else % If 'SimpleFlag' is active: simple drawing

      PlotColor = CenterColor;
      plot3(Circle(:,1), Circle(:,2), Circle(:,3), ...
        '.', 'Color', PlotColor, 'MarkerSize', MS);

    end
  end
end

%%

function ToggleLabel(h, eventdata)
TXTH = findobj(gcf, 'Type', 'text');
if strcmp(get(TXTH(1), 'Visible'), 'on') == 1
  VisibleFlag = 'off';
else
  VisibleFlag = 'on';
end

% for i = 1:size(TXTH)
%   set(TXTH(i), 'Visible', VisibleFlag);
% end

set(TXTH, repmat({'visible'}, 1, length(TXTH)), repmat({VisibleFlag}, 1, length(TXTH)));

%%

function ChangeColor(h, eventdata)
Handles = get(gcf, 'UserData');
C = uisetcolor('Set color for NaN plot');
set(Handles.NaN, 'Color', C, 'Visible', 'on');
set(Handles.X, 'Visible', 'off'); % Delete 'X' symbol

%%

function [D_List, Fruit] = DistBtw ( MA, MB, Num )

% DISTBTW - Computes distance between vectors.
%
% USAGE.
%   [D_List, Fruit] = DistBtw ( MA, MB, Num )
%
% DESCRIPTION.
%   This function computes distances between two vectors or vector and
%   matrix or two matrix with the same size. If the input is only single
%   matrix (e.g. sorted apical points creating arch ) it computes
%   distances & cumulative distances between matrix points.
%   If there are three inputs (first input is vector, second one is matrix,
%   third one is number) it selects number of matrix points
%   closest to the vector.
%
% INPUTS.
%   MA is vector or matrix [n,3], e.g. MA = [ x, y, z ];
%   MB is vector or matrix [n,3]
%   Num is number determining the number for the selection of the closest
%   points from matrix MB to vector MA
%
% OUTPUTS.
%   D_List is [n,1] list of distances or coordinates of the closest points.
%   Fruit is [n,1] list of distances or cumulative distances.

%   Brain Research Group
%   Food Physics Laboratory, Division of Food Function,
%   National Food Research Institute,  Tsukuba,  Japan
%   WEB: http://brain.job.affrc.go.jp,  EMAIL: dan@nfri.affrc.go.jp
%   AUTHOR: Valer Jurcak,   DATE: 12.jan.2006,    VERSION: 1.3
%-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-


% CHECK INPUTS & OUTPUTS
% ..............................................................  ^..^ )~
% Check size
if nargin == 1 & size(MA,1) < 3
  error('For one input minimum size of matrix is [3,3].')
end

if nargin == 2
  sA = size(MA,1); sB = size(MB,1);
  if sA < 2 | sB < 2 | sA==sB
    % OK
  else
    error('Can compute distance between "vector & vector", "vector & matrix" or "matrix & matrix" with the same size.')
  end
end

if nargout > 2
  error('Too many output arguments.');
end


% Keep [n,3] size
if nargin == 1
  MA = MA(:,1:3);
end

% Keep [n,3] size
if nargin == 2 | nargin == 3
  MA = MA(:,1:3);
  MB = MB(:,1:3);
end


% MAKE OUTPUT
% ..............................................................  ^..^ )~
Fruit = [];



% ONE INPUT -> COMPUTES DISTANCES AND CUMULATIVE DISTANCES BETWEEN
% SORTED POINTS e.g. apical points which create arch
% ..............................................................  ^..^ )~
if nargin == 1

  % Distances between points in MA matrix
  for n = 1 : size(MA, 1) - 1
    PreD = MA(n,:) - MA(n+1,:);
    PreD2 = PreD.^2;
    PreD3 = sum(PreD2, 2);
    PreD4(n,1) = PreD3 .^ 0.5;
  end
  D_List = PreD4;
  clear n

  % Cumulative distances
  Fruit = zeros(size(MA,1),1);
  Fruit(2:end) = PreD4;
  Fruit = cumsum(Fruit); % cumulative distances
end



% TWO INPUTS -> COMPUTE DISTANCES BETWEEN (VECTOR or MATRIX) & MATRIX
% ..............................................................  ^..^ )~
if nargin == 2

  % Check size
  if size(MA,1) > size(MB,1);
    DxM = MB;  DxN = MA;
  else
    DxM = MA;  DxN = MB;
  end

  % Distances between matrix & matrix
  if size(DxM,1) == size(DxN,1)
    PreD = DxN - DxM;
    PreD2 = PreD.^2;
    PreD3 = sum(PreD2, 2);
    D_List = PreD3 .^ 0.5; % Now distance is found
  else
    % Distances between vector & matrix
    Echo = repmat(DxM, size(DxN,1), 1);
    PreD  = DxN - Echo;
    PreD2 = PreD.^2;
    PreD3 = sum(PreD2, 2);
    D_List = PreD3 .^ 0.5; % Now distance is found
  end
end



% THREE INPUTS -> SELECT Num OF CLOSEST VECTORS TO MA
% ..............................................................  ^..^ )~
if nargin == 3

  % Distances between vector & matrix
  Echo = repmat( MA, size(MB, 1), 1 );
  PreD  = MB - Echo;
  PreD2 = PreD.^2;
  PreD3 = sum(PreD2, 2);

  PreD4 = PreD3 .^ 0.5; % Now distance is found

  % Sort matrix according to distances
  MatRiX = [MB, PreD4]; % add forth column of distances
  MatRiX = sortrows(MatRiX, 4); % sorting
  D_List = MatRiX(1:Num, 1:3); % number of closest points
  Fruit = MatRiX(1:Num, 4); % actual distances
end
