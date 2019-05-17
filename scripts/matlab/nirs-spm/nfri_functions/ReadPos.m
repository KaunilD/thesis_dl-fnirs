function [Label, XYZ, XYZ_AER_NXYZ] = ReadPos(FileName);

% ReadPos - Read '*.pos' file and extract each XYZ coordinate and label.
%
% Usage
%  [Label, XYZ, XYZ_AER_NXYZ] = ReadPos(FileName);
%
% Input
%  Filename ... *.pos file to be read.
%
% Output
%  Label ... Label of measured positions.
%  XYZ ... Coordinates of measured positions.
%  XYZ_AER_NXYZ ... Coordinates, orientation matrix in Euler angle, and
%                   mysterios XYZ parameters of measured positions.
%
% Example
%  [Label, XYZ, XYZ_AER_NXYZ] = ReadPos('anyfile.pos');
    
    XYZ = [];
    XYZ_AER_NXYZ = [];
    Rows = 1;
    
    fid = fopen(FileName);
    while 1
        tline = fgetl(fid);
        if ~isstr(tline), break, end

        if size(tline, 2) > 2
            flag = tline(1, 1:2);
        end

        if strcmp(flag, 'X=')
            Label{Rows, 1} = otline;
            Rows = Rows + 1;
            
            X = GetValue(tline);
            tline = fgetl(fid); Y = GetValue(tline);
            tline = fgetl(fid); Z = GetValue(tline);
            XYZ = [XYZ; X Y Z];
            
            tline = fgetl(fid);
            if ~isstr(tline), break, end
            
            flag = tline(1, 1:2);
            if strcmp(flag, 'A=')
                A = GetValue(tline);
                tline = fgetl(fid); E = GetValue(tline);
                tline = fgetl(fid); R = GetValue(tline);

                tline = fgetl(fid); NX = GetValue(tline);
                tline = fgetl(fid); NY = GetValue(tline);
                tline = fgetl(fid); NZ = GetValue(tline);
                
                XYZ_AER_NXYZ = [XYZ_AER_NXYZ; X Y Z A E R NX NY NZ];
            end
                
        end
        otline = tline;
    end
    fclose(fid);
    
    % MyPlot(Label, XYZ);
    MyPlot2(Label, XYZ_AER_NXYZ);
    
function Value = GetValue(Str)
    [T, R] = strtok(Str, '=');
    R(findstr(num2str(R), '=')) = '';
    Value = str2num(R);

    
function MyPlot(Label, XYZ);
    Padding = -5;
    figure;
    plot3(XYZ(:,1), XYZ(:,2), XYZ(:,3), 'r.', 'MarkerSize', 10);
    for i = 1 : size(Label, 1)
        text(XYZ(i,1)+Padding, XYZ(i,2)+Padding, XYZ(i,3), Label{i});
    end
    axis equal;

    
function MyPlot2(Label, Mat);
    Padding = -5;
    figure;
    plot3(Mat(:,1), Mat(:,2), Mat(:,3), 'r.', 'MarkerSize', 10); hold on;
    plot3(Mat(:,7), Mat(:,8), Mat(:,9), 'b.', 'MarkerSize', 10);
    for i = 1:size(Mat, 1)
        Line = [Mat(i,1), Mat(i,2), Mat(i,3); Mat(i,7), Mat(i,8), Mat(i,9)];
        plot3(Line(:,1), Line(:,2), Line(:,3), '-');
    end
    axis equal;
    