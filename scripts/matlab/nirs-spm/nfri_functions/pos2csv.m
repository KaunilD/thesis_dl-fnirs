function pos2csv

[filename, pathname, filterindex] = uigetfile('*.pos', 'Pick an POS-file');
[l, XYZ, XYZ_AER_NXYZ] = ReadPos([pathname filename]);

[PATH, NAME, EXT] = fileparts(filename);
originfile = [pathname, NAME, '_origin.csv'];
othersfile = [pathname, NAME, '_others.csv'];

fid = fopen(originfile, 'w');
fprintf(fid, ',,,,\n'); % Header
fprintf(fid, '"Nz",%f,%f,%f\n', XYZ(3, :)); % Nz
fprintf(fid, '"Iz",%f,%f,%f\n', XYZ(4, :)); % Iz
fprintf(fid, '"AR",%f,%f,%f\n', XYZ(2, :)); % AR
fprintf(fid, '"AL",%f,%f,%f\n', XYZ(1, :)); % AL
for k = 1:7
  fprintf(fid, ',,,,\n'); % Fp1, Fp2, Fz, F3, F4, F7, F8
end
fprintf(fid, '"Cz",%f,%f,%f\n', XYZ(5, :)); % Cz
for k = 1:11
  fprintf(fid, ',,,,\n'); % C3, C4, T3, T4, Pz, P3, P4, T5, T6, O1, O2
end
fclose(fid);

fid = fopen(othersfile, 'w');
for k = 6:size(XYZ, 1)-1 % Last one line is transmitter itself? So I remove it.
  fprintf(fid, ',,%f,%f,%f\n', XYZ(k, :));
end
fclose(fid);