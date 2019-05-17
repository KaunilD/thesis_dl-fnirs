%%%SurfaceCenterFind3%%%
%Last-modified by Dan,041104
%Dependent on BrainSurfEdgeMNI.mat, HeadSurfEdgeMNI.mat, BackProjectionf.m,
%VarCalcf.m
%This program searches the centroid of the input matrix on the cortical
%surface or head surface by back projection
%First, find the average. 
%Then, project it onto the averaged head or cortical surfaces.
%Data of the Nth subject should be stored in a book with the name, 00N

HorC=input('Please input which to work on, 0.Brain, or 1.Head > ');
FileName=input('Please input the Excel data file name as ''filename''  > ');%Read Excel file
SbjN=input('Please input the number of subjects > ');%The number of subjects is mannualy input

tic

DataCell=cell(1,SbjN);

%Reading excel files and sheet
for cc=1:SbjN
    SheetName=sprintf('%03d',cc);
    A=['B=xlsread(''' [FileName] ''',''' [SheetName] ''');'];
    eval(A);
    DataCell{cc}=B;
end
%Read data are stored in a cell structure, DataCell

%Now, arranging the data to subject-wise manner
EntryN=size(DataCell{1},1);
SbjWiseCell=cell(1,EntryN);
for cc=1:EntryN
    TempData=ones(SbjN,3);
    for cca=1:SbjN
        TempData(cca,:)=DataCell{cca}(cc,:);
    end
    SbjWiseCell{cc}=TempData;
end
%Arranged subject-wise data are stored in a cell structure, SbjWiseCell

%The between-subject surface centroids are calculated on brain or head surface
%But first finding averages 
if HorC==0;%Brain
    BrainMeanList=ones(EntryN,3);
    BrainSurfaceCenterList=ones(EntryN,3);
    BrainVarList=ones(EntryN,4);
    load BrainSurfEdgeMNI;%Brain surface edge data
    for cc=1:EntryN
        cc;
        AA=mean(SbjWiseCell{cc},1);
        BrainMeanList(cc,:)=AA;%Simple mean calculation
        BB=BackProjectionf(xallBEM, yallBEM, zallBEM, AA);%Surface transformation
        BrainSurfaceCenterList(cc,:)=BB;
        %Variance calculation
        VV=VarCalcf(SbjWiseCell{cc},BB);
        BrainVarList(cc,:)=VV;
        clear AA BB VV;
    end
else%Head
    HeadMeanList=ones(EntryN,3);
    HeadSurfaceCenterList=ones(EntryN,3);
    HeadVarList=ones(EntryN,4);
    load HeadSurfEdgeMNI;%Head surface data
    for cc=1:EntryN
        cc;
        AA=mean(SbjWiseCell{cc},1);%Simple mean calculation
        HeadMeanList(cc,:)=AA;
        BB=BackProjectionf(xallHEM, yallHEM, zallHEM, AA);%Surface transformation
        HeadSurfaceCenterList(cc,:)=BB;
        %Variance calculation
        VV=VarCalcf(SbjWiseCell{cc},BB);
        HeadVarList(cc,:)=VV;
        clear AA BB VV;
    end
end

fprintf('Type DataCell{N} to get the original input data of Nth input \n');
fprintf('Specify the range of input as DataCell{N:L} to get the original input data of Nth to Lth inputs \n');
fprintf('Type SbjWiseCell{M} to get the sbuject-wise arranged data for Mth subject. \n');
fprintf('Specify the range of subjects as SbjWiseCell{M:K} to get the the sbuject-wise arranged data for Mth to Kth subjects \n');
fprintf('Type HeadSurfaceCenterList to get the calculated results for head surface. \n');
fprintf('Type BrainSurfaceCenterList to get the calculated results for brain surface. \n');
fprintf('Type HeadMeanList to get the mean before surface transformation for head surface. \n');
fprintf('Type BrainMeanList to get the mean before surface transformation for brain surface. \n');
fprintf('Type HeadVarList to get the variance from the surface-transformed point for Head surface. \n');
fprintf('Type BrainVarList to get the variance from the surface-transformed point for Brain surface. \n');
fprintf('In either variance, the first three values indicate x, y, z variances and 4th value, composite variance,r^2. \n');

toc


