warning off all;

DefaultDir=cd;

RepFolderPath=uigetdir;
if(RepFolderPath==0)
    error('Selecting a folder is cancelled');
    return;
end
fullfile(RepFolderPath,'*.xls');

LstFile=dir(fullfile(RepFolderPath,'*.xls')); %Structure is made
SbjNbr=size(LstFile,1); %The number of subjects
BSdf=SbjNbr-1; %(n-1)

cd(RepFolderPath);
RefBNbr=xlsread(LstFile(1).name,'Info','A2'); % The number of reference brain.
cd(DefaultDir);

BRefBdf=RefBNbr-1; %m-1
WSdf=SbjNbr*BRefBdf; %n(m-1)
TTLdf=SbjNbr*RefBNbr-1; %nm-1

ISSwsCs=cell(1,SbjNbr); %SSws for INDIVIDUAL subject brain
%MSwsCs=cell(1,SbjNbr)
WShutCs=cell(1,SbjNbr);

cd(RepFolderPath);

% DataL=58; % tsuzuki commented this out. This value shouldn't be fixed.
% tsuzuki add 2 lines below.
TmpRows=xlsread(LstFile(1).name,'SSwsC');
DataL = size(TmpRows, 1);

%Set the size for TSSwsCs
% AA=xlsread(LstFile(1).name,'SSwsC')
% PSSwsCs=zeros(size(AA))%Pooled SSws for Group subjects' brains
PSSwsCs=zeros(DataL,4);

%%%Extra setting

for CC=1:SbjNbr
    % A=LstFile(CC).name
    % B=xlsread(A,'SSwsC')
    ISSwsCs{CC}=xlsread(LstFile(CC).name,'SSwsC');
    ISSwsCs{CC}=ISSwsCs{CC}(1:DataL,:);
    %MSwsCs{CC}=SSwsCs{CC}./WSdf
    WShatCs{CC}=xlsread(LstFile(CC).name,'WShatC');
    WShatCs{CC}=WShatCs{CC}(1:DataL,:);
    PSSwsCs=PSSwsCs+ISSwsCs{CC};
end

PMSwsCs=PSSwsCs./WSdf;

cd(DefaultDir);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These data will be transferred
SbjN=SbjNbr;
HorC=0;

DataCell=WShatCs;

%%%SurfaceCenterFind3%%%
%Last-modified by Dan,041104
%Dependent on BrainSurfEdgeMNI.mat, HeadSurfEdgeMNI.mat, BackProjectionf.m,
%VarCalcf.m
%This program searches the centroid of the input matrix on the cortical
%surface or head surface by back projection
%First, find the average. 
%Then, project it onto the averaged head or cortical surfaces.
%Data of the Nth subject should be stored in a book with the name, 00N

%HorC=input('Please input which to work on, 0.Brain, or 1.Head > ');
%FileName=input('Please input the Excel data file name as ''filename''  > ');%Read Excel file
%SbjN=input('Please input the number of subjects > ');%The number of subjects is mannualy input


% DataCell=cell(1,SbjN);
% 
% %Reading excel files and sheet
% for cc=1:SbjN
%     SheetName=sprintf('%03d',cc);
%     A=['B=xlsread(''' [FileName] ''',''' [SheetName] ''');'];
%     eval(A);
%     DataCell{cc}=B;
% end
% %Read data are stored in a cell structure, DataCell

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

%BrainVarList is MSbs
%SSbs is MSbs*(n-1)

PMSwsCs
PSSwsCs

MSbsC=RefBNbr*BrainVarList%multiplied by m
SSbsC=MSbsC.*BSdf%MS*df

SStC=SSbsC+PSSwsCs
MStC=SStC./TTLdf

HATtC=BrainSurfaceCenterList
SDtC=MStC(:,4).^0.5


% fprintf('Type DataCell{N} to get the original input data of Nth input \n');
% fprintf('Specify the range of input as DataCell{N:L} to get the original input data of Nth to Lth inputs \n');
% fprintf('Type SbjWiseCell{M} to get the sbuject-wise arranged data for Mth subject. \n');
% fprintf('Specify the range of subjects as SbjWiseCell{M:K} to get the the sbuject-wise arranged data for Mth to Kth subjects \n');
% fprintf('Type HeadSurfaceCenterList to get the calculated results for head surface. \n');
% fprintf('Type BrainSurfaceCenterList to get the calculated results for brain surface. \n');
% fprintf('Type HeadMeanList to get the mean before surface transformation for head surface. \n');
% fprintf('Type BrainMeanList to get the mean before surface transformation for brain surface. \n');
% fprintf('Type HeadVarList to get the variance from the surface-transformed point for Head surface. \n');
% fprintf('Type BrainVarList to get the variance from the surface-transformed point for Brain surface. \n');
% fprintf('In either variance, the first three values indicate x, y, z variances and 4th value, composite variance,r^2. \n');
% 
% toc

%%%%%%%%%%%%%%%%%%%%%%%%%%Above from Surface Center Find3

% 
% %%%%%Origin files are read from GUI%%%%%%
% [FileNamOrigin, PathNamOrigin]=uigetfile('*.xls','Locate the origin xls file')%Name & path of the origin excel file are selected
% %%%Cancel message%%%
% if(PathNamOrigin == 0)
%     error('File selection is cancelled')
%     return;
% end
% %%%%%%%%%%%%%%%%%%%%
% cd(PathNamOrigin)
% [D T]=xlsread(FileNamOrigin)
% T=T(SelIndx)%Reference points are selected
% D=D(SelIndx,:)%Reference points are selected
% cd(DefaultDir)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %%%%Averaged reference points on MNI Head are read%%%
% load DMNIH
% T1=DMNIHLvl(SelIndx)%Reference points are selected
% DD=DMNIHAve(SelIndx,:)%Reference points are selected
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %%%%%Other points to be examined are read from GUI%%%%%%
% [FileNamOthers, PathNamOthers]=uigetfile('*.xls','Locate the corresponding Others xls file')%Name & path of the MNI excel file are selected
% %%%Cancel message%%%
% if(PathNamOthers == 0)
%     error('File selection is cancelled')
%     return;
% end
% %%%%%%%%%%%%%%%%%%%%
% cd(PathNamOthers)
% [DDD TTT]=xlsread(FileNamOthers)
% cd(DefaultDir)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %%%%%Report file is determined by GUI%%%%%%
% [FileNamRep, PathNamRep]=uigetfile('*.xls','Locate the report xls file')%Name & path of the Report excel file are selected
% %%%Cancel message%%%
% if(PathNamRep == 0)
%     error('File selection is cancelled')
%     return;
% end
% %%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% %AffineEstSurface4
% 
% %[OtherH, OtherC, OtherHSD, OtherCSD, SSwsH, SSwsC, RefN]
% [ROtherH, ROtherC, ROtherHSD, ROtherCSD, RSSwsH, RSSwsC, RRefN]=AffineEstSurface4(D, DD, DDD, T, TTT, SelIndx)
% 
% %%%Excel Report file%%%%
% cd(PathNamRep)
% xlswrite(FileNamRep, ROtherH, 'WShutH')
% xlswrite(FileNamRep, ROtherC, 'WShutC')
% xlswrite(FileNamRep, ROtherHSD, 'WS_SDH')
% xlswrite(FileNamRep, ROtherCSD, 'WS_SDC')
% xlswrite(FileNamRep, RSSwsH, 'SSwsH')
% xlswrite(FileNamRep, RSSwsC, 'SSwsC')
% xlswrite(FileNamRep, TTT, 'OtherPointLabels')
% TextA=cell(1)
% TextA{1}=['Reference brain number']
% xlswrite(FileNamRep, TextA, 'Info','A1')
% xlswrite(FileNamRep, RRefN, 'Info','A2')
% TextB=cell(1)
% TextB{1}=['Refference points used']
% xlswrite(FileNamRep, TextB, 'Info','A3')
% xlswrite(FileNamRep, T, 'Info','A4')
% 
% 
% 
% cd(DefaultDir)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % 
% % ReadFolderPath=uigetdir('', 'Where to save the processed data'); %Result folder name is defined
% % SaveFolderPath=uigetdir('', 'Where to save the processed data'); %Result folder name is defined
% % 
% % cd(strFolderPath)
% % Nindex=1
% % CurFileN=LstFile(Nindex).name
% % [D T]=xlsread(CurFileN)
% % ResFileN=['Res' CurFileN]
% % ExcelOut={T D}
% % 
% % cd (SaveFolderPath)
% % xlswrite(ResFileN,T,'Data', 'A1')
% % xlswrite(ResFileN,D,'Data', 'B1')
% % cd(DefaultDir)
