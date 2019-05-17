DefaultDir=cd

%%%Initialize SelMat%%%
load SelMat
SelMat(:)=1
save SelMat
clear SelMat

GUIexample

fprintf('Press any key to move on.\n')
pause

load SelMat
SelIndx=find(SelMat==1)

%%%%%Origin files are read from GUI%%%%%%
[FileNamOrigin, PathNamOrigin]=uigetfile('*.xls','Locate the origin xls file')%Name & path of the origin excel file are selected
%%%Cancel message%%%
if(PathNamOrigin == 0)
    error('File selection is cancelled')
    return;
end
%%%%%%%%%%%%%%%%%%%%
cd(PathNamOrigin)
[D T]=xlsread(FileNamOrigin)
T=T(SelIndx)%Reference points are selected
D=D(SelIndx,:)%Reference points are selected
cd(DefaultDir)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%Averaged reference points on MNI Head are read%%%
load DMNIH
T1=DMNIHLvl(SelIndx)%Reference points are selected
DD=DMNIHAve(SelIndx,:)%Reference points are selected
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%Other points to be examined are read from GUI%%%%%%
[FileNamOthers, PathNamOthers]=uigetfile('*.xls','Locate the corresponding Others xls file')%Name & path of the MNI excel file are selected
%%%Cancel message%%%
if(PathNamOthers == 0)
    error('File selection is cancelled')
    return;
end
%%%%%%%%%%%%%%%%%%%%
cd(PathNamOthers)
[DDD TTT]=xlsread(FileNamOthers)
cd(DefaultDir)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%Report file is determined by GUI%%%%%%
[FileNamRep, PathNamRep]=uigetfile('*.xls','Locate the report xls file')%Name & path of the Report excel file are selected
%%%Cancel message%%%
if(PathNamRep == 0)
    error('File selection is cancelled')
    return;
end
%%%%%%%%%%%%%%%%%%%%



%AffineEstSurface4

%[OtherH, OtherC, OtherHSD, OtherCSD, SSwsH, SSwsC, RefN]
[ROtherH, ROtherC, ROtherHSD, ROtherCSD, RSSwsH, RSSwsC, RRefN]=AffineEstSurface4(D, DD, DDD, T, TTT, SelIndx)

%%%Excel Report file%%%%
cd(PathNamRep)
xlswrite(FileNamRep, ROtherH, 'WShutH')
xlswrite(FileNamRep, ROtherC, 'WShutC')
xlswrite(FileNamRep, ROtherHSD, 'WS_SDH')
xlswrite(FileNamRep, ROtherCSD, 'WS_SDC')
xlswrite(FileNamRep, RSSwsH, 'SSwsH')
xlswrite(FileNamRep, RSSwsC, 'SSwsC')
xlswrite(FileNamRep, TTT, 'OtherPointLabels')
TextA=cell(1)
TextA{1}=['Reference brain number']
xlswrite(FileNamRep, TextA, 'Info','A1')
xlswrite(FileNamRep, RRefN, 'Info','A2')
TextB=cell(1)
TextB{1}=['Refference points used']
xlswrite(FileNamRep, TextB, 'Info','A3')
xlswrite(FileNamRep, T, 'Info','A4')



cd(DefaultDir)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 
% ReadFolderPath=uigetdir('', 'Where to save the processed data'); %Result folder name is defined
% SaveFolderPath=uigetdir('', 'Where to save the processed data'); %Result folder name is defined
% 
% cd(strFolderPath)
% Nindex=1
% CurFileN=LstFile(Nindex).name
% [D T]=xlsread(CurFileN)
% ResFileN=['Res' CurFileN]
% ExcelOut={T D}
% 
% cd (SaveFolderPath)
% xlswrite(ResFileN,T,'Data', 'A1')
% xlswrite(ResFileN,D,'Data', 'B1')
% cd(DefaultDir)
