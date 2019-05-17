function varargout = fMRI_Results_Viewer(varargin)

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @fMRI_Results_Viewer_OpeningFcn, ...
    'gui_OutputFcn',  @fMRI_Results_Viewer_OutputFcn, ...
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
% End initialization code - DO NOT EDIT


% --- Executes just before fMRI_Results_Viewer is made visible.
function fMRI_Results_Viewer_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;

% Update handles structure
guidata(hObject, handles);


global fmri_result_path
cur_dir = cd;
cd(fmri_result_path);
load fmri_result;

cd(cur_dir);

leng = length(fmri_result)-1;

set(handles.edit_pvalue,'string',fmri_result{1}.thresDesc);
set(handles.edit_spmdir,'string',fmri_result{1}.swd);

axes(handles.axes1)
load Split;
colormap(split);
if isempty(eventdata) == 1
    set(handles.slider1,'sliderstep',[1/leng,1/leng],'max',leng,'min',0 ,'Value',3);
    image(fmri_result{4}.img*64);
elseif isempty(eventdata) == 0
    if eventdata > 99

        set(handles.checkbox1,'value',0);
        set(handles.slider1,'enable','on');
        set(handles.slider1,'visible','on');
        set(handles.slider1,'sliderstep',[1/leng,1/leng],'max',leng,'min',0 ,'Value',eventdata-100);
        image(fmri_result{eventdata-99}.img*64);

    else

        set(handles.checkbox1,'value',1);
        set(handles.slider1,'enable','off');
        set(handles.slider1,'visible','off');
        set(handles.slider1,'sliderstep',[1/leng,1/leng],'max',leng,'min',0 ,'Value',eventdata);
        image(fmri_result{eventdata+1}.img*64);

    end
end
axis off
axis image

% --- Outputs from this function are returned to the command line.
function varargout = fMRI_Results_Viewer_OutputFcn(hObject, eventdata, handles)
% Get default command line output from handles structure
varargout{1} = handles.output;
varargout{2} = handles;


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)

sli_value = round(get(handles.slider1,'value'));
set(handles.slider1,'value',sli_value);

global fmri_result_path
cur_dir = cd;
cd(fmri_result_path);
load fmri_result;
cd(cur_dir);

axes(handles.axes1)
load Split;
colormap(split);

image(fmri_result{sli_value+1}.img*64);
axis off

% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function edit_spmdir_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_spmdir_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_pvalue_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function edit_pvalue_CreateFcn(hObject, eventdata, handles)
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


