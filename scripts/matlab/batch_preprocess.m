files = transpose(dir('data/*.csv'))
for i = 1:size(files, 2)
    disp([i, size(files, 2)]);
    data = data_conversion([files(1, i).folder, '\', files(1, i).name]);
    nirs_data = preprocess(data);
    save([files(1, i).name, '.mat'], 'nirs_data');
    struct2csv(nirs_data, files(1, i).name);
end