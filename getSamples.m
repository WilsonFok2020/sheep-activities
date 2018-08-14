function tables = getSamples(ss, HOME_DIR)
%% get the treated sheep
table = struct();
tables = repmat(table, 5,1);
counter = 1;
for s=1:length(ss)
    
    sheepNum = char(strcat('Sheep ',{' '},num2str(cell2mat(ss(s)))));
	subfolder = fullfile(HOME_DIR,sheepNum)
    d = dir(subfolder)
    
    for j=4:length(d)
        path = fullfile(subfolder, d(j).name);
        time = d(j).name(1) ;
        content = dir(path);
        csv = content(3).name;
        [filePath, name, ext] = fileparts(csv);
        if strcmp(ext, '.csv')
            csvPath = fullfile(path, content(3).name);
            tables(counter).sheep = cell2mat(ss(s));
            tables(counter).name = csvPath;
            tables(counter).time = time;
            counter = counter + 1;

        end
    end
end
end

