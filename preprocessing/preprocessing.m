
% Set all paths to external dependencies this could be done in my startup
% file.

path_fieldtrip = '/Users/borjanmilinkovic/Documents/toolboxes/fieldtrip';
path_LSCPtools = '/Users/borjanmilinkovic/Documents/toolboxes/LSCPtools';
path_rawdata = '/Volumes/dataSets/restEEGHealthySubjects/rawData';
path_preproc = '/Volumes/dataSets/restEEGHealthySubjects/preprocessedData';


addpath(path_fieldtrip);
ft_defaults;
addpath(genpath(path_LSCPtools));

files = dir([path_rawdata filesep '*.mat']);
output_dir = path_preproc;

if exist(output_dir) == 0
    mkdir(output_dir)
end

% looping over all subjects here.

segmet = 1;
for nfiles = 1:length(files)
    filename = files(nfiles).name;
    foldername = files(nfiles).folder;
    subID = filename(1:end-4);
    
    tic;
    fprintf('...working on %s (%g/%g) \n', filename, nfiles, length(files))
    
    % set up fieldtrip configuration
    
    cfg = [];
    cfg.subID = subID;
    cfg.dataset = [foldername filesep filename];
    cfg.trialdef.lengthSegments = 2;
    cfg = ft_definetrial(cfg);
end

    
    
    

