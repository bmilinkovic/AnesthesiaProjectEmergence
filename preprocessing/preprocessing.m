%% First get the list of files from our Raw data directory and set up our ouput directory of our preprocessing

files = dir([path_rawdata filesep '*.mat']);
output_dir = path_preproc;

if exist(output_dir) == 0
    mkdir(output_dir)
end

%% Start preprocessing: Looping over all subjects.

segment = 1;
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





    
    

