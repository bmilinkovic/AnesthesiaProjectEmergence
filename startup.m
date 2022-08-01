global emergenceDir;
emergenceDir = fileparts(mfilename('fullpath'));
addpath(emergenceDir);

addpath(fullfile(emergenceDir, 'preprocessing'));
addpath(genpath(fullfile(emergenceDir, 'mvgcfuncs')));
addpath(fullfile(emergenceDir, 'results'));
addpath(fullfile(emergenceDir, 'simulation'));
addpath(fullfile(emergenceDir, 'src'));
addpath(fullfile(emergenceDir, 'utils'));
addpath(fullfile(emergenceDir, 'test'));

global ssdiDir;
ssdiDir = '/Users/borjanmilinkovic/Documents/gitdir/ssdi';
addpath((ssdiDir));
cd(ssdiDir);
startup;
cd(emergenceDir);
fprintf('[Emergence Pipeline startup] Added path to State-Space Dynamical Indpendence toolbox: %s\n',ssdiDir);

global ceDir;
ceDir = '/Users/borjanmilinkovic/Documents/gitdir/ReconcilingEmergences';
addpath(genpath(ceDir));
fprintf('[Emergence Pipeline startup] Added path to Strong Causal Emergence toolbox: %s\n',ceDir);





