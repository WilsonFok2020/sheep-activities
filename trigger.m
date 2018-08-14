% clear workspace
clc;
clear;
close all;

addpath('Quaternions');
addpath('ximu_matlab_library');
addpath('quaternion_library');      % include quaternion library
addpath('videoSensorPairs');
HOME_DIR = pwd;
addpath(HOME_DIR);


% noise prevention
first_seconds = 10;
last_seconds = 10;

% a directory that holds all the results
%output_dir = fullfile(HOME_DIR, 'output');
%output_dir = fullfile(HOME_DIR, 'control');
output_dir = fullfile(HOME_DIR, 'treated');

% post-op sheep numbers
ss = {3, 5, 13, 25};
% control sheep numbers
%ss = {1,7,19,21};

csvfiles = getSamples(ss, HOME_DIR);
num_files = size(csvfiles);
% use Madgwick filter to process the sample one by one
for k=1:num_files(1,1)
    csvfiles(k).name
	saveName = char(strcat(num2str(csvfiles(k).sheep), {'_'}, csvfiles(k).time))
    [acc, euler] = cal_a_quat(csvfiles(k).name, saveName, output_dir, first_seconds, last_seconds);
    % save the rotations and accelerations into a single mat
    
    f2 = fullfile(output_dir, strcat(saveName, '.mat'));
    save(f2,'acc','euler','-v6') % version is compatible with Python Scipy
    close all
    
end
