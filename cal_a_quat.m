function [acc, euler] = cal_a_quat(filePath, saveName, output_dir, first_seconds, last_seconds)
% This function calculates rotation referenced by pitch, yaw and roll (three Euler angles) and linear accelerations (x,y,z) based on Madgwick filter. 

% Madgwick filter optimizes the direction of the gyroscope measurements based on measurements from linear accelerometer and magnetometer because it uses quaternion representation so the relationship of the three can be analytically derived. The optimization algorithm chosen in the paper is gradient descent algorithm. Its benefits over other existing orientation estimation methods are speed, accuracy and lack of parameter tuning.

% The filter is publicly available at http://x-io.co.uk/open-source-imu-and-ahrs-algorithms/.
% S. Madgwick, “An efficient orientation filter for inertial and inertial/magnetic sensor arrays,” Report x-io and University of Bristol (UK), vol. 25, 2010

% filePath = full csv file path
% output_dir = save to whatever directory
% first_seconds = remove noise that starts at time (in second)
% last_seconds = remove noise that ends at time (in second)


% skip header
M = csvread(filePath,1,0);

% first column is time
time = M(:,1);
max_time = M(end,1);
% remove samples due to noise at the beginning and end of the recording
start_idx = find( time > first_seconds, 1, 'first');
end_idx = find(time > (max_time - last_seconds), 1, 'first');
% assign values from columns
Gyroscope = M(start_idx:end_idx,5:7);
Accelerometer = M(start_idx:end_idx,2:4);
Magnetometer = M(start_idx:end_idx,8:10);

time = M(start_idx:end_idx, 1);

samplePeriod = M(2,1) - M(1,1);

h = figure('Name', 'Sensor Data');
axis(1) = subplot(3,1,1);
hold on;
plot(time, Gyroscope(:,1), 'r');
plot(time, Gyroscope(:,2), 'g');
plot(time, Gyroscope(:,3), 'b');
legend('X', 'Y', 'Z');
xlabel('Time (s)');
ylabel('Angular rate (deg/s)');
title('Gyroscope');
hold off;
axis(2) = subplot(3,1,2);
hold on;
plot(time, Accelerometer(:,1), 'r');
plot(time, Accelerometer(:,2), 'g');
plot(time, Accelerometer(:,3), 'b');
legend('X', 'Y', 'Z');
xlabel('Time (s)');
ylabel('Acceleration (m/s/s)');
title('Accelerometer');
hold off;
axis(3) = subplot(3,1,3);
hold on;
plot(time, Magnetometer(:,1), 'r');
plot(time, Magnetometer(:,2), 'g');
plot(time, Magnetometer(:,3), 'b');
legend('X', 'Y', 'Z');
xlabel('Time (s)');
ylabel('Flux (G)');
title('Magnetometer');
hold off;
linkaxes(axis, 'x');
f = fullfile(output_dir, strcat(saveName, 'raw.fig'));
savefig(h, f, 'compact')
% 

%% Process sensor data through algorithm
AHRS = MadgwickAHRS('SamplePeriod', samplePeriod, 'Beta', 0.1);


quaternion = zeros(length(time), 4);
for t = 1:length(time)
    AHRS.Update(Gyroscope(t,:) * (pi/180), Accelerometer(t,:), Magnetometer(t,:));	% gyroscope units must be radians
    quaternion(t, :) = AHRS.Quaternion;
end

%% Plot algorithm output as Euler angles
% The first and third Euler angles in the sequence (phi and psi) become
% unreliable when the middle angles of the sequence (theta) approaches �90
% degrees. This problem commonly referred to as Gimbal Lock.
% See: http://en.wikipedia.org/wiki/Gimbal_lock

euler = quatern2euler(quaternConj(quaternion)) * (180/pi);	% use conjugate for sensor frame relative to Earth and convert to degrees.

h = figure('Name', 'Euler Angles');
hold on;
plot(time, euler(:,1), 'r');
plot(time, euler(:,2), 'g');
plot(time, euler(:,3), 'b');
title('Euler angles');
xlabel('Time (s)');
ylabel('Angle (deg)');
legend('\phi', '\theta', '\psi');
hold off;
f = fullfile(output_dir, strcat(saveName, 'euler.fig'));
savefig(h, f, 'compact')


% -------------------------------------------------------------------------
%% Compute translational accelerations

% Rotate body accelerations to Earth frame
acc = quaternRotate([Accelerometer(:,1) Accelerometer(:,2) Accelerometer(:,3)], quaternConj(quaternion));

% Plot translational accelerations
h = figure('Name', 'translational/ Earth accelerations' );
hold on;
plot(time, acc(:,1), 'r');
plot(time, acc(:,2), 'g');
plot(time, acc(:,3), 'b');
title('Acceleration');
xlabel('Time (s)');
ylabel('Acceleration (m/s/s)');
legend('X', 'Y', 'Z');
hold off;
f = fullfile(output_dir, strcat(saveName, 'acc.fig'));
savefig(h, f, 'compact');
end
