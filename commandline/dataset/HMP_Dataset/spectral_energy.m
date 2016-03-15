function spectral_density()
% filename  = 'Gyro_data_cucd.txt'  
% filename2 = 'Gyro_data_igag.txt'
% filename3 = 'Gyro_data_qsuz.txt'
% filename4 = 'Gyro_data_shnj.txt'
% filename5 = 'Gyro_data_vbcz.txt'
close all;
filenames = [];
file_list = ls;
rows = size(file_list,1);
path = pwd;
folders = strsplit(path, '\');
action = folders(end);
name = folders(end-1);

for index = 1:rows
    found = strfind(file_list(index,:), 'Accelerometer' );
    if( found == 1)
        filenames = [filenames; file_list(index,:)];
    end
end

rows = size(filenames,1);


% x_fig = figure;
% y_fig = figure;
% z_fig = figure;
% 
% x_title = 'spectral x-axis ';
% y_title = 'spectral y-axis ';
% z_title = 'spectral z-axis ';
% 
% x_title = [x_title, name, action];
% y_title = [y_title, name, action];
% z_title = [z_title, name, action];
% 
% x_title = horzcat(x_title);
% y_title = horzcat(y_title);
% z_title = horzcat(z_title);
% 
% xaxis = [1:500];
% colors = ['r', 'g', 'b', 'm', 'k'];

for index = 1:rows
    delimiterIn = ' ';
    outputfile_name = 'spectral_energy_';
    [data] = importdata(filenames(index,:),delimiterIn,1);
    accel_data = data.data;
    display(length(accel_data));
    
    outputfile_name = strcat(outputfile_name, filenames(index,:));
    %outputfile_name = strtrim(outputfile_name);
    outputfile_name = char(outputfile_name);
    outputfile = fopen(outputfile_name, 'w');

    x_axis = accel_data(:,1);
    y_axis = accel_data(:,2);
    z_axis = accel_data(:,end);

    L = size(x_axis);
    L = L(1);
    
    bandwidth = (2^(nextpow2(L) + 1))/2;
    
    x_sample = zeros(bandwidth,1);
    y_sample = zeros(bandwidth,1);
    z_sample = zeros(bandwidth,1);
    
    x_sample(1:L,1) = x_axis;
    y_sample(1:L,1) = y_axis;
    z_sample(1:L,1) = z_axis;
    
    Fs = 32;
    T = 1/Fs;
    
    t = (0:L-1)*T;
    %f = Fs/L*(0:255);
    
    x_result = fft(x_sample,bandwidth);
    P_x = x_result.*conj(x_result)/bandwidth;
    
    y_result = fft(y_sample,bandwidth);
    P_y = y_result.*conj(y_result)/bandwidth;
    
    z_result = fft(z_sample,bandwidth);
    P_z = z_result.*conj(z_result)/bandwidth;
    
%     figure(x_fig);
%     title(x_title);
%     plot(f, P_x(1:256), colors(index));
%     hold on;
% 
%     figure(y_fig);
%     title(y_title);
%     plot(f, P_y(1:256), colors(index));
%     hold on;
% 
%     figure(z_fig);
%     title(z_title);
%     plot(f, P_z(1:256), colors(index));
%     hold on;
    
    x_sum = sum(P_x(2:bandwidth));
    y_sum = sum(P_y(2:bandwidth));
    z_sum = sum(P_z(2:bandwidth));
    
    fprintf(outputfile, '%f %f %f\n', x_sum, y_sum, z_sum);
    fclose(outputfile);
   
end

% figure(x_fig);
% hold off;
% figure(y_fig);
% hold off;
% figure(z_fig);
% hold off;

