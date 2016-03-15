close all;
filenames = [];
file_list = ls;
rows = size(file_list,1);
path = pwd;
folders = strsplit(path, '\');
action = folders(end);
name = folders(end-1);

for index = 1:rows
    found = strfind(file_list(index,:), 'Gyro_data' );
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
    outputfile_name = strtrim(outputfile_name);
    outputfile_name = char(outputfile_name);
    outputfile = fopen(outputfile_name, 'w');

    x_axis = accel_data(:,1);
    y_axis = accel_data(:,2);
    z_axis = accel_data(:,end);

    z_axis = z_axis(1:500);
    x_axis = x_axis(1:500);
    y_axis = y_axis(1:500); 
    
    Fs = 100;
    T = 1/Fs;
    L = 500;
    t = (0:L-1)*T;
    f = 100/L*(0:255);
    
    x_result = fft(x_axis,L);
    P_x = x_result.*conj(x_result)/L;
    
    y_result = fft(y_axis,L);
    P_y = y_result.*conj(y_result)/L;
    
    z_result = fft(z_axis,L);
    P_z = z_result.*conj(z_result)/L;
    
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
    
    x_sum = sum(P_x(2:256));
    y_sum = sum(P_y(2:256));
    z_sum = sum(P_z(2:256));
    
    fprintf(outputfile, '%f %f %f\n', x_sum, y_sum, z_sum);
    fclose(outputfile);
%     P2_X = abs(x_result/L);
%     P1_X = P2_X(1:L/2+1);
%     P1_X(2:end-1) = 2*P1_X(2:end-1);
%     
%     y_result = fft(y_axis);
%     P2_Y = abs(y_result/L);
%     P1_Y = P2_Y(1:L/2+1);
%     P1_Y(2:end-1) = 2*P1_Y(2:end-1);
%     
%     z_result = fft(z_axis);
%     P2_Z = abs(z_result/L);
%     P1_Z = P2_Z(1:L/2+1);
%     P1_Z(2:end-1) = 2*P1_Z(2:end-1);
%     
%     freq_axis = Fs*(0:(L/2))/L;
% 
%     figure(x_fig);
%     title(x_title);
%     plot(freq_axis, P1_X, colors(index));
%     hold on;
%     
%     figure(y_fig);
%     title(y_title);
%     plot(freq_axis, P1_Y, colors(index));
%     hold on;
%     
%     figure(z_fig);
%     title(z_title);
%     plot(freq_axis, P1_Z, colors(index));
%     hold on;
   
end

% figure(x_fig);
% hold off;
% figure(y_fig);
% hold off;
% figure(z_fig);
% hold off;

