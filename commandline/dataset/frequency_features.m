
function frequency_features()

    clear;
    close all;

    run_dir = pwd;

    os_sep = '/';

    datasets   = {'HMP_Dataset', 'LG_Watch_Urbane'};
    patterns   = {'Accelerometer', 'Gyro'};
    classes    = {'by_action', 'by_user'};
    class_dirs = {
        {
            {'Brush_teeth', 'Comb_hair', 'Drink_glass', 'Eat_soup', 'Liedown_bed', 'Sitdown_chair', 'Use_telephone', 'Climb_stairs', 'Descend_stairs', 'Eat_meat', 'Getup_bed', 'Pour_water', 'Standup_chair', 'Walk'},
            {'f1', 'f2', 'f3', 'f4', 'f5', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11'}
        },
        {
            {'cup', 'door', 'typing', 'walking', 'watch'},
            {'andrew', 'chris', 'derrick', 'scott', 'sebastian', 'matt', 'justine', 'jackie', 'jennifer', 'sabrina'}
        }
    };

    for i = 1:size(datasets, 2)
        for j = 1:size(classes, 2)
            for k = 1:size(class_dirs{i}{j}, 2)
                dataset = datasets{i};
                pattern = patterns{i};
                class = classes{j};
                class_dir = class_dirs{i}{j}{k};

                path = strcat(dataset, os_sep, class, os_sep, class_dir);
                cd (path)

                spectral_data_calc(pattern);

                cd (run_dir)
            end
        end
    end

end

function spectral_data_calc( pattern )

    file_list = dir;
    rows = size(file_list, 1);

    filenames = {};
    ptr = 1;
    for index = 1:rows
        found = strfind(file_list(index).name, pattern);
        if found == 1
            filenames{ptr} = file_list(index).name;
            ptr = ptr + 1;
        end
    end

    rows = size(filenames, 2);

    for index = 1:rows
        disp(filenames(index));

        % Import data from file
        [data] = importdata(char(filenames(index)), ' ', 1);
        accel_data = data.data;

        % Open output file
        output_filename = char(strtrim(strcat('spectral_energy_', char(filenames(index)))));
        output_fid = fopen(output_filename, 'w');

        x_axis = accel_data(:,1);
        y_axis = accel_data(:,2);
        z_axis = accel_data(:,3);

        % -- SPECTRAL ENERGY -- %

        L = size(x_axis, 1);

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

        x_result = fft(x_sample,bandwidth);
        P_x = x_result.*conj(x_result)/bandwidth;

        y_result = fft(y_sample,bandwidth);
        P_y = y_result.*conj(y_result)/bandwidth;

        z_result = fft(z_sample,bandwidth);
        P_z = z_result.*conj(z_result)/bandwidth;

        energy_x_sum = sum(P_x(2:bandwidth));
        energy_y_sum = sum(P_y(2:bandwidth));
        energy_z_sum = sum(P_z(2:bandwidth));

        % -- FOURIER COEFFICIENTS -- %

        time = [1:length(x_axis),];
        time = time';

        x_fit = fit(time, x_axis, 'fourier8');
        x_coeff = coeffvalues(x_fit);
        coef_x_sum = sum(x_coeff(1:end-1));

        y_fit = fit(time, y_axis, 'fourier8');
        y_coeff = coeffvalues(y_fit);
        coef_y_sum = sum(y_coeff(1:end-1));

        z_fit = fit(time, z_axis, 'fourier8');
        z_coeff = coeffvalues(z_fit);
        coef_z_sum = sum(z_coeff(1:end-1));

        % -- SPECTRAL ENTROPY -- %

        x_length = length(x_axis);
        x_dft = fft(x_axis);

        s_xdft = sum(abs(x_dft));
        x_dft = x_dft/s_xdft;
        x_entropy = 0;
        for i=1:x_length
            x_entropy = x_entropy + abs(x_dft(i))*log(1/abs(x_dft(i)));
        end

        y_length = length(y_axis);
        y_dft = fft(y_axis);

        s_ydft = sum(abs(y_dft));
        y_dft = x_dft/s_ydft;
        y_entropy = 0;
        for i=1:y_length
            y_entropy = y_entropy + abs(y_dft(i))*log(1/abs(y_dft(i)));
        end

        z_length = length(z_axis);
        z_dft = fft(z_axis);

        s_zdft = sum(abs(z_dft));
        z_dft = z_dft/s_zdft;
        z_entropy = 0;
        for i=1:z_length
            z_entropy = z_entropy + abs(z_dft(i))*log(1/abs(z_dft(i)));
        end

        % Print to file and close
        fprintf(output_fid, '%f %f %f %f %f %f %f %f %f\n', energy_x_sum, energy_y_sum, energy_z_sum, coef_x_sum, coef_y_sum, coef_z_sum, x_entropy, y_entropy, z_entropy);
        fclose(output_fid);
    end

end

