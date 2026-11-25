clc; clear; close all;

%% 
fig = figure('Color', 'none');
set(gca, 'XTick', [], 'YTick', [])
axis off
hold on;

view(-5, 35); 

h_euc  =  0;     
h_disk = .27;    
h_top  = .65;    

color_e = [0.55, 0.65, 0.75];     
color_p = [0.75, 0.65, 0.50];     
color_point = [0.7, 0.7, 0.7];    
point_edge  = [0.45, 0.45, 0.45];  
boundary_color =[0.4, 0.4, 0.4];  

colors = [
    0.2980, 0.4471, 0.6902; 
    0.8667, 0.5176, 0.3216; 
    0.3333, 0.6588, 0.4078; 
    0.7686, 0.3059, 0.3216; 
    0.5059, 0.4471, 0.7020;
    0.5765, 0.4706, 0.3765; 
    0.8549, 0.5451, 0.7647; 
    0.5490, 0.6353, 0.8039; 
    0.8157, 0.7333, 0.4039; 
    0.2353, 0.7020, 0.4431; 
    0.9569, 0.6431, 0.3765; 
    0.3451, 0.3373, 0.5294; 
];

n_clusters = 12;
n_per_cluster = [15,20,25,30,25,20,15,20,25,30,25,20,15,20,25,30,25,20]; 
euc_limit = 1; 

%% 
grid_step = 0.2;
lines = -euc_limit:grid_step:euc_limit;
sampling_density = 20;

box_x = [-euc_limit, euc_limit, euc_limit, -euc_limit];
box_y = [-euc_limit, -euc_limit, euc_limit, euc_limit];
fill3(box_x, box_y, ones(size(box_x))*h_euc-0.005, [1,1,1], ...
    'EdgeColor', boundary_color,'FaceAlpha', 0., 'LineWidth', 1);

h_lines_x = []; h_lines_y = [];
for y = lines
    x_vals = linspace(-euc_limit, euc_limit, sampling_density);
    y_vals = ones(1, sampling_density) * y;
    h_lines_x = [h_lines_x; x_vals];
    h_lines_y = [h_lines_y; y_vals];
end

v_lines_x = []; v_lines_y = [];
for x = lines
    y_vals = linspace(-euc_limit, euc_limit, sampling_density);
    x_vals = ones(1, sampling_density) * x;
    v_lines_x = [v_lines_x; x_vals];
    v_lines_y = [v_lines_y; y_vals];
end

plot3(h_lines_x', h_lines_y', -0.005*ones(size(h_lines_x)), 'Color', [color_e, 0.5], 'LineWidth', 1.5);
plot3(v_lines_x', v_lines_y', -0.005*ones(size(v_lines_x)), 'Color', [color_e, 0.5], 'LineWidth', 1.5);

theta = linspace(0, 2*pi, 300);
fill3(cos(theta), sin(theta), ones(size(theta))*h_disk-0.005, [0.98, 0.98, 0.98], ...
    'EdgeColor', boundary_color, 'FaceAlpha', 0.6, 'LineWidth', 1.5);

lines_val = linspace(-euc_limit, euc_limit, 8);
fine_sample = linspace(-euc_limit, euc_limit, 100);
for v = lines_val
    [hx, hy] = map_exp(fine_sample, ones(size(fine_sample))*v);
    plot3(hx, hy, ones(size(hx))*h_disk, 'Color', [color_p, 0.8], 'LineWidth', 1.5);

    [vx, vy] = map_exp(ones(size(fine_sample))*v, fine_sample);
    plot3(vx, vy, ones(size(vx))*h_disk, 'Color', [color_p, 0.8], 'LineWidth', 1.5);
end

fill3(box_x*1.2, box_y*1.2, ones(size(box_x))*h_top, [1 1 1], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.1);

%% 
cluster_x_all = [];
cluster_y_all = [];

top_nodes = []; 
point_edge = [0.8, 0.8, 0.8];

rng(42); 
for c = 1:n_clusters
    angle_c = (c-1) * 2*pi / n_clusters;
    r_center = 0.7; 

    center_x = r_center * cos(angle_c);
    center_y = r_center * sin(angle_c);

    noise = (rand(n_per_cluster(c), 2) - 0.5) * 0.4; 
    cluster_x = center_x + noise(:,1);
    cluster_y = center_y + noise(:,2);

    scatter3(cluster_x, cluster_y, ones(n_per_cluster(c),1)*h_euc, ...
        100, 'filled', 'MarkerFaceColor', color_point, ...
        'MarkerEdgeColor',point_edge,'MarkerEdgeAlpha', 0.9);

    cluster_x_all = [cluster_x_all; cluster_x];
    cluster_y_all = [cluster_y_all; cluster_y];

    r_vals = sqrt(cluster_x.^2 + cluster_y.^2);
    theta_vals = atan2(cluster_y, cluster_x);

    scale_factor = .9;
    bias = 0.4; 
    r_hyp = tanh(scale_factor * r_vals + bias);

    r_hyp(r_hyp > 0.95) = 0.95; 
    r_hyp(r_hyp < 0.1) = 0.2; 

    [hyp_x, hyp_y] = pol2cart(theta_vals, r_hyp);

    scatter3(hyp_x, hyp_y, ones(n_per_cluster(c),1)*h_disk, ...
        100, 'filled', 'MarkerFaceColor', colors(c,:), 'MarkerEdgeColor',point_edge, ...
        'LineWidth', 1);

    draw_intra_connections(hyp_x, hyp_y, h_disk, colors(c,:), 2, 0.9);

    super_x = mean(hyp_x);
    super_y = mean(hyp_y);
    top_nodes = [top_nodes; super_x, super_y];

    scatter3(super_x, super_y, h_top, 250, 'filled', ...
        'MarkerFaceColor', colors(c,:), 'MarkerEdgeColor', point_edge, 'LineWidth', 1.5);
    idx_sample = 1:3:n_per_cluster(c);
    for k = idx_sample
        plot3([cluster_x(k), hyp_x(k)], [cluster_y(k), hyp_y(k)], [h_euc, h_disk], ...
            ':', 'Color', [[.6,.6,.6], .8],'LineWidth', .7);
    end
    idx_sample = 1:1:n_per_cluster(c);
    for k = idx_sample
        plot3([hyp_x(k), super_x], [hyp_y(k), super_y], [h_disk, h_top], ...
            '-', 'Color', [colors(c,:), .4], 'LineWidth', 1);
    end
end

draw_intra_connections(cluster_x_all(1:10:end), cluster_y_all(1:10:end), h_euc, point_edge, 1.5, 0.9);

%% 
top_nodes = [top_nodes;top_nodes];
num_super = size(top_nodes, 1);
for i = 1:num_super
    for j = i+1:num_super
        is_neighbor = abs(i-j) == 1 || abs(i-j) == 4;
        if is_neighbor
            p1 = top_nodes(i,:);
            p2 = top_nodes(j,:);
            plot3([p1(1), p2(1)], [p1(2), p2(2)], [h_top, h_top], ...
                '-', 'Color', [0.7 0.7 0.7, 0.8], 'LineWidth', 2);
        end
    end
end
zlim([h_euc-.1, h_top+.1]);


function draw_intra_connections(x, y, z, col, line_width, alpha_val)
cx = mean(x); cy = mean(y);
for i = 1:length(x)
    plot3([x(i), cx], [y(i), cy], [z, z], ...
        '-', 'Color', [col, alpha_val], 'LineWidth', 0.5);
end
n = length(x);
for k = 1:n
    neighbor = randi(n);
    plot3([x(k), x(neighbor)], [y(k), y(neighbor)], [z, z], ...
        '-', 'Color', [col, alpha_val], 'LineWidth', line_width);
end
end

function [x_new, y_new] = map_exp(x, y)
r_euc = sqrt(x.^2 + y.^2);
scale_factor = tanh(r_euc) ./ (r_euc + 1e-9); 
scale_factor(r_euc < 1e-9) = 1;
x_new = x .* scale_factor;
y_new = y .* scale_factor;
end