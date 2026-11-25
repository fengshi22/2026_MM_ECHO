clc; clear; close all;

%%
fig = figure('Color', 'w', 'Position',[100, 50, 900, 800]); 
set(gca, 'XTick', [], 'YTick',[])
axis off
hold on;
view(-45, 30); 

h_euc  =  0;      
h_top  = .3;    

color_e =[0.55, 0.65, 0.75];    
point_edge  = [0.45, 0.45, 0.45];
boundary_color =[0.4, 0.4, 0.4]; 

colors =[
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
n_per_cluster =[15,20,25,30,25,20,15,20,25,30,25,20]; 
euc_limit = 1;  

%% 
grid_step = 0.2;
lines = -euc_limit:grid_step:euc_limit;
sampling_density = 20;

box_x =[-euc_limit, euc_limit, euc_limit, -euc_limit];
box_y =[-euc_limit, -euc_limit, euc_limit, euc_limit];
fill3(box_x, box_y, ones(size(box_x))*h_euc-0.005, [1,1,1], ...
    'EdgeColor', boundary_color,'FaceAlpha', 0, 'LineWidth', 1);

h_lines_x = []; h_lines_y = [];
for y = lines
    h_lines_x =[h_lines_x; linspace(-euc_limit, euc_limit, sampling_density)];
    h_lines_y = [h_lines_y; ones(1, sampling_density) * y];
end
v_lines_x = []; v_lines_y =[];
for x = lines
    v_lines_x = [v_lines_x; ones(1, sampling_density) * x];
    v_lines_y =[v_lines_y; linspace(-euc_limit, euc_limit, sampling_density)];
end

plot3(h_lines_x', h_lines_y', -0.005*ones(size(h_lines_x)), 'Color',[color_e, 0.5], 'LineWidth', 1.5);
plot3(v_lines_x', v_lines_y', -0.005*ones(size(v_lines_x)), 'Color', [color_e, 0.5], 'LineWidth', 1.5);

fill3(box_x*1.2, box_y*1.2, ones(size(box_x))*h_top, [1 1 1], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.05);

%% 
top_nodes =[]; 

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
        80, 'filled', 'MarkerFaceColor', colors(c,:), ...
        'MarkerEdgeColor', point_edge, 'MarkerEdgeAlpha', 0.9);

    draw_intra_connections(cluster_x, cluster_y, h_euc, colors(c,:), 1.0, 0.5);
    super_x = mean(cluster_x);
    super_y = mean(cluster_y);
    top_nodes =[top_nodes; super_x, super_y];

    scatter3(super_x, super_y, h_top, 250, 'filled', ...
        'MarkerFaceColor', colors(c,:), 'MarkerEdgeColor', point_edge, 'LineWidth', 1.5);

    for k = 1:n_per_cluster(c)
        plot3([cluster_x(k), super_x], [cluster_y(k), super_y],[h_euc, h_top], ...
            '-', 'Color', [colors(c,:), 0.35], 'LineWidth', 1);
    end
end

%% 
top_nodes = [top_nodes;top_nodes];
num_super = size(top_nodes, 1);
for i = 1:num_super
    for j = i+1:num_super
        is_neighbor = abs(i-j) == 1 || abs(i-j) == 4 || abs(i-j) == num_super-1;
        
        if is_neighbor
            p1 = top_nodes(i,:);
            p2 = top_nodes(j,:);
            plot3([p1(1), p2(1)],[p1(2), p2(2)], [h_top, h_top], ...
                '-', 'Color',[0.7 0.7 0.7, 0.8], 'LineWidth', 2);
        end
    end
end
zlim([h_euc-.1, h_top+.1]);

function draw_intra_connections(x, y, z, col, line_width, alpha_val)
    cx = mean(x); cy = mean(y);
    for i = 1:length(x)
        plot3([x(i), cx], [y(i), cy],[z, z], ...
            '-', 'Color', [col, alpha_val], 'LineWidth', 0.5);
    end
    n = length(x);
    for k = 1:n
        neighbor = randi(n);
        plot3([x(k), x(neighbor)], [y(k), y(neighbor)], [z, z], ...
            '-', 'Color', [col, alpha_val], 'LineWidth', line_width);
    end
end