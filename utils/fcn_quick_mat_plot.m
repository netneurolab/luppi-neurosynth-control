function[h] = fcn_quick_mat_plot(mat, optional_title_string, optional_col_names, optional_row_names);
% This function plots a matrix as heatmap, with optional title and X-axis and/or Y-axis labels
%
% INPUTS:
% mat: a N-by-M array (double); example: rand(10,2);
% optional_title_string: a string for the title; Optional: can leave empty
% optional_col_names, optional_row_names: cell arrays of strings; for example,
% {'First row', 'second row'}; optional: can leave empty
%
% example use: fcn_quick_mat_plot(rand(2,100), '', '', {'First row', 'second row'})
%
% if ENIGMA Toolbox (https://github.com/MICA-MNI/ENIGMA.git) is present
% on the MATLAB path, this function will try to plot red-white-blue colormap
% (or red-only if exclusively positive, and blue-only if exclusively negative)

h=figure; imagesc(mat);

try
    colormap(rdbu_sym(h)); colorbar;
end

if exist('optional_title_string', 'var')
    if not(isempty(optional_title_string))
        title(optional_title_string);
    end
end


if exist('optional_col_names', 'var') && not(isempty(optional_col_names))
    set(gca, 'xtick', 1:numel(optional_col_names), 'xticklabel',optional_col_names, 'xticklabelrotation', 25);
end

if exist('optional_row_names', 'var') && not(isempty(optional_row_names))
    set(gca, 'ytick', 1:numel(optional_row_names), 'yticklabel',optional_row_names);
end
